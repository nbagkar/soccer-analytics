"""Monte Carlo league simulation.

Plays a season out many times by sampling each remaining fixture's scoreline from the
Poisson model, then tallies the distribution of final tables -- title, top-N and
relegation probabilities, plus expected points. Vectorised with numpy so ten thousand
runs of a full fixture list finish in well under a second.

Two framings the CLI exposes:
* Season replay (no cutoff): every fixture unplayed, standings start at zero -- a
  preseason projection given the fitted strengths.
* Rest of season (cutoff date): matches before the cutoff set the current table and the
  fitted strengths; matches on/after it are simulated.

Rating uncertainty: a point-estimate rating replayed ten thousand times is overconfident --
the ratings themselves are uncertain (transfers, promoted sides, plain estimation noise),
most of all before a ball is kicked. `rating_noise` draws one log-normal shock per team per
simulated season, and `season_rating_noise` sets how big it is from how much of the season
is already played (measured -- see SEASON_NOISE_START).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np


class ScorelineModel(Protocol):
    """Structural type for anything `simulate_season` can run on -- PoissonModel and
    DixonColesModel both satisfy it; only `expected_goals` is actually used here."""

    def expected_goals(self, home: str, away: str) -> tuple[float, float]: ...


class Outcome(Protocol):
    """A played result: who played whom, and the score."""

    @property
    def home_norm(self) -> str: ...
    @property
    def away_norm(self) -> str: ...
    @property
    def fthg(self) -> int: ...
    @property
    def ftag(self) -> int: ...


# Rating-uncertainty schedule: noise sd = START * (1 - fraction played) ** POWER. Backtested
# 2026-10-06 on 2005/06-2025/26, projecting at 0 / 25 / 50 / 75% of each season and scoring
# title / top-4 / relegation against the real final tables (`season_backtest`). Tuned on 13
# leagues (4,963 team-seasons): summed log loss pre-season 0.893 -> 0.830, quarter-way 0.608
# -> 0.606, neutral from halfway (no noise is already calibrated there). Confirmed untuned on
# E2/E3/I2/F2 (1,763): pre-season 1.354 -> 1.099. Without it, a pre-season "61% top four"
# came true 45% of the time; flat noise of any size hurt from halfway on, hence the fade.
#
# Per club, not flat (later 2026-10-06): the same doubt for everyone left established top
# clubs UNDER-confident (pre-season title calls of 50%+ averaged 0.60 and came true 0.76),
# while a promoted side really is uncertain. Doubt now shrinks with the matches behind the
# rating: sd = START * sqrt(HISTORY / (HISTORY + games in the fitting window)) -- ~0.16 for
# a club with three seasons in the division, the full 0.4 for one with none. Pre-season
# summed log loss 0.828 -> 0.814 (13 tuning leagues), 1.112 -> 1.079 (untuned E2/E3/I2/F2);
# title 50%+ calls now 0.63 predicted / 0.67 actual. Cost: relegation log loss slightly worse
# (0.370 -> 0.376 tuning, 0.410 -> 0.445 untuned) -- newcomers carry more doubt. Capping the
# doubt at the old 0.25 gave the gain straight back. Neutral from a quarter played on.
SEASON_NOISE_START = 0.4
SEASON_NOISE_HISTORY = 20.0
SEASON_NOISE_POWER = 2.0


def season_rating_noise(fraction_played: float, games: float = 0.0) -> float:
    """A club's rating noise (log-scale sd), `fraction_played` (0-1) into the season, with
    `games` matches behind its rating in the fitting window."""
    remaining = min(max(1.0 - fraction_played, 0.0), 1.0)
    history = math.sqrt(SEASON_NOISE_HISTORY / (SEASON_NOISE_HISTORY + max(games, 0.0)))
    return SEASON_NOISE_START * history * math.pow(remaining, SEASON_NOISE_POWER)


def games_by_team(rows: Iterable[Outcome]) -> dict[str, int]:
    """Matches each club played in `rows` -- the history behind its rating."""
    games: dict[str, int] = {}
    for o in rows:
        games[o.home_norm] = games.get(o.home_norm, 0) + 1
        games[o.away_norm] = games.get(o.away_norm, 0) + 1
    return games


def standings(played: Iterable[Outcome]) -> tuple[dict[str, int], dict[str, int]]:
    """(points, goal difference) per normalized team from results already played."""
    points: dict[str, int] = {}
    goal_diff: dict[str, int] = {}
    for o in played:
        hp = 3 if o.fthg > o.ftag else 1 if o.fthg == o.ftag else 0
        ap = 3 if o.ftag > o.fthg else 1 if o.fthg == o.ftag else 0
        points[o.home_norm] = points.get(o.home_norm, 0) + hp
        points[o.away_norm] = points.get(o.away_norm, 0) + ap
        goal_diff[o.home_norm] = goal_diff.get(o.home_norm, 0) + (o.fthg - o.ftag)
        goal_diff[o.away_norm] = goal_diff.get(o.away_norm, 0) + (o.ftag - o.fthg)
    return points, goal_diff


def remaining_round_robin(
    teams: Iterable[str], played: Iterable[Outcome]
) -> tuple[list[tuple[str, str]], float]:
    """(unplayed home/away pairs of a double round-robin, fraction of it already played).

    Every team meets every other once at home and once away; a pair that has been played
    is done. Leagues that add a split or extra rounds are approximated by the double
    round-robin -- the same assumption the season projections have always made.
    """
    roster = sorted(set(teams))
    done = {(o.home_norm, o.away_norm) for o in played}
    pairs = [(h, a) for h in roster for a in roster if h != a]
    remaining = [p for p in pairs if p not in done]
    fraction = 1.0 - len(remaining) / len(pairs) if pairs else 0.0
    return remaining, fraction


@dataclass(frozen=True)
class TeamProjection:
    team: str
    title_pct: float
    top_pct: float
    relegation_pct: float
    expected_points: float
    avg_position: float
    points_low: float = 0.0
    """10th percentile of simulated final points -- with `points_high`, an 80% range."""
    points_high: float = 0.0


@dataclass(frozen=True)
class SimulationResult:
    n_sims: int
    top_n: int
    relegation: int
    projections: list[TeamProjection]
    """Sorted by title probability, then average finishing position."""


def simulate_season(
    model: ScorelineModel,
    remaining: list[tuple[str, str]],
    *,
    points_start: dict[str, int] | None = None,
    goal_diff_start: dict[str, int] | None = None,
    teams: list[str] | None = None,
    n_sims: int = 10_000,
    top_n: int = 4,
    relegation: int = 3,
    seed: int | None = None,
    rating_noise: float | Mapping[str, float] = 0.0,
) -> SimulationResult:
    """Simulate the remaining fixtures `n_sims` times and summarise final tables.

    Team keys are normalized names (matching the model). `remaining` is (home, away)
    pairs. Teams are the union of `teams`, the starting standings, and the fixtures.
    `rating_noise` (log-scale sd, 0 = off; one number for every team or a per-team mapping)
    gives each team one attack and one defence shock per simulated season, so the spread of
    outcomes includes doubt about the ratings themselves, not just match luck -- see
    `season_rating_noise`.
    """
    points_start = points_start or {}
    goal_diff_start = goal_diff_start or {}

    roster = set(teams or [])
    roster.update(points_start)
    for home, away in remaining:
        roster.update((home, away))
    order = sorted(roster)
    if not order:
        raise ValueError("no teams to simulate")
    index = {team: i for i, team in enumerate(order)}
    n_teams = len(order)

    rng = np.random.default_rng(seed)

    # Per-fixture rates and the team columns they credit.
    home_idx = np.array([index[h] for h, _ in remaining], dtype=np.intp)
    away_idx = np.array([index[a] for _, a in remaining], dtype=np.intp)
    lam = np.array([model.expected_goals(h, a)[0] for h, a in remaining])
    mu = np.array([model.expected_goals(h, a)[1] for h, a in remaining])
    if isinstance(rating_noise, Mapping):
        sd = np.array([rating_noise.get(team, 0.0) for team in order])
    else:
        sd = np.full(n_teams, float(rating_noise))
    if remaining and np.any(sd > 0):
        # attack scales a team's own goals; defence scales what its opponents score
        attack = np.exp(rng.normal(0.0, 1.0, (n_sims, n_teams)) * sd[None, :])
        defence = np.exp(rng.normal(0.0, 1.0, (n_sims, n_teams)) * sd[None, :])
        lam = lam[None, :] * attack[:, home_idx] * defence[:, away_idx]
        mu = mu[None, :] * attack[:, away_idx] * defence[:, home_idx]

    points = np.zeros((n_sims, n_teams))
    goal_diff = np.zeros((n_sims, n_teams))
    for team, pts in points_start.items():
        points[:, index[team]] = pts
    for team, gd in goal_diff_start.items():
        goal_diff[:, index[team]] = gd

    if remaining:
        home_goals = rng.poisson(lam, size=(n_sims, len(remaining)))
        away_goals = rng.poisson(mu, size=(n_sims, len(remaining)))
        home_pts = np.where(home_goals > away_goals, 3, np.where(home_goals == away_goals, 1, 0))
        away_pts = np.where(away_goals > home_goals, 3, np.where(home_goals == away_goals, 1, 0))
        margin = home_goals - away_goals

        # Credit each fixture to its two teams. A short loop over fixtures, each step a
        # vectorised column add over all simulations.
        for f in range(len(remaining)):
            points[:, home_idx[f]] += home_pts[:, f]
            points[:, away_idx[f]] += away_pts[:, f]
            goal_diff[:, home_idx[f]] += margin[:, f]
            goal_diff[:, away_idx[f]] -= margin[:, f]

    # Rank by points, then goal difference. points dominate; gd is bounded well under
    # the multiplier, so a single sort key orders the table correctly.
    key = points * 10_000 + goal_diff
    best_first = np.argsort(-key, axis=1)
    position = np.empty_like(best_first)
    np.put_along_axis(position, best_first, np.arange(1, n_teams + 1), axis=1)

    title = (position == 1).mean(axis=0)
    top = (position <= top_n).mean(axis=0)
    releg = (position > n_teams - relegation).mean(axis=0)
    exp_points = points.mean(axis=0)
    low, high = np.percentile(points, [10, 90], axis=0)
    avg_pos = position.mean(axis=0)

    projections = [
        TeamProjection(
            team=order[i],
            title_pct=float(title[i]),
            top_pct=float(top[i]),
            relegation_pct=float(releg[i]),
            expected_points=float(exp_points[i]),
            avg_position=float(avg_pos[i]),
            points_low=float(low[i]),
            points_high=float(high[i]),
        )
        for i in range(n_teams)
    ]
    projections.sort(key=lambda p: (-p.title_pct, p.avg_position))
    return SimulationResult(
        n_sims=n_sims, top_n=top_n, relegation=relegation, projections=projections
    )
