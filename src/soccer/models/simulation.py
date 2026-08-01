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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from soccer.models.poisson import PoissonModel


@dataclass(frozen=True)
class TeamProjection:
    team: str
    title_pct: float
    top_pct: float
    relegation_pct: float
    expected_points: float
    avg_position: float


@dataclass(frozen=True)
class SimulationResult:
    n_sims: int
    top_n: int
    relegation: int
    projections: list[TeamProjection]
    """Sorted by title probability, then average finishing position."""


def simulate_season(
    model: PoissonModel,
    remaining: list[tuple[str, str]],
    *,
    points_start: dict[str, int] | None = None,
    goal_diff_start: dict[str, int] | None = None,
    teams: list[str] | None = None,
    n_sims: int = 10_000,
    top_n: int = 4,
    relegation: int = 3,
    seed: int | None = None,
) -> SimulationResult:
    """Simulate the remaining fixtures `n_sims` times and summarise final tables.

    Team keys are normalized names (matching the model). `remaining` is (home, away)
    pairs. Teams are the union of `teams`, the starting standings, and the fixtures.
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
    avg_pos = position.mean(axis=0)

    projections = [
        TeamProjection(
            team=order[i],
            title_pct=float(title[i]),
            top_pct=float(top[i]),
            relegation_pct=float(releg[i]),
            expected_points=float(exp_points[i]),
            avg_position=float(avg_pos[i]),
        )
        for i in range(n_teams)
    ]
    projections.sort(key=lambda p: (-p.title_pct, p.avg_position))
    return SimulationResult(
        n_sims=n_sims, top_n=top_n, relegation=relegation, projections=projections
    )
