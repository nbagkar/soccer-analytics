"""Backtest the season projections: title / top-N / relegation odds vs real final tables.

The single-match forecasts are scored against the closing line every matchday; the season
projections (what a league's title race "looks like") had never been checked at all. This
replays past seasons: at each checkpoint (a fraction of the season's matches played) it fits
the ratings on only what had been played by then, projects the final table from the banked
points exactly as the app does, and scores the probabilities against what actually happened.

A no-skill baseline -- every team rated league-average, same banked points -- shows how much
the ratings add beyond the table itself (late in a season the table does most of the work).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from soccer.models.simulation import (
    ScorelineModel,
    remaining_round_robin,
    season_rating_noise,
    simulate_season,
    standings,
)

EVENTS = ("title", "top", "relegation")


@dataclass(frozen=True)
class EventScore:
    log_loss: float
    brier: float


@dataclass(frozen=True)
class CheckpointScore:
    checkpoint: float
    """Fraction of each season's matches already played when the projection was made."""
    n_team_seasons: int
    model: dict[str, EventScore]  # keyed by EVENTS
    baseline: dict[str, EventScore]  # every team league-average, same banked points
    points_rmse: float
    baseline_points_rmse: float

    @property
    def skill(self) -> float:
        """1 - model/baseline summed log loss: the share of the baseline's error removed."""
        model = sum(s.log_loss for s in self.model.values())
        base = sum(s.log_loss for s in self.baseline.values())
        return 1.0 - model / base if base else 0.0


@dataclass(frozen=True)
class SeasonBacktest:
    seasons: list[str]
    checkpoints: list[CheckpointScore]


class _Flat:
    """League-average for everyone: home and away goal rates from the fitting window."""

    def __init__(self, home: float, away: float) -> None:
        self._rates = (home, away)

    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        return self._rates


def final_positions(rows: Sequence[Any]) -> tuple[dict[str, int], dict[str, int]]:
    """(position, points) per team from a season's full results, ranked by points then GD."""
    points, goal_diff = standings(rows)
    order = sorted(points, key=lambda t: (-points[t], -goal_diff[t], t))
    return {t: i + 1 for i, t in enumerate(order)}, points


def _event_score(pairs: list[tuple[float, bool]]) -> EventScore:
    n = len(pairs)
    clip = 1e-4  # a 0% call that happens costs log(1e-4), not infinity
    log_loss = -sum(math.log(max(p if hit else 1.0 - p, clip)) for p, hit in pairs) / n
    brier = sum((p - hit) ** 2 for p, hit in pairs) / n
    return EventScore(log_loss=log_loss, brier=brier)


def backtest_season_projections(
    seasons: Sequence[tuple[str, Sequence[Any]]],
    fit: Callable[[list[Any], list[str]], ScorelineModel],
    *,
    checkpoints: Sequence[float] = (0.0, 0.25, 0.5, 0.75),
    history_seasons: int = 2,
    n_sims: int = 4000,
    top_n: int = 4,
    relegation: int = 3,
    seed: int = 1,
) -> SeasonBacktest | None:
    """Score projections made part-way through each season against its final table.

    `seasons` is (season code, results) in chronological order; the earlier entries supply
    fitting history (up to `history_seasons` before each scored season). `fit(window, teams)`
    returns the model, fit only on `window` (everything before the checkpoint). Seasons
    that are not a clean double round-robin (splits, play-offs, unfinished) are skipped --
    their final table is not what the projection describes. None if nothing was scored.
    """
    per: dict[float, dict[str, list[Any]]] = {cp: {"model": [], "base": []} for cp in checkpoints}
    scored: list[str] = []
    for i, (code, season_rows) in enumerate(seasons):
        rows = sorted(season_rows, key=lambda o: o.match_date)
        teams = sorted({o.home_norm for o in rows} | {o.away_norm for o in rows})
        n = len(teams)
        if i == 0 or n < 4 or len(rows) != n * (n - 1):
            continue
        position, final_points = final_positions(rows)
        history = [o for _c, prior in seasons[max(0, i - history_seasons) : i] for o in prior]
        scored.append(code)
        for cp in checkpoints:
            k = round(cp * len(rows))
            while 0 < k < len(rows) and rows[k].match_date == rows[k - 1].match_date:
                k += 1  # cut at a date boundary, never part-way through a matchday
            played = rows[:k]
            window = history + played
            points, goal_diff = standings(played)
            remaining, fraction = remaining_round_robin(teams, played)
            model = fit(window, teams)
            matches = len(window) or 1
            flat = _Flat(
                sum(o.fthg for o in window) / matches, sum(o.ftag for o in window) / matches
            )
            for key, m, noise in (
                ("model", model, season_rating_noise(fraction)),
                ("base", flat, 0.0),
            ):
                sim = simulate_season(
                    m,
                    remaining,
                    points_start=points,
                    goal_diff_start=goal_diff,
                    teams=teams,
                    n_sims=n_sims,
                    top_n=top_n,
                    relegation=relegation,
                    seed=seed,
                    rating_noise=noise,
                )
                for p in sim.projections:
                    per[cp][key].append(
                        (
                            (p.title_pct, position[p.team] == 1),
                            (p.top_pct, position[p.team] <= top_n),
                            (p.relegation_pct, position[p.team] > n - relegation),
                            (p.expected_points, final_points[p.team]),
                        )
                    )
    if not scored:
        return None

    def scores(entries: list[Any]) -> dict[str, EventScore]:
        return {ev: _event_score([e[j] for e in entries]) for j, ev in enumerate(EVENTS)}

    def rmse(entries: list[Any]) -> float:
        return math.sqrt(sum((e[3][0] - e[3][1]) ** 2 for e in entries) / len(entries))

    return SeasonBacktest(
        seasons=scored,
        checkpoints=[
            CheckpointScore(
                checkpoint=cp,
                n_team_seasons=len(per[cp]["model"]),
                model=scores(per[cp]["model"]),
                baseline=scores(per[cp]["base"]),
                points_rmse=rmse(per[cp]["model"]),
                baseline_points_rmse=rmse(per[cp]["base"]),
            )
            for cp in checkpoints
        ],
    )
