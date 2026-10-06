"""Season-projection backtest: scoring title / top-N / relegation odds against final tables."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pytest

from soccer.models.poisson import fit_poisson
from soccer.models.season_backtest import backtest_season_projections, final_positions

TEAMS = ["a", "b", "c", "d", "e", "f"]
STRENGTH = {t: i for i, t in enumerate(reversed(TEAMS))}  # a strongest ... f weakest


@dataclass(frozen=True)
class Played:
    match_date: date
    home_norm: str
    away_norm: str
    fthg: int
    ftag: int


def season(start: date, teams: list[str] = TEAMS) -> list[Played]:
    """A double round-robin in which the stronger side always wins by its strength gap."""
    rows, day = [], start
    for h in teams:
        for a in teams:
            if h == a:
                continue
            gap = STRENGTH[h] - STRENGTH[a]
            rows.append(Played(day, h, a, max(gap, 0) + 1, max(-gap, 0)))
            day += timedelta(days=3)
    return rows


def fit(window: list[Played], teams: list[str]):  # type: ignore[no-untyped-def]
    return fit_poisson(window)


def test_final_positions_rank_by_points_then_goal_difference() -> None:
    position, points = final_positions(season(date(2020, 8, 1)))
    assert position["a"] == 1 and position["f"] == 6
    assert points["a"] > points["f"]


def test_ratings_beat_the_flat_baseline_on_a_predictable_league() -> None:
    seasons = [(f"{y}", season(date(2000 + y, 8, 1))) for y in range(4)]
    report = backtest_season_projections(
        seasons, fit, checkpoints=(0.0, 0.5), n_sims=500, top_n=2, relegation=2
    )
    assert report is not None
    assert report.seasons == ["1", "2", "3"]  # the first season only supplies history
    pre, mid = report.checkpoints
    assert pre.n_team_seasons == mid.n_team_seasons == 3 * len(TEAMS)
    assert pre.skill > 0 and mid.skill > 0
    assert mid.points_rmse < pre.points_rmse + 1e-9  # banked points only help
    for cp in report.checkpoints:
        for score in cp.model.values():
            assert 0 <= score.brier <= 1 and score.log_loss >= 0


def test_unclean_or_missing_seasons_are_skipped() -> None:
    first = ("0", season(date(2000, 8, 1)))
    unfinished = ("1", season(date(2001, 8, 1))[:-5])
    assert backtest_season_projections([first, unfinished], fit, n_sims=200) is None


def test_checkpoints_cut_at_a_date_boundary() -> None:
    # Two matches share every date; a cut must never split them, so the projection at
    # "half played" always starts from whole matchdays.
    rows = season(date(2001, 8, 1))
    paired = [
        Played(rows[i - i % 2].match_date, r.home_norm, r.away_norm, r.fthg, r.ftag)
        for i, r in enumerate(rows)
    ]
    seasons = [("0", season(date(2000, 8, 1))), ("1", paired)]
    report = backtest_season_projections(seasons, fit, checkpoints=(0.5,), n_sims=200)
    assert report is not None
    assert report.checkpoints[0].n_team_seasons == len(TEAMS)


@pytest.mark.parametrize("cp", [0.0, 0.25])
def test_skill_property_matches_its_definition(cp: float) -> None:
    seasons = [(f"{y}", season(date(2000 + y, 8, 1))) for y in range(2)]
    report = backtest_season_projections(seasons, fit, checkpoints=(cp,), n_sims=300)
    assert report is not None
    (score,) = report.checkpoints
    model = sum(s.log_loss for s in score.model.values())
    base = sum(s.log_loss for s in score.baseline.values())
    assert score.skill == pytest.approx(1 - model / base)
