"""Season-projection backtest: scoring title / top-N / relegation odds against final tables."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pytest

from soccer.models.poisson import fit_poisson
from soccer.models.season_backtest import (
    backtest_season_projections,
    final_positions,
    skill_range,
)

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


def fit(window: list[Played], teams: list[str], code: str):  # type: ignore[no-untyped-def]
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


class TestSkillRange:
    def test_brackets_a_known_skill_and_narrows_with_more_seasons(self) -> None:
        import random

        rng = random.Random(3)

        def seasons(n: int) -> list[tuple[list[float], list[float]]]:
            # true skill 0.2: model loss 0.8x the baseline's, with season-to-season noise
            out = []
            for _ in range(n):
                base = rng.uniform(5, 15)
                out.append(([0.8 * base * rng.uniform(0.7, 1.3)], [base]))
            return out

        lo20, hi20 = skill_range(seasons(20))
        lo200, hi200 = skill_range(seasons(200))
        assert lo20 < 0.2 < hi20 and lo200 < 0.2 < hi200
        assert hi200 - lo200 < (hi20 - lo20) / 2  # ~1/sqrt(10) as wide with 10x the seasons

    def test_identical_seasons_have_no_spread_and_one_season_has_no_range(self) -> None:
        lo, hi = skill_range([([4.0], [5.0])] * 10)
        assert lo == pytest.approx(0.2) and hi == pytest.approx(0.2)
        assert all(x != x for x in skill_range([([4.0], [5.0])]))  # nan: nothing to resample


def test_backtest_reports_a_range_around_its_skill() -> None:
    seasons = [(f"{y}", season(date(2000 + y, 8, 1))) for y in range(5)]
    report = backtest_season_projections(seasons, fit, checkpoints=(0.0, 0.5), n_sims=200)
    assert report is not None
    for score in report.checkpoints:
        lo, hi = score.skill_range
        assert lo <= score.skill + 1e-9 and score.skill - 1e-9 <= hi
