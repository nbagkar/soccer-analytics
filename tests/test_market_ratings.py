"""Market-implied ratings: inverting closing odds, and feeding them into the team fit."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pytest

from soccer.models.poisson import (
    DEFAULT_RHO,
    fit_poisson_shots,
    implied_goal_rates,
    market_expected_goals,
    score_grid,
)


@pytest.mark.parametrize(("ph", "pa"), [(0.45, 0.28), (0.70, 0.12), (0.20, 0.55), (0.36, 0.36)])
def test_inversion_reproduces_the_market_probabilities(ph: float, pa: float) -> None:
    lam, mu = implied_goal_rates(ph, pa)
    grid = score_grid(lam, mu, DEFAULT_RHO)
    assert sum(p for (x, y), p in grid.items() if x > y) == pytest.approx(ph, abs=1e-6)
    assert sum(p for (x, y), p in grid.items() if x < y) == pytest.approx(pa, abs=1e-6)


def test_a_stronger_home_price_implies_more_home_goals() -> None:
    weak, strong = implied_goal_rates(0.35, 0.35), implied_goal_rates(0.65, 0.15)
    assert strong[0] > weak[0] and strong[1] < weak[1]


@dataclass
class Played:
    match_date: date
    home_norm: str
    away_norm: str
    fthg: int
    ftag: int
    close_home_odds: float | None = None
    close_draw_odds: float | None = None
    close_away_odds: float | None = None


def test_missing_or_invalid_odds_mean_no_market_view() -> None:
    assert market_expected_goals(Played(date(2026, 1, 1), "a", "b", 1, 0)) is None
    assert market_expected_goals(Played(date(2026, 1, 1), "a", "b", 1, 0, 1.0, 3.0, 4.0)) is None
    lam, mu = market_expected_goals(Played(date(2026, 1, 1), "a", "b", 1, 0, 1.5, 4.5, 7.0))
    assert lam > mu  # the home side was the clear favourite


def test_market_weight_moves_ratings_toward_the_market() -> None:
    # "unlucky" keeps losing 0-1 despite being priced as a heavy favourite every time;
    # goals alone call it weak, the market calls it strong.
    teams = ["unlucky", "b", "c", "d"]
    games, day = [], date(2026, 1, 1)
    for _round in range(3):
        for opp in teams[1:]:
            games.append(Played(day, "unlucky", opp, 0, 1, 1.4, 4.8, 8.0))
            games.append(Played(day + timedelta(days=1), opp, "unlucky", 1, 0, 6.0, 4.2, 1.6))
            day += timedelta(days=2)
        for i, h in enumerate(teams[1:]):
            for a in teams[1:][i + 1 :]:
                games.append(Played(day, h, a, 1, 1, 2.6, 3.2, 2.8))
                day += timedelta(days=1)
    goals_only = fit_poisson_shots(games, alpha=1.0)
    market = fit_poisson_shots(games, alpha=1.0, market_weight=1.0)
    assert goals_only.strengths["unlucky"].attack < 1.0 < market.strengths["unlucky"].attack


def test_results_carry_their_closing_odds(tmp_path) -> None:
    from soccer.storage.analytics_db import AnalyticsDB
    from tests.test_integrity import _row

    path = tmp_path / "a.duckdb"
    row = _row("2526", "E0", date(2026, 1, 3), "Arsenal", "Chelsea")
    from dataclasses import replace

    with AnalyticsDB(path) as adb:
        adb.load_results(
            [replace(row, close_home_odds=1.9, close_draw_odds=3.6, close_away_odds=4.2)]
        )
        (out,) = adb.outcomes_for("2526", "E0")
        (recent,) = adb.recent_outcomes_through("E0", "2526", n_seasons=1)
    assert (out.close_home_odds, out.close_away_odds) == (1.9, 4.2)
    assert recent.close_draw_odds == 3.6


def test_extreme_favourite_gets_a_sane_closest_fit() -> None:
    # Burnley v Man City at 15.78 / 8.1 / 1.15: an ~85% away favourite with an ~11% draw has
    # no exact grid. A diverging solver once returned a 7-20 "expected score" here, which
    # then skewed every later rating fit for both clubs.
    inv = [1 / 15.78, 1 / 8.1, 1 / 1.15]
    total = sum(inv)
    lam, mu = implied_goal_rates(round(inv[0] / total, 3), round(inv[2] / total, 3))
    assert 0.3 < lam < 2.0 and 2.0 < mu < 5.0
