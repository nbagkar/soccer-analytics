"""Market-implied ratings: inverting closing odds, and feeding them into the team fit."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pytest

from soccer.models.poisson import (
    DEFAULT_RHO,
    fit_poisson_shots,
    implied_goal_rates,
    implied_goal_rates_with_total,
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
    close_over25_odds: float | None = None
    close_under25_odds: float | None = None


def _market_probs(lam: float, mu: float) -> tuple[float, float, float]:
    grid = score_grid(lam, mu, DEFAULT_RHO)
    return (
        sum(p for (x, y), p in grid.items() if x > y),
        sum(p for (x, y), p in grid.items() if x < y),
        sum(p for (x, y), p in grid.items() if x + y >= 3),
    )


@pytest.mark.parametrize(("lam", "mu"), [(1.5, 1.1), (2.4, 0.6), (0.9, 1.7), (1.0, 0.9)])
def test_inversion_with_total_recovers_the_goal_rates(lam: float, mu: float) -> None:
    lam2, mu2 = implied_goal_rates_with_total(*_market_probs(lam, mu))
    assert (lam2, mu2) == (pytest.approx(lam, abs=1e-4), pytest.approx(mu, abs=1e-4))


def test_the_over_under_line_moves_the_goal_total_not_the_gap() -> None:
    # Same 1X2, but one match is priced for goals and the other for a low-scoring grind:
    # the 1X2-only inversion cannot tell them apart; with the total line it can.
    ph, pa, _ = _market_probs(1.6, 1.1)
    open_game = implied_goal_rates_with_total(round(ph, 3), round(pa, 3), 0.62)
    tight_game = implied_goal_rates_with_total(round(ph, 3), round(pa, 3), 0.40)
    assert sum(open_game) > sum(tight_game) + 0.4
    assert open_game[0] > open_game[1] and tight_game[0] > tight_game[1]


def test_market_view_uses_the_over_under_when_present() -> None:
    base = Played(date(2026, 1, 1), "a", "b", 1, 0, 2.0, 3.5, 4.0)
    goals_heavy = Played(date(2026, 1, 1), "a", "b", 1, 0, 2.0, 3.5, 4.0, 1.5, 2.6)
    assert market_expected_goals(base) == implied_goal_rates(*_vig_free(2.0, 4.0, 3.5))
    assert sum(market_expected_goals(goals_heavy)) > sum(market_expected_goals(base))  # type: ignore[arg-type]
    # an unusable over/under falls back to the 1X2-only view
    broken = Played(date(2026, 1, 1), "a", "b", 1, 0, 2.0, 3.5, 4.0, 1.0, 2.6)
    assert market_expected_goals(broken) == market_expected_goals(base)


def _vig_free(home: float, away: float, draw: float) -> tuple[float, float]:
    inv = [1 / home, 1 / draw, 1 / away]
    total = sum(inv)
    return round(inv[0] / total, 3), round(inv[2] / total, 3)


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
            [
                replace(
                    row,
                    close_home_odds=1.9,
                    close_draw_odds=3.6,
                    close_away_odds=4.2,
                    close_over25_odds=1.8,
                    close_under25_odds=2.1,
                )
            ]
        )
        (out,) = adb.outcomes_for("2526", "E0")
        (recent,) = adb.recent_outcomes_through("E0", "2526", n_seasons=1)
        (odds,) = adb.outcomes_with_odds("2526", "E0")
    assert (out.close_home_odds, out.close_away_odds) == (1.9, 4.2)
    assert recent.close_draw_odds == 3.6
    assert (out.close_over25_odds, recent.close_under25_odds) == (1.8, 2.1)
    assert (odds.close_over25_odds, odds.close_under25_odds) == (1.8, 2.1)


def test_extreme_favourite_gets_a_sane_closest_fit() -> None:
    # Burnley v Man City at 15.78 / 8.1 / 1.15: an ~85% away favourite with an ~11% draw has
    # no exact grid. A diverging solver once returned a 7-20 "expected score" here, which
    # then skewed every later rating fit for both clubs.
    inv = [1 / 15.78, 1 / 8.1, 1 / 1.15]
    total = sum(inv)
    lam, mu = implied_goal_rates(round(inv[0] / total, 3), round(inv[2] / total, 3))
    assert 0.3 < lam < 2.0 and 2.0 < mu < 5.0


class TestCalibrateModel:
    def _model(self):
        from soccer.models.poisson import PoissonModel, TeamStrength

        return PoissonModel(
            strengths={
                "top": TeamStrength(1.5, 0.7),
                "mid": TeamStrength(1.0, 1.0),
                "low": TeamStrength(0.7, 1.4),
            },
            home_avg=1.5,
            away_avg=1.2,
        )

    def test_identity_settings_change_nothing(self) -> None:
        from soccer.models.poisson import calibrate_model

        base = self._model()
        same = calibrate_model(base)
        assert same.expected_goals("top", "low") == pytest.approx(base.expected_goals("top", "low"))
        assert same.rho == base.rho

    def test_spread_widens_the_gap_and_leaves_an_average_side_alone(self) -> None:
        from soccer.models.poisson import calibrate_model

        base, wide = self._model(), calibrate_model(self._model(), spread=1.1)
        assert wide.forecast("top", "low").prob_home > base.forecast("top", "low").prob_home
        assert wide.expected_goals("mid", "mid") == pytest.approx(base.expected_goals("mid", "mid"))

    def test_home_shift_moves_goals_home_and_rho_replaces_the_grid_correlation(self) -> None:
        from soccer.models.poisson import calibrate_model

        base = self._model()
        shifted = calibrate_model(base, home_shift=0.05, rho=-0.05)
        (bh, ba), (sh, sa) = base.expected_goals("mid", "mid"), shifted.expected_goals("mid", "mid")
        assert sh > bh and sa < ba
        assert shifted.rho == -0.05
        assert (
            base.forecast("mid", "mid").prob_draw
            > calibrate_model(base, rho=-0.05).forecast("mid", "mid").prob_draw
        )  # a weaker low-score correlation, fewer draws
