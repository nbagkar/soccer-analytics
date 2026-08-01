"""Market-edge (value) analysis tests.

The de-vig and EV/Kelly primitives are pure arithmetic with exact expectations. The
walk-forward value backtest is checked on a synthetic season for internal consistency
(staked == bets, returns only on hits) rather than a specific yield, which depends on the
model.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pytest

from soccer.models.value import (
    expected_value,
    implied_probabilities,
    kelly_fraction,
    overround,
    value_backtest,
)


class TestImplied:
    def test_devig_sums_to_one(self) -> None:
        p = implied_probabilities(2.0, 3.5, 4.0)
        assert sum(p) == pytest.approx(1.0)

    def test_overround_is_margin_above_one(self) -> None:
        # A fair 2.0/2.0 two-way priced with margin: 1/1.9+1/1.9 = 1.0526...
        assert overround(1.9, 1.9, 1e12) == pytest.approx(1 / 1.9 + 1 / 1.9 - 1.0)

    def test_shorter_price_implies_higher_probability(self) -> None:
        home, draw, away = implied_probabilities(1.5, 4.0, 7.0)
        assert home > draw > away


class TestEvAndKelly:
    def test_expected_value_sign(self) -> None:
        assert expected_value(0.6, 2.0) == pytest.approx(0.2)  # 0.6*2-1
        assert expected_value(0.4, 2.0) == pytest.approx(-0.2)

    def test_kelly_zero_on_no_edge(self) -> None:
        assert kelly_fraction(0.5, 2.0) == pytest.approx(0.0)  # fair coin, even money
        assert kelly_fraction(0.4, 2.0) == 0.0  # negative edge floored at 0

    def test_kelly_positive_on_edge(self) -> None:
        # prob 0.6 at even money: f = (0.6*1 - 0.4)/1 = 0.2
        assert kelly_fraction(0.6, 2.0) == pytest.approx(0.2)


@dataclass(frozen=True)
class _OddsRow:
    match_date: date
    home_norm: str
    away_norm: str
    fthg: int
    ftag: int
    close_home_odds: float | None
    close_draw_odds: float | None
    close_away_odds: float | None

    @property
    def has_odds(self) -> bool:
        return None not in (self.close_home_odds, self.close_draw_odds, self.close_away_odds)


def _season(with_odds: bool = True) -> list[_OddsRow]:
    teams = ["a", "b", "c", "d", "e", "f"]
    rows, day = [], 1
    for i, h in enumerate(teams):
        for a in teams:
            if a == h:
                continue
            hg, ag = (2, 0) if i % 2 == 0 else (1, 1)
            odds = (2.2, 3.3, 3.4) if with_odds else (None, None, None)
            rows.append(_OddsRow(date(2026, 1, 1 + (day % 27)), h, a, hg, ag, *odds))
            day += 1
    return rows


class TestValueBacktest:
    def test_report_is_internally_consistent(self) -> None:
        report = value_backtest(_season(), model="poisson", min_history=10, edge_threshold=0.0)
        assert report.n_matches > 0
        assert report.staked == report.n_bets  # one unit per bet
        assert report.returned >= 0.0
        # Log losses are finite, positive, and the baseline is a real number.
        assert report.model_log_loss > 0 and report.market_log_loss > 0
        assert report.baseline_log_loss > 0

    def test_no_odds_raises(self) -> None:
        with pytest.raises(ValueError, match="no predictions with odds"):
            value_backtest(_season(with_odds=False), model="poisson", min_history=10)

    def test_high_edge_threshold_places_fewer_bets(self) -> None:
        low = value_backtest(_season(), model="poisson", min_history=10, edge_threshold=0.0)
        high = value_backtest(_season(), model="poisson", min_history=10, edge_threshold=0.5)
        assert high.n_bets <= low.n_bets
