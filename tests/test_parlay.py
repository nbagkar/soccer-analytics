"""Accumulator-backtest tests.

The walk-forward fit is the same machinery `evaluation.py`/`value.py` already exercise, so
these focus on the parlay-specific bookkeeping: weekly grouping, leg selection, and the
identity that a 1-leg "accumulator" must reduce to a straight bet.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from soccer.models.parlay import backtest_accumulator


def _synthetic_rows(n_rounds: int = 6):
    """Round-robins among six ranked teams, with vig-loaded odds tracking strength.

    One match per day, so every 7 consecutive matches falls in one ISO week -- enough
    matches per week for a multi-leg accumulator once past the min_history warmup.
    """
    from soccer.domain.names import normalize_name
    from soccer.storage.analytics_db import OddsRow

    teams = ["A", "B", "C", "D", "E", "F"]  # A strongest .. F weakest
    strength = {t: 6 - i for i, t in enumerate(teams)}
    rows, day = [], 0
    for _rnd in range(n_rounds):
        for h in teams:
            for a in teams:
                if h == a:
                    continue
                day += 1
                hs, as_ = strength[h] + 1, strength[a]  # +1 home edge
                hg, ag = (2, 0) if hs > as_ else (0, 2) if hs < as_ else (1, 1)
                eh, ed, ea = math.exp(hs / 3), math.exp(1), math.exp(as_ / 3)
                tot = eh + ed + ea
                rows.append(
                    OddsRow(
                        match_date=date(2025, 1, 6) + timedelta(days=day),  # a Monday
                        home=h,
                        away=a,
                        home_norm=normalize_name(h),
                        away_norm=normalize_name(a),
                        fthg=hg,
                        ftag=ag,
                        close_home_odds=round(tot / eh * 1.05, 2),
                        close_draw_odds=round(tot / ed * 1.05, 2),
                        close_away_odds=round(tot / ea * 1.05, 2),
                    )
                )
    return rows


class TestAccumulatorBacktest:
    def test_report_is_internally_consistent(self) -> None:
        report = backtest_accumulator(_synthetic_rows(), legs_per_bet=2, min_history=20)
        assert report is not None
        assert report.n_bets > 0
        assert report.staked == report.n_bets  # one unit per accumulator
        assert report.straight_staked == report.n_bets * report.legs_per_bet
        assert 0 <= report.hits <= report.n_bets
        assert report.returned >= 0.0
        assert report.straight_returned >= 0.0
        assert 0.0 <= report.hit_rate <= 1.0

    def test_returned_only_when_every_leg_hits(self) -> None:
        """hits counts full-accumulator wins; returned can only be nonzero on those weeks,
        so it can never exceed hits times the largest plausible combined price."""
        report = backtest_accumulator(_synthetic_rows(), legs_per_bet=3, min_history=20)
        assert report is not None
        if report.hits == 0:
            assert report.returned == 0.0

    def test_single_leg_accumulator_is_a_straight_bet(self) -> None:
        """The defining sanity check: a 1-leg 'accumulator' has nothing to combine, so its
        staked/returned must exactly equal the straight-bet tally of the same picks."""
        report = backtest_accumulator(_synthetic_rows(), legs_per_bet=1, min_history=20)
        assert report is not None
        assert report.staked == pytest.approx(report.straight_staked)
        assert report.returned == pytest.approx(report.straight_returned)
        assert report.hits == pytest.approx(report.hit_rate * report.n_bets)

    def test_more_legs_never_increases_bets_placed(self) -> None:
        """A week needs at least `legs_per_bet` eligible matches to qualify at all, so
        raising the requirement can only shrink (or hold) the number of weeks that qualify."""
        rows = _synthetic_rows()
        two = backtest_accumulator(rows, legs_per_bet=2, min_history=20)
        four = backtest_accumulator(rows, legs_per_bet=4, min_history=20)
        assert two is not None and four is not None
        assert four.n_bets <= two.n_bets

    def test_none_when_too_little_history(self) -> None:
        assert backtest_accumulator(_synthetic_rows(n_rounds=1), min_history=1000) is None

    def test_none_when_no_odds(self) -> None:
        from dataclasses import replace

        rows = [
            replace(r, close_home_odds=None, close_draw_odds=None, close_away_odds=None)
            for r in _synthetic_rows()
        ]
        assert backtest_accumulator(rows, min_history=20) is None
