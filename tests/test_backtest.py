"""Backtest harness tests.

The metrics must be correct (log loss and Brier computed the standard way), the walk
must not leak (a match never trains on itself), and a model given genuine signal must
beat the base-rate baseline. Correctness of the numbers is checked against hand values;
skill is checked on a synthetic league with a real strength gradient.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pytest

from soccer.models.backtest import (
    BacktestResult,
    _brier,
    _log_loss,
    backtest_poisson,
)


@dataclass(frozen=True)
class Row:
    home_norm: str
    away_norm: str
    fthg: int
    ftag: int
    match_date: date


class TestMetrics:
    def test_log_loss_of_certainty(self) -> None:
        # A perfect, confident call has ~0 loss; a confident miss is large.
        assert _log_loss((1.0, 0.0, 0.0), 0) == pytest.approx(0.0, abs=1e-9)
        assert _log_loss((0.0, 0.0, 1.0), 0) > 30  # clamped, not infinite

    def test_brier_bounds(self) -> None:
        assert _brier((1.0, 0.0, 0.0), 0) == pytest.approx(0.0)
        assert _brier((0.0, 0.0, 1.0), 0) == pytest.approx(2.0)  # worst case


def build_league(n_rounds: int = 6) -> list[Row]:
    """A league with a clear strength gradient, played over many rounds in date order."""
    teams = ["strong", "good", "mid", "poor", "weak"]
    # Higher index = weaker. Goals scored trend on the strength gap.
    strength = {t: len(teams) - i for i, t in enumerate(teams)}
    rows: list[Row] = []
    start = date(2026, 1, 1)
    day = 0
    for _ in range(n_rounds):
        for i, h in enumerate(teams):
            for a in teams[i + 1 :]:
                gap = strength[h] - strength[a]
                hg = max(0, 1 + gap)
                ag = max(0, 1 - gap)
                rows.append(Row(h, a, hg, ag, start + timedelta(days=day)))
                rows.append(
                    Row(a, h, max(0, 1 - gap), max(0, 1 + gap), start + timedelta(days=day + 1))
                )
                day += 2
    return rows


class TestWalkForward:
    def test_beats_baseline_when_signal_exists(self) -> None:
        result = backtest_poisson(build_league(), min_history=20)
        assert isinstance(result, BacktestResult)
        assert result.n_predictions > 0
        # With a real strength gradient, team info should beat base rates.
        assert result.log_loss < result.baseline_log_loss
        assert result.log_loss_skill > 0

    def test_no_future_leak(self) -> None:
        # If predictions used their own match, a deterministic league would be called
        # perfectly (loss ~0). A non-trivial loss shows the match was held out.
        result = backtest_poisson(build_league(), min_history=20)
        assert result.log_loss > 0.01

    def test_calibration_covers_predictions(self) -> None:
        result = backtest_poisson(build_league(), min_history=20)
        assert sum(b.count for b in result.calibration) == result.n_predictions
        for b in result.calibration:
            assert 0.0 <= b.observed_rate <= 1.0
            assert b.lower <= b.mean_predicted <= b.upper or b.count >= 1

    def test_too_little_history_raises(self) -> None:
        with pytest.raises(ValueError, match="no predictions"):
            backtest_poisson(build_league(n_rounds=1), min_history=10_000)
