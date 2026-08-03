"""Forecast-evaluation tests: RPS, the log-opinion blend, and the walk-forward report."""

from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from soccer.models.evaluation import blend_probs, evaluate_forecasts, rps


class TestRPS:
    def test_perfect_and_worst(self) -> None:
        assert rps((1.0, 0.0, 0.0), 0) == 0.0
        assert rps((0.0, 0.0, 1.0), 0) == 1.0  # predicted away, home won -> worst

    def test_rewards_ordinal_closeness(self) -> None:
        # Home actually won: calling a draw is a smaller error than calling an away win.
        assert rps((0.0, 1.0, 0.0), 0) < rps((0.0, 0.0, 1.0), 0)


class TestBlend:
    def test_endpoints_are_pure_inputs(self) -> None:
        m, k = (0.5, 0.3, 0.2), (0.2, 0.3, 0.5)
        assert blend_probs(m, k, 1.0) == pytest.approx(m)
        assert blend_probs(m, k, 0.0) == pytest.approx(k)

    def test_geometric_midpoint_and_normalized(self) -> None:
        m, k = (0.5, 0.3, 0.2), (0.2, 0.3, 0.5)
        b = blend_probs(m, k, 0.5)
        assert sum(b) == pytest.approx(1.0)
        raw = [math.sqrt(mi * ki) for mi, ki in zip(m, k, strict=True)]
        total = sum(raw)
        assert b == pytest.approx(tuple(r / total for r in raw))


def _synthetic_rows():
    """A few round-robins among six ranked teams, with vig-loaded odds tracking strength."""
    from soccer.domain.names import normalize_name
    from soccer.storage.analytics_db import OddsRow

    teams = ["A", "B", "C", "D", "E", "F"]  # A strongest .. F weakest
    strength = {t: 6 - i for i, t in enumerate(teams)}
    rows, day = [], 0
    for _rnd in range(4):
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
                        match_date=date(2025, 1, 1) + timedelta(days=day),
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


class TestEvaluateForecasts:
    def test_report_structure_and_blend_optimality(self) -> None:
        report = evaluate_forecasts(_synthetic_rows(), min_history=20, weight_steps=21)
        assert report is not None
        assert report.n >= 30
        assert len(report.blend_curve) == 21
        assert 0.0 <= report.best_weight <= 1.0
        for s in (report.model, report.market, report.baseline, report.blend):
            assert s.log_loss > 0 and 0.0 <= s.rps <= 1.0 and s.brier >= 0.0
        # The weight grid includes pure market (0) and pure model (1), so the argmin blend
        # can never be worse than the better of the two on log loss.
        assert report.blend.log_loss <= min(report.model.log_loss, report.market.log_loss) + 1e-9
        distances = [d.distance for d in report.divergences]
        assert distances == sorted(distances, reverse=True)

    def test_none_when_too_little_data(self) -> None:
        assert evaluate_forecasts(_synthetic_rows()[:20], min_history=60) is None
