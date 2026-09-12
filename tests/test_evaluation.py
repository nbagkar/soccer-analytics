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


def _synthetic_rows(start: date = date(2025, 1, 1)):
    """A few round-robins among six ranked teams, with vig-loaded odds tracking strength.

    All dates land within `start`'s season -- four rounds among six teams is 120 days, so
    keep `start` well clear of a season boundary (e.g. not late June/early July) for that
    to hold.
    """
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
                        match_date=start + timedelta(days=day),
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

    def test_calibration_covers_all_three_outcomes(self) -> None:
        report = evaluate_forecasts(_synthetic_rows(), min_history=20)
        assert report is not None
        assert [oc.label for oc in report.calibration_by_outcome] == ["Home", "Draw", "Away"]
        for oc in report.calibration_by_outcome:
            # Every scored match lands in exactly one bin of each outcome's reliability diagram.
            assert sum(b.count for b in oc.bins) == report.n
            assert oc.ece >= 0.0
            for b in oc.bins:
                assert 0.0 <= b.mean_predicted <= 1.0
                assert 0.0 <= b.observed_rate <= 1.0

    def test_track_record_is_most_recent_first_and_matches_the_hit_rate(self) -> None:
        """The synthetic rows have a real skill signal (odds track true strength), so the
        model should call the right outcome noticeably more than a random 3-way guess."""
        report = evaluate_forecasts(_synthetic_rows(), min_history=20)
        assert report is not None
        assert report.recent  # single-season fixture -> every scored match is "recent"
        assert len(report.recent) == report.n
        dates = [r.match_date for r in report.recent]
        assert dates == sorted(dates, reverse=True)  # newest first
        assert 0.0 <= report.hit_rate <= 1.0
        assert report.hit_rate > 0.4  # clears a random 3-way guess by a wide margin
        assert 0.0 <= report.goals_within_one_rate <= 1.0
        for r in report.recent:
            assert r.correct == (r.predicted == r.actual)
            assert r.actual in (0, 1, 2)
            assert r.goals_within_one == (
                abs(round(r.home_expected + r.away_expected) - (r.home_goals + r.away_goals)) <= 1
            )

    def test_recent_track_record_is_limited_to_the_latest_season(self) -> None:
        """Two full seasons of results -- the track record should only show the later one,
        not spill into the earlier season the way a fixed match-count cap would."""
        from soccer.sources.football_data_co_uk import current_season_code

        earlier = _synthetic_rows(start=date(2023, 8, 1))  # season "2324"
        later = _synthetic_rows(start=date(2024, 8, 1))  # season "2425"
        report = evaluate_forecasts(earlier + later, min_history=20)
        assert report is not None
        assert len(report.recent) < report.n  # excludes the earlier season entirely
        seasons_in_recent = {current_season_code(r.match_date) for r in report.recent}
        assert seasons_in_recent == {current_season_code(date(2024, 9, 1))}
        dates = [r.match_date for r in report.recent]
        assert dates == sorted(dates, reverse=True)  # newest first
