"""Market-slate tests.

Every market is a sum over the scoreline grid, so the invariants are strict: mutually
exclusive, exhaustive markets sum to 1 (or 2 for the three double chances); a stronger
side is favoured across correlated markets; fair odds are the reciprocal.
"""

from __future__ import annotations

import pytest

from soccer.models.markets import compute_markets

# A clear favourite: home scores ~2.0, away ~0.6.
SLATE = compute_markets("Home", "Away", lam=2.0, mu=0.6, rho=-0.13)
EVEN = compute_markets("Home", "Away", lam=1.3, mu=1.3, rho=-0.13)


class TestDistributions:
    def test_result_sums_to_one(self) -> None:
        assert sum(m.probability for m in SLATE.result) == pytest.approx(1.0, abs=1e-6)

    def test_each_over_under_line_sums_to_one(self) -> None:
        for ou in SLATE.over_under:
            assert ou.over + ou.under == pytest.approx(1.0, abs=1e-6)

    def test_btts_sums_to_one(self) -> None:
        assert sum(m.probability for m in SLATE.btts) == pytest.approx(1.0, abs=1e-6)

    def test_total_goals_sums_to_one(self) -> None:
        assert sum(m.probability for m in SLATE.total_goals) == pytest.approx(1.0, abs=1e-6)

    def test_double_chance_sums_to_two(self) -> None:
        # 1X + 12 + X2 = 2*(H+D+A) = 2.
        assert sum(m.probability for m in SLATE.double_chance) == pytest.approx(2.0, abs=1e-6)


class TestFavouriteBehaviour:
    def test_home_favoured_across_markets(self) -> None:
        result = {m.name: m.probability for m in SLATE.result}
        assert result["Home"] > result["Away"]
        cs = {m.name: m.probability for m in SLATE.clean_sheet}
        assert cs["Home"] > cs["Away"]  # the stronger side keeps more clean sheets

    def test_higher_line_has_lower_over(self) -> None:
        overs = [ou.over for ou in SLATE.over_under]
        assert overs == sorted(overs, reverse=True)  # over 0.5 > over 1.5 > ...

    def test_even_match_is_balanced(self) -> None:
        result = {m.name: m.probability for m in EVEN.result}
        assert result["Home"] == pytest.approx(result["Away"], abs=0.05)


class TestOddsAndScores:
    def test_fair_odds_is_reciprocal(self) -> None:
        m = SLATE.result[0]
        assert m.fair_odds == pytest.approx(1.0 / m.probability)

    def test_most_likely_score_is_the_top_correct_score(self) -> None:
        assert SLATE.most_likely_score == SLATE.correct_scores[0]
        # For a strong favourite the modal score is a home win.
        x, y, _ = SLATE.most_likely_score
        assert x >= y

    def test_correct_scores_sorted_desc(self) -> None:
        probs = [p for _, _, p in SLATE.correct_scores]
        assert probs == sorted(probs, reverse=True)

    def test_win_to_nil_below_plain_win(self) -> None:
        win = {m.name: m.probability for m in SLATE.result}["Home"]
        wtn = {m.name: m.probability for m in SLATE.win_to_nil}["Home"]
        assert 0 < wtn < win  # winning to nil is a subset of winning
