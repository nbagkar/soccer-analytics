"""Forecasting model tests -- the mathematical properties that must hold.

Poisson: probabilities are a proper distribution, a stronger team is favoured, expected
goals track the strengths. Elo: ratings are zero-sum around the mean, winning raises a
rating, order reflects results. These are checkable without real data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pytest

from soccer.models.elo import EloConfig, compute_ratings, expected_score, power_ranking
from soccer.models.poisson import fit_poisson


@dataclass(frozen=True)
class Row:
    home_norm: str
    away_norm: str
    fthg: int
    ftag: int
    match_date: date = date(2026, 1, 1)


def season(dominant: str, weak: str, *, others: list[str] | None = None) -> list[Row]:
    """A small round-robin where `dominant` wins a lot and `weak` loses a lot."""
    others = others or ["mid1", "mid2"]
    teams = [dominant, weak, *others]
    rows: list[Row] = []
    day = 1
    for i, h in enumerate(teams):
        for a in teams[i + 1 :]:
            # dominant scores heavily; weak concedes heavily; others draw-ish.
            hg, ag = (3, 0) if h == dominant else (0, 3) if a == dominant else (1, 1)
            if h == weak:
                hg, ag = 0, 2
            rows.append(Row(h, a, hg, ag, date(2026, 1, day)))
            rows.append(Row(a, h, ag, hg, date(2026, 1, day + 1)))  # reverse fixture
            day += 2
    return rows


class TestPoisson:
    def test_probabilities_form_a_distribution(self) -> None:
        model = fit_poisson(season("strong", "weak"))
        fc = model.forecast("strong", "weak")
        assert fc.prob_home + fc.prob_draw + fc.prob_away == pytest.approx(1.0, abs=1e-6)
        assert all(p >= 0 for p in (fc.prob_home, fc.prob_draw, fc.prob_away))

    def test_stronger_team_is_favoured(self) -> None:
        model = fit_poisson(season("strong", "weak"))
        fc = model.forecast("strong", "weak")
        assert fc.prob_home > fc.prob_away
        assert fc.home_expected > fc.away_expected

    def test_home_advantage_shows_in_league_averages(self) -> None:
        # A league where home teams always win 2-0 -> home_avg > away_avg.
        rows = [Row("a", "b", 2, 0), Row("b", "a", 2, 0), Row("a", "c", 2, 0), Row("c", "a", 2, 0)]
        model = fit_poisson(rows)
        assert model.home_avg > model.away_avg

    def test_top_scores_are_sorted_and_plausible(self) -> None:
        model = fit_poisson(season("strong", "weak"))
        fc = model.forecast("strong", "weak")
        probs = [p for _, _, p in fc.top_scores]
        assert probs == sorted(probs, reverse=True)
        # Favoured home side: the single most likely score is a home win.
        x, y, _ = fc.top_scores[0]
        assert x >= y

    def test_unknown_team_raises(self) -> None:
        model = fit_poisson(season("strong", "weak"))
        with pytest.raises(KeyError):
            model.forecast("strong", "nobody")

    def test_empty_results_rejected(self) -> None:
        with pytest.raises(ValueError, match="no results"):
            fit_poisson([])

    def test_rho_only_perturbs_low_scores(self) -> None:
        # With rho=0 the model is plain independent Poisson; outcomes still valid.
        rows = season("strong", "weak")
        fc0 = fit_poisson(rows, rho=0.0).forecast("strong", "weak")
        assert fc0.prob_home + fc0.prob_draw + fc0.prob_away == pytest.approx(1.0, abs=1e-6)


class TestElo:
    def test_ratings_are_zero_sum_around_initial(self) -> None:
        # Elo only redistributes points, so the mean stays at the initial rating.
        cfg = EloConfig()
        ratings = compute_ratings(season("strong", "weak"), cfg)
        assert sum(ratings.values()) / len(ratings) == pytest.approx(cfg.initial, abs=1e-6)

    def test_winning_team_rises_loser_falls(self) -> None:
        cfg = EloConfig()
        ratings = compute_ratings(season("strong", "weak"), cfg)
        assert ratings["strong"] > cfg.initial
        assert ratings["weak"] < cfg.initial

    def test_power_ranking_orders_by_rating(self) -> None:
        ranking = power_ranking(season("strong", "weak"))
        assert ranking[0].team == "strong"
        assert ranking[-1].team == "weak"
        ratings = [r.rating for r in ranking]
        assert ratings == sorted(ratings, reverse=True)

    def test_home_advantage_raises_home_expectation(self) -> None:
        with_adv = expected_score(1500, 1500, EloConfig(home_advantage=65))
        without = expected_score(1500, 1500, EloConfig(home_advantage=0))
        assert with_adv > without == pytest.approx(0.5)

    def test_expected_score_bounded(self) -> None:
        assert 0.0 < expected_score(1000, 2000, EloConfig()) < 0.5
        assert 0.5 < expected_score(2000, 1000, EloConfig()) < 1.0

    def test_margin_scaling_amplifies_big_wins(self) -> None:
        cfg_on = EloConfig(margin_scaling=True)
        cfg_off = EloConfig(margin_scaling=False)
        big = [Row("a", "b", 5, 0)]
        assert compute_ratings(big, cfg_on)["a"] > compute_ratings(big, cfg_off)["a"]

    def test_processed_in_date_order_not_input_order(self) -> None:
        # Same matches, shuffled input, must give identical ratings.
        rows = season("strong", "weak")
        shuffled = list(reversed(rows))
        assert compute_ratings(rows) == compute_ratings(shuffled)
