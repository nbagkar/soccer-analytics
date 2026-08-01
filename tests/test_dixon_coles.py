"""Dixon-Coles MLE model tests.

The fitted model must be a valid probabilistic forecaster (probabilities sum to one, a
stronger team is favoured), a drop-in for PoissonModel (same interface, works in the
simulation), and time decay must actually re-weight matches. Recovering the *true*
parameters is checked on synthetic data generated from known strengths.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pytest

from soccer.models.dixon_coles import fit_dixon_coles
from soccer.models.simulation import simulate_season


@dataclass(frozen=True)
class Row:
    home_norm: str
    away_norm: str
    fthg: int
    ftag: int
    match_date: date = date(2026, 1, 1)


def league(rounds: int = 8) -> list[Row]:
    """A round-robin with a real strength gradient and a home boost, spread over dates."""
    teams = ["strong", "good", "mid", "weak"]
    strength = {t: len(teams) - i for i, t in enumerate(teams)}
    rows: list[Row] = []
    day = 0
    for _ in range(rounds):
        for i, h in enumerate(teams):
            for a in teams[i + 1 :]:
                gap = strength[h] - strength[a]
                # +1 to whoever is home, so a home advantage genuinely exists to recover.
                rows.append(
                    Row(
                        h,
                        a,
                        max(0, 1 + gap) + 1,
                        max(0, 1 - gap),
                        date(2026, 1, 1) + timedelta(day),
                    )
                )
                rows.append(
                    Row(
                        a,
                        h,
                        max(0, 1 - gap) + 1,
                        max(0, 1 + gap),
                        date(2026, 1, 1) + timedelta(day + 1),
                    )
                )
                day += 2
    return rows


class TestFit:
    def test_returns_a_model_with_all_teams(self) -> None:
        model = fit_dixon_coles(league())
        assert set(model.teams) == {"strong", "good", "mid", "weak"}

    def test_stronger_team_has_higher_attack(self) -> None:
        model = fit_dixon_coles(league())
        assert model._attack["strong"] > model._attack["weak"]

    def test_home_advantage_is_positive(self) -> None:
        # The synthetic league has home teams score more, so home advantage > 0.
        assert fit_dixon_coles(league()).home_advantage > 0

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValueError, match="no results"):
            fit_dixon_coles([])


class TestForecast:
    def test_probabilities_form_a_distribution(self) -> None:
        fc = fit_dixon_coles(league()).forecast("strong", "weak")
        assert fc.prob_home + fc.prob_draw + fc.prob_away == pytest.approx(1.0, abs=1e-6)

    def test_stronger_team_favoured(self) -> None:
        fc = fit_dixon_coles(league()).forecast("strong", "weak")
        assert fc.prob_home > fc.prob_away
        assert fc.home_expected > fc.away_expected

    def test_unknown_team_raises(self) -> None:
        with pytest.raises(KeyError):
            fit_dixon_coles(league()).expected_goals("strong", "nobody")


class TestDropInInterface:
    def test_has_poisson_model_interface(self) -> None:
        model = fit_dixon_coles(league())
        assert hasattr(model, "strengths")  # membership checks
        assert hasattr(model, "expected_goals")
        assert hasattr(model, "forecast")
        assert "strong" in model.strengths

    def test_works_in_simulation(self) -> None:
        model = fit_dixon_coles(league())
        fixtures = [(h, a) for h in model.teams for a in model.teams if h != a]
        result = simulate_season(model, fixtures, n_sims=1000, seed=1)
        assert sum(p.title_pct for p in result.projections) == pytest.approx(1.0, abs=1e-6)
        assert result.projections[0].team == "strong"


class TestTimeDecay:
    def test_decay_shifts_the_fit_toward_recent_form(self) -> None:
        # "riser" is thrashed early (scores 0) and dominant late (scores 3), home and
        # away vs 3 opponents. Weighting recent high-scoring games raises its attack.
        opponents = ["o1", "o2", "o3"]
        rows: list[Row] = []
        day = 0
        for hg, ag in [(0, 3)] * 3 + [(3, 0)] * 3:  # early losses, then late wins
            for opp in opponents:
                rows.append(Row("riser", opp, hg, ag, date(2026, 1, 1) + timedelta(day)))
                rows.append(Row(opp, "riser", ag, hg, date(2026, 1, 1) + timedelta(day + 1)))
                day += 2

        no_decay = fit_dixon_coles(rows, time_decay=0.0)
        with_decay = fit_dixon_coles(rows, time_decay=0.05)  # aggressive recency
        assert with_decay._attack["riser"] > no_decay._attack["riser"]
