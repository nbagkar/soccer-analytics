"""League simulation tests.

Monte Carlo output is random, so the tests assert invariants that hold regardless of
the draw: probabilities that must sum exactly (one champion, N in the top bucket),
determinism under a fixed seed, and that a stronger team and a points head start both
raise title odds.
"""

from __future__ import annotations

import pytest

from soccer.models.poisson import PoissonModel, TeamStrength
from soccer.models.simulation import simulate_season


def model(strengths: dict[str, tuple[float, float]]) -> PoissonModel:
    return PoissonModel(
        strengths={t: TeamStrength(a, d) for t, (a, d) in strengths.items()},
        home_avg=1.5,
        away_avg=1.1,
    )


def round_robin(teams: list[str]) -> list[tuple[str, str]]:
    return [(h, a) for h in teams for a in teams if h != a]


FOUR = model({"strong": (1.6, 0.7), "b": (1.0, 1.0), "c": (1.0, 1.0), "weak": (0.6, 1.5)})
FIXTURES = round_robin(["strong", "b", "c", "weak"])


class TestInvariants:
    def test_exactly_one_champion_per_sim(self) -> None:
        result = simulate_season(FOUR, FIXTURES, n_sims=2000, seed=1)
        assert sum(p.title_pct for p in result.projections) == pytest.approx(1.0, abs=1e-9)

    def test_top_bucket_sums_to_bucket_size(self) -> None:
        result = simulate_season(FOUR, FIXTURES, n_sims=2000, top_n=2, seed=1)
        assert sum(p.top_pct for p in result.projections) == pytest.approx(2.0, abs=1e-9)

    def test_relegation_sums_to_count(self) -> None:
        result = simulate_season(FOUR, FIXTURES, n_sims=2000, relegation=1, seed=1)
        assert sum(p.relegation_pct for p in result.projections) == pytest.approx(1.0, abs=1e-9)

    def test_probabilities_in_range(self) -> None:
        for p in simulate_season(FOUR, FIXTURES, n_sims=1000, seed=1).projections:
            assert 0.0 <= p.title_pct <= 1.0
            assert 1.0 <= p.avg_position <= 4.0


class TestDeterminism:
    def test_same_seed_same_result(self) -> None:
        a = simulate_season(FOUR, FIXTURES, n_sims=1000, seed=42)
        b = simulate_season(FOUR, FIXTURES, n_sims=1000, seed=42)
        assert [p.title_pct for p in a.projections] == [p.title_pct for p in b.projections]


class TestSensitivity:
    def test_stronger_team_wins_title_more(self) -> None:
        result = simulate_season(FOUR, FIXTURES, n_sims=3000, seed=1)
        title = {p.team: p.title_pct for p in result.projections}
        assert title["strong"] == max(title.values())
        assert title["strong"] > title["weak"]
        assert result.projections[0].team == "strong"  # sorted by title odds

    def test_points_head_start_raises_odds(self) -> None:
        even = model({"a": (1.0, 1.0), "b": (1.0, 1.0)})
        fixtures = round_robin(["a", "b"])
        # b starts 20 points clear with a handful of games left.
        result = simulate_season(even, fixtures, points_start={"b": 20}, n_sims=2000, seed=1)
        title = {p.team: p.title_pct for p in result.projections}
        assert title["b"] > title["a"]

    def test_no_remaining_fixtures_reflects_standings(self) -> None:
        # Season already decided: no fixtures, b leads on points -> b is champion always.
        even = model({"a": (1.0, 1.0), "b": (1.0, 1.0)})
        result = simulate_season(
            even, [], points_start={"a": 10, "b": 20}, teams=["a", "b"], n_sims=100, seed=1
        )
        title = {p.team: p.title_pct for p in result.projections}
        assert title["b"] == pytest.approx(1.0)


def test_no_teams_raises() -> None:
    with pytest.raises(ValueError, match="no teams"):
        simulate_season(model({}), [], n_sims=10)
