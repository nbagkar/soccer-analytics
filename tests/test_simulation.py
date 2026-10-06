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


class TestRatingNoise:
    def test_noise_keeps_the_invariants(self) -> None:
        result = simulate_season(FOUR, FIXTURES, n_sims=2000, seed=1, rating_noise=0.3)
        assert sum(p.title_pct for p in result.projections) == pytest.approx(1.0, abs=1e-9)

    def test_noise_pulls_a_favourite_toward_the_field(self) -> None:
        # Doubt about the ratings means the best-rated side is less of a sure thing.
        sure = simulate_season(FOUR, FIXTURES, n_sims=4000, seed=1)
        doubt = simulate_season(FOUR, FIXTURES, n_sims=4000, seed=1, rating_noise=0.4)
        title = {p.team: p.title_pct for p in sure.projections}
        title_doubt = {p.team: p.title_pct for p in doubt.projections}
        assert title_doubt["strong"] < title["strong"]
        assert title_doubt["weak"] > title["weak"]

    def test_zero_noise_is_the_plain_simulation(self) -> None:
        a = simulate_season(FOUR, FIXTURES, n_sims=500, seed=3)
        b = simulate_season(FOUR, FIXTURES, n_sims=500, seed=3, rating_noise=0.0)
        assert a == b

    def test_schedule_fades_to_nothing_as_the_season_is_played(self) -> None:
        from soccer.models.simulation import SEASON_NOISE_START, season_rating_noise

        assert season_rating_noise(0.0) == pytest.approx(SEASON_NOISE_START)
        assert season_rating_noise(0.5) < season_rating_noise(0.25) < season_rating_noise(0.0)
        assert season_rating_noise(1.0) == 0.0
        assert season_rating_noise(1.5) == 0.0  # out-of-range input is clamped


class TestTableHelpers:
    def test_standings_and_remaining_pairs(self) -> None:
        from dataclasses import dataclass

        from soccer.models.simulation import remaining_round_robin, standings

        @dataclass(frozen=True)
        class Played:
            home_norm: str
            away_norm: str
            fthg: int
            ftag: int

        played = [Played("a", "b", 2, 0), Played("c", "a", 1, 1)]
        points, gd = standings(played)
        assert points == {"a": 4, "b": 0, "c": 1}
        assert gd == {"a": 2, "b": -2, "c": 0}

        remaining, fraction = remaining_round_robin(["a", "b", "c"], played)
        assert len(remaining) == 4 and ("a", "b") not in remaining and ("b", "a") in remaining
        assert fraction == pytest.approx(2 / 6)
