"""Built-in assistant intent-routing tests.

The value is in the routing and entity resolution: a plain question reaching the right
intent and pulling the right slice, plus honest fallbacks. Data is seeded into a temp
store so the checks are deterministic.
"""

from __future__ import annotations

from soccer.dashboard.assistant import answer
from tests.test_dashboard_data import seed_player_events, seed_results


def _seed(tmp_path):
    path = tmp_path / "analytics.duckdb"
    seed_results(path, division="E0", teams=["Arsenal", "Chelsea", "Fulham", "Brentford"])
    seed_player_events(path)  # adds Messi (Argentina) + Otamendi with match_meta-less events
    return path


class TestRouting:
    def test_help(self, tmp_path) -> None:
        reply = answer("what can you do?", _seed(tmp_path))
        assert "assistant" in reply.text.lower()
        assert reply.suggestions

    def test_standings(self, tmp_path) -> None:
        reply = answer("who is top of the premier league?", _seed(tmp_path))
        assert "Arsenal" in reply.text  # strongest seeded team leads
        assert reply.table and reply.table[0]["#"] == 1

    def test_standings_with_apostrophe_and_no_question_mark(self, tmp_path) -> None:
        # "who's" (apostrophe) and no trailing "?" must still reach standings.
        reply = answer("who's top of the Premier League", _seed(tmp_path))
        assert "Arsenal" in reply.text
        assert reply.table

    def test_team_position(self, tmp_path) -> None:
        reply = answer("where are brentford in the table", _seed(tmp_path))
        assert "Brentford" in reply.text
        assert "th" in reply.text or "st" in reply.text or "nd" in reply.text  # an ordinal

    def test_forecast_keeps_question_order(self, tmp_path) -> None:
        reply = answer("Chelsea vs Arsenal who wins?", _seed(tmp_path))
        assert reply.text.startswith("**Chelsea vs Arsenal")  # order preserved

    def test_forecast_needs_two_teams(self, tmp_path) -> None:
        # Only one team named -> not a forecast; should not crash, routes elsewhere/fallback.
        reply = answer("will arsenal win", _seed(tmp_path))
        assert reply.text  # some answer, no exception

    def test_player_lookup_by_common_name(self, tmp_path) -> None:
        reply = answer("how many goals did messi score", _seed(tmp_path))
        assert "Messi" in reply.text
        assert "goals" in reply.text.lower()

    def test_top_scorers(self, tmp_path) -> None:
        reply = answer("top scorers", _seed(tmp_path))
        assert reply.table is not None
        assert "goals" in reply.text.lower()

    def test_best_player_by_involvement(self, tmp_path) -> None:
        # "best player" (no "scorer"/"goals") must still reach the leaderboard, ranked by
        # goal involvement rather than falling through to the honest fallback.
        reply = answer("who's the best player", _seed(tmp_path))
        assert "not sure" not in reply.text.lower()
        assert "Messi" in reply.text
        assert reply.table is not None

    def test_best_playmaker_ranks_by_assists(self, tmp_path) -> None:
        reply = answer("who's the best playmaker", _seed(tmp_path))
        assert "not sure" not in reply.text.lower()
        assert "assist" in reply.text.lower()

    def test_match_forecasts_prompts_for_teams(self, tmp_path) -> None:
        # Bare "match forecasts" names no teams -> guide the user instead of falling back.
        reply = answer("match forecasts", _seed(tmp_path))
        assert "not sure" not in reply.text.lower()
        assert "two teams" in reply.text.lower()
        assert reply.suggestions

    def test_title_odds_for_a_named_team(self, tmp_path) -> None:
        # "chances of winning" phrasing + a named club -> that club's own projection.
        reply = answer("What are Arsenal's chances of winning the new season", _seed(tmp_path))
        assert "not sure" not in reply.text.lower()
        assert "Arsenal" in reply.text
        assert "title" in reply.text.lower()
        assert "%" in reply.text

    def test_fallback_is_honest(self, tmp_path) -> None:
        reply = answer("what is the weather tomorrow", _seed(tmp_path))
        assert "not sure" in reply.text.lower()
        assert reply.suggestions

    def test_no_data_message(self, tmp_path) -> None:
        reply = answer("who is top?", tmp_path / "missing.duckdb")
        assert "data" in reply.text.lower()

    def test_head_to_head(self, tmp_path) -> None:
        # "vs" also triggers forecast, but the h2h keyword must win (checked first).
        reply = answer("Arsenal vs Chelsea head to head", _seed(tmp_path))
        assert "head to head" in reply.text.lower()
        assert "Arsenal" in reply.text and "Chelsea" in reply.text
        assert reply.table  # recent meetings listed

    def test_overperformance_without_shots_is_honest(self, tmp_path) -> None:
        # The seed carries no shots on target, so xP can't be computed -> honest note.
        reply = answer("who is overperforming their xg in the premier league", _seed(tmp_path))
        assert "shot data" in reply.text.lower()

    def test_second_division_league_alias(self, tmp_path) -> None:
        path = tmp_path / "analytics.duckdb"
        seed_results(path, division="E1", teams=["Leeds", "Leicester", "Norwich", "Watford"])
        reply = answer("who's top of the championship?", path)
        assert "Championship" in reply.text
        assert reply.table

    def test_past_season_resolution(self, tmp_path) -> None:
        path = tmp_path / "analytics.duckdb"
        teams = ["Arsenal", "Chelsea", "Fulham", "Brentford"]
        seed_results(path, division="E0", teams=teams, season="2425")
        seed_results(path, division="E0", teams=teams, season="2526")
        reply = answer("premier league last season table", path)
        assert "2024/25" in reply.text  # the season before the latest, not this one
