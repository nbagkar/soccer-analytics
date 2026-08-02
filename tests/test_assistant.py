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

    def test_fallback_is_honest(self, tmp_path) -> None:
        reply = answer("what is the weather tomorrow", _seed(tmp_path))
        assert "not sure" in reply.text.lower()
        assert reply.suggestions

    def test_no_data_message(self, tmp_path) -> None:
        reply = answer("who is top?", tmp_path / "missing.duckdb")
        assert "data" in reply.text.lower()
