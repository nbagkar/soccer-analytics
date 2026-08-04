"""Built-in assistant intent-routing tests.

The value is in the routing and entity resolution: a plain question reaching the right
intent and pulling the right slice, plus honest fallbacks. Data is seeded into a temp
store so the checks are deterministic.
"""

from __future__ import annotations

import re

from soccer.dashboard.assistant import answer
from tests.test_dashboard_data import seed_player_events, seed_results


def _seed(tmp_path):
    path = tmp_path / "analytics.duckdb"
    seed_results(path, division="E0", teams=["Arsenal", "Chelsea", "Fulham", "Brentford"])
    seed_player_events(path)  # adds Messi (Argentina) + Otamendi with match_meta-less events
    return path


def _seed_shots(path):
    """A single StatsBomb match (Argentina v France) with a few shots, for the match centre."""
    from soccer.sources.statsbomb import Shot
    from soccer.storage.analytics_db import AnalyticsDB

    with AnalyticsDB(path) as adb:
        adb.load_shots(
            [
                Shot(
                    1,
                    "Argentina",
                    "Messi",
                    23,
                    1,
                    110.0,
                    40.0,
                    0.35,
                    "Goal",
                    True,
                    False,
                    "Left Foot",
                ),
                Shot(1, "France", "Mbappé", 80, 2, 108.0, 44.0, 0.76, "Goal", True, True, None),
                Shot(
                    1,
                    "Argentina",
                    "Di María",
                    60,
                    1,
                    100.0,
                    40.0,
                    0.20,
                    "Saved",
                    False,
                    False,
                    None,
                ),
            ]
        )


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
        assert reply.chart and reply.chart["kind"] == "result_bar"
        assert len(reply.chart["data"]) == 3  # home / draw / away

    def test_season_compare(self, tmp_path) -> None:
        path = tmp_path / "analytics.duckdb"
        teams = ["Arsenal", "Chelsea", "Fulham", "Brentford"]
        seed_results(path, division="E0", teams=teams, season="2425")
        seed_results(path, division="E0", teams=teams, season="2526")
        reply = answer("Arsenal this season vs last", path)
        assert "not sure" not in reply.text.lower()
        assert "Arsenal" in reply.text
        assert reply.table and len(reply.table) == 2  # one row per season
        assert {r["Season"] for r in reply.table} == {"2024/25", "2025/26"}

    def test_season_compare_needs_two_seasons(self, tmp_path) -> None:
        # Only one season loaded -> not enough to compare; falls through, no crash.
        reply = answer("Arsenal this season vs last", _seed(tmp_path))
        assert reply.text  # some answer (dossier/fallback), no exception


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

    def test_match_centre_shot_log(self, tmp_path) -> None:
        path = _seed(tmp_path)
        _seed_shots(path)  # Argentina v France
        reply = answer("shot log for Argentina vs France", path)
        assert "Argentina" in reply.text and "France" in reply.text
        assert "xG" in reply.text  # the xG race line
        assert reply.table and "Player" in reply.table[0]
        assert reply.chart and reply.chart["kind"] == "xg_race" and reply.chart["data"]

    def test_team_dossier_has_trajectory_chart(self, tmp_path) -> None:
        reply = answer("tell me about Arsenal", _seed(tmp_path))
        assert "Arsenal" in reply.text
        assert reply.chart and reply.chart["kind"] == "trajectory" and reply.chart["data"]

    def test_match_centre_ignores_alias_collision(self, tmp_path) -> None:
        # A shot ask with a single team named must not invent a match; fall through cleanly.
        path = _seed(tmp_path)
        _seed_shots(path)
        reply = answer("shot map for Argentina", path)  # only one side named
        assert "Argentina v France" not in reply.text  # did not fabricate the match

    def test_scout_percentiles(self, tmp_path) -> None:
        reply = answer("scouting report for Messi", _seed(tmp_path))
        assert "not sure" not in reply.text.lower()
        assert "Messi" in reply.text
        assert "pct" in reply.text.lower()
        assert reply.table is not None
        assert reply.chart and reply.chart["kind"] == "percentiles" and reply.chart["data"]

    def test_team_fixtures_query_not_stolen_by_dossier(self, tmp_path) -> None:
        # "<club> fixtures" must reach the fixtures intent (honest note without a live DB),
        # not the team dossier -- the two-word query used to trip the dossier's short branch.
        reply = answer("Arsenal fixtures", _seed(tmp_path))
        assert "fixtures" in reply.text.lower()
        assert "1st" not in reply.text  # not the standings/dossier line

    def test_value_backtest_routes_and_is_honest(self, tmp_path) -> None:
        # A betting-value ask reaches the backtest intent (never the fallback) and answers
        # honestly -- either the yield or the "no odds loaded" note, both mentioning odds.
        reply = answer("are there any value bets in the premier league", _seed(tmp_path))
        assert "not sure" not in reply.text.lower()
        assert "odds" in reply.text.lower() or "yield" in reply.text.lower()

    def test_league_compare_two_leagues(self, tmp_path) -> None:
        path = tmp_path / "analytics.duckdb"
        seed_results(path, division="E0", teams=["Arsenal", "Chelsea", "Fulham", "Brentford"])
        seed_results(path, division="SP1", teams=["Barca", "Madrid", "Sevilla", "Valencia"])
        reply = answer("compare the premier league and la liga", path)
        assert "not sure" not in reply.text.lower()
        assert reply.table and len(reply.table) >= 2
        assert "Goals/g" in reply.table[0]

    def test_which_league_ranks_all_loaded(self, tmp_path) -> None:
        path = tmp_path / "analytics.duckdb"
        seed_results(path, division="E0", teams=["Arsenal", "Chelsea", "Fulham", "Brentford"])
        seed_results(path, division="SP1", teams=["Barca", "Madrid", "Sevilla", "Valencia"])
        reply = answer("which league scores the most goals", path)
        assert "goals per game" in reply.text.lower()
        assert reply.table and len(reply.table) >= 2


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

    def test_new_league_alias(self, tmp_path) -> None:
        # A newly added league (Eredivisie) resolves from its plain name.
        path = tmp_path / "analytics.duckdb"
        seed_results(path, division="N1", teams=["Ajax", "PSV", "Feyenoord", "AZ"])
        reply = answer("who's top of the eredivisie?", path)
        assert "Eredivisie" in reply.text
        assert reply.table

    def test_cup_standings_carry_a_caveat(self, tmp_path) -> None:
        # Champions League table is reachable but honestly flagged (knockouts decide it).
        path = tmp_path / "analytics.duckdb"
        seed_results(path, division="UCL", teams=["Real Madrid", "Bayern", "PSG", "Inter"])
        reply = answer("champions league standings", path)
        assert "Champions League" in reply.text
        assert "trophy winner" in reply.text.lower()  # the caveat
        assert reply.table

    def test_who_wins_a_cup_is_honest(self, tmp_path) -> None:
        path = tmp_path / "analytics.duckdb"
        seed_results(path, division="UCL", teams=["Real Madrid", "Bayern", "PSG", "Inter"])
        reply = answer("who will win the champions league", path)
        assert "knockout" in reply.text.lower()  # no fake title projection

    def test_cup_does_not_hijack_domestic_resolution(self, tmp_path) -> None:
        # A club that plays in both its league and a cup must resolve to its league.
        path = tmp_path / "analytics.duckdb"
        seed_results(path, division="E0", teams=["Arsenal", "Chelsea", "Fulham", "Brentford"])
        seed_results(path, division="UCL", teams=["Arsenal", "Real Madrid", "Bayern", "PSG"])
        reply = answer("tell me about Arsenal", path)
        assert "Premier League" in reply.text  # its league, not the Champions League

    def test_team_resolution_is_whole_word(self) -> None:
        # "Aris" (Greek club) must not match inside "Paris"; "Arsenal" must still resolve.
        from soccer.dashboard.assistant import _resolve_teams

        index = {
            "aris": ("Aris", "G1", "2526"),
            "arsenal": ("Arsenal", "E0", "2526"),
            "paris sg": ("Paris SG", "F1", "2526"),
        }
        names = [t[0] for t in _resolve_teams("paris sg vs arsenal head to head", index)]
        assert names[:2] == ["Paris SG", "Arsenal"]
        assert "Aris" not in names

    def test_united_city_suffix_does_not_summon_manchester(self) -> None:
        # "Sheffield United" must not drag in Man United (the "united" nickname alias), and a
        # suffix the index omits ("Newcastle United" stored as "Newcastle") must stay itself.
        from soccer.dashboard.assistant import _resolve_teams

        index = {
            "sheffield united": ("Sheffield United", "E0", "2526"),
            "man united": ("Man United", "E0", "2526"),
            "newcastle": ("Newcastle", "E0", "2526"),
            "leicester": ("Leicester", "E0", "2526"),
            "man city": ("Man City", "E0", "2526"),
            "arsenal": ("Arsenal", "E0", "2526"),
        }
        # A real "United"/"City" club must resolve to itself and NOTHING from Manchester.
        assert [t[0] for t in _resolve_teams("sheffield united vs arsenal", index)] == [
            "Sheffield United",
            "Arsenal",
        ]
        assert [t[0] for t in _resolve_teams("newcastle united vs leicester city", index)] == [
            "Newcastle",
            "Leicester",
        ]
        # But the bare colloquial reference must STILL reach the Manchester clubs.
        assert [t[0] for t in _resolve_teams("united vs city", index)] == [
            "Man United",
            "Man City",
        ]

    def test_past_season_resolution(self, tmp_path) -> None:
        path = tmp_path / "analytics.duckdb"
        teams = ["Arsenal", "Chelsea", "Fulham", "Brentford"]
        seed_results(path, division="E0", teams=teams, season="2425")
        seed_results(path, division="E0", teams=teams, season="2526")
        reply = answer("premier league last season table", path)
        assert "2024/25" in reply.text  # the season before the latest, not this one

    def test_forecast_gives_a_predicted_scoreline(self, tmp_path) -> None:
        reply = answer("what's the predicted score for Chelsea vs Arsenal", _seed(tmp_path))
        assert "Expected goals" in reply.text  # leads with the differentiated signal
        assert "scoreline" in reply.text.lower()
        assert re.search(r"\d-\d", reply.text)  # an actual scoreline is still shown

    def test_forecast_fires_on_two_teams_without_a_keyword(self, tmp_path) -> None:
        # "scoreline" + two clubs must forecast, not get hijacked by a player-name collision.
        reply = answer("predicted scoreline chelsea arsenal", _seed(tmp_path))
        assert reply.text.startswith("**Chelsea vs Arsenal")

    def test_team_dossier(self, tmp_path) -> None:
        reply = answer("tell me about Arsenal", _seed(tmp_path))
        assert reply.text.startswith("**Arsenal**")
        assert reply.table  # recent results listed
        assert "not sure" not in reply.text.lower()

    def test_bare_club_name_is_a_dossier(self, tmp_path) -> None:
        # Just naming a club should give its dossier, not fall through.
        reply = answer("Brentford", _seed(tmp_path))
        assert reply.text.startswith("**Brentford**")

    def test_all_time_honours(self, tmp_path) -> None:
        reply = answer("who has won the most titles in the premier league", _seed(tmp_path))
        assert "all-time" in reply.text.lower()
        assert reply.table and "Titles" in reply.table[0]

    def test_model_accuracy_is_honest(self, tmp_path) -> None:
        # No odds in the seed -> honest "need odds" note; real point is it does not fall back.
        reply = answer("how accurate is your model", _seed(tmp_path))
        assert "not sure" not in reply.text.lower()
        assert "odds" in reply.text.lower()

    def test_compare_two_players(self, tmp_path) -> None:
        reply = answer("compare Messi and Otamendi", _seed(tmp_path))
        assert "Messi" in reply.text and "Otamendi" in reply.text
        assert reply.table and reply.table[0]["Metric"] == "Matches"
