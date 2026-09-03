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


def _seed_squad(path, team="Arsenal"):
    """Seed a small squad for `team` so 'squad for X' has a roster to read back."""
    from soccer.domain.names import normalize_name
    from soccer.storage.analytics_db import AnalyticsDB, SquadMember

    roster = [
        ("Raya", "Goalkeeper", "Spain", "1995-09-15"),
        ("Saliba", "Centre-Back", "France", "2001-03-24"),
        ("Rice", "Defensive Midfield", "England", "1999-01-14"),
        ("Saka", "Right Winger", "England", "2001-09-05"),
    ]
    members = [
        SquadMember(
            competition="PL",
            team=team,
            team_norm=normalize_name(team),
            team_id=1,
            player=player,
            player_norm=normalize_name(player),
            position=position,
            nationality=nationality,
            date_of_birth=dob,
            fetched_at="2026-08-01",
        )
        for player, position, nationality, dob in roster
    ]
    with AnalyticsDB(path) as adb:
        adb.load_squads(members)


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


def _seed_availability(live_path, rows):
    """Seed FPL-style availability into a live store. `rows` are (team, player, status, chance,
    news) tuples; team_norm is derived exactly as the real ingest does so club lookups match."""
    from soccer.domain.availability import AvailabilityStore, PlayerAvailability, status_label
    from soccer.domain.names import normalize_name
    from soccer.storage.live_db import LiveDB

    records = [
        PlayerAvailability(
            source="fpl",
            team=team,
            team_norm=normalize_name(team),
            player=player,
            full_name=None,
            status=status,
            availability=status_label(status),
            chance=chance,
            news=news or None,
            news_added=None,
            fetched_at="2026-08-04",
        )
        for team, player, status, chance, news in rows
    ]
    with LiveDB(live_path) as db:
        AvailabilityStore(db).replace_source("fpl", records)


def _seed_priced_squad(live_path, team, players):
    """Seed a priced squad for the forecast adjustment. `players` are
    (name, status, element_type, price, chance) tuples -- element_type/price are what let the
    nudge weight a loss by role and quality, so this helper carries them where the 5-tuple
    availability seeder does not."""
    from soccer.domain.availability import AvailabilityStore, PlayerAvailability, status_label
    from soccer.domain.names import normalize_name
    from soccer.storage.live_db import LiveDB

    records = [
        PlayerAvailability(
            source="fpl",
            team=team,
            team_norm=normalize_name(team),
            player=name,
            full_name=None,
            status=status,
            availability=status_label(status),
            chance=chance,
            news=None,
            news_added=None,
            fetched_at="2026-08-04",
            element_type=etype,
            price=price,
        )
        for (name, status, etype, price, chance) in players
    ]
    with LiveDB(live_path) as db:
        AvailabilityStore(db).replace_source("fpl", records)


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
        assert "all-time" in reply.text.lower()  # scoped honestly, not implied "right now"

    def test_top_scorers_this_season_gets_an_explicit_caveat(self, tmp_path) -> None:
        # The free player-level archive has no current-season coverage -- "this season"
        # cannot be honoured, so the answer must say so rather than silently ignoring it.
        reply = answer("top scorers this season", _seed(tmp_path))
        assert reply.table is not None
        assert "doesn't cover the current season" in reply.text

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

    def test_squad_lists_the_roster(self, tmp_path) -> None:
        path = _seed(tmp_path)
        _seed_squad(path, "Arsenal")
        reply = answer("squad for Arsenal", path)
        assert "Arsenal" in reply.text and "squad" in reply.text.lower()
        assert reply.table and {r["Player"] for r in reply.table} >= {"Raya", "Saka"}
        assert reply.table[0]["Pos"] == "Goalkeeper"  # keepers sort first

    def test_who_plays_for_reaches_squad(self, tmp_path) -> None:
        path = _seed(tmp_path)
        _seed_squad(path, "Arsenal")
        reply = answer("who plays for arsenal", path)
        assert reply.table and any(r["Player"] == "Saka" for r in reply.table)

    def test_squad_is_honest_when_none_loaded(self, tmp_path) -> None:
        # A club is named but no squads are loaded -> point at the action, never fall back.
        reply = answer("squad for Arsenal", _seed(tmp_path))
        assert "not sure" not in reply.text.lower()
        assert "update squads" in reply.text.lower()

    def test_dossier_notes_squad_when_loaded(self, tmp_path) -> None:
        path = _seed(tmp_path)
        _seed_squad(path, "Arsenal")
        reply = answer("tell me about Arsenal", path)
        assert "squad" in reply.text.lower()
        assert "Arsenal squad" in reply.suggestions

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


class TestAvailability:
    def test_club_injuries_lists_only_the_flagged(self, tmp_path) -> None:
        path = _seed(tmp_path)
        live = tmp_path / "live.sqlite"
        _seed_availability(
            live,
            [
                ("Arsenal", "Saka", "i", 25, "Hamstring - back in 2 weeks"),
                ("Arsenal", "Rice", "d", 75, "Knock, doubtful"),
                ("Arsenal", "Raya", "a", None, ""),  # fit -> must not appear
            ],
        )
        reply = answer("Arsenal injuries", path, live)
        assert "team news" in reply.text.lower()
        assert reply.table and {r["Player"] for r in reply.table} == {"Saka", "Rice"}

    def test_is_player_fit(self, tmp_path) -> None:
        path = _seed(tmp_path)
        live = tmp_path / "live.sqlite"
        _seed_availability(live, [("Arsenal", "Saka", "i", 25, "Hamstring injury")])
        reply = answer("is Saka fit?", path, live)
        assert "Saka" in reply.text
        assert "injured" in reply.text.lower()

    def test_league_wide_injury_news(self, tmp_path) -> None:
        path = _seed(tmp_path)
        live = tmp_path / "live.sqlite"
        _seed_availability(
            live,
            [("Arsenal", "Saka", "i", 25, "Hamstring"), ("Chelsea", "James", "d", 50, "Knock")],
        )
        reply = answer("premier league injury news", path, live)
        assert "flagged" in reply.text.lower()
        assert reply.table and len(reply.table) == 2

    def test_clean_bill_of_health_when_nobody_flagged(self, tmp_path) -> None:
        path = _seed(tmp_path)
        live = tmp_path / "live.sqlite"
        _seed_availability(live, [("Arsenal", "Raya", "a", None, "")])
        reply = answer("Arsenal injuries", path, live)
        assert "clean bill of health" in reply.text.lower()

    def test_honest_when_no_availability_loaded(self, tmp_path) -> None:
        # Keyword present but nothing loaded -> point at the action, not the generic fallback.
        reply = answer("injury news", _seed(tmp_path))
        assert "not sure" not in reply.text.lower()
        assert "update injuries" in reply.text.lower()

    def test_dossier_notes_team_news(self, tmp_path) -> None:
        path = _seed(tmp_path)
        live = tmp_path / "live.sqlite"
        _seed_availability(live, [("Arsenal", "Saka", "i", 25, "Hamstring")])
        reply = answer("tell me about Arsenal", path, live)
        assert "team news" in reply.text.lower()
        assert "Arsenal injuries" in reply.suggestions


class TestForecastTeamNews:
    """A PL forecast folds in current injuries as a transparent prior, shown against the raw
    model -- but only when someone is actually flagged, and only when a live store is passed."""

    _BRENTFORD = (
        ("Flekken", "a", 1, 45, None),
        ("Collins", "a", 2, 45, None),
        ("Janelt", "a", 3, 50, None),
        ("Mbeumo", "i", 4, 75, None),  # top striker out
        ("Wissa", "i", 4, 65, None),  # second striker out
    )

    def test_forecast_shows_adjusted_vs_raw_when_flagged(self, tmp_path) -> None:
        analytics = _seed(tmp_path)
        live = tmp_path / "live.sqlite"
        _seed_priced_squad(live, "Brentford", self._BRENTFORD)
        reply = answer("Arsenal vs Brentford who wins?", analytics, live)
        assert "Adjusted for team news" in reply.text
        assert "Mbeumo" in reply.text  # names the absence driving it
        assert "→" in reply.text  # raw -> adjusted, shown side by side
        assert "not a backtested edge" in reply.text  # honest labelling

    def test_no_adjustment_block_when_nobody_flagged(self, tmp_path) -> None:
        analytics = _seed(tmp_path)
        live = tmp_path / "live.sqlite"
        fit = [(n, "a", e, p, c) for (n, _s, e, p, c) in self._BRENTFORD]  # everyone available
        _seed_priced_squad(live, "Brentford", fit)
        reply = answer("Arsenal vs Brentford who wins?", analytics, live)
        assert "Adjusted for team news" not in reply.text  # strict no-op
        assert reply.text.startswith("**Arsenal vs Brentford")  # base forecast intact

    def test_no_adjustment_block_without_a_live_store(self, tmp_path) -> None:
        # No live_db passed -> the base forecast still works, just no team-news nudge.
        reply = answer("Arsenal vs Brentford who wins?", _seed(tmp_path))
        assert "Adjusted for team news" not in reply.text
        assert reply.chart and reply.chart["kind"] == "result_bar"

    def test_adjustment_block_suppressed_when_it_does_not_visibly_move_anything(
        self, tmp_path, monkeypatch
    ) -> None:
        """A nudge can be technically nonzero (`is_material`, at its 1e-9 threshold) and still
        round away to nothing at the block's own display precision. Printing "X is missing
        players" followed by identical before/after numbers reads as a broken feature, not a
        real (if modest) update -- so the block must not appear at all in that case."""
        import soccer.dashboard.data as data_module
        from soccer.domain.availability import NEUTRAL_ADJUSTMENT, AvailabilityAdjustment

        analytics = _seed(tmp_path)
        live = tmp_path / "live.sqlite"
        _seed_priced_squad(live, "Brentford", self._BRENTFORD)

        slate = data_module.forecast_slate(analytics, "2526", "E0", "Arsenal", "Brentford")
        assert slate is not None
        invisible_nudge = AvailabilityAdjustment(
            attack_factor=1.0 - 1e-6,
            leak_factor=1.0,
            lost_attack=1e-6,
            lost_defence=0.0,
            missing=("Bench Player",),
        )
        assert invisible_nudge.is_material  # nonzero -- just not visibly so

        def fake_adjusted(*args, **kwargs):
            return data_module.AdjustedForecast(
                raw=slate, adjusted=slate, home_adj=NEUTRAL_ADJUSTMENT, away_adj=invisible_nudge
            )

        monkeypatch.setattr(data_module, "availability_adjusted_slate", fake_adjusted)

        reply = answer("Arsenal vs Brentford who wins?", analytics, live)
        assert "Adjusted for team news" not in reply.text
        assert reply.text.startswith("**Arsenal vs Brentford")  # base forecast intact


class TestConversationContext:
    """Multi-turn follow-ups: a bare question inherits the previous turn's team or league, so
    'tell me about Arsenal' then 'how's their form?' resolves without re-naming the club."""

    def test_reply_carries_the_resolved_subject(self, tmp_path) -> None:
        reply = answer("tell me about Arsenal", _seed(tmp_path))
        assert reply.context is not None
        assert any(display == "Arsenal" for display, _d, _s in reply.context.teams)
        assert reply.context.division == "E0"

    def test_pronoun_follow_up_resolves_the_last_team(self, tmp_path) -> None:
        path = _seed(tmp_path)
        first = answer("tell me about Arsenal", path)
        reply = answer("how is their form?", path, context=first.context)
        assert "Arsenal" in reply.text  # 'their' -> Arsenal, not a league-wide form list

    def test_explicitly_named_team_overrides_the_context(self, tmp_path) -> None:
        path = _seed(tmp_path)
        first = answer("tell me about Arsenal", path)
        reply = answer("how is Chelsea's form?", path, context=first.context)
        assert "Chelsea" in reply.text
        assert "Arsenal" not in reply.text  # a named club always wins over the remembered one

    def test_league_carries_to_a_bare_follow_up(self, tmp_path) -> None:
        path = tmp_path / "analytics.duckdb"
        seed_results(path, division="E0", teams=["Arsenal", "Chelsea", "Fulham", "Brentford"])
        seed_results(
            path, division="SP1", teams=["Barcelona", "Real Madrid", "Sevilla", "Valencia"]
        )
        first = answer("who is top of la liga?", path)
        assert first.context is not None and first.context.division == "SP1"
        # 'who is in form?' names no league, so it should stay in La Liga, not snap to the default.
        reply = answer("who is in form?", path, context=first.context)
        assert "La Liga" in reply.text

    def test_pronoun_follow_up_reaches_team_news(self, tmp_path) -> None:
        path = _seed(tmp_path)
        live = tmp_path / "live.sqlite"
        _seed_availability(live, [("Arsenal", "Saka", "i", None, "Ankle knock")])
        first = answer("tell me about Arsenal", path, live)
        reply = answer("are they injured?", path, live, context=first.context)
        assert "Saka" in reply.text + str(reply.table)

    def test_context_chain_keeps_subject_across_several_turns(self, tmp_path) -> None:
        path = _seed(tmp_path)
        r1 = answer("tell me about Arsenal", path)
        r2 = answer("how is their form?", path, context=r1.context)
        r3 = answer("what about their fixtures?", path, context=r2.context)
        # Turn 3 has no team of its own and only a pronoun; the subject must survive turn 2.
        assert r3.context is not None
        assert any(display == "Arsenal" for display, _d, _s in r3.context.teams)
