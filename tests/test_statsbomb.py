"""StatsBomb adapter and shots-store tests.

The parser is the logic worth testing (tolerant extraction of shots from event JSON);
the fetch path is checked with a mock transport. The DuckDB shots methods get their own
xG-aggregation checks.
"""

from __future__ import annotations

import httpx
import pytest

from soccer.sources.statsbomb import Shot, StatsBomb, parse_player_stats, parse_shots
from soccer.storage.analytics_db import AnalyticsDB
from soccer.storage.raw import RawStore


@pytest.fixture
def adb(tmp_path) -> AnalyticsDB:
    return AnalyticsDB(tmp_path / "a.duckdb")


EVENTS = [
    {"type": {"name": "Pass"}, "team": {"name": "A"}},  # not a shot
    {
        "type": {"name": "Shot"},
        "team": {"name": "Argentina"},
        "player": {"name": "Messi"},
        "minute": 23,
        "period": 1,
        "location": [110.0, 40.0],
        "shot": {
            "statsbomb_xg": 0.35,
            "outcome": {"name": "Goal"},
            "type": {"name": "Open Play"},
            "body_part": {"name": "Left Foot"},
        },
    },
    {
        "type": {"name": "Shot"},
        "team": {"name": "France"},
        "player": {"name": "Mbappé"},
        "minute": 80,
        "period": 2,
        "location": [108.0, 44.0],
        "shot": {
            "statsbomb_xg": 0.76,
            "outcome": {"name": "Goal"},
            "type": {"name": "Penalty"},
        },
    },
]


class TestParser:
    def test_extracts_only_shots(self) -> None:
        shots = parse_shots(EVENTS, match_id=1)
        assert len(shots) == 2  # the Pass is skipped

    def test_shot_fields(self) -> None:
        messi = parse_shots(EVENTS, match_id=1)[0]
        assert messi.player == "Messi"
        assert messi.team == "Argentina"
        assert messi.xg == pytest.approx(0.35)
        assert messi.is_goal is True
        assert messi.is_penalty is False
        assert (messi.x, messi.y) == (110.0, 40.0)
        assert messi.body_part == "Left Foot"

    def test_penalty_flagged(self) -> None:
        mbappe = parse_shots(EVENTS, match_id=1)[1]
        assert mbappe.is_penalty is True

    def test_missing_subfields_tolerated(self) -> None:
        sparse = [{"type": {"name": "Shot"}, "shot": {"statsbomb_xg": 0.1}}]
        shot = parse_shots(sparse, match_id=1)[0]
        assert shot.player == "Unknown"
        assert shot.xg == pytest.approx(0.1)
        assert shot.x is None

    def test_empty_events(self) -> None:
        assert parse_shots([], match_id=1) == []


class TestAdapter:
    def test_fetch_parses_and_snapshots(self, tmp_path) -> None:
        raw = RawStore(tmp_path / "raw")
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json=EVENTS))
        with StatsBomb(raw, client=httpx.Client(transport=transport)) as sb:
            shots = sb.fetch_shots(9999)
        assert len(shots) == 2
        from soccer.sources.registry import SourceId

        assert raw.latest(SourceId.STATSBOMB, "events_9999") is not None

    def test_missing_match_returns_empty(self, tmp_path) -> None:
        raw = RawStore(tmp_path / "raw")
        transport = httpx.MockTransport(lambda request: httpx.Response(404))
        with StatsBomb(raw, client=httpx.Client(transport=transport)) as sb:
            assert sb.fetch_shots(1) == []


class TestShotsStore:
    def _shots(self) -> list[Shot]:
        return [
            *parse_shots(EVENTS, match_id=1),
            Shot(
                1,
                "Argentina",
                "Di María",
                36,
                1,
                100.0,
                38.0,
                0.2,
                "Saved",
                False,
                False,
                "Right Foot",
            ),
        ]

    def test_load_and_team_xg(self, adb: AnalyticsDB) -> None:
        assert adb.load_shots(self._shots()) == 3
        teams = adb.team_xg(1)
        by = {r.name: r for r in teams}
        assert by["Argentina"].xg == pytest.approx(0.55)  # 0.35 + 0.2
        assert by["Argentina"].goals == 1
        assert by["Argentina"].shots == 2
        assert teams[0].name == "France"  # France's penalty (0.76) outranks Argentina's 0.55

    def test_reload_is_idempotent(self, adb: AnalyticsDB) -> None:
        adb.load_shots(self._shots())
        adb.load_shots(self._shots())
        assert sum(r.shots for r in adb.team_xg(1)) == 3

    def test_top_shooters(self, adb: AnalyticsDB) -> None:
        adb.load_shots(self._shots())
        top = adb.top_shooters(1, limit=5)
        assert top[0].name == "Mbappé"  # highest xG (penalty 0.76)

    def test_shots_for_has_locations(self, adb: AnalyticsDB) -> None:
        adb.load_shots(self._shots())
        shots = adb.shots_for(1)
        assert all("x" in s and "y" in s for s in shots)
        assert adb.matches_with_shots() == [1]

    def test_player_leaderboard(self, adb: AnalyticsDB) -> None:
        adb.load_shots(self._shots())
        board = adb.player_leaderboard(min_shots=1)
        by = {r.player: r for r in board}
        # Mbappé scored a penalty (0.76 xg, 1 goal); npxG excludes it.
        assert by["Mbappé"].goals == 1
        assert by["Mbappé"].npxg == 0.0  # only shot was a penalty
        # Messi: 0.35 xg goal, non-penalty.
        assert by["Messi"].npxg == pytest.approx(0.35)
        assert by["Messi"].xg_diff == pytest.approx(1 - 0.35)  # 1 goal - 0.35 xg
        assert adb.player_count() == 3

    def test_leaderboard_ordering_and_min_shots(self, adb: AnalyticsDB) -> None:
        adb.load_shots(self._shots())
        # min_shots filters everyone out here (each player has 1 shot).
        assert adb.player_leaderboard(min_shots=2) == []
        by_goals = adb.player_leaderboard(min_shots=1, order="goals")
        assert by_goals[0].goals >= by_goals[-1].goals


# A compact events stream exercising the tricky parsing rules: pass completion inferred
# from the ABSENCE of an outcome, progressive thresholds, xA/key-pass linkage via a shot's
# key_pass_id, a won tackle, a card, and minutes derived from Starting XI + Substitution.
PLAYER_EVENTS = [
    {
        "type": {"name": "Starting XI"},
        "team": {"name": "A"},
        "tactics": {
            "lineup": [
                {"player": {"name": "P1"}, "position": {"name": "Center Midfield"}},
                {"player": {"name": "P2"}, "position": {"name": "Striker"}},
            ]
        },
    },
    # P1 completed, progressive pass (no `outcome` key => completed; 40->90 up-pitch).
    {
        "id": "pass1",
        "type": {"name": "Pass"},
        "player": {"name": "P1"},
        "team": {"name": "A"},
        "location": [40.0, 40.0],
        "pass": {"end_location": [90.0, 40.0]},
    },
    # P1 incomplete pass (has an outcome => not completed).
    {
        "id": "pass2",
        "type": {"name": "Pass"},
        "player": {"name": "P1"},
        "team": {"name": "A"},
        "location": [40.0, 40.0],
        "pass": {"end_location": [50.0, 40.0], "outcome": {"name": "Incomplete"}},
    },
    # P1's assist: a pass flagged goal_assist that a shot references by key_pass_id.
    {
        "id": "pass3",
        "type": {"name": "Pass"},
        "player": {"name": "P1"},
        "team": {"name": "A"},
        "location": [80.0, 40.0],
        "pass": {"end_location": [100.0, 40.0], "goal_assist": True},
    },
    {
        "type": {"name": "Shot"},
        "player": {"name": "P2"},
        "team": {"name": "A"},
        "minute": 10,
        "shot": {"statsbomb_xg": 0.5, "key_pass_id": "pass3", "outcome": {"name": "Goal"}},
    },
    # P2 progressive carry (60->80).
    {
        "type": {"name": "Carry"},
        "player": {"name": "P2"},
        "team": {"name": "A"},
        "location": [60.0, 40.0],
        "carry": {"end_location": [80.0, 40.0]},
    },
    {
        "type": {"name": "Duel"},
        "player": {"name": "P1"},
        "team": {"name": "A"},
        "duel": {"type": {"name": "Tackle"}, "outcome": {"name": "Won"}},
    },
    {"type": {"name": "Interception"}, "player": {"name": "P1"}, "team": {"name": "A"}},
    {
        "type": {"name": "Foul Committed"},
        "player": {"name": "P1"},
        "team": {"name": "A"},
        "foul_committed": {"card": {"name": "Yellow Card"}},
    },
    # P1 off at 70, P3 on -- minutes: P1 70, P3 20 (to match end), P2 90 (never subbed).
    {
        "type": {"name": "Substitution"},
        "player": {"name": "P1"},
        "team": {"name": "A"},
        "minute": 70,
        "substitution": {"replacement": {"name": "P3"}},
    },
    {
        "type": {"name": "Pass"},
        "player": {"name": "P3"},
        "team": {"name": "A"},
        "minute": 90,
        "location": [40.0, 40.0],
        "pass": {"end_location": [50.0, 40.0]},
    },
]


class TestPlayerStatsParser:
    def _by_name(self) -> dict:
        return {p.player: p for p in parse_player_stats(PLAYER_EVENTS, match_id=1)}

    def test_pass_completion_and_progression(self) -> None:
        p1 = self._by_name()["P1"]
        assert p1.passes == 3
        assert p1.passes_completed == 2  # pass2 had an outcome => incomplete
        assert p1.progressive_passes == 2  # pass1 and pass3 both gain up-pitch

    def test_xa_and_key_pass_linked_from_shot(self) -> None:
        p1 = self._by_name()["P1"]
        assert p1.key_passes == 1
        assert p1.xa == pytest.approx(0.5)  # the shot's xG credited to the passer
        assert p1.assists == 1  # goal_assist flag

    def test_defensive_and_discipline_counts(self) -> None:
        p1 = self._by_name()["P1"]
        assert (p1.tackles, p1.tackles_won) == (1, 1)
        assert p1.interceptions == 1
        assert p1.yellow_cards == 1

    def test_carry_progression(self) -> None:
        p2 = self._by_name()["P2"]
        assert p2.carries == 1
        assert p2.progressive_carries == 1

    def test_minutes_from_lineup_and_subs(self) -> None:
        by = self._by_name()
        assert by["P1"].minutes == 70  # started, off at 70
        assert by["P2"].minutes == 90  # started, played to the end (last event minute)
        assert by["P3"].minutes == 20  # on at 70, to the end


class TestPlayerProfiles:
    def test_profile_joins_shots_and_events(self, adb: AnalyticsDB) -> None:
        # A shooter (P2) and a non-shooting defender (P1) both get a profile.
        adb.load_player_stats(parse_player_stats(PLAYER_EVENTS, match_id=1))
        adb.load_shots(
            [Shot(1, "A", "P2", 10, 1, 100.0, 40.0, 0.5, "Goal", True, False, "Right Foot")]
        )
        profiles = {p.player: p for p in adb.player_profiles(min_minutes=1, order="contributions")}

        striker = profiles["P2"]
        assert striker.goals == 1 and striker.shots == 1  # from the shots table
        assert striker.xg == pytest.approx(0.5)

        mid = profiles["P1"]
        assert mid.goals == 0  # never shot -> LEFT JOIN yields zero, not dropped
        assert mid.assists == 1 and mid.tackles == 1
        assert mid.pass_pct == pytest.approx(200 / 3)  # 2 of 3 completed

    def test_single_profile_and_per90(self, adb: AnalyticsDB) -> None:
        adb.load_player_stats(parse_player_stats(PLAYER_EVENTS, match_id=1))
        p1 = adb.player_profile("P1")
        assert p1 is not None
        assert p1.minutes == 70
        # 1 interception in 70 minutes -> ~1.29 per 90.
        assert p1.per90(p1.interceptions) == pytest.approx(90 / 70)
        assert adb.player_profile("Nobody") is None


class TestMatchMeta:
    def test_metadata_and_competition_filter(self, adb: AnalyticsDB) -> None:
        from soccer.sources.statsbomb import PlayerMatchStats, parse_match_meta
        from soccer.storage.analytics_db import MatchMeta

        # Two matches, two competitions; one player appears in each.
        adb.load_player_stats(parse_player_stats(PLAYER_EVENTS, match_id=1))
        other = [
            PlayerMatchStats(
                match_id=2,
                player="P1",
                team="A",
                position="CM",
                minutes=90,
                passes=10,
                passes_completed=9,
                key_passes=0,
                assists=0,
                xa=0.0,
                progressive_passes=0,
                carries=0,
                progressive_carries=0,
                dribbles=0,
                dribbles_completed=0,
                tackles=9,
                tackles_won=9,
                interceptions=0,
                blocks=0,
                clearances=0,
                ball_recoveries=0,
                pressures=0,
                fouls=0,
                fouled=0,
                yellow_cards=0,
                red_cards=0,
                touches=10,
            )
        ]
        adb.load_player_stats(other)
        adb.load_match_meta(
            [
                MatchMeta(
                    **parse_match_meta(
                        {
                            "match_id": 1,
                            "competition": {"competition_name": "World Cup", "competition_id": 43},
                            "season": {"season_name": "2022", "season_id": 106},
                            "home_team": {"home_team_name": "A"},
                            "away_team": {"away_team_name": "B"},
                            "match_date": "2022-12-18",
                        }
                    )
                ),
                MatchMeta(
                    **parse_match_meta(
                        {
                            "match_id": 2,
                            "competition": {"competition_name": "La Liga"},
                            "season": {"season_name": "2020/2021"},
                            "home_team": {"home_team_name": "A"},
                            "away_team": {"away_team_name": "C"},
                        }
                    )
                ),
            ]
        )

        assert dict(adb.competitions_loaded()) == {"World Cup": 1, "La Liga": 1}
        assert adb.match_label(1) == "A v B (World Cup 2022)"

        # Filtered to La Liga, P1's tackles come only from match 2 (9), not match 1.
        laliga = adb.player_profile("P1", competition="La Liga")
        assert laliga is not None
        assert laliga.tackles == 9 and laliga.matches == 1
        wc = adb.player_profile("P1", competition="World Cup")
        assert wc.tackles == 1  # only match 1's single tackle

        # Season filter narrows further within a competition.
        assert dict(adb.competition_seasons("World Cup")) == {"2022": 1}
        wc_2022 = adb.player_profiles(min_minutes=1, competition="World Cup", season="2022")
        assert {p.player for p in wc_2022} >= {"P1"}
        # A season that does not exist yields no rows.
        assert adb.player_profile("P1", competition="World Cup", season="1900") is None

    def test_unknown_match_label_is_none(self, adb: AnalyticsDB) -> None:
        assert adb.match_label(999) is None
