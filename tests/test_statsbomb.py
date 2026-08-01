"""StatsBomb adapter and shots-store tests.

The parser is the logic worth testing (tolerant extraction of shots from event JSON);
the fetch path is checked with a mock transport. The DuckDB shots methods get their own
xG-aggregation checks.
"""

from __future__ import annotations

import httpx
import pytest

from soccer.sources.statsbomb import Shot, StatsBomb, parse_shots
from soccer.storage.analytics_db import AnalyticsDB
from soccer.storage.raw import RawStore

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
    @pytest.fixture
    def adb(self, tmp_path) -> AnalyticsDB:
        return AnalyticsDB(tmp_path / "a.duckdb")

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
