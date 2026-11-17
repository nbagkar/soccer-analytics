"""Team-xG backfill: resumable, quota-bounded, bridged to the results table's names."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import httpx

from soccer.ingest.api_football_xg import backfill_team_xg, season_code
from soccer.storage.analytics_db import AnalyticsDB
from tests.test_api_football import make, ok
from tests.test_market import _result


def _fixture(fid: int, home: tuple[int, str], away: tuple[int, str], day: str) -> dict[str, Any]:
    return {
        "fixture": {"id": fid, "date": f"{day}T15:00:00+00:00", "status": {"short": "FT"}},
        "teams": {
            "home": {"id": home[0], "name": home[1]},
            "away": {"id": away[0], "name": away[1]},
        },
    }


FIXTURES = [
    _fixture(1, (33, "Manchester United"), (66, "Aston Villa"), "2025-05-25"),
    _fixture(2, (50, "Manchester City"), (33, "Manchester United"), "2025-05-18"),
    _fixture(3, (66, "Aston Villa"), (50, "Manchester City"), "2025-05-11"),
    {  # not finished -> never requested
        "fixture": {"id": 4, "date": "2025-05-30T15:00:00+00:00", "status": {"short": "PST"}},
        "teams": {"home": {"id": 33, "name": "x"}, "away": {"id": 66, "name": "y"}},
    },
]


def _stats(home_id: int, away_id: int, hx: str | None, ax: str | None) -> list[dict[str, Any]]:
    def side(tid: int, x: str | None) -> dict[str, Any]:
        stats = [{"type": "Total Shots", "value": 10}]
        if x is not None:
            stats.append({"type": "expected_goals", "value": x})
        return {"team": {"id": tid}, "statistics": stats}

    return [side(home_id, hx), side(away_id, ax)]


def handler(calls: list[str]) -> Any:
    stats = {
        "1": _stats(33, 66, "2.95", "1.10"),
        "2": _stats(50, 33, "1.80", "0.70"),
        "3": _stats(66, 50, None, None),  # provider had no xG for this match
    }

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(f"{request.url.path}?{request.url.query.decode()}")
        if request.url.path == "/fixtures":
            return httpx.Response(200, json=ok(FIXTURES))
        return httpx.Response(200, json=ok(stats[request.url.params["fixture"]]))

    return respond


def seeded(tmp_path: Path) -> AnalyticsDB:
    adb = AnalyticsDB(tmp_path / "a.duckdb")
    rows = [
        _result(
            0,
            season="2425",
            home="Man United",
            away="Aston Villa",
            home_norm="man united",
            away_norm="aston villa",
        ),
        _result(
            1,
            season="2425",
            home="Man City",
            away="Man United",
            home_norm="man city",
            away_norm="man united",
        ),
    ]
    adb.load_results(rows)
    return adb


def test_season_code() -> None:
    assert season_code(2024) == "2425"
    assert season_code(1999) == "9900"


def test_backfill_stores_xg_bridges_names_and_records_absences(tmp_path: Path) -> None:
    calls: list[str] = []
    api = make(tmp_path, handler(calls))
    with seeded(tmp_path) as adb:
        result = backfill_team_xg(api, adb, league_id=39, api_season=2024, division="E0")
        rows = {r.fixture_id: r for r in adb.team_xg_for("E0", "2425")}
    assert result.finished == 3 and result.stored == 3 and result.complete
    assert result.with_xg == 2 and result.requests == 4  # list + three matches
    first = rows[1]
    assert (first.home_norm, first.away_norm) == ("man united", "aston villa")
    assert (first.home_xg, first.away_xg) == (2.95, 1.10)
    assert first.match_date == date(2025, 5, 25)
    assert rows[3].home_xg is None  # stored, so it isn't re-requested every day
    assert not any("fixture=4" in c for c in calls)


def test_resumes_without_respending_and_stops_at_the_budget(tmp_path: Path) -> None:
    calls: list[str] = []
    api = make(tmp_path, handler(calls), daily=3)  # list + 2 matches, then out
    with seeded(tmp_path) as adb:
        first = backfill_team_xg(api, adb, league_id=39, api_season=2024, division="E0")
        assert first.stopped_for_quota and first.stored == 2 and not first.complete
        # Next day: the list comes from the raw store (no request), done matches are skipped.
        api2 = make(tmp_path, handler(calls), daily=100)  # a bigger budget, same ledger
        second = backfill_team_xg(api2, adb, league_id=39, api_season=2024, division="E0")
    assert second.complete and second.requests == 1
    assert sum("/fixtures?" in c for c in calls) == 1
    assert sum("fixture=" in c for c in calls) == 3  # each match exactly once
