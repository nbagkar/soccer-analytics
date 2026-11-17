"""API-Football adapter, quota ledger and coverage probe -- all against mocked HTTP.

The failure paths matter most: an `errors` object beside an empty `response` (the shape the
real API returned for a keyless request on 2026-11-16) must raise, never read as "no data";
and a request that would breach the daily quota must be refused before it is sent.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from soccer.config import Settings
from soccer.ingest.quota import QuotaExhaustedError, QuotaLedger
from soccer.sources.api_football import ApiFootball, ApiFootballError, from_settings
from soccer.sources.api_football_coverage import (
    DENIED,
    EMPTY,
    OK,
    SKIPPED,
    coverage_markdown,
    probe_coverage,
)
from soccer.sources.errors import SourceUnavailableError
from soccer.sources.registry import SOURCES, SourceId
from soccer.storage.raw import RawStore

# Verbatim shape of the real keyless answer (HTTP 403).
MISSING_KEY = {
    "get": "",
    "parameters": [],
    "errors": {"token": "Missing application key, Check our documentation."},
    "results": 0,
    "paging": {"current": 1, "total": 1},
    "response": [],
}


def ok(response: Any, **extra: Any) -> dict[str, Any]:
    n = len(response) if isinstance(response, list) else 1
    return {
        "errors": [],
        "results": n,
        "paging": {"current": 1, "total": 1},
        "response": response,
    } | extra


class Clock:
    def __init__(self) -> None:
        self.t = 0.0
        self.slept: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.slept.append(s)
        self.t += s


def make(
    tmp_path: Path,
    handler: Any,
    *,
    daily: int = 100,
    reserve: int = 0,
    clock: Clock | None = None,
) -> ApiFootball:
    clock = clock or Clock()
    ledger = QuotaLedger(
        tmp_path / "quota.sqlite", "api_football", daily_limit=daily, reserve=reserve
    )
    return ApiFootball(
        "test-key",
        RawStore(tmp_path / "raw"),
        ledger,
        per_minute=10,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        clock=clock.now,
        sleep=clock.sleep,
    )


class TestEnvelope:
    def test_errors_beside_an_empty_response_raise(self, tmp_path: Path) -> None:
        api = make(tmp_path, lambda r: httpx.Response(403, json=MISSING_KEY))
        with pytest.raises(ApiFootballError) as info:
            api.get("/fixtures", {"league": 39})
        assert info.value.kinds == ["token"]

    def test_errors_with_http_200_still_raise(self, tmp_path: Path) -> None:
        body = ok([]) | {"errors": {"plan": "Free plans do not have access to this season."}}
        api = make(tmp_path, lambda r: httpx.Response(200, json=body))
        with pytest.raises(ApiFootballError) as info:
            api.get("/fixtures", {"league": 39, "season": 2026})
        assert info.value.is_plan_restriction

    def test_error_list_form_raises(self, tmp_path: Path) -> None:
        body = ok([]) | {"errors": ["something went wrong"]}
        api = make(tmp_path, lambda r: httpx.Response(200, json=body))
        with pytest.raises(ApiFootballError):
            api.get("/status")

    def test_non_json_and_missing_response_are_failures(self, tmp_path: Path) -> None:
        api = make(tmp_path, lambda r: httpx.Response(502, text="<html>bad gateway</html>"))
        with pytest.raises(SourceUnavailableError):
            api.get("/status")
        api = make(tmp_path, lambda r: httpx.Response(200, json={"errors": []}))
        with pytest.raises(SourceUnavailableError):
            api.get("/status")

    def test_a_genuinely_empty_success_is_allowed_and_snapshotted(self, tmp_path: Path) -> None:
        api = make(tmp_path, lambda r: httpx.Response(200, json=ok([])))
        r = api.get("/odds", {"league": 39, "season": 2026})
        assert r.results == 0 and r.response == []
        key = ApiFootball.snapshot_key("/odds", {"league": 39, "season": 2026})
        assert key == "odds__league-39_season-2026"
        assert RawStore(tmp_path / "raw").latest(SourceId.API_FOOTBALL, key) is not None

    def test_key_header_is_sent(self, tmp_path: Path) -> None:
        seen: list[str | None] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request.headers.get("x-apisports-key"))
            return httpx.Response(200, json=ok([]))

        make(tmp_path, handler).get("/status")
        assert seen == ["test-key"]


RATE_LIMITED: dict[str, Any] = {
    "errors": {"rateLimit": "Too many requests. ... or upgrade your plan for higher limits."},
    "results": 0,
    "response": [],
}


class TestRateLimit:
    def test_rate_limit_is_not_a_plan_refusal_and_is_retried_once(self, tmp_path: Path) -> None:
        replies = [httpx.Response(200, json=RATE_LIMITED), httpx.Response(200, json=ok([1]))]
        clock = Clock()
        api = make(tmp_path, lambda r: replies.pop(0), clock=clock)
        assert api.get("/standings").response == [1]
        assert 61.0 in clock.slept

    def test_a_second_rate_limit_is_reported_not_misread(self, tmp_path: Path) -> None:
        api = make(tmp_path, lambda r: httpx.Response(200, json=RATE_LIMITED))
        with pytest.raises(ApiFootballError) as info:
            api.get("/standings")
        assert info.value.is_rate_limited and not info.value.is_plan_restriction


class TestQuota:
    def test_refuses_before_sending_when_the_day_is_spent(self, tmp_path: Path) -> None:
        sent: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            sent.append(request.url.path)
            return httpx.Response(200, json=ok([]))

        api = make(tmp_path, handler, daily=2)
        api.get("/a")
        api.get("/b")
        with pytest.raises(QuotaExhaustedError):
            api.get("/c")
        assert sent == ["/a", "/b"]

    def test_server_count_wins(self, tmp_path: Path) -> None:
        headers = {"x-ratelimit-requests-limit": "100", "x-ratelimit-requests-remaining": "7"}
        api = make(tmp_path, lambda r: httpx.Response(200, json=ok([]), headers=headers))
        api.get("/status")
        st = api.ledger.status()
        assert (st.used, st.limit, st.remaining) == (93, 100, 7)

    def test_reserve_is_held_back_from_jobs_but_not_manual_checks(self, tmp_path: Path) -> None:
        api = make(tmp_path, lambda r: httpx.Response(200, json=ok([])), daily=3, reserve=2)
        api.get("/a")
        with pytest.raises(QuotaExhaustedError):
            api.get("/b")  # 2 left, both reserved
        api.get("/b", use_reserve=True)
        api.get("/c", use_reserve=True)
        with pytest.raises(QuotaExhaustedError):
            api.get("/d", use_reserve=True)  # the hard limit is never crossed

    def test_days_roll_over_at_utc_midnight(self, tmp_path: Path) -> None:
        now = [datetime(2026, 11, 16, 23, 59, tzinfo=UTC)]
        ledger = QuotaLedger(tmp_path / "q.sqlite", "x", daily_limit=1, clock=lambda: now[0])
        ledger.record()
        with pytest.raises(QuotaExhaustedError):
            ledger.check()
        now[0] += timedelta(minutes=2)
        ledger.check()  # a new UTC day

    def test_a_failed_request_still_counts(self, tmp_path: Path) -> None:
        api = make(tmp_path, lambda r: httpx.Response(403, json=MISSING_KEY))
        with pytest.raises(ApiFootballError):
            api.get("/status")
        assert api.ledger.status().used == 1


class TestPacingAndCache:
    def test_requests_are_spaced_under_the_per_minute_cap(self, tmp_path: Path) -> None:
        clock = Clock()
        api = make(tmp_path, lambda r: httpx.Response(200, json=ok([])), clock=clock)
        for _ in range(3):
            api.get("/status")
        assert clock.slept == [6.0, 6.0]  # 10/min -> one every 6s

    def test_fresh_snapshot_is_served_without_spending_quota(self, tmp_path: Path) -> None:
        calls: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(1)
            return httpx.Response(200, json=ok([{"id": 1}]))

        api = make(tmp_path, handler)
        api.get("/leagues", {"id": 39})
        hit = api.get("/leagues", {"id": 39}, max_age=timedelta(hours=1))
        assert hit.from_cache and hit.response == [{"id": 1}]
        assert len(calls) == 1 and api.ledger.status().used == 1


class TestSettings:
    def test_off_without_a_key_and_key_never_in_repr(self, tmp_path: Path) -> None:
        off = Settings(data_dir=tmp_path, api_football_key=None)
        assert not off.is_enabled(SourceId.API_FOOTBALL)
        with pytest.raises(SourceUnavailableError):
            from_settings(off)
        on = Settings(data_dir=tmp_path, api_football_key="s3cret-value")  # type: ignore[arg-type]
        assert on.is_enabled(SourceId.API_FOOTBALL)
        assert "s3cret-value" not in repr(on)
        assert from_settings(on).ledger.status().limit == 100

    def test_registry_grants_no_capability_until_probed(self) -> None:
        src = SOURCES[SourceId.API_FOOTBALL]
        assert src.capabilities == frozenset()
        assert not src.enabled_by_default


def _fixture(fid: int, short: str, when: str) -> dict[str, Any]:
    return {
        "fixture": {"id": fid, "date": when, "status": {"short": short}},
        "teams": {"home": {"name": "Arsenal"}, "away": {"name": "Chelsea"}},
    }


def fake_api(denied_league: int | None = None) -> Any:
    """A handler shaped like the real API: one league refused by plan, the rest served."""

    def handler(request: httpx.Request) -> httpx.Response:
        path, q = request.url.path, dict(request.url.params)
        if path == "/status":
            return httpx.Response(
                200,
                json=ok(
                    {
                        "subscription": {"plan": "Free", "active": True},
                        "requests": {"current": 1, "limit_day": 100},
                    }
                ),
            )
        if denied_league is not None and q.get("league", q.get("id")) == str(denied_league):
            return httpx.Response(
                200,
                json=ok([]) | {"errors": {"plan": "Free plans do not have access to this season."}},
            )
        if path == "/leagues":
            return httpx.Response(
                200,
                json=ok(
                    [
                        {
                            "seasons": [
                                {
                                    "year": 2026,
                                    "start": "2026-08-15",
                                    "end": "2027-05-23",
                                    "coverage": {"fixtures": {"lineups": True}, "odds": True},
                                }
                            ]
                        }
                    ]
                ),
            )
        if path == "/fixtures":
            return httpx.Response(
                200,
                json=ok(
                    [
                        _fixture(1, "FT", "2026-11-08T15:00:00+00:00"),
                        _fixture(2, "NS", "2026-11-21T15:00:00+00:00"),
                    ]
                ),
            )
        if path == "/fixtures/statistics":
            stats = [
                {"type": "Shots on Goal", "value": 5},
                {"type": "expected_goals", "value": "1.2"},
            ]
            return httpx.Response(200, json=ok([{"statistics": stats}, {"statistics": stats}]))
        if path == "/odds":
            bets = [{"name": "Match Winner"}, {"name": "Both Teams Score"}]
            return httpx.Response(200, json=ok([{"bookmakers": [{"bets": bets}]}]))
        if path == "/injuries":
            return httpx.Response(200, json=ok([]))
        return httpx.Response(200, json=ok([{"x": 1}]))

    return handler


class TestCoverageProbe:
    def test_a_past_season_is_probed_at_its_end_not_today(self, tmp_path: Path) -> None:
        asked: list[dict[str, str]] = []
        handler = fake_api()

        def spy(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/fixtures":
                asked.append(dict(request.url.params))
            return handler(request)

        api = make(tmp_path, spy)
        probe_coverage(api, season=2026, leagues=[39], today=date(2027, 11, 1))
        assert asked[0]["from"] == "2027-05-09" and asked[0]["to"] == "2027-05-23"

    def test_matrix_classifies_ok_empty_and_denied(self, tmp_path: Path) -> None:
        api = make(tmp_path, fake_api(denied_league=2))
        report = probe_coverage(api, season=2026, leagues=[39, 2], today=date(2026, 11, 16))
        pl, ucl = report.cells[39], report.cells[2]
        assert report.account["plan"] == "Free"
        assert pl["season"].status == OK
        assert pl["statistics"].note == "xG present"
        assert pl["odds"].note == "BTTS priced"
        assert pl["injuries"].status == EMPTY
        assert "Arsenal v Chelsea (1)" in pl["fixtures"].note  # the finished one, not the NS
        assert ucl["season"].status == DENIED and "Free plans" in ucl["season"].note
        # A refused season stops that league: every other query names the same season.
        assert all(c.status == SKIPPED for ep, c in ucl.items() if ep != "season")
        assert report.declared[39]["odds"] is True
        assert report.requests_used == 1 + 10 + 1  # status, the full PL sweep, UCL's refusal
        md = coverage_markdown(report)
        assert "| statistics | ✅ 2 | · |" in md and "⛔" in md

    def test_running_out_of_quota_marks_the_rest_skipped(self, tmp_path: Path) -> None:
        api = make(tmp_path, fake_api(), daily=5)
        report = probe_coverage(api, season=2026, leagues=[39, 140], today=date(2026, 11, 16))
        assert report.requests_used == 5
        assert all(c.status == SKIPPED for c in report.cells[140].values())
        assert json.dumps(coverage_markdown(report))  # renders
