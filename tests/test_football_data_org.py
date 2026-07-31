"""Adapter tests, against mocked transports.

The behaviours worth testing here are the failure paths, not the happy path: budget
exhaustion, tier filtering, and stale-cache fallback. Those are what stand between a
working pipeline and one that silently reports empty data as success.
"""

from __future__ import annotations

from datetime import date
from itertools import pairwise
from pathlib import Path

import httpx
import pytest

from soccer.sources.football_data_org import (
    FREE_COMPETITIONS,
    PARTIAL_COMPETITIONS,
    FootballDataOrg,
    SourceUnavailableError,
)
from soccer.sources.registry import SourceId
from soccer.storage.raw import RawStore


@pytest.fixture
def store(tmp_path: Path) -> RawStore:
    return RawStore(tmp_path / "raw")


def make_adapter(
    store: RawStore, handler: httpx.MockTransport, **kwargs: object
) -> FootballDataOrg:
    client = httpx.AsyncClient(
        base_url="https://api.football-data.org/v4",
        transport=handler,
        headers={"X-Auth-Token": "test"},
    )
    return FootballDataOrg("test", store, client=client, **kwargs)  # type: ignore[arg-type]


COMPETITIONS_PAYLOAD = {
    "count": 13,
    "competitions": [
        {"id": 2021, "code": "PL", "name": "Premier League", "plan": "TIER_ONE"},
        {"id": 2013, "code": "BSA", "name": "Série A", "plan": "TIER_ONE"},
        # Labelled TIER_FOUR, so excluded from the accessible list -- but see
        # test_partial_access_competition_is_allowed_not_rejected: the label does
        # not actually predict access.
        {"id": 2152, "code": "CLI", "name": "Copa Libertadores", "plan": "TIER_FOUR"},
    ],
}


class TestTierFiltering:
    async def test_paid_competitions_are_excluded(self, store: RawStore) -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json=COMPETITIONS_PAYLOAD)
        )
        async with make_adapter(store, transport) as adapter:
            competitions, is_stale = await adapter.competitions()

        codes = {c["code"] for c in competitions}
        assert codes == {"PL", "BSA"}
        assert "CLI" not in codes
        assert not is_stale

    async def test_unknown_competition_rejected_without_spending_budget(
        self, store: RawStore
    ) -> None:
        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(200, json={"matches": []})

        async with make_adapter(store, httpx.MockTransport(handler)) as adapter:
            with pytest.raises(ValueError, match=r"(?i)unknown competition"):
                await adapter.matches(competitions=["XYZ"])
            with pytest.raises(ValueError, match=r"(?i)unknown competition"):
                await adapter.standings("XYZ")

        assert calls == [], "should reject locally, not spend a request"

    async def test_partial_access_competition_is_allowed_not_rejected(
        self, store: RawStore
    ) -> None:
        # CLI reports TIER_FOUR but its standings return full populated tables.
        # The provider's tier label does not predict access, so we must not
        # pre-emptively block it.
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json={"standings": [{"table": []}]})
        )
        async with make_adapter(store, transport) as adapter:
            result = await adapter.standings("CLI")
        assert not result.is_stale

    async def test_free_competition_list_matches_verified_reality(self) -> None:
        # Twelve TIER_ONE competitions, confirmed against the live API.
        assert len(FREE_COMPETITIONS) == 12
        assert "CLI" not in FREE_COMPETITIONS
        assert "CLI" in PARTIAL_COMPETITIONS


class TestRateLimitHandling:
    async def test_server_budget_headers_are_tracked(self, store: RawStore) -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={"matches": []},
                headers={
                    "X-Requests-Available-Minute": "4",
                    "X-RequestCounter-Reset": "60",
                },
            )
        )
        async with make_adapter(store, transport) as adapter:
            await adapter.matches(date_from=date(2026, 7, 31))
            assert adapter.requests_remaining == 4

    async def test_429_retries_then_succeeds(self, store: RawStore) -> None:
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            if attempts["n"] == 1:
                return httpx.Response(429, headers={"Retry-After": "0"}, json={})
            return httpx.Response(200, json={"matches": [{"id": 1}]})

        async with make_adapter(store, httpx.MockTransport(handler)) as adapter:
            result = await adapter.matches(date_from=date(2026, 7, 31))

        assert attempts["n"] == 2
        assert result.payload == {"matches": [{"id": 1}]}
        assert not result.is_stale


class TestFailureHandling:
    async def test_403_is_not_retried(self, store: RawStore) -> None:
        # Tier restrictions are permanent; retrying burns budget we cannot spare.
        attempts = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["n"] += 1
            return httpx.Response(403, json={"message": "restricted"})

        async with make_adapter(store, httpx.MockTransport(handler)) as adapter:
            with pytest.raises(httpx.HTTPStatusError):
                await adapter.matches(date_from=date(2026, 7, 31))

        assert attempts["n"] == 1

    async def test_falls_back_to_cache_and_flags_staleness(self, store: RawStore) -> None:
        good = httpx.MockTransport(
            lambda request: httpx.Response(200, json={"matches": [{"id": 7}]})
        )
        async with make_adapter(store, good) as adapter:
            fresh = await adapter.matches(date_from=date(2026, 7, 31))
        assert not fresh.is_stale

        broken = httpx.MockTransport(lambda request: httpx.Response(500, json={"error": "boom"}))
        async with make_adapter(store, broken, max_retries=1) as adapter:
            stale = await adapter.matches(date_from=date(2026, 7, 31))

        # The critical property: degraded, flagged, and still useful.
        assert stale.is_stale
        assert stale.payload == {"matches": [{"id": 7}]}

    async def test_raises_rather_than_returning_empty_when_no_cache(self, store: RawStore) -> None:
        # An empty result and a broken API must never look identical downstream.
        broken = httpx.MockTransport(lambda request: httpx.Response(500, json={}))
        async with make_adapter(store, broken, max_retries=1) as adapter:
            with pytest.raises(SourceUnavailableError):
                await adapter.matches(date_from=date(2026, 7, 31))


class TestDateRangeCap:
    """The API rejects ranges over 10 days with a 400. Enforce locally, spend nothing."""

    async def test_oversized_range_rejected_without_a_request(self, store: RawStore) -> None:
        calls: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(200, json={"matches": []})

        async with make_adapter(store, httpx.MockTransport(handler)) as adapter:
            with pytest.raises(ValueError, match="10-day limit"):
                await adapter.matches(date_from=date(2026, 8, 1), date_to=date(2026, 8, 31))

        assert calls == []

    async def test_exactly_ten_days_is_allowed(self, store: RawStore) -> None:
        # Inclusive boundary: Aug 1-10 is 10 days and the live API accepts it.
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"matches": []}))
        async with make_adapter(store, transport) as adapter:
            result = await adapter.matches(date_from=date(2026, 8, 1), date_to=date(2026, 8, 10))
        assert not result.is_stale

    async def test_range_helper_chunks_by_ten_days(self, store: RawStore) -> None:
        windows: list[tuple[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            params = request.url.params
            windows.append((params["dateFrom"], params["dateTo"]))
            return httpx.Response(200, json={"matches": [], "n": len(windows)})

        async with make_adapter(store, httpx.MockTransport(handler)) as adapter:
            results = await adapter.matches_over_range(date(2026, 8, 1), date(2026, 8, 25))

        assert len(results) == 3
        assert windows == [
            ("2026-08-01", "2026-08-10"),
            ("2026-08-11", "2026-08-20"),
            ("2026-08-21", "2026-08-25"),
        ]

    async def test_chunks_do_not_overlap_or_gap(self, store: RawStore) -> None:
        windows: list[tuple[date, date]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            params = request.url.params
            windows.append(
                (
                    date.fromisoformat(params["dateFrom"]),
                    date.fromisoformat(params["dateTo"]),
                )
            )
            return httpx.Response(200, json={"matches": []})

        async with make_adapter(store, httpx.MockTransport(handler)) as adapter:
            await adapter.matches_over_range(date(2026, 8, 1), date(2026, 9, 15))

        for (_, prev_end), (next_start, _) in pairwise(windows):
            assert (next_start - prev_end).days == 1, "chunks must be contiguous"
        assert windows[0][0] == date(2026, 8, 1)
        assert windows[-1][1] == date(2026, 9, 15)

    async def test_reversed_range_rejected(self, store: RawStore) -> None:
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"matches": []}))
        async with make_adapter(store, transport) as adapter:
            with pytest.raises(ValueError, match="precedes"):
                await adapter.matches_over_range(date(2026, 8, 10), date(2026, 8, 1))


class TestRawSnapshots:
    async def test_every_response_is_stored(self, store: RawStore) -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json={"matches": [{"id": 1}]})
        )
        async with make_adapter(store, transport) as adapter:
            result = await adapter.matches(date_from=date(2026, 7, 31))

        assert result.snapshot.path.exists()
        assert result.snapshot.was_new
        assert store.latest(SourceId.FOOTBALL_DATA_ORG, "matches") is not None

    async def test_unchanged_response_is_deduplicated(self, store: RawStore) -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json={"matches": [{"id": 1}]})
        )
        async with make_adapter(store, transport) as adapter:
            first = await adapter.matches(date_from=date(2026, 7, 31))
            second = await adapter.matches(date_from=date(2026, 7, 31))

        assert first.snapshot.was_new
        assert not second.snapshot.was_new
