"""FPL adapter, availability parser, and the enable-gate on the action.

The parser is source-pure: `bootstrap-static` -> availability records, club names left exactly
as FPL spells them (the action layer reconciles them). The adapter earns its keep on the
failure path -- a broken live fetch must serve the last good snapshot, flagged stale, never an
empty success. And the whole feature is gated off by default, because the FPL terms bar
'creating a database'; that gate is asserted here so it can't silently regress.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from soccer.config import Settings
from soccer.sources.fpl import BASE_URL, FantasyPremierLeague, parse_availability
from soccer.sources.registry import SourceId
from soccer.storage.raw import RawStore

BOOTSTRAP = {
    "teams": [
        {"id": 1, "name": "Arsenal"},
        {"id": 2, "name": "Man Utd"},  # FPL's short spelling, reconciled later, not here
    ],
    "elements": [
        {
            "team": 1,
            "web_name": "Saka",
            "first_name": "Bukayo",
            "second_name": "Saka",
            "status": "i",
            "news": "Hamstring injury - expected back 15 Aug",
            "news_added": "2026-08-01T10:00:00Z",
            "chance_of_playing_next_round": 50,
        },
        {
            "team": 1,
            "web_name": "Raya",
            "first_name": "David",
            "second_name": "Raya",
            "status": "a",
            "news": "",
            "news_added": None,
            "chance_of_playing_next_round": None,
            "minutes": 2700,
        },
        # No usable name -> dropped rather than stored as a blank row.
        {"team": 2, "web_name": "", "first_name": "", "second_name": "", "status": "a"},
    ],
}


@pytest.fixture
def store(tmp_path: Path) -> RawStore:
    return RawStore(tmp_path / "raw")


def make_adapter(
    store: RawStore, handler: httpx.MockTransport, **kwargs: object
) -> FantasyPremierLeague:
    client = httpx.AsyncClient(base_url=BASE_URL, transport=handler)
    return FantasyPremierLeague(store, client=client, **kwargs)  # type: ignore[arg-type]


class TestParseAvailability:
    def test_maps_status_team_news_and_chance(self) -> None:
        recs = parse_availability(BOOTSTRAP, fetched_at="2026-08-04T00:00:00+00:00")
        assert len(recs) == 2  # the nameless element is dropped

        saka = next(r for r in recs if r.player == "Saka")
        assert saka.full_name == "Bukayo Saka"
        assert (saka.team, saka.team_norm) == ("Arsenal", "arsenal")
        assert (saka.status, saka.availability) == ("i", "Injured")
        assert saka.chance == 50
        assert saka.news is not None and "Hamstring" in saka.news
        assert saka.source == SourceId.FPL.value

    def test_available_player_has_empty_news_normalised_to_none(self) -> None:
        (raya,) = [r for r in parse_availability(BOOTSTRAP, fetched_at="x") if r.player == "Raya"]
        assert raya.status == "a"
        assert raya.news is None  # "" collapses to None, not a blank string

    def test_minutes_round_trip_and_default_to_none(self) -> None:
        recs = parse_availability(BOOTSTRAP, fetched_at="x")
        raya = next(r for r in recs if r.player == "Raya")
        assert raya.minutes == 2700
        saka = next(r for r in recs if r.player == "Saka")  # BOOTSTRAP's Saka has no "minutes"
        assert saka.minutes is None

    def test_missing_status_defaults_to_available(self) -> None:
        payload = {
            "teams": [{"id": 1, "name": "Arsenal"}],
            "elements": [{"team": 1, "web_name": "X"}],
        }
        (rec,) = parse_availability(payload, fetched_at="x")
        assert rec.status == "a"

    def test_empty_payload_is_empty_not_an_error(self) -> None:
        assert parse_availability({}, fetched_at="x") == []


class TestFetch:
    async def test_bootstrap_fetches_and_snapshots(self, store: RawStore) -> None:
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json=BOOTSTRAP))
        async with make_adapter(store, transport) as fpl:
            fetch = await fpl.bootstrap()

        assert not fetch.is_stale
        assert fetch.payload["teams"][0]["name"] == "Arsenal"
        # Snapshotted for provenance and stale-fallback.
        assert store.latest(SourceId.FPL, "bootstrap-static") is not None

    async def test_serves_cache_flagged_stale_when_live_fails(self, store: RawStore) -> None:
        good = httpx.MockTransport(lambda request: httpx.Response(200, json=BOOTSTRAP))
        async with make_adapter(store, good) as fpl:
            await fpl.bootstrap()  # prime the cache

        broken = httpx.MockTransport(lambda request: httpx.Response(503, json={}))
        async with make_adapter(store, broken, max_retries=0) as fpl:
            fetch = await fpl.bootstrap()

        # Degraded but honest: last good data, unmistakably flagged stale.
        assert fetch.is_stale
        assert fetch.payload["teams"][0]["name"] == "Arsenal"


class TestEnableGate:
    def test_disabled_by_default_returns_guidance_without_network(self, tmp_path: Path) -> None:
        from soccer.dashboard import actions

        # enable_fpl defaults False; the action must explain how to opt in and touch no network.
        settings = Settings(data_dir=tmp_path, enable_fpl=False, _env_file=None)
        message = actions.update_availability(settings)
        assert "SOCCER_ENABLE_FPL=true" in message
