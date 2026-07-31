"""TheSportsDB adapter tests.

Weighted toward malformed input and failure paths. This is the only free live source
and its free access is undocumented, so the behaviour that matters is what happens
when it misbehaves or goes away.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from soccer.sources.registry import SourceId
from soccer.sources.thesportsdb import (
    MatchStatus,
    SourceUnavailableError,
    TheSportsDB,
    parse_live_match,
    parse_progress,
    parse_status,
)
from soccer.storage.raw import RawStore


@pytest.fixture
def store(tmp_path: Path) -> RawStore:
    return RawStore(tmp_path / "raw")


def make_adapter(store: RawStore, transport: httpx.MockTransport, **kwargs: object) -> TheSportsDB:
    client = httpx.AsyncClient(
        base_url="https://www.thesportsdb.com/api/v1/json/123", transport=transport
    )
    return TheSportsDB(store, client=client, **kwargs)  # type: ignore[arg-type]


# Shape taken from a real response.
ROW = {
    "idLiveScore": "14188380",
    "idEvent": "2439366",
    "strSport": "Soccer",
    "idLeague": "4957",
    "strLeague": "Ecuadorian Serie B",
    "idHomeTeam": "151314",
    "idAwayTeam": "138222",
    "strHomeTeam": "22 de Julio",
    "strAwayTeam": "El Nacional",
    "intHomeScore": 0,
    "intAwayScore": 0,
    "strStatus": "1H",
    "strProgress": "24",
    "strTimestamp": "2026-07-31T20:30:00",
    "updated": "2026-07-31 21:55:31",
}


class TestProgressParsing:
    """`strProgress` is a string that can be '24', '90+3', '' or missing."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("24", (24, None)),
            ("90+3", (90, 3)),
            ("90 + 7", (90, 7)),
            ("  45  ", (45, None)),
            ("", (None, None)),
            (None, (None, None)),
            ("HT", (None, None)),
            ("abc", (None, None)),
            (24, (24, None)),
        ],
    )
    def test_parses_without_raising(
        self, value: object, expected: tuple[int | None, int | None]
    ) -> None:
        assert parse_progress(value) == expected

    def test_display_minute_renders_stoppage(self) -> None:
        match = parse_live_match({**ROW, "strProgress": "90+3"})
        assert match is not None
        assert match.display_minute == "90+3"

    def test_display_minute_plain_in_normal_time(self) -> None:
        match = parse_live_match({**ROW, "strProgress": "24"})
        assert match is not None
        assert match.display_minute == "24"


class TestStatusMapping:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("1H", MatchStatus.FIRST_HALF),
            ("2H", MatchStatus.SECOND_HALF),
            ("HT", MatchStatus.HALF_TIME),
            ("FT", MatchStatus.FINISHED),
            ("AET", MatchStatus.FINISHED),
            ("P", MatchStatus.PENALTIES),
            ("NS", MatchStatus.NOT_STARTED),
            ("ft", MatchStatus.FINISHED),
            (" FT ", MatchStatus.FINISHED),
            ("SOMETHING_NEW", MatchStatus.UNKNOWN),
            (None, MatchStatus.UNKNOWN),
        ],
    )
    def test_maps_provider_codes(self, raw: object, expected: MatchStatus) -> None:
        assert parse_status(raw) == expected

    def test_only_in_play_statuses_count_as_live(self) -> None:
        assert MatchStatus.FIRST_HALF.is_in_play
        assert MatchStatus.HALF_TIME.is_in_play
        assert not MatchStatus.FINISHED.is_in_play
        assert not MatchStatus.NOT_STARTED.is_in_play
        assert not MatchStatus.UNKNOWN.is_in_play


class TestRowParsing:
    def test_parses_a_real_row(self) -> None:
        match = parse_live_match(ROW)
        assert match is not None
        assert match.event_id == "2439366"
        assert match.home_team == "22 de Julio"
        assert match.home_score == 0
        assert match.status is MatchStatus.FIRST_HALF
        assert match.minute == 24
        assert match.is_in_play

    def test_naive_timestamps_become_utc_aware(self) -> None:
        match = parse_live_match(ROW)
        assert match is not None
        assert match.updated_at == datetime(2026, 7, 31, 21, 55, 31, tzinfo=UTC)
        assert match.kickoff == datetime(2026, 7, 31, 20, 30, tzinfo=UTC)

    def test_null_scores_survive(self) -> None:
        match = parse_live_match({**ROW, "intHomeScore": None, "intAwayScore": ""})
        assert match is not None
        assert match.home_score is None
        assert match.away_score is None

    def test_row_without_an_id_is_rejected(self) -> None:
        row = {k: v for k, v in ROW.items() if k not in ("idEvent", "idLiveScore")}
        assert parse_live_match(row) is None

    def test_missing_team_names_do_not_raise(self) -> None:
        match = parse_live_match({"idEvent": "1"})
        assert match is not None
        assert match.home_team == "Unknown"


class TestLiveEndpoint:
    async def test_in_play_excludes_finished_matches(self, store: RawStore) -> None:
        # The endpoint returns recently-finished games; 41 of 53 were FT in one
        # real sample. Presenting those as live would be wrong.
        payload = {
            "livescore": [
                {**ROW, "idEvent": "1", "strStatus": "1H"},
                {**ROW, "idEvent": "2", "strStatus": "FT"},
                {**ROW, "idEvent": "3", "strStatus": "HT"},
                {**ROW, "idEvent": "4", "strStatus": "postponed"},
            ]
        }
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
        async with make_adapter(store, transport) as adapter:
            result = await adapter.livescore()

        assert len(result.matches) == 4
        assert {m.event_id for m in result.in_play} == {"1", "3"}

    async def test_one_bad_row_does_not_discard_the_rest(self, store: RawStore) -> None:
        payload = {
            "livescore": [
                {**ROW, "idEvent": "1"},
                {"no_id": True},
                "not a dict",
                {**ROW, "idEvent": "2", "strProgress": "garbage"},
            ]
        }
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
        async with make_adapter(store, transport) as adapter:
            result = await adapter.livescore()

        assert {m.event_id for m in result.matches} == {"1", "2"}

    async def test_empty_feed_is_not_an_error(self, store: RawStore) -> None:
        # A genuinely quiet moment. Distinct from a failure, which raises.
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json={"livescore": None})
        )
        async with make_adapter(store, transport) as adapter:
            result = await adapter.livescore()
        assert result.matches == []
        assert not result.is_stale


class TestDegradation:
    async def test_403_falls_back_to_cache_and_flags_stale(self, store: RawStore) -> None:
        # The scenario this adapter exists for: free access withdrawn.
        good = httpx.MockTransport(
            lambda request: httpx.Response(200, json={"livescore": [{**ROW, "idEvent": "9"}]})
        )
        async with make_adapter(store, good) as adapter:
            await adapter.livescore()

        forbidden = httpx.MockTransport(lambda request: httpx.Response(403, json={}))
        async with make_adapter(store, forbidden) as adapter:
            result = await adapter.livescore()

        assert result.is_stale
        assert [m.event_id for m in result.matches] == ["9"]

    async def test_403_without_cache_raises_rather_than_returning_empty(
        self, store: RawStore
    ) -> None:
        forbidden = httpx.MockTransport(lambda request: httpx.Response(403, json={}))
        async with make_adapter(store, forbidden) as adapter:
            with pytest.raises(SourceUnavailableError):
                await adapter.livescore()

    async def test_non_json_response_is_a_failure_not_empty_data(self, store: RawStore) -> None:
        # An HTML error page must not parse as "no matches today".
        html = httpx.MockTransport(
            lambda request: httpx.Response(200, text="<html>rate limited</html>")
        )
        async with make_adapter(store, html, max_retries=0) as adapter:
            with pytest.raises(SourceUnavailableError):
                await adapter.livescore()

    async def test_raw_snapshot_written_on_success(self, store: RawStore) -> None:
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json={"livescore": [ROW]})
        )
        async with make_adapter(store, transport) as adapter:
            result = await adapter.livescore()

        assert result.snapshot.path.exists()
        assert store.latest(SourceId.THESPORTSDB, "livescore") is not None
