"""Ingestion tests: mappers, state precedence, and the pipeline end to end.

The behaviours that matter: source quirks are normalized at the mapper edge, a more
live-capable source is not overwritten by a delayed one regardless of fetch order, and
the pipeline collapses repeated ingests to stable state.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from soccer.domain.match_state import MatchState, MatchStateStore, MatchStatus
from soccer.ingest.mappers import map_football_data_match, map_thesportsdb_live
from soccer.sources.registry import SourceId
from soccer.sources.thesportsdb import parse_live_match
from soccer.storage.live_db import LiveDB

OBSERVED = datetime(2026, 8, 8, 18, 5, tzinfo=UTC)

FD_MATCH = {
    "id": 558214,
    "utcDate": "2026-08-08T18:00:00Z",
    "status": "FINISHED",
    "competition": {"id": 2003, "name": "Eredivisie", "area": {"name": "Netherlands"}},
    "homeTeam": {"id": 674, "name": "PSV"},
    "awayTeam": {"id": 1919, "name": "Fortuna Sittard"},
    "score": {"fullTime": {"home": 3, "away": 1}, "halfTime": {"home": 1, "away": 0}},
}

# A not-yet-drawn knockout tie: the API returns null teams. A wide fixture sweep hits these.
FD_TBD_MATCH = {
    "id": 999999,
    "utcDate": "2026-08-09T18:00:00Z",
    "status": "SCHEDULED",
    "competition": {"id": 2001, "name": "UEFA Champions League", "area": {"name": "Europe"}},
    "homeTeam": {"id": None, "name": None},
    "awayTeam": {"id": None, "name": None},
    "score": {"fullTime": {"home": None, "away": None}, "halfTime": {"home": None, "away": None}},
}


class TestFootballDataMapper:
    def test_maps_identity_and_state(self) -> None:
        obs, state = map_football_data_match(FD_MATCH, observed_at=OBSERVED, is_stale=False)
        assert obs.source_match_id == "558214"
        assert obs.home.name == "PSV"
        assert obs.home.country == "Netherlands"  # taken from competition area
        assert obs.kickoff == datetime(2026, 8, 8, 18, 0, tzinfo=UTC)
        assert state.status is MatchStatus.FINISHED
        assert (state.home_score, state.away_score) == (3, 1)
        assert state.minute is None  # not a live source

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("SCHEDULED", MatchStatus.NOT_STARTED),
            ("TIMED", MatchStatus.NOT_STARTED),
            ("IN_PLAY", MatchStatus.IN_PLAY),
            ("PAUSED", MatchStatus.HALF_TIME),
            ("FINISHED", MatchStatus.FINISHED),
            ("AWARDED", MatchStatus.AWARDED),
            ("SOMETHING_NEW", MatchStatus.UNKNOWN),
        ],
    )
    def test_status_vocabulary(self, raw: str, expected: MatchStatus) -> None:
        _, state = map_football_data_match(
            {**FD_MATCH, "status": raw}, observed_at=OBSERVED, is_stale=False
        )
        assert state.status is expected

    def test_unplayed_match_has_null_scores(self) -> None:
        record = {
            **FD_MATCH,
            "status": "SCHEDULED",
            "score": {"fullTime": {"home": None, "away": None}},
        }
        _, state = map_football_data_match(record, observed_at=OBSERVED, is_stale=False)
        assert state.home_score is None


class TestTheSportsDBMapper:
    def test_maps_live_match(self) -> None:
        live = parse_live_match(
            {
                "idEvent": "2439366",
                "idLeague": "4957",
                "strLeague": "Ecuadorian Serie B",
                "strHomeTeam": "22 de Julio",
                "strAwayTeam": "El Nacional",
                "intHomeScore": 1,
                "intAwayScore": 0,
                "strStatus": "2H",
                "strProgress": "67",
                "strTimestamp": "2026-07-31T20:30:00",
                "updated": "2026-07-31 21:37:00",
            }
        )
        assert live is not None
        obs, state = map_thesportsdb_live(live, observed_at=OBSERVED, is_stale=False)
        assert obs.source_match_id == "2439366"
        assert obs.home.country is None  # TheSportsDB gives no country
        assert state.status is MatchStatus.SECOND_HALF
        assert state.minute == "67"
        assert state.source == SourceId.THESPORTSDB


class TestStatePrecedence:
    """A delayed source must not clobber a more live source's state."""

    @pytest.fixture
    def store(self, tmp_path) -> MatchStateStore:
        db = LiveDB(tmp_path / "live.sqlite")
        # match_state.match_id is a foreign key into canonical_match, so the match must
        # exist first. Create a bare one for the precedence tests to attach state to.
        db.connection.execute(
            "INSERT INTO canonical_match "
            "(internal_id, competition_id, home_team_id, away_team_id, kickoff_utc, created_at) "
            "VALUES ('m1', 'c', 'h', 'a', '2026-08-08T18:00:00+00:00', '2026-08-08T00:00:00+00:00')"
        )
        return MatchStateStore(db)

    def _state(self, source: str, status: MatchStatus, when: datetime) -> MatchState:
        return MatchState(
            status=status,
            home_score=1,
            away_score=0,
            minute=None,
            source=source,
            observed_at=when,
            is_stale=False,
        )

    def test_same_source_always_updates(self, store: MatchStateStore) -> None:
        assert store.upsert(
            "m1", self._state(SourceId.FOOTBALL_DATA_ORG, MatchStatus.NOT_STARTED, OBSERVED)
        )
        assert store.upsert(
            "m1", self._state(SourceId.FOOTBALL_DATA_ORG, MatchStatus.FINISHED, OBSERVED)
        )
        assert store.get("m1").status is MatchStatus.FINISHED

    def test_live_source_beats_delayed_even_if_fetched_earlier(
        self, store: MatchStateStore
    ) -> None:
        # football-data.org (no live latency) is fetched LATER...
        store.upsert(
            "m1",
            self._state(
                SourceId.THESPORTSDB,
                MatchStatus.SECOND_HALF,
                datetime(2026, 8, 8, 18, 4, tzinfo=UTC),
            ),
        )
        changed = store.upsert(
            "m1",
            self._state(
                SourceId.FOOTBALL_DATA_ORG,
                MatchStatus.NOT_STARTED,
                datetime(2026, 8, 8, 18, 5, tzinfo=UTC),
            ),
        )
        # ...but must NOT overwrite the live source's in-play state.
        assert changed is False
        assert store.get("m1").source == SourceId.THESPORTSDB
        assert store.get("m1").status is MatchStatus.SECOND_HALF

    def test_live_source_supersedes_existing_delayed(self, store: MatchStateStore) -> None:
        store.upsert(
            "m1", self._state(SourceId.FOOTBALL_DATA_ORG, MatchStatus.NOT_STARTED, OBSERVED)
        )
        changed = store.upsert(
            "m1", self._state(SourceId.THESPORTSDB, MatchStatus.SECOND_HALF, OBSERVED)
        )
        assert changed is True
        assert store.get("m1").source == SourceId.THESPORTSDB


class TestPipelineIntegration:
    async def test_pipeline_ingests_resolves_and_persists(self, tmp_path) -> None:
        import httpx

        from soccer.ingest.pipeline import IngestPipeline
        from soccer.sources.football_data_org import FootballDataOrg
        from soccer.storage.raw import RawStore

        # A football-data.org adapter backed by a mock transport -- exercises the real
        # wired path (fetch -> map -> resolve -> persist), no network.
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json={"matches": [FD_MATCH]})
        )
        client = httpx.AsyncClient(
            base_url="https://api.football-data.org/v4",
            transport=transport,
            headers={"X-Auth-Token": "test"},
        )
        adapter = FootballDataOrg("test", RawStore(tmp_path / "raw"), client=client)

        db = LiveDB(tmp_path / "live.sqlite")
        pipeline = IngestPipeline(db)
        target = date(2026, 8, 8)

        first = await pipeline.ingest_football_data(adapter, target, target)
        assert first.matches_seen == 1
        assert first.matches_created == 1
        assert first.state_changed == 1

        # Re-ingest the same data: identity and state both idempotent.
        second = await pipeline.ingest_football_data(adapter, target, target)
        assert second.matches_seen == 1
        assert second.matches_created == 0

        view = MatchStateStore(db).list_current()
        assert len(view) == 1
        assert view[0].home == "PSV"
        assert view[0].score == "3-1"
        assert view[0].competition == "Eredivisie"
        assert view[0].status is MatchStatus.FINISHED
        await adapter.aclose()
        db.close()

    async def test_pipeline_skips_unresolvable_fixture(self, tmp_path) -> None:
        # A wide fixture sweep can include a not-yet-drawn tie (null teams). It must be
        # skipped, not abort the whole ingest, and the real match still lands.
        import httpx

        from soccer.ingest.pipeline import IngestPipeline
        from soccer.sources.football_data_org import FootballDataOrg
        from soccer.storage.raw import RawStore

        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json={"matches": [FD_TBD_MATCH, FD_MATCH]})
        )
        client = httpx.AsyncClient(
            base_url="https://api.football-data.org/v4",
            transport=transport,
            headers={"X-Auth-Token": "test"},
        )
        adapter = FootballDataOrg("test", RawStore(tmp_path / "raw"), client=client)
        db = LiveDB(tmp_path / "live.sqlite")
        target = date(2026, 8, 8)

        summary = await IngestPipeline(db).ingest_football_data(adapter, target, target)
        assert summary.skipped == 1  # the TBD tie skipped, not fatal
        assert summary.matches_seen == 1  # the real match still ingested
        assert summary.matches_created == 1
        await adapter.aclose()
        db.close()
