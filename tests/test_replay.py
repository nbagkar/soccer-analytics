"""Replay-from-raw tests.

The point of replay: re-derive all live state from immutable snapshots. The behaviours
that matter are retroactive alias application (the reason we built it), idempotence
(replay of unchanged raw yields identical state), alias preservation across the wipe,
and atomicity (a failure leaves the old state intact, and an empty raw store cannot
silently erase anything).
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from soccer.domain.aliases import AliasStore
from soccer.ingest.pipeline import IngestPipeline
from soccer.sources.football_data_org import FootballDataOrg
from soccer.sources.registry import SourceId
from soccer.storage.live_db import LiveDB
from soccer.storage.raw import RawStore


def fd_match_payload(match_id: int, home: str, away: str) -> dict:
    return {
        "id": match_id,
        "utcDate": "2026-08-08T18:00:00Z",
        "status": "FINISHED",
        "competition": {"id": 2003, "name": "Eredivisie", "area": {"name": "Netherlands"}},
        "homeTeam": {"id": None, "name": home},
        "awayTeam": {"id": None, "name": away},
        "score": {"fullTime": {"home": 2, "away": 1}},
    }


async def ingest_via_adapter(raw: RawStore, db: LiveDB, matches: list[dict]) -> None:
    """Fetch `matches` through a mocked football-data.org adapter, writing raw + state."""
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"matches": matches}))
    client = httpx.AsyncClient(
        base_url="https://api.football-data.org/v4",
        transport=transport,
        headers={"X-Auth-Token": "t"},
    )
    adapter = FootballDataOrg("t", raw, client=client)
    from datetime import date

    await IngestPipeline(db).ingest_football_data(adapter, date(2026, 8, 8), date(2026, 8, 8))
    await adapter.aclose()


class TestRetroactiveAlias:
    async def test_replay_applies_an_alias_added_after_ingest(self, tmp_path) -> None:
        # The headline case. Two sources name a club differently and no alias exists,
        # so it splits into two entities. Adding an alias + rebuilding heals the split.
        raw = RawStore(tmp_path / "raw")
        db = LiveDB(tmp_path / "live.sqlite")

        # Ingest the same fixture from two "sources": one calls the home team "PSV",
        # the other "PSV Eindhoven". Simulate via two payloads through the adapter
        # (different match ids so both persist).
        await ingest_via_adapter(raw, db, [fd_match_payload(1, "PSV", "Ajax")])
        # A second snapshot under the same endpoint with the appended-city spelling.
        await ingest_via_adapter(raw, db, [fd_match_payload(2, "PSV Eindhoven", "Feyenoord")])

        entities_before = db.connection.execute(
            "SELECT COUNT(*) FROM canonical_entity WHERE entity_type='team' "
            "AND normalized_name LIKE 'psv%'"
        ).fetchone()[0]
        assert entities_before == 2  # "psv" and "psv eindhoven" are split

        # Curate the equivalence, then rebuild.
        AliasStore(db).add("team", "PSV Eindhoven", "PSV", country="Netherlands")
        summary = IngestPipeline(db).replay_from_raw(raw)

        entities_after = db.connection.execute(
            "SELECT COUNT(*) FROM canonical_entity WHERE entity_type='team' "
            "AND normalized_name='psv'"
        ).fetchone()[0]
        split_after = db.connection.execute(
            "SELECT COUNT(*) FROM canonical_entity WHERE normalized_name='psv eindhoven'"
        ).fetchone()[0]
        assert entities_after == 1
        assert split_after == 0  # the split entity is gone
        assert summary.snapshots >= 2
        db.close()

    async def test_alias_survives_the_wipe(self, tmp_path) -> None:
        raw = RawStore(tmp_path / "raw")
        db = LiveDB(tmp_path / "live.sqlite")
        await ingest_via_adapter(raw, db, [fd_match_payload(1, "PSV", "Ajax")])
        AliasStore(db).add("team", "PSV Eindhoven", "PSV")

        IngestPipeline(db).replay_from_raw(raw)

        assert len(AliasStore(db).all("team")) == 1  # not deleted by the rebuild
        db.close()


class TestIdempotence:
    async def test_replay_of_unchanged_raw_reproduces_state(self, tmp_path) -> None:
        raw = RawStore(tmp_path / "raw")
        db = LiveDB(tmp_path / "live.sqlite")
        await ingest_via_adapter(
            raw, db, [fd_match_payload(1, "PSV", "Ajax"), fd_match_payload(2, "Feyenoord", "AZ")]
        )

        def snapshot_state() -> list:
            return db.connection.execute(
                "SELECT home_score, away_score, status FROM match_state ORDER BY match_id"
            ).fetchall()

        before_matches = db.connection.execute("SELECT COUNT(*) FROM canonical_match").fetchone()[0]
        IngestPipeline(db).replay_from_raw(raw)
        after_matches = db.connection.execute("SELECT COUNT(*) FROM canonical_match").fetchone()[0]

        assert before_matches == after_matches == 2
        # State content is reproduced.
        assert len(snapshot_state()) == 2
        db.close()


class TestSafety:
    async def test_refuses_to_wipe_when_no_snapshots(self, tmp_path) -> None:
        raw = RawStore(tmp_path / "raw")
        db = LiveDB(tmp_path / "live.sqlite")
        # Put some state in directly, with no raw snapshots backing it.
        db.connection.execute(
            "INSERT INTO canonical_entity VALUES "
            "('e1','team','PSV','psv','Netherlands','2026-07-31')"
        )
        with pytest.raises(ValueError, match="no raw snapshots"):
            IngestPipeline(db).replay_from_raw(raw)
        # State must be untouched -- the guard fired before any DELETE.
        assert db.connection.execute("SELECT COUNT(*) FROM canonical_entity").fetchone()[0] == 1
        db.close()

    async def test_failure_rolls_back_to_previous_state(self, tmp_path, monkeypatch) -> None:
        raw = RawStore(tmp_path / "raw")
        db = LiveDB(tmp_path / "live.sqlite")
        await ingest_via_adapter(raw, db, [fd_match_payload(1, "PSV", "Ajax")])
        before = db.connection.execute("SELECT COUNT(*) FROM canonical_match").fetchone()[0]
        assert before == 1

        # Make the resolver blow up midway through the rebuild.
        pipeline = IngestPipeline(db)
        boom = RuntimeError("resolver exploded")

        def explode(*args, **kwargs):
            raise boom

        monkeypatch.setattr(pipeline._matches, "resolve", explode)
        with pytest.raises(RuntimeError, match="exploded"):
            pipeline.replay_from_raw(raw)

        # The transaction rolled back: the wipe did not stick.
        after = db.connection.execute("SELECT COUNT(*) FROM canonical_match").fetchone()[0]
        assert after == before == 1
        db.close()


class TestRawIteration:
    def test_snapshots_yield_oldest_first(self, tmp_path) -> None:
        raw = RawStore(tmp_path / "raw")
        raw.write(
            SourceId.THESPORTSDB,
            "livescore",
            {"n": 1},
            fetched_at=datetime(2026, 7, 29, 10, 0, tzinfo=UTC),
        )
        raw.write(
            SourceId.THESPORTSDB,
            "livescore",
            {"n": 2},
            fetched_at=datetime(2026, 7, 31, 9, 0, tzinfo=UTC),
        )
        raw.write(
            SourceId.THESPORTSDB,
            "livescore",
            {"n": 3},
            fetched_at=datetime(2026, 7, 29, 18, 0, tzinfo=UTC),
        )
        order = [s.payload["n"] for s in raw.iter_snapshots(SourceId.THESPORTSDB, "livescore")]
        assert order == [1, 3, 2]  # 07-29 10:00, 07-29 18:00, 07-31 09:00
