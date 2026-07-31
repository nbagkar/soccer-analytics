"""Raw store tests -- the deduplication and crash-safety behaviour specifically."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from soccer.sources.registry import SourceId
from soccer.storage.raw import RawStore


@pytest.fixture
def store(tmp_path: Path) -> RawStore:
    return RawStore(tmp_path / "raw")


class TestDeduplication:
    def test_first_write_is_new(self, store: RawStore) -> None:
        snapshot = store.write(SourceId.THESPORTSDB, "livescore", {"events": []})
        assert snapshot.was_new
        assert snapshot.path.exists()

    def test_identical_payload_is_not_rewritten(self, store: RawStore) -> None:
        payload = {"events": [{"id": 1, "score": "0-0"}]}
        first = store.write(SourceId.THESPORTSDB, "livescore", payload)
        second = store.write(SourceId.THESPORTSDB, "livescore", payload)

        assert first.was_new
        assert not second.was_new
        assert first.path == second.path

    def test_changed_payload_creates_a_new_snapshot(self, store: RawStore) -> None:
        first = store.write(SourceId.THESPORTSDB, "livescore", {"score": "0-0"})
        second = store.write(SourceId.THESPORTSDB, "livescore", {"score": "1-0"})

        assert second.was_new
        assert first.path != second.path

    def test_key_order_does_not_count_as_a_change(self, store: RawStore) -> None:
        # Providers serialize dicts nondeterministically; without canonical hashing
        # every poll would store a duplicate and defeat the point of the cache.
        first = store.write(SourceId.THESPORTSDB, "livescore", {"a": 1, "b": 2})
        second = store.write(SourceId.THESPORTSDB, "livescore", {"b": 2, "a": 1})

        assert not second.was_new
        assert first.payload_hash == second.payload_hash

    def test_same_payload_from_different_sources_stays_separate(self, store: RawStore) -> None:
        payload = {"score": "1-0"}
        first = store.write(SourceId.THESPORTSDB, "match", payload)
        second = store.write(SourceId.OPENLIGADB, "match", payload)

        assert second.was_new
        assert first.path != second.path


class TestRoundTrip:
    def test_payload_survives_storage(self, store: RawStore) -> None:
        payload = {"events": [{"id": 1, "minute": 15}], "unicode": "Köln"}
        snapshot = store.write(SourceId.OPENLIGADB, "getmatchdata", payload)
        assert snapshot.payload == payload

    def test_latest_returns_most_recent(self, store: RawStore) -> None:
        store.write(
            SourceId.THESPORTSDB,
            "livescore",
            {"minute": 15},
            fetched_at=datetime(2026, 7, 31, 10, 0, tzinfo=UTC),
        )
        store.write(
            SourceId.THESPORTSDB,
            "livescore",
            {"minute": 16},
            fetched_at=datetime(2026, 7, 31, 10, 1, tzinfo=UTC),
        )

        latest = store.latest(SourceId.THESPORTSDB, "livescore")
        assert latest is not None
        assert latest.payload == {"minute": 16}

    def test_latest_is_none_when_nothing_stored(self, store: RawStore) -> None:
        # Must not raise: callers use this to decide whether to serve stale data.
        assert store.latest(SourceId.FOOTBALL_DATA_ORG, "matches") is None


class TestRetention:
    """Live endpoints accumulate ~525k files a year; bounded scans and pruning matter."""

    def test_dedup_scan_is_bounded_to_recent_days(self, store: RawStore) -> None:
        # An identical payload from long ago is a new observation, not a duplicate.
        old = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
        recent = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)
        payload = {"score": "1-0"}

        first = store.write(SourceId.THESPORTSDB, "livescore", payload, fetched_at=old)
        second = store.write(SourceId.THESPORTSDB, "livescore", payload, fetched_at=recent)

        assert first.was_new
        assert second.was_new, "payload outside the dedup window must be re-recorded"
        assert first.path != second.path

    def test_dedup_still_applies_within_the_window(self, store: RawStore) -> None:
        when = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)
        payload = {"score": "1-0"}

        first = store.write(SourceId.THESPORTSDB, "livescore", payload, fetched_at=when)
        second = store.write(
            SourceId.THESPORTSDB,
            "livescore",
            payload,
            fetched_at=when.replace(hour=18),
        )

        assert first.was_new
        assert not second.was_new

    def test_latest_finds_newest_across_date_directories(self, store: RawStore) -> None:
        store.write(
            SourceId.THESPORTSDB,
            "livescore",
            {"n": 1},
            fetched_at=datetime(2026, 7, 29, 12, 0, tzinfo=UTC),
        )
        store.write(
            SourceId.THESPORTSDB,
            "livescore",
            {"n": 2},
            fetched_at=datetime(2026, 7, 31, 9, 0, tzinfo=UTC),
        )

        latest = store.latest(SourceId.THESPORTSDB, "livescore")
        assert latest is not None
        assert latest.payload == {"n": 2}

    def test_prune_removes_old_days_only(self, store: RawStore) -> None:
        for day in (10, 20, 29, 30, 31):
            store.write(
                SourceId.THESPORTSDB,
                "livescore",
                {"day": day},
                fetched_at=datetime(2026, 7, day, 12, 0, tzinfo=UTC),
            )

        removed = store.prune(SourceId.THESPORTSDB, "livescore", keep_days=2)

        assert removed == 3
        remaining = list(store.root.rglob("*.json.gz"))
        assert len(remaining) == 2
        # Newest survives and is still readable.
        latest = store.latest(SourceId.THESPORTSDB, "livescore")
        assert latest is not None
        assert latest.payload == {"day": 31}

    def test_prune_is_a_noop_when_nothing_is_old(self, store: RawStore) -> None:
        store.write(SourceId.THESPORTSDB, "livescore", {"a": 1})
        assert store.prune(SourceId.THESPORTSDB, "livescore", keep_days=7) == 0

    def test_prune_on_empty_endpoint_does_not_raise(self, store: RawStore) -> None:
        assert store.prune(SourceId.FOOTBALL_DATA_ORG, "matches", keep_days=1) == 0


class TestCrashSafety:
    def test_no_temp_files_survive_a_successful_write(self, store: RawStore) -> None:
        store.write(SourceId.THESPORTSDB, "livescore", {"a": 1})
        assert list(store.root.rglob("*.tmp")) == []

    def test_endpoint_names_with_slashes_do_not_escape_the_store(self, store: RawStore) -> None:
        snapshot = store.write(SourceId.FOOTBALL_DATA_ORG, "../../etc/passwd", {"a": 1})
        assert store.root.resolve() in snapshot.path.resolve().parents
