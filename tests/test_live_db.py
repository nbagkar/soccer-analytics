"""LiveDB schema-migration tests -- fresh stamp, additive/idempotent column reconcile,
and the refusal to run against a newer schema."""

from __future__ import annotations

import pytest

from soccer.storage.live_db import SCHEMA_VERSION, LiveDB


def _columns(db: LiveDB, table: str) -> set[str]:
    return {row[1] for row in db.connection.execute(f"PRAGMA table_info({table})")}


def test_fresh_db_is_stamped_current(tmp_path) -> None:
    with LiveDB(tmp_path / "live.sqlite") as db:
        assert db.connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_migration_creates_player_availability(tmp_path) -> None:
    # The ephemeral FPL availability cache is a plain new table, so a bare migration must
    # materialize it (and its lookup index) on a fresh stamp.
    with LiveDB(tmp_path / "live.sqlite") as db:
        assert "player_availability" in {
            row[0]
            for row in db.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert "team_norm" in _columns(db, "player_availability")


def test_ensure_column_adds_missing_and_is_idempotent(tmp_path) -> None:
    # The safeguard for the case a bare CREATE TABLE IF NOT EXISTS can't handle: adding a
    # column to an already-existing table. Must add once and be a no-op thereafter.
    with LiveDB(tmp_path / "live.sqlite") as db:
        assert "extra_note" not in _columns(db, "match_state")
        db._ensure_column("match_state", "extra_note", "TEXT")
        assert "extra_note" in _columns(db, "match_state")
        db._ensure_column("match_state", "extra_note", "TEXT")  # second call must not raise
        assert "extra_note" in _columns(db, "match_state")


def test_refuses_future_schema(tmp_path) -> None:
    path = tmp_path / "live.sqlite"
    with LiveDB(path) as db:
        db.connection.execute(f"PRAGMA user_version={SCHEMA_VERSION + 1}")
    with pytest.raises(RuntimeError, match="newer than this code"):
        LiveDB(path)
