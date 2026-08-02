"""SQLite live-state store.

Holds mutable relational state: canonical entities and the source crosswalk that maps
each provider's identifiers onto them. Distinct from the raw snapshot store (immutable
provider payloads) and from the DuckDB analytics store (columnar history).

Opened in WAL mode with a busy timeout so the scheduler can write while the dashboard
reads -- the original plan called out cricket-mcp's single-writer conflict as a thing
to avoid, and WAL plus a timeout is the standard answer at this scale.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA_VERSION = 4

_SCHEMA = """
CREATE TABLE IF NOT EXISTS canonical_entity (
    internal_id     TEXT PRIMARY KEY,
    entity_type     TEXT NOT NULL,
    canonical_name  TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    country         TEXT,
    created_at      TEXT NOT NULL
);

-- Name resolution looks up candidates by (type, normalized name); country is checked
-- in code because the match rule (equal, or either side unknown) is not a plain SQL
-- equality.
CREATE INDEX IF NOT EXISTS idx_entity_norm
    ON canonical_entity (entity_type, normalized_name);

CREATE TABLE IF NOT EXISTS entity_crosswalk (
    entity_type      TEXT NOT NULL,
    source           TEXT NOT NULL,
    source_entity_id TEXT NOT NULL,
    internal_id      TEXT NOT NULL REFERENCES canonical_entity(internal_id),
    source_name      TEXT,
    method           TEXT NOT NULL,      -- source_id | exact_name | manual
    confidence       REAL NOT NULL,
    first_seen       TEXT NOT NULL,
    last_seen        TEXT NOT NULL,
    PRIMARY KEY (entity_type, source, source_entity_id)
);

-- Reverse lookup: every source identifier attached to one canonical entity.
CREATE INDEX IF NOT EXISTS idx_crosswalk_internal
    ON entity_crosswalk (entity_type, internal_id);

-- A match's identity is its components, not a name: which competition, which two
-- teams, and when. Sources share no match ids, so cross-source recognition is by
-- (competition, home, away, kickoff-within-tolerance); see domain/matches.py.
CREATE TABLE IF NOT EXISTS canonical_match (
    internal_id     TEXT PRIMARY KEY,
    competition_id  TEXT,
    home_team_id    TEXT NOT NULL,
    away_team_id    TEXT NOT NULL,
    kickoff_utc     TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_match_components
    ON canonical_match (competition_id, home_team_id, away_team_id);

CREATE TABLE IF NOT EXISTS match_crosswalk (
    source          TEXT NOT NULL,
    source_match_id TEXT NOT NULL,
    internal_id     TEXT NOT NULL REFERENCES canonical_match(internal_id),
    method          TEXT NOT NULL,      -- source_id | components | manual
    confidence      REAL NOT NULL,
    first_seen      TEXT NOT NULL,
    last_seen       TEXT NOT NULL,
    PRIMARY KEY (source, source_match_id)
);

CREATE INDEX IF NOT EXISTS idx_match_crosswalk_internal
    ON match_crosswalk (internal_id);

-- Current state (score/status) per canonical match: one row, materialized for fast
-- reads. Full history stays in the raw snapshot store. Precedence between sources is
-- enforced in domain/match_state.py, not here.
CREATE TABLE IF NOT EXISTS match_state (
    match_id    TEXT PRIMARY KEY REFERENCES canonical_match(internal_id),
    status      TEXT NOT NULL,
    home_score  INTEGER,
    away_score  INTEGER,
    minute      TEXT,
    source      TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    is_stale    INTEGER NOT NULL,
    updated_at  TEXT NOT NULL
);

-- Curated known-equivalents that normalization cannot bridge: appended cities
-- ("PSV" / "PSV Eindhoven") and translations ("Köln" / "Cologne"). Consulted by the
-- resolver to route a variant spelling onto its canonical form. country_key is the
-- normalized country or '' for a country-agnostic alias.
CREATE TABLE IF NOT EXISTS entity_alias (
    entity_type          TEXT NOT NULL,
    alias_normalized     TEXT NOT NULL,
    country_key          TEXT NOT NULL,
    canonical_name       TEXT NOT NULL,
    canonical_normalized TEXT NOT NULL,
    note                 TEXT,
    created_at           TEXT NOT NULL,
    PRIMARY KEY (entity_type, alias_normalized, country_key)
);
"""


# Columns added to an existing table after its first release, as (table, column, "type ...").
# Registered here so _migrate reconciles them onto already-created databases via ALTER TABLE
# (a bare CREATE TABLE IF NOT EXISTS cannot). Empty today -- the mechanism is the safeguard;
# bump SCHEMA_VERSION and add an entry here whenever you add a column to an existing table.
_ADDED_COLUMNS: tuple[tuple[str, str, str], ...] = ()


class LiveDB:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if self.path.parent != Path():
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._configure()
        self._migrate()

    def _configure(self) -> None:
        # WAL for reader/writer concurrency; busy_timeout so a contended write waits
        # rather than raising "database is locked"; foreign_keys for crosswalk
        # integrity. isolation_level=None above puts us in autocommit, so explicit
        # transactions are opened with BEGIN where needed.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA synchronous=NORMAL")

    def _migrate(self) -> None:
        current = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if current == SCHEMA_VERSION:
            return
        if current > SCHEMA_VERSION:
            raise RuntimeError(
                f"Database at {self.path} is schema v{current}, newer than this code "
                f"(v{SCHEMA_VERSION}). Refusing to run against a future schema."
            )
        self._conn.executescript(_SCHEMA)  # new tables / indexes
        # A bare CREATE TABLE IF NOT EXISTS is a no-op on an existing table, so a column
        # added by editing _SCHEMA would never reach already-created databases -- while
        # user_version still advanced, leaving the DB "current" but missing the column.
        # Reconcile such columns explicitly (SQLite's ADD COLUMN has no IF NOT EXISTS).
        for table, column, ddl in _ADDED_COLUMNS:
            self._ensure_column(table, column, ddl)
        self._conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    def _ensure_column(self, table: str, column: str, ddl: str) -> None:
        existing = {row[1] for row in self._conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Run a block atomically. Everything on this connection commits or rolls back
        together -- used by replay to wipe and rebuild derived state without leaving a
        half-rebuilt database if something fails partway.
        """
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        else:
            self._conn.execute("COMMIT")

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> LiveDB:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
