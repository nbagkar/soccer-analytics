"""Local usage log: which pages, actions and assistant intents actually get used.

Feature decisions were being made without evidence -- several features shipped and were
never touched (an empty bet ledger, a lineup signal that never fired). This records just
enough to tell used from unused: one row per page visit, dashboard action, or assistant
question, tagged with what it was. It lives in the local live DB and never leaves the
machine.

Deliberately coarse. Question text is kept only when the assistant could not answer
(intent "fallback") -- that is the unmet demand worth reading; answered questions are
recorded by intent alone. `soccer usage` reports it.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from soccer.config import Settings

logger = logging.getLogger(__name__)

PAGE = "page"
ACTION = "action"
ASK = "ask"
KINDS = (PAGE, ACTION, ASK)

FALLBACK_INTENT = "fallback"


@dataclass(frozen=True)
class UsageCount:
    kind: str
    name: str
    count: int
    last_at: datetime


@dataclass(frozen=True)
class UsageSummary:
    counts: list[UsageCount]
    """Every (kind, name) seen in the window, most-used first within each kind."""
    unanswered: list[tuple[datetime, str]]
    """Assistant questions that fell through to the fallback, newest first."""
    tracking_since: datetime | None
    """The earliest event ever recorded -- how much history the counts can rest on."""

    def for_kind(self, kind: str) -> list[UsageCount]:
        return [c for c in self.counts if c.kind == kind]

    def unused(self, kind: str, known: Iterable[str]) -> list[str]:
        """Names from `known` with no events of `kind` in the window, in `known`'s order."""
        seen = {c.name for c in self.for_kind(kind)}
        return [name for name in known if name not in seen]


_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage_event (
    id     INTEGER PRIMARY KEY,
    at     TEXT NOT NULL,
    kind   TEXT NOT NULL,       -- page | action | ask
    name   TEXT NOT NULL,       -- page key, action name, or assistant intent
    detail TEXT
);
CREATE INDEX IF NOT EXISTS idx_usage_kind_at ON usage_event (kind, at);
"""


class UsageDB:
    """Its own small SQLite file (`Settings.usage_db`), separate from the live store.

    It lived in live.sqlite at first, and every logged page view changed that file's mtime --
    which keys the dashboard's fixture/season caches -- so ordinary browsing kept throwing
    away 3-10s of cached forecasts. Data caches must move only when data does.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> UsageDB:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class UsageLog:
    def __init__(self, db: UsageDB) -> None:
        self._conn = db.connection

    def record(
        self, kind: str, name: str, detail: str | None = None, *, at: datetime | None = None
    ) -> None:
        if kind not in KINDS:
            raise ValueError(f"unknown usage kind {kind!r}; expected one of {KINDS}")
        when = (at or datetime.now(UTC)).isoformat()
        with self._conn:
            self._conn.execute(
                "INSERT INTO usage_event (at, kind, name, detail) VALUES (?, ?, ?, ?)",
                (when, kind, name, detail),
            )

    def summary(self, since: datetime) -> UsageSummary:
        cutoff = since.isoformat()
        rows = self._conn.execute(
            "SELECT kind, name, COUNT(*) AS n, MAX(at) AS last_at FROM usage_event "
            "WHERE at >= ? GROUP BY kind, name ORDER BY kind, n DESC, name",
            (cutoff,),
        ).fetchall()
        counts = [
            UsageCount(r["kind"], r["name"], r["n"], datetime.fromisoformat(r["last_at"]))
            for r in rows
        ]
        unanswered = [
            (datetime.fromisoformat(r["at"]), r["detail"])
            for r in self._conn.execute(
                "SELECT at, detail FROM usage_event WHERE kind = ? AND name = ? AND at >= ? "
                "AND detail IS NOT NULL ORDER BY at DESC",
                (ASK, FALLBACK_INTENT, cutoff),
            ).fetchall()
        ]
        first = self._conn.execute("SELECT MIN(at) AS t FROM usage_event").fetchone()
        since_at = datetime.fromisoformat(first["t"]) if first and first["t"] else None
        return UsageSummary(counts=counts, unanswered=unanswered, tracking_since=since_at)


def track(settings: Settings, kind: str, name: str, detail: str | None = None) -> None:
    """Record one event, never letting a logging failure break the page that called it."""
    if not settings.usage_tracking:
        return
    try:
        with UsageDB(settings.usage_db) as db:
            UsageLog(db).record(kind, name, detail)
    except (sqlite3.Error, OSError) as exc:
        logger.warning("usage tracking failed for %s/%s: %s", kind, name, exc)
