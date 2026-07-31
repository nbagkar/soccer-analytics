"""Immutable raw snapshot store.

Every provider response is written here verbatim before anything parses it. Nothing
in this module ever mutates or deletes an existing snapshot -- normalization reads
from here and writes elsewhere. That gives us three things the original plan asked
for: replay after a parser bug, provenance for every derived value, and evidence when
two sources disagree.

Layout:
    raw/<source>/<endpoint>/<YYYY-MM-DD>/<fetched_at>-<hash12>.json

Content-addressed by payload hash, so re-fetching unchanged data is a cheap no-op
rather than a duplicate file. That matters against football-data.org's 10 req/min
budget, where we poll far more often than the data actually changes.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from soccer.sources.registry import SourceId

SCHEMA_VERSION = 1

# How many date directories back to search when deduplicating. Bounds the per-write
# cost on high-frequency endpoints; see `_find_by_hash`.
DEDUP_WINDOW_DAYS = 2


@dataclass(frozen=True)
class Snapshot:
    """A single stored provider response."""

    source: SourceId
    endpoint: str
    fetched_at: datetime
    payload_hash: str
    path: Path
    was_new: bool
    """False if an identical payload was already stored -- the fetch found no change."""

    @property
    def payload(self) -> Any:
        with gzip.open(self.path, "rt", encoding="utf-8") as handle:
            return json.load(handle)["payload"]


def _canonical_hash(payload: Any) -> str:
    """Stable hash of a payload, independent of key order and whitespace.

    Sorting keys matters: several providers serialize dicts in nondeterministic order,
    and without this every poll would look like a change and store a duplicate.
    """
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _safe_segment(value: str) -> str:
    """Make an endpoint name safe for a path without collapsing distinct names."""
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


class RawStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _dir_for(self, source: SourceId, endpoint: str, when: datetime) -> Path:
        return self.root / source.value / _safe_segment(endpoint) / when.strftime("%Y-%m-%d")

    def write(
        self,
        source: SourceId,
        endpoint: str,
        payload: Any,
        *,
        fetched_at: datetime | None = None,
        request_meta: dict[str, Any] | None = None,
    ) -> Snapshot:
        """Store a response. Returns the existing snapshot if the payload is unchanged."""
        fetched_at = fetched_at or datetime.now(UTC)
        payload_hash = _canonical_hash(payload)
        directory = self._dir_for(source, endpoint, fetched_at)

        existing = self._find_by_hash(source, endpoint, payload_hash, fetched_at)
        if existing is not None:
            return Snapshot(
                source=source,
                endpoint=endpoint,
                fetched_at=fetched_at,
                payload_hash=payload_hash,
                path=existing,
                was_new=False,
            )

        directory.mkdir(parents=True, exist_ok=True)
        filename = f"{fetched_at.strftime('%H%M%S')}-{payload_hash[:12]}.json.gz"
        path = directory / filename

        envelope = {
            "schema_version": SCHEMA_VERSION,
            "source": source.value,
            "endpoint": endpoint,
            "fetched_at": fetched_at.isoformat(),
            "payload_hash": payload_hash,
            "request_meta": request_meta or {},
            "payload": payload,
        }

        # Write-then-rename so a crash mid-write cannot leave a torn snapshot that
        # later reads as valid.
        temp_path = path.with_suffix(".tmp")
        with gzip.open(temp_path, "wt", encoding="utf-8") as handle:
            json.dump(envelope, handle, default=str)
        temp_path.replace(path)

        return Snapshot(
            source=source,
            endpoint=endpoint,
            fetched_at=fetched_at,
            payload_hash=payload_hash,
            path=path,
            was_new=True,
        )

    def _date_dirs_newest_first(self, source: SourceId, endpoint: str) -> list[Path]:
        base = self.root / source.value / _safe_segment(endpoint)
        if not base.is_dir():
            return []
        # Directory names are YYYY-MM-DD, so lexical sort is chronological.
        return sorted((d for d in base.iterdir() if d.is_dir()), reverse=True)

    def _find_by_hash(
        self,
        source: SourceId,
        endpoint: str,
        payload_hash: str,
        when: datetime,
    ) -> Path | None:
        """Locate a recent snapshot with this payload, if any.

        Scoped to snapshots within `DEDUP_WINDOW_DAYS` of `when`, for two reasons.
        Performance: a live endpoint polled every 60s accumulates ~525k files a year,
        and scanning all of them on every write is unbounded work for a near-zero hit
        rate. Correctness: an identical payload recurring months later is a new
        observation, not a duplicate of the old one.

        The window is measured in calendar days, not directory count -- with sparse
        polling those differ, and counting directories would keep an ancient one in
        scope indefinitely.

        Matching is by filename; the hash prefix is in the name precisely so change
        detection costs a directory listing rather than a read.
        """
        pattern = f"*-{payload_hash[:12]}.json.gz"
        cutoff = when.date() - timedelta(days=DEDUP_WINDOW_DAYS)

        for directory in self._date_dirs_newest_first(source, endpoint):
            try:
                directory_date = date.fromisoformat(directory.name)
            except ValueError:
                continue
            if directory_date < cutoff:
                break  # Sorted newest-first, so everything beyond here is older.
            match = next(directory.glob(pattern), None)
            if match is not None:
                return match
        return None

    def latest(self, source: SourceId, endpoint: str) -> Snapshot | None:
        """Most recent snapshot for an endpoint, for serving stale data on failure.
        Walks date directories newest-first and stops at the first hit, so cost is
        independent of how much history has accumulated.
        """
        for directory in self._date_dirs_newest_first(source, endpoint):
            candidates = sorted(directory.glob("*.json.gz"))
            if not candidates:
                continue

            # Filenames start with HHMMSS, so the last is the latest that day.
            path = candidates[-1]
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                envelope = json.load(handle)

            return Snapshot(
                source=source,
                endpoint=endpoint,
                fetched_at=datetime.fromisoformat(envelope["fetched_at"]),
                payload_hash=envelope["payload_hash"],
                path=path,
                was_new=False,
            )
        return None

    def iter_snapshots(self, source: SourceId, endpoint: str) -> Iterator[Snapshot]:
        """Yield every snapshot for an endpoint, oldest first.

        Chronological order is what makes replay correct: identity resolution is
        idempotent, and match-state precedence converges to the latest observation when
        snapshots are folded oldest-to-newest. Date directories sort chronologically and
        filenames start with HHMMSS, so nested sorting gives true fetch order.
        """
        base = self.root / source.value / _safe_segment(endpoint)
        if not base.is_dir():
            return
        for directory in sorted(d for d in base.iterdir() if d.is_dir()):
            for path in sorted(directory.glob("*.json.gz")):
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    envelope = json.load(handle)
                yield Snapshot(
                    source=source,
                    endpoint=endpoint,
                    fetched_at=datetime.fromisoformat(envelope["fetched_at"]),
                    payload_hash=envelope["payload_hash"],
                    path=path,
                    was_new=False,
                )

    def count_snapshots(self, source: SourceId, endpoint: str) -> int:
        base = self.root / source.value / _safe_segment(endpoint)
        return sum(1 for _ in base.rglob("*.json.gz")) if base.is_dir() else 0

    def prune(self, source: SourceId, endpoint: str, *, keep_days: int) -> int:
        """Delete snapshots older than `keep_days`. Returns the number removed.

        High-frequency endpoints need this: livescore at 60s intervals produces about
        2.5 GB and half a million files a year, and a finished match's minute-by-minute
        history has little replay value once results are confirmed elsewhere. Slow
        endpoints (competitions, standings) should never be pruned -- they are cheap
        and are the provenance record.
        """
        directories = self._date_dirs_newest_first(source, endpoint)
        removed = 0
        for directory in directories[keep_days:]:
            for path in directory.glob("*.json.gz"):
                path.unlink()
                removed += 1
            if not any(directory.iterdir()):
                directory.rmdir()
        return removed
