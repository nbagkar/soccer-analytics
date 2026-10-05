"""Data freshness: when results and fixtures were last checked, and whether that's stale.

Results only advance when something fetches them, and nothing did unless `soccer serve` was
running -- so forecasts silently aged (12 days, when this was added). The fix is to make
staleness visible and act on it: each refresh job stamps `refresh_log`, and the dashboard
refreshes (or warns) once the last *check* is older than `STALE_AFTER`.

Staleness is judged on the last check, not the latest match date: during an international
break or the off-season the newest result is legitimately old, and nagging then would be
noise. The latest match date is still reported, so the user sees how current the data is.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from soccer.storage.live_db import LiveDB

RESULTS = "results"
FIXTURES = "fixtures"
INJURIES = "injuries"
SQUADS = "squads"
JOBS = (RESULTS, FIXTURES, INJURIES, SQUADS)

STALE_AFTER = timedelta(hours=24)
# Rosters change at transfer windows, not daily -- a weekly re-pull is plenty.
SQUADS_STALE_AFTER = timedelta(days=7)


@dataclass(frozen=True)
class RefreshRun:
    job: str
    ran_at: datetime
    ok: bool
    message: str | None


class RefreshLog:
    def __init__(self, db: LiveDB) -> None:
        self._conn = db.connection

    def mark(
        self, job: str, *, ok: bool, message: str | None = None, at: datetime | None = None
    ) -> None:
        if job not in JOBS:
            raise ValueError(f"unknown refresh job {job!r}; expected one of {JOBS}")
        with self._conn:
            self._conn.execute(
                "INSERT INTO refresh_log (job, ran_at, ok, message) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(job) DO UPDATE SET ran_at=excluded.ran_at, ok=excluded.ok, "
                "message=excluded.message",
                (job, (at or datetime.now(UTC)).isoformat(), int(ok), message),
            )

    def last(self, job: str) -> RefreshRun | None:
        row = self._conn.execute(
            "SELECT job, ran_at, ok, message FROM refresh_log WHERE job = ?", (job,)
        ).fetchone()
        if row is None:
            return None
        return RefreshRun(
            row["job"], datetime.fromisoformat(row["ran_at"]), bool(row["ok"]), row["message"]
        )

    def last_success(self, job: str) -> datetime | None:
        run = self.last(job)
        return run.ran_at if run is not None and run.ok else None


@dataclass(frozen=True)
class Freshness:
    results_through: date | None
    """The newest match date in the results store -- how current the data actually is."""
    results_checked_at: datetime | None
    """Last *successful* results refresh; None if never recorded."""
    fixtures_checked_at: datetime | None
    """Last *successful* fixtures refresh; None if never recorded."""
    last_error: str | None
    """The most recent failed refresh's message, if the latest run of any job failed."""
    now: datetime
    injuries_as_of: datetime | None = None
    """When the stored injury news was fetched from its source (the data's own timestamp,
    so a cache fallback still reads as old). None if none is stored."""
    squads_as_of: date | None = None
    """When the stored squads were fetched. None if none are stored."""
    headline_through: date | None = None
    """The newest Premier League result. Shown beside the store-wide date when it lags, so
    a busy MLS/Brazil calendar can't make the headline league look more current than it is
    (football-data.co.uk publishes the European files on its own, sometimes slower, cadence)."""

    @property
    def is_stale(self) -> bool:
        checked = self.results_checked_at
        return checked is None or self.now - checked > STALE_AFTER

    @property
    def fixtures_stale(self) -> bool:
        checked = self.fixtures_checked_at
        return checked is None or self.now - checked > STALE_AFTER

    @property
    def injuries_stale(self) -> bool:
        return self.injuries_as_of is None or self.now - self.injuries_as_of > STALE_AFTER

    @property
    def squads_stale(self) -> bool:
        if self.squads_as_of is None:
            return True
        return self.now.date() - self.squads_as_of > SQUADS_STALE_AFTER

    @property
    def checked_label(self) -> str:
        return age_label(self.results_checked_at, self.now)


def age_label(when: datetime | None, now: datetime) -> str:
    """'just now' / '5h ago' / '3 days ago' / 'never' -- for staleness captions."""
    if when is None:
        return "never"
    hours = (now - when).total_seconds() / 3600
    if hours < 1:
        return "just now"
    if hours < 48:
        return f"{int(hours)}h ago"
    return f"{int(hours // 24)} days ago"


def freshness(
    live_db: LiveDB,
    results_through: date | None,
    *,
    squads_as_of: date | None = None,
    headline_through: date | None = None,
    now: datetime | None = None,
) -> Freshness:
    log = RefreshLog(live_db)
    errors = [r.message for r in (log.last(j) for j in JOBS) if r is not None and not r.ok]
    return Freshness(
        results_through=results_through,
        results_checked_at=log.last_success(RESULTS),
        fixtures_checked_at=log.last_success(FIXTURES),
        last_error=errors[0] if errors else None,
        injuries_as_of=injuries_as_of(live_db),
        squads_as_of=squads_as_of,
        headline_through=headline_through,
        now=now or datetime.now(UTC),
    )


def injuries_as_of(live_db: LiveDB) -> datetime | None:
    row = live_db.connection.execute(
        "SELECT MAX(fetched_at) AS t FROM player_availability"
    ).fetchone()
    return datetime.fromisoformat(row["t"]) if row and row["t"] else None
