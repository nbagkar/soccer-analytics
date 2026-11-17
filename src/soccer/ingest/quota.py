"""Daily request ledger for quota'd APIs.

A paid or free-tier API with a hard daily quota needs two things a per-minute pacer can't
give: a refusal BEFORE a request that would breach the budget, and a durable count that
survives restarts and is shared by every process (dashboard, `soccer serve`, the CLI).

The server's own count wins whenever a response reports it (API-Football sends
`x-ratelimit-requests-remaining` / `-limit` on every reply), because it also sees requests
made from elsewhere with the same key. Days are UTC, matching the provider's reset.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS api_quota (
    source        TEXT NOT NULL,
    day           TEXT NOT NULL,     -- UTC date, YYYY-MM-DD
    used          INTEGER NOT NULL,  -- requests made (server's count when it reported one)
    server_limit  INTEGER,
    updated_at    TEXT NOT NULL,
    PRIMARY KEY (source, day)
);
"""


class QuotaExhaustedError(RuntimeError):
    """The request would breach today's budget -- refused before it was sent."""


@dataclass(frozen=True)
class QuotaStatus:
    day: str
    used: int
    limit: int
    reserve: int

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    @property
    def available_to_jobs(self) -> int:
        """What automated jobs may spend: the remainder minus the held-back reserve."""
        return max(0, self.remaining - self.reserve)


class QuotaLedger:
    def __init__(
        self,
        path: Path,
        source: str,
        *,
        daily_limit: int,
        reserve: int = 0,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.source = source
        self.daily_limit = daily_limit
        self.reserve = reserve
        self._clock = clock
        with self._connect() as con:
            con.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    def _day(self) -> str:
        return self._clock().astimezone(UTC).date().isoformat()

    def status(self) -> QuotaStatus:
        day = self._day()
        with self._connect() as con:
            row = con.execute(
                "SELECT used, server_limit FROM api_quota WHERE source=? AND day=?",
                (self.source, day),
            ).fetchone()
        used, server_limit = row if row else (0, None)
        return QuotaStatus(day, used, server_limit or self.daily_limit, self.reserve)

    def check(self, cost: int = 1, *, use_reserve: bool = False) -> None:
        """Raise QuotaExhaustedError if spending `cost` now would breach the budget.

        Automated jobs stop at the reserve; a deliberate manual action (`use_reserve`)
        may spend it, but never past the day's hard limit.
        """
        st = self.status()
        allowed = st.remaining if use_reserve else st.available_to_jobs
        if cost > allowed:
            raise QuotaExhaustedError(
                f"{self.source}: {st.used}/{st.limit} requests used on {st.day} (UTC); "
                f"{allowed} left for this kind of request, {cost} needed"
            )

    def record(
        self, *, server_remaining: int | None = None, server_limit: int | None = None
    ) -> None:
        """Count one request; adopt the server's count when the response carried it."""
        day = self._day()
        now = self._clock().isoformat()
        with self._connect() as con:
            row = con.execute(
                "SELECT used, server_limit FROM api_quota WHERE source=? AND day=?",
                (self.source, day),
            ).fetchone()
            used = (row[0] if row else 0) + 1
            limit = server_limit or (row[1] if row else None)
            if server_remaining is not None and limit:
                used = max(0, limit - server_remaining)
            con.execute(
                "INSERT INTO api_quota VALUES (?,?,?,?,?) "
                "ON CONFLICT(source, day) DO UPDATE SET used=excluded.used, "
                "server_limit=excluded.server_limit, updated_at=excluded.updated_at",
                (self.source, day, used, limit, now),
            )
