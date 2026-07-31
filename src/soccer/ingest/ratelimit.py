"""Rate limiting driven by server-reported budget.

A blind client-side token bucket drifts: it cannot see requests made by another
process, a previous run, or a retry that was counted server-side but failed locally.
football-data.org reports its own accounting on every response --
`X-Requests-Available-Minute` and `X-RequestCounter-Reset` -- so we track locally for
pacing but treat the server's number as authoritative whenever it arrives.

Sources that report no budget fall back to pure local pacing.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field


@dataclass
class RateLimiter:
    """Paces requests to stay under a per-minute ceiling.

    `reserve` is the safety margin held back from the ceiling so that retries and
    a final confirmation fetch never trip the limit. Against a 10/min budget the
    difference between 10 and 8 is the difference between an occasional 429 storm
    and a pipeline that simply keeps working.
    """

    limit_per_minute: int
    reserve: int = 2

    _timestamps: list[float] = field(default_factory=list, repr=False)
    _server_remaining: int | None = field(default=None, repr=False)
    _server_reset_at: float | None = field(default=None, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    @property
    def effective_limit(self) -> int:
        return max(1, self.limit_per_minute - self.reserve)

    def _prune(self, now: float) -> None:
        cutoff = now - 60.0
        self._timestamps = [t for t in self._timestamps if t > cutoff]

    def _wait_seconds(self, now: float) -> float:
        # The server's own count wins when it says we are out, since it sees requests
        # this process did not make.
        if (
            self._server_remaining is not None
            and self._server_remaining <= 0
            and self._server_reset_at is not None
            and self._server_reset_at > now
        ):
            return self._server_reset_at - now

        self._prune(now)
        if len(self._timestamps) < self.effective_limit:
            return 0.0

        # Wait until the oldest request in the window ages out.
        oldest = min(self._timestamps)
        return max(0.0, (oldest + 60.0) - now)

    async def acquire(self) -> None:
        """Block until a request may be made."""
        while True:
            async with self._lock:
                now = time.monotonic()
                wait = self._wait_seconds(now)
                if wait <= 0:
                    self._timestamps.append(now)
                    if self._server_remaining is not None:
                        self._server_remaining -= 1
                    return
            await asyncio.sleep(min(wait, 60.0))

    def observe(self, *, remaining: int | None, reset_seconds: int | None) -> None:
        """Sync with the server's accounting from a response header."""
        if remaining is not None:
            self._server_remaining = remaining
        if reset_seconds is not None:
            self._server_reset_at = time.monotonic() + reset_seconds

    def penalize(self, retry_after_seconds: float) -> None:
        """Apply a server-instructed backoff after a 429."""
        self._server_remaining = 0
        self._server_reset_at = time.monotonic() + retry_after_seconds

    @property
    def remaining(self) -> int:
        """Best estimate of requests still available this minute."""
        if self._server_remaining is not None:
            return max(0, self._server_remaining)
        self._prune(time.monotonic())
        return max(0, self.effective_limit - len(self._timestamps))
