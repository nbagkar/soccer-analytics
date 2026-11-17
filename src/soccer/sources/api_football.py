"""API-Football (api-sports.io) adapter -- quota-aware, failure-explicit, snapshot-first.

What is verified (2026-11-16, by direct request): the v3 host answers a keyless request with
HTTP 403 and an envelope whose `errors` object explains the failure while `response` is an
EMPTY array. That shape is the central hazard: read naively, a refused request looks like a
successful "no data". So this adapter treats ANY non-empty `errors` (a dict or a list) as a
failure, regardless of HTTP status, and raises -- it never returns an empty success for a
request the server didn't fulfil.

What is NOT verified yet: which endpoints, competitions and seasons a given plan serves.
The vendor's docs and pricing pages sit behind a bot check; `probe_coverage` (and
`soccer probe-api-football`) establish it from real responses instead of from claims.

Every request is (1) checked against the daily quota ledger before it is sent, (2) paced
under the per-minute cap, (3) counted, adopting the server's own count from its response
headers, and (4) snapshotted to the raw store with its parameters, so any answer can be
re-read later without spending quota.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import httpx

from soccer.ingest.quota import QuotaLedger
from soccer.sources.errors import SourceUnavailableError
from soccer.sources.registry import SourceId
from soccer.storage.raw import RawStore

if TYPE_CHECKING:
    from soccer.config import Settings

logger = logging.getLogger(__name__)

BASE_URL = "https://v3.football.api-sports.io"
KEY_HEADER = "x-apisports-key"
RATE_LIMIT_WAIT_SECONDS = 61.0

# The competitions the upgrade plan prioritises, by API-Football league id, mapped to this
# project's division codes. Ids are the vendor's long-standing ones; the probe confirms each.
PRIORITY_LEAGUES: dict[int, tuple[str, str]] = {
    39: ("E0", "Premier League"),
    140: ("SP1", "La Liga"),
    78: ("D1", "Bundesliga"),
    135: ("I1", "Serie A"),
    61: ("F1", "Ligue 1"),
    2: ("UCL", "Champions League"),
}


class ApiFootballError(SourceUnavailableError):
    """The API answered, but with errors instead of (or beside) data."""

    def __init__(self, endpoint: str, errors: Any, status_code: int) -> None:
        self.endpoint = endpoint
        self.errors = errors
        self.status_code = status_code
        super().__init__(f"{endpoint}: HTTP {status_code}, errors={errors!r}")

    @property
    def kinds(self) -> list[str]:
        """The error keys ("token", "plan", "rateLimit", ...), for classification."""
        if isinstance(self.errors, dict):
            return [str(k) for k in self.errors]
        return ["error"] if self.errors else []

    @property
    def is_plan_restriction(self) -> bool:
        """The plan refuses this request. Keyed on the `plan` error ONLY: a rate-limit
        message also says "upgrade your plan" and must not read as a refusal."""
        return "plan" in self.kinds

    @property
    def is_rate_limited(self) -> bool:
        return "rateLimit" in self.kinds or self.status_code == 429


def _error_values(errors: Any) -> list[Any]:
    if isinstance(errors, dict):
        return list(errors.values())
    if isinstance(errors, list):
        return list(errors)
    return [errors]


def _has_errors(errors: Any) -> bool:
    if errors is None:
        return False
    if isinstance(errors, dict | list):
        return len(errors) > 0
    return bool(errors)


def _int_header(headers: httpx.Headers, name: str) -> int | None:
    raw = headers.get(name)
    try:
        return int(raw) if raw is not None else None
    except ValueError:
        return None


@dataclass(frozen=True)
class ApiResponse:
    endpoint: str
    params: dict[str, Any]
    response: list[Any]
    results: int
    page: int
    pages: int
    fetched_at: datetime
    from_cache: bool = False


class ApiFootball:
    def __init__(
        self,
        key: str,
        raw_store: RawStore,
        ledger: QuotaLedger,
        *,
        per_minute: int = 10,
        timeout: float = 20.0,
        client: httpx.Client | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not key:
            raise ValueError("API-Football needs a key (SOCCER_API_FOOTBALL_KEY)")
        self._raw = raw_store
        self._ledger = ledger
        self._client = client or httpx.Client(timeout=timeout)
        self._headers = {KEY_HEADER: key}
        self._interval = 60.0 / max(1, per_minute)
        self._clock = clock
        self._sleep = sleep
        self._last_sent: float | None = None

    @property
    def ledger(self) -> QuotaLedger:
        return self._ledger

    def __enter__(self) -> ApiFootball:
        return self

    def __exit__(self, *_: object) -> None:
        self._client.close()

    @staticmethod
    def snapshot_key(endpoint: str, params: dict[str, Any]) -> str:
        """Raw-store endpoint name: path plus sorted params, so each query has its own log."""
        query = "_".join(f"{k}-{params[k]}" for k in sorted(params))
        return endpoint.strip("/").replace("/", "-") + (f"__{query}" if query else "")

    def cached(
        self, endpoint: str, params: dict[str, Any], max_age: timedelta
    ) -> ApiResponse | None:
        """The last successful answer to this exact query, if younger than `max_age`."""
        snap = self._raw.latest(SourceId.API_FOOTBALL, self.snapshot_key(endpoint, params))
        if snap is None or datetime.now(UTC) - snap.fetched_at > max_age:
            return None
        return self._envelope(endpoint, params, snap.payload, snap.fetched_at, from_cache=True)

    def get(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        *,
        max_age: timedelta | None = None,
        use_reserve: bool = False,
        counts: bool = True,
    ) -> ApiResponse:
        """One request. Raises ApiFootballError / QuotaExhaustedError / httpx errors.

        `max_age` serves a fresh-enough snapshot instead of spending quota. `counts=False`
        is for endpoints the vendor exempts from the quota (`/status`) -- still paced.
        """
        params = dict(params or {})
        if max_age is not None:
            hit = self.cached(endpoint, params, max_age)
            if hit is not None:
                return hit
        try:
            return self._get_once(endpoint, params, use_reserve=use_reserve, counts=counts)
        except ApiFootballError as exc:
            if not exc.is_rate_limited:
                raise
            # The per-minute window is shared with any other process using the key; one
            # full minute's wait clears it. A second refusal is reported, not retried.
            logger.info("%s rate-limited; waiting %ss", endpoint, RATE_LIMIT_WAIT_SECONDS)
            self._sleep(RATE_LIMIT_WAIT_SECONDS)
            return self._get_once(endpoint, params, use_reserve=use_reserve, counts=counts)

    def _get_once(
        self, endpoint: str, params: dict[str, Any], *, use_reserve: bool, counts: bool
    ) -> ApiResponse:
        if counts:
            self._ledger.check(use_reserve=use_reserve)
        self._pace()
        response = self._client.get(f"{BASE_URL}{endpoint}", params=params, headers=self._headers)
        if counts:
            self._ledger.record(
                server_remaining=_int_header(response.headers, "x-ratelimit-requests-remaining"),
                server_limit=_int_header(response.headers, "x-ratelimit-requests-limit"),
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise SourceUnavailableError(
                f"{endpoint}: HTTP {response.status_code}, body is not JSON"
            ) from exc
        if not isinstance(body, dict):
            raise SourceUnavailableError(f"{endpoint}: unexpected body type {type(body).__name__}")
        errors = body.get("errors")
        if _has_errors(errors) or response.status_code >= 400:
            raise ApiFootballError(endpoint, errors, response.status_code)
        if "response" not in body:
            raise SourceUnavailableError(f"{endpoint}: envelope has no 'response' field")

        fetched_at = datetime.now(UTC)
        self._raw.write(
            SourceId.API_FOOTBALL,
            self.snapshot_key(endpoint, params),
            body,
            fetched_at=fetched_at,
            request_meta={"endpoint": endpoint, "params": params},
        )
        return self._envelope(endpoint, params, body, fetched_at)

    def _pace(self) -> None:
        now = self._clock()
        if self._last_sent is not None:
            wait = self._last_sent + self._interval - now
            if wait > 0:
                self._sleep(wait)
                now = self._clock()
        self._last_sent = now

    @staticmethod
    def _envelope(
        endpoint: str,
        params: dict[str, Any],
        body: dict[str, Any],
        fetched_at: datetime,
        *,
        from_cache: bool = False,
    ) -> ApiResponse:
        items = body.get("response")
        items = items if isinstance(items, list) else [items] if items else []
        paging = body.get("paging") or {}
        return ApiResponse(
            endpoint=endpoint,
            params=params,
            response=items,
            results=int(body.get("results") or len(items)),
            page=int(paging.get("current") or 1),
            pages=int(paging.get("total") or 1),
            fetched_at=fetched_at,
            from_cache=from_cache,
        )


def from_settings(settings: Settings) -> ApiFootball:
    """An adapter wired to the configured key, raw store and quota ledger.

    Raises SourceUnavailableError when no key is configured -- callers decide whether that
    is a skip (scheduled jobs) or a message (the CLI).
    """
    if settings.api_football_key is None:
        raise SourceUnavailableError("API-Football is off: set SOCCER_API_FOOTBALL_KEY in .env")
    settings.ensure_dirs()
    ledger = QuotaLedger(
        settings.quota_db,
        SourceId.API_FOOTBALL.value,
        daily_limit=settings.api_football_daily_limit,
        reserve=settings.api_football_reserve,
    )
    return ApiFootball(
        settings.api_football_key.get_secret_value(),
        RawStore(settings.raw_dir),
        ledger,
        per_minute=settings.api_football_rpm,
        timeout=settings.http_timeout_seconds,
    )
