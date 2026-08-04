"""football-data.org adapter.

Scope is deliberately narrow, matching what the free tier actually grants: fixtures,
results and standings for the twelve TIER_ONE competitions. No lineups, scorers or
squads -- those require the EUR 29/mo tier and are absent from this source's declared
capabilities.

Two behaviours matter more than the endpoint coverage:

1. **Tier filtering.** `/competitions` returns competitions the token cannot access
   (Copa Libertadores comes back as TIER_FOUR on a free key). Filtering on `plan`
   turns a confusing runtime 403 into a clean, visible absence.

2. **Batched fetching.** At 10 requests/minute, per-match polling is not viable --
   one busy Saturday would exhaust the budget on a single matchday. Everything here
   fetches by date range or competition, never per match.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx

from soccer.ingest.ratelimit import RateLimiter
from soccer.sources.errors import SourceUnavailableError
from soccer.sources.registry import SourceId
from soccer.storage.raw import RawStore, Snapshot

logger = logging.getLogger(__name__)

BASE_URL = "https://api.football-data.org/v4"
FREE_PLAN = "TIER_ONE"

# Hard server-side cap on /matches. Exceeding it returns 400 "Specified period must
# not exceed 10 days." Enforced locally so a bad range costs zero requests.
MAX_DATE_RANGE_DAYS = 10

# The twelve competitions a free token can fully reach, confirmed against the live
# API. Kept as a constant so `doctor` can report coverage without spending a request.
FREE_COMPETITIONS: dict[str, str] = {
    "BSA": "Campeonato Brasileiro Série A",
    "ELC": "Championship",
    "PL": "Premier League",
    "CL": "UEFA Champions League",
    "EC": "European Championship",
    "FL1": "Ligue 1",
    "BL1": "Bundesliga",
    "SA": "Serie A",
    "DED": "Eredivisie",
    "PPL": "Primeira Liga",
    "PD": "Primera Division",
    "WC": "FIFA World Cup",
}

# Competitions the API labels above the free tier but which are partially reachable
# anyway. Copa Libertadores reports `plan: TIER_FOUR`, yet its standings return full
# populated group tables and its matches appear in unfiltered date sweeps -- only an
# explicit `competitions=CLI` filter comes back empty.
#
# The provider's own tier labels therefore do not predict access. Probe and observe
# rather than trusting the label, but never let a feature depend on this: it is
# undocumented behaviour that can be closed at any time.
PARTIAL_COMPETITIONS: dict[str, str] = {
    "CLI": "Copa Libertadores",
}

ALL_KNOWN_COMPETITIONS = FREE_COMPETITIONS | PARTIAL_COMPETITIONS


@dataclass
class FetchResult:
    payload: Any
    snapshot: Snapshot
    is_stale: bool
    """True when the live fetch failed and this came from cache. Must reach the UI."""

    fetched_at: datetime


class FootballDataOrg:
    def __init__(
        self,
        token: str,
        raw_store: RawStore,
        *,
        rate_limit_per_minute: int = 10,
        timeout: float = 20.0,
        max_retries: int = 3,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._token = token
        self._raw = raw_store
        self._limiter = RateLimiter(limit_per_minute=rate_limit_per_minute)
        self._max_retries = max_retries
        self._client = client or httpx.AsyncClient(
            base_url=BASE_URL,
            headers={"X-Auth-Token": token},
            timeout=timeout,
        )

    async def __aenter__(self) -> FootballDataOrg:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    @property
    def requests_remaining(self) -> int:
        return self._limiter.remaining

    # --- Core fetch ---------------------------------------------------------

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> FetchResult:
        """Fetch, store the raw response, and fall back to cache on failure.

        A failed source must produce a visible stale-data warning, never an empty
        success-shaped response -- an empty fixture list and a broken API look
        identical downstream otherwise.
        """
        endpoint = path.strip("/")
        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            await self._limiter.acquire()
            try:
                response = await self._client.get(path, params=params)
            except httpx.HTTPError as exc:
                last_error = exc
                logger.warning("%s request failed (attempt %d): %s", path, attempt + 1, exc)
                await asyncio.sleep(self._backoff(attempt))
                continue

            self._sync_budget(response)

            if response.status_code == 429:
                # Retry-After is usually seconds, but the spec also allows an HTTP-date;
                # float() on a date would raise, so fall back to a sane default.
                try:
                    retry_after = float(response.headers.get("Retry-After", 60))
                except (TypeError, ValueError):
                    retry_after = 60.0
                self._limiter.penalize(retry_after)
                last_error = httpx.HTTPStatusError(
                    "rate limited", request=response.request, response=response
                )
                logger.warning("%s rate limited, waiting %.0fs", path, retry_after)
                continue

            if response.status_code in (403, 404):
                # Not transient: a tier restriction or a genuinely absent resource.
                # Retrying wastes budget we cannot spare.
                response.raise_for_status()

            if response.is_server_error:
                last_error = httpx.HTTPStatusError(
                    f"server error {response.status_code}",
                    request=response.request,
                    response=response,
                )
                await asyncio.sleep(self._backoff(attempt))
                continue

            response.raise_for_status()
            payload = response.json()
            snapshot = self._raw.write(
                SourceId.FOOTBALL_DATA_ORG,
                endpoint,
                payload,
                request_meta={"params": params or {}, "status": response.status_code},
            )
            return FetchResult(
                payload=payload,
                snapshot=snapshot,
                is_stale=False,
                fetched_at=snapshot.fetched_at,
            )

        cached = self._raw.latest(SourceId.FOOTBALL_DATA_ORG, endpoint)
        if cached is not None:
            logger.warning("%s failed, serving cached data from %s", path, cached.fetched_at)
            return FetchResult(
                payload=cached.payload,
                snapshot=cached,
                is_stale=True,
                fetched_at=cached.fetched_at,
            )

        raise SourceUnavailableError(
            f"football-data.org {endpoint} failed and no cache exists"
        ) from last_error

    @staticmethod
    def _backoff(attempt: int) -> float:
        # Jittered exponential. Deterministic jitter from the attempt number so tests
        # stay reproducible without patching the clock.
        return min(30.0, (2**attempt) + (attempt * 0.37))

    def _sync_budget(self, response: httpx.Response) -> None:
        def header_int(name: str) -> int | None:
            raw = response.headers.get(name)
            try:
                return int(raw) if raw is not None else None
            except ValueError:
                return None

        self._limiter.observe(
            remaining=header_int("X-Requests-Available-Minute"),
            reset_seconds=header_int("X-RequestCounter-Reset"),
        )

    # --- Endpoints ----------------------------------------------------------

    async def competitions(self) -> tuple[list[dict[str, Any]], bool]:
        """Accessible competitions only. Returns (competitions, is_stale)."""
        result = await self._get("/competitions")
        accessible = [
            competition
            for competition in result.payload.get("competitions", [])
            if competition.get("plan") == FREE_PLAN
        ]

        skipped = len(result.payload.get("competitions", [])) - len(accessible)
        if skipped:
            logger.info(
                "Skipped %d competition(s) above the free tier -- not a failure, "
                "they are simply not accessible with this token.",
                skipped,
            )
        return accessible, result.is_stale

    async def matches(
        self,
        *,
        date_from: date | None = None,
        date_to: date | None = None,
        competitions: list[str] | None = None,
    ) -> FetchResult:
        """Matches across competitions in one request.

        This is the batched endpoint that makes the 10/min budget survivable. Never
        replace it with a per-match loop.
        """
        params: dict[str, Any] = {}
        if date_from and date_to:
            span = (date_to - date_from).days + 1
            if span > MAX_DATE_RANGE_DAYS:
                raise ValueError(
                    f"Range of {span} days exceeds the API's {MAX_DATE_RANGE_DAYS}-day "
                    f"limit. Use matches_over_range() to fetch in chunks -- note it "
                    f"costs one request per chunk against a "
                    f"{self._limiter.limit_per_minute}/min budget."
                )
        if date_from:
            params["dateFrom"] = date_from.isoformat()
        if date_to:
            params["dateTo"] = date_to.isoformat()
        if competitions:
            unknown = set(competitions) - set(ALL_KNOWN_COMPETITIONS)
            if unknown:
                raise ValueError(
                    f"Unknown competition code(s): {sorted(unknown)}. "
                    f"Known: {sorted(ALL_KNOWN_COMPETITIONS)}"
                )
            # Filtering on a partial-access competition returns HTTP 200 with zero
            # results rather than an error -- indistinguishable from a genuinely
            # empty week unless we say so.
            partial = set(competitions) & set(PARTIAL_COMPETITIONS)
            if partial:
                logger.warning(
                    "Filtering by %s returns an empty result on the free tier even "
                    "though the competition exists. Its matches do appear in "
                    "unfiltered date sweeps.",
                    sorted(partial),
                )
            params["competitions"] = ",".join(competitions)

        return await self._get("/matches", params)

    async def today(self) -> FetchResult:
        today = datetime.now(UTC).date()
        return await self.matches(date_from=today, date_to=today)

    async def matches_over_range(
        self,
        date_from: date,
        date_to: date,
        *,
        competitions: list[str] | None = None,
    ) -> list[FetchResult]:
        """Fetch a range longer than the API's 10-day cap, in chunks.

        Costs one request per chunk. Kept as a separate method rather than making
        `matches()` silently chunk, because against a 10/min budget the difference
        between one request and nine is something the caller must choose knowingly.
        """
        if date_to < date_from:
            raise ValueError(f"date_to {date_to} precedes date_from {date_from}")

        chunks: list[tuple[date, date]] = []
        cursor = date_from
        while cursor <= date_to:
            chunk_end = min(cursor + timedelta(days=MAX_DATE_RANGE_DAYS - 1), date_to)
            chunks.append((cursor, chunk_end))
            cursor = chunk_end + timedelta(days=1)

        logger.info(
            "Fetching %s to %s as %d chunk(s); budget remaining before start: %d",
            date_from,
            date_to,
            len(chunks),
            self._limiter.remaining,
        )

        return [
            await self.matches(date_from=start, date_to=end, competitions=competitions)
            for start, end in chunks
        ]

    async def standings(self, competition: str, season: int | None = None) -> FetchResult:
        """Standings for a competition.

        Without `season`, the API returns the most recently *completed* season until
        the new one has fixtures played -- so a request in July returns last season's
        final table, not an empty current one. Callers displaying this must show the
        season, or users will read a finished table as a live one.
        """
        if competition not in ALL_KNOWN_COMPETITIONS:
            raise ValueError(
                f"Unknown competition code {competition!r}. Known: {sorted(ALL_KNOWN_COMPETITIONS)}"
            )
        if competition in PARTIAL_COMPETITIONS:
            logger.info(
                "%s is labelled above the free tier but its standings are reachable. "
                "This is undocumented -- treat it as a bonus, not a guarantee.",
                competition,
            )
        params = {"season": season} if season is not None else None
        return await self._get(f"/competitions/{competition}/standings", params)

    async def competition_matches(self, competition: str, season: int | None = None) -> FetchResult:
        """All matches for one competition-season (e.g. Champions League).

        The competition-scoped endpoint returns a whole season at once, so it sidesteps the
        10-day cap on the global `/matches`. The free tier only serves recent seasons for a
        given competition (older ones 403); callers should skip a 403 rather than fail.
        """
        if competition not in ALL_KNOWN_COMPETITIONS:
            raise ValueError(
                f"Unknown competition code {competition!r}. Known: {sorted(ALL_KNOWN_COMPETITIONS)}"
            )
        params = {"season": season} if season is not None else None
        return await self._get(f"/competitions/{competition}/matches", params)
