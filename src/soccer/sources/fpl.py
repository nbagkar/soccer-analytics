"""Fantasy Premier League adapter -- player availability (injuries, suspensions, doubts).

Uniquely valuable and uniquely constrained. FPL's `bootstrap-static/` is the only free
source of current injury/availability news with a chance-of-playing percentage, but it is
Premier League only and GENUINELY GREY: the terms contemplate private personal use yet bar
'creating a database'. This adapter is therefore built to stay on the right side of that:

  - OFF by default (`config.enable_fpl=False`); enabling it is the operator's knowing choice.
  - One undocumented endpoint, fetched gently and never per-player -- `bootstrap-static/`
    returns the entire game state (every club, every player) in a single request, so there
    is never a reason to poll harder.
  - Cached only ephemerally: a raw snapshot for stale-fallback, plus a replace-on-refresh
    table in the live store. Nothing accumulates into a committed or redistributed dataset.

No API key: the endpoint is public. We still send a descriptive User-Agent and rate-limit
politely, because "undocumented and unauthenticated" is a reason to be more careful, not less.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx

from soccer.domain.availability import PlayerAvailability, status_label
from soccer.domain.names import normalize_name
from soccer.ingest.ratelimit import RateLimiter
from soccer.sources.errors import SourceUnavailableError
from soccer.sources.registry import SourceId
from soccer.storage.raw import RawStore, Snapshot

logger = logging.getLogger(__name__)

BASE_URL = "https://fantasy.premierleague.com/api"
BOOTSTRAP_ENDPOINT = "bootstrap-static"

# Politeness: FPL publishes no rate limit, so we invent a conservative one. We only ever make
# one request per refresh, so this is belt-and-braces rather than a real constraint.
DEFAULT_RPM = 10


@dataclass
class FplFetch:
    payload: Any
    snapshot: Snapshot
    is_stale: bool
    """True when the live fetch failed and this came from cache. Must reach the UI."""

    fetched_at: datetime


class FantasyPremierLeague:
    def __init__(
        self,
        raw_store: RawStore,
        *,
        rate_limit_per_minute: int = DEFAULT_RPM,
        timeout: float = 20.0,
        max_retries: int = 3,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._raw = raw_store
        self._limiter = RateLimiter(limit_per_minute=rate_limit_per_minute)
        self._max_retries = max_retries
        self._client = client or httpx.AsyncClient(
            base_url=BASE_URL,
            headers={"User-Agent": "soccer-analytics/1.0 (personal, non-commercial)"},
            timeout=timeout,
        )

    async def __aenter__(self) -> FantasyPremierLeague:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _get(self, path: str, endpoint: str) -> FplFetch:
        """Fetch, store the raw response, fall back to cache on failure.

        Same discipline as the other adapters: a failed source produces a visible stale-data
        result served from the last good snapshot, never an empty success-shaped response.
        """
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            await self._limiter.acquire()
            try:
                response = await self._client.get(path)
            except httpx.HTTPError as exc:
                last_error = exc
                logger.warning("FPL %s failed (attempt %d): %s", path, attempt + 1, exc)
                await asyncio.sleep(self._backoff(attempt))
                continue

            if response.status_code == 429:
                try:
                    retry_after = float(response.headers.get("Retry-After", 60))
                except (TypeError, ValueError):
                    retry_after = 60.0
                self._limiter.penalize(retry_after)
                last_error = httpx.HTTPStatusError(
                    "rate limited", request=response.request, response=response
                )
                logger.warning("FPL rate limited, waiting %.0fs", retry_after)
                continue

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
                SourceId.FPL,
                endpoint,
                payload,
                request_meta={"status": response.status_code},
            )
            return FplFetch(
                payload=payload,
                snapshot=snapshot,
                is_stale=False,
                fetched_at=snapshot.fetched_at,
            )

        cached = self._raw.latest(SourceId.FPL, endpoint)
        if cached is not None:
            logger.warning("FPL %s failed, serving cached data from %s", path, cached.fetched_at)
            return FplFetch(
                payload=cached.payload, snapshot=cached, is_stale=True, fetched_at=cached.fetched_at
            )
        raise SourceUnavailableError(
            f"FPL {endpoint} failed and no cache exists"
        ) from last_error

    @staticmethod
    def _backoff(attempt: int) -> float:
        # Jittered exponential; deterministic jitter from the attempt keeps tests reproducible.
        return min(30.0, float(2**attempt) + (attempt * 0.37))

    async def bootstrap(self) -> FplFetch:
        """The whole game state -- clubs and players with their availability -- in one request."""
        return await self._get("/bootstrap-static/", BOOTSTRAP_ENDPOINT)


def parse_availability(payload: dict[str, Any], *, fetched_at: str) -> list[PlayerAvailability]:
    """Turn a `bootstrap-static` payload into one availability record per player.

    Each `element` carries a status code (a/d/i/s/u/n), a free-text `news` blurb and a
    `chance_of_playing_next_round`; `team` is an id into the `teams` array. Club names are
    left as FPL spells them here ("Man Utd", "Spurs") -- the action layer reconciles them onto
    the loaded domestic spelling, exactly as squads are, so this mapper stays source-pure.
    """
    teams = {t.get("id"): t.get("name") or "Unknown" for t in payload.get("teams", [])}
    out: list[PlayerAvailability] = []
    for element in payload.get("elements", []):
        web_name = (element.get("web_name") or "").strip()
        second = (element.get("second_name") or "").strip()
        player = web_name or second
        if not player:
            continue  # a player with no usable name is not worth a row
        team_name = teams.get(element.get("team"), "Unknown")
        status = (element.get("status") or "a").strip() or "a"
        first = (element.get("first_name") or "").strip()
        full_name = f"{first} {second}".strip() or None
        news = (element.get("news") or "").strip() or None
        out.append(
            PlayerAvailability(
                source=SourceId.FPL.value,
                team=team_name,
                team_norm=normalize_name(team_name),
                player=player,
                full_name=full_name,
                status=status,
                availability=status_label(status),
                chance=element.get("chance_of_playing_next_round"),
                news=news,
                news_added=element.get("news_added"),
                fetched_at=fetched_at,
                # element_type routes the loss (GK/DEF/MID/FWD); now_cost (price x10) weights
                # it by quality. Both drive the forecast adjustment; kept as FPL sends them.
                element_type=element.get("element_type"),
                price=element.get("now_cost"),
                # Season-to-date minutes -- the "who's a regular starter" proxy behind the
                # confirmed-squad-gap adjustment (domain/availability.py).
                minutes=element.get("minutes"),
            )
        )
    return out
