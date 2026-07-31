"""TheSportsDB adapter -- the only free live source.

Two things shape this module.

**It can disappear.** Free access to `livescore.php` appears undocumented: the pricing
page frames livescore as a paid feature. Everything here is therefore written so a
sudden 403 or an empty response degrades visibly rather than silently. Callers get
`SourceUnavailable`, never a plausible-looking empty list.

**Its free tier is live-only.** Bulk endpoints are crippled -- `all_leagues` returns
5 leagues, a full-season query returns 15 events of 380. This adapter deliberately
exposes only the live endpoint; backfill must come from football-data.org or the
open datasets. Adding a bulk method here would produce quietly truncated data.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import httpx

from soccer.domain.match_state import MatchStatus
from soccer.ingest.ratelimit import RateLimiter
from soccer.sources.errors import SourceUnavailableError
from soccer.sources.registry import SourceId
from soccer.storage.raw import RawStore, Snapshot

logger = logging.getLogger(__name__)

# Re-exported for callers that import MatchStatus from this adapter; the canonical
# definition now lives in the domain so football-data.org can share it.
__all__ = ["ATTRIBUTION", "LiveMatch", "LiveResult", "MatchStatus", "TheSportsDB"]

BASE_URL = "https://www.thesportsdb.com/api/v1/json"
ATTRIBUTION = "Data from TheSportsDB (https://www.thesportsdb.com)"

# "90+3" in stoppage time, "24" in normal play, occasionally empty.
_PROGRESS = re.compile(r"^\s*(\d+)(?:\s*\+\s*(\d+))?\s*$")


# Provider status codes, lowercased. Unknown codes map to UNKNOWN rather than raising:
# a status we have not seen must not take down the live centre.
_STATUS_MAP: dict[str, MatchStatus] = {
    "ns": MatchStatus.NOT_STARTED,
    "not started": MatchStatus.NOT_STARTED,
    "1h": MatchStatus.FIRST_HALF,
    "ht": MatchStatus.HALF_TIME,
    "2h": MatchStatus.SECOND_HALF,
    "et": MatchStatus.EXTRA_TIME,
    "aet": MatchStatus.FINISHED,
    "pen": MatchStatus.PENALTIES,
    # "P" appears in the live feed with an empty minute alongside level scores in
    # knockout competitions (verified: 3-3 in MLS Next Pro, which decides draws by
    # shootout) -- it is a penalty shootout in progress, NOT postponement. Postponement
    # has its own codes below.
    "p": MatchStatus.PENALTIES,
    "ap": MatchStatus.FINISHED,
    "ft": MatchStatus.FINISHED,
    "match finished": MatchStatus.FINISHED,
    "postp": MatchStatus.POSTPONED,
    "postponed": MatchStatus.POSTPONED,
    "canc": MatchStatus.CANCELLED,
    "cancelled": MatchStatus.CANCELLED,
    "abd": MatchStatus.CANCELLED,
}


@dataclass(frozen=True)
class LiveMatch:
    event_id: str
    league_id: str | None
    league: str
    home_team: str
    away_team: str
    home_team_id: str | None
    away_team_id: str | None
    home_score: int | None
    away_score: int | None
    status: MatchStatus
    raw_status: str
    minute: int | None
    """Normal-time minute. 90 for anything in stoppage."""
    stoppage: int | None
    """Added minutes, when the provider reports '90+3'. None otherwise."""
    kickoff: datetime | None
    updated_at: datetime | None

    @property
    def display_minute(self) -> str:
        if self.minute is None:
            return ""
        return f"{self.minute}+{self.stoppage}" if self.stoppage else str(self.minute)

    @property
    def is_in_play(self) -> bool:
        return self.status.is_in_play


def parse_progress(value: Any) -> tuple[int | None, int | None]:
    """Parse `strProgress` into (minute, stoppage).

    The provider sends a string that may be '24', '90+3', '', or absent. Returning a
    tuple rather than raising keeps one malformed row from failing a whole poll.
    """
    if value is None:
        return None, None
    match = _PROGRESS.match(str(value))
    if not match:
        return None, None
    minute = int(match.group(1))
    stoppage = int(match.group(2)) if match.group(2) else None
    return minute, stoppage


def parse_status(value: Any) -> MatchStatus:
    if value is None:
        return MatchStatus.UNKNOWN
    return _STATUS_MAP.get(str(value).strip().lower(), MatchStatus.UNKNOWN)


def _parse_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_timestamp(value: Any) -> datetime | None:
    """Parse a provider timestamp, which is naive. Assumed UTC and made explicit."""
    if not value:
        return None
    text = str(value).strip().replace(" ", "T")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed


def parse_live_match(row: dict[str, Any]) -> LiveMatch | None:
    """Build a LiveMatch from one payload row, or None if it is unusable.

    Returning None rather than raising is deliberate: a single malformed row must not
    discard the other fifty.
    """
    event_id = row.get("idEvent") or row.get("idLiveScore")
    if not event_id:
        return None

    minute, stoppage = parse_progress(row.get("strProgress"))
    return LiveMatch(
        event_id=str(event_id),
        league_id=str(row["idLeague"]) if row.get("idLeague") else None,
        league=str(row.get("strLeague") or "Unknown"),
        home_team=str(row.get("strHomeTeam") or "Unknown"),
        away_team=str(row.get("strAwayTeam") or "Unknown"),
        home_team_id=str(row["idHomeTeam"]) if row.get("idHomeTeam") else None,
        away_team_id=str(row["idAwayTeam"]) if row.get("idAwayTeam") else None,
        home_score=_parse_int(row.get("intHomeScore")),
        away_score=_parse_int(row.get("intAwayScore")),
        status=parse_status(row.get("strStatus")),
        raw_status=str(row.get("strStatus") or ""),
        minute=minute,
        stoppage=stoppage,
        kickoff=_parse_timestamp(row.get("strTimestamp")),
        updated_at=_parse_timestamp(row.get("updated")),
    )


@dataclass
class LiveResult:
    matches: list[LiveMatch]
    snapshot: Snapshot
    is_stale: bool
    fetched_at: datetime

    @property
    def in_play(self) -> list[LiveMatch]:
        """Only genuinely in-play matches.

        The endpoint returns recently-finished games too -- in one sample, 41 of 53
        were already FT. Presenting all of them as "live" would be wrong.
        """
        return [m for m in self.matches if m.is_in_play]


class TheSportsDB:
    def __init__(
        self,
        raw_store: RawStore,
        *,
        api_key: str = "123",
        rate_limit_per_minute: int = 20,
        timeout: float = 20.0,
        max_retries: int = 2,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._raw = raw_store
        self._key = api_key
        self._limiter = RateLimiter(limit_per_minute=rate_limit_per_minute, reserve=5)
        self._max_retries = max_retries
        self._client = client or httpx.AsyncClient(
            base_url=f"{BASE_URL}/{api_key}", timeout=timeout
        )

    async def __aenter__(self) -> TheSportsDB:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def livescore(self, sport: str = "Soccer") -> LiveResult:
        """Current live scores. Falls back to the last snapshot if the feed fails."""
        endpoint = "livescore"
        last_error: Exception | None = None

        for attempt in range(self._max_retries + 1):
            await self._limiter.acquire()
            try:
                response = await self._client.get("/livescore.php", params={"s": sport})
            except httpx.HTTPError as exc:
                last_error = exc
                logger.warning("livescore fetch failed (attempt %d): %s", attempt + 1, exc)
                continue

            if response.status_code == 403:
                # The scenario this adapter exists to survive: free access withdrawn.
                logger.error(
                    "TheSportsDB returned 403. Free access to livescore.php is "
                    "undocumented and may have been withdrawn -- the live surface "
                    "should degrade to delayed sources."
                )
                last_error = httpx.HTTPStatusError(
                    "forbidden", request=response.request, response=response
                )
                break

            if response.status_code != 200:
                last_error = httpx.HTTPStatusError(
                    f"status {response.status_code}",
                    request=response.request,
                    response=response,
                )
                continue

            try:
                payload = response.json()
            except ValueError as exc:
                # A truncated or HTML response -- treat as failure, not empty data.
                last_error = exc
                logger.warning("livescore returned non-JSON")
                continue

            snapshot = self._raw.write(
                SourceId.THESPORTSDB, endpoint, payload, request_meta={"sport": sport}
            )
            return LiveResult(
                matches=self._parse(payload),
                snapshot=snapshot,
                is_stale=False,
                fetched_at=snapshot.fetched_at,
            )

        cached = self._raw.latest(SourceId.THESPORTSDB, endpoint)
        if cached is not None:
            logger.warning("livescore failed, serving cached data from %s", cached.fetched_at)
            return LiveResult(
                matches=self._parse(cached.payload),
                snapshot=cached,
                is_stale=True,
                fetched_at=cached.fetched_at,
            )

        raise SourceUnavailableError(
            "TheSportsDB livescore failed and no cache exists"
        ) from last_error

    @staticmethod
    def _parse(payload: Any) -> list[LiveMatch]:
        if not isinstance(payload, dict):
            return []
        # The key is `livescore`; older docs say `events`. Accept both.
        rows = payload.get("livescore") or payload.get("events") or []
        if not isinstance(rows, list):
            return []

        parsed = [parse_live_match(row) for row in rows if isinstance(row, dict)]
        matches = [m for m in parsed if m is not None]

        dropped = len(rows) - len(matches)
        if dropped:
            logger.warning("Dropped %d unparseable livescore row(s)", dropped)

        unknown = {m.raw_status for m in matches if m.status is MatchStatus.UNKNOWN}
        if unknown:
            # Worth surfacing: an unmapped status silently excludes matches from the
            # in-play view, which looks like missing data rather than a mapping gap.
            logger.warning("Unmapped status code(s): %s", sorted(unknown))

        return matches
