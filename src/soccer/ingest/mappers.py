"""Source payload -> canonical observation + state.

Pure functions, one per source, isolating every source-specific quirk (status codes,
score locations, id vs name availability) at the edge so the resolver and store see
one uniform shape. Each returns `(MatchObservation, MatchState)`: identity for the
resolver, current state for the store.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from soccer.domain.match_state import MatchState, MatchStatus
from soccer.domain.matches import MatchObservation, SourceRef
from soccer.sources.registry import SourceId
from soccer.sources.thesportsdb import LiveMatch

# football-data.org status vocabulary -> canonical. It reports IN_PLAY/PAUSED without a
# half, so those map to the generic live statuses. Verified against the live API:
# SCHEDULED, TIMED, FINISHED, AWARDED seen; the rest are documented.
_FD_STATUS: dict[str, MatchStatus] = {
    "SCHEDULED": MatchStatus.NOT_STARTED,
    "TIMED": MatchStatus.NOT_STARTED,
    "IN_PLAY": MatchStatus.IN_PLAY,
    "PAUSED": MatchStatus.HALF_TIME,
    "FINISHED": MatchStatus.FINISHED,
    "AWARDED": MatchStatus.AWARDED,
    "SUSPENDED": MatchStatus.SUSPENDED,
    "POSTPONED": MatchStatus.POSTPONED,
    "CANCELLED": MatchStatus.CANCELLED,
}


def _fd_country(area: dict[str, Any] | None) -> str | None:
    return area.get("name") if isinstance(area, dict) else None


def map_football_data_match(
    match: dict[str, Any], *, observed_at: datetime, is_stale: bool
) -> tuple[MatchObservation, MatchState]:
    """One football-data.org match record -> observation + state."""
    competition = match.get("competition", {})
    home = match.get("homeTeam", {})
    away = match.get("awayTeam", {})
    # Country lives on competition.area; team records here carry no area, so the
    # competition's country is the best available disambiguator for both teams.
    country = _fd_country(competition.get("area"))

    observation = MatchObservation(
        source=SourceId.FOOTBALL_DATA_ORG,
        source_match_id=str(match["id"]),
        competition=SourceRef(
            id=str(competition["id"]) if competition.get("id") else None,
            name=competition.get("name", "Unknown"),
            country=country,
        ),
        home=SourceRef(
            id=str(home["id"]) if home.get("id") else None,
            name=home.get("name", "Unknown"),
            country=country,
        ),
        away=SourceRef(
            id=str(away["id"]) if away.get("id") else None,
            name=away.get("name", "Unknown"),
            country=country,
        ),
        kickoff=datetime.fromisoformat(match["utcDate"].replace("Z", "+00:00")),
    )

    full_time = match.get("score", {}).get("fullTime", {})
    state = MatchState(
        status=_FD_STATUS.get(match.get("status", ""), MatchStatus.UNKNOWN),
        home_score=full_time.get("home"),
        away_score=full_time.get("away"),
        minute=None,  # football-data.org is not live; it reports no match minute
        source=SourceId.FOOTBALL_DATA_ORG,
        observed_at=observed_at,
        is_stale=is_stale,
    )
    return observation, state


def map_thesportsdb_live(
    match: LiveMatch, *, observed_at: datetime, is_stale: bool
) -> tuple[MatchObservation, MatchState]:
    """One TheSportsDB LiveMatch -> observation + state.

    TheSportsDB gives no country, so entities resolve without one -- fine, since its
    live coverage rarely overlaps football-data.org's, so cross-source country clashes
    are not the binding concern here.
    """
    observation = MatchObservation(
        source=SourceId.THESPORTSDB,
        source_match_id=match.event_id,
        competition=SourceRef(id=match.league_id, name=match.league),
        home=SourceRef(id=match.home_team_id, name=match.home_team),
        away=SourceRef(id=match.away_team_id, name=match.away_team),
        kickoff=match.kickoff or observed_at,
    )
    state = MatchState(
        status=match.status,
        home_score=match.home_score,
        away_score=match.away_score,
        minute=match.display_minute or None,
        source=SourceId.THESPORTSDB,
        observed_at=match.updated_at or observed_at,
        is_stale=is_stale,
    )
    return observation, state


def _utcnow() -> datetime:
    return datetime.now(UTC)
