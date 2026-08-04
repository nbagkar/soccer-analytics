"""Canonical match state: what is happening in a match right now.

Separate from identity (`domain/matches.py`, which answers *which* match). This answers
*what state* it is in -- score, status, minute -- and materializes one current row per
canonical match for fast reads, with the raw snapshots remaining the full history.

The precedence rule when two sources describe the same match is the plan's
"canonicalization layer" made concrete: **the more live-capable source wins**, ranked
by the registry's declared latency, not by which fetch happened to land last. A delayed
source polled at 10:05 must not clobber a live source's 10:04 in-play data. Only within
one source, or between equally-live sources, does fresher fetch time decide.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from soccer.sources.registry import SOURCES, SourceId
from soccer.storage.live_db import LiveDB


class MatchStatus(StrEnum):
    NOT_STARTED = "not_started"
    IN_PLAY = "in_play"
    """Generic live -- for sources (football-data.org) that do not report the half."""
    FIRST_HALF = "first_half"
    HALF_TIME = "half_time"
    SECOND_HALF = "second_half"
    EXTRA_TIME = "extra_time"
    PENALTIES = "penalties"
    FINISHED = "finished"
    AWARDED = "awarded"
    """Result decided administratively (walkover). Concluded, not played out."""
    POSTPONED = "postponed"
    SUSPENDED = "suspended"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"

    @property
    def is_in_play(self) -> bool:
        return self in _IN_PLAY

    @property
    def is_concluded(self) -> bool:
        return self in {MatchStatus.FINISHED, MatchStatus.AWARDED}


_IN_PLAY = frozenset(
    {
        MatchStatus.IN_PLAY,
        MatchStatus.FIRST_HALF,
        MatchStatus.HALF_TIME,
        MatchStatus.SECOND_HALF,
        MatchStatus.EXTRA_TIME,
        MatchStatus.PENALTIES,
    }
)


@dataclass(frozen=True)
class MatchState:
    status: MatchStatus
    home_score: int | None
    away_score: int | None
    minute: str | None
    """Display minute like '45+2', when a live source reports it. None otherwise."""
    source: str
    observed_at: datetime
    is_stale: bool
    """True when this came from a source's cache rather than a fresh fetch."""


def _source_latency(source: str) -> float:
    """Registry latency for a source; +inf for unknown or static sources.

    Lower is more live-capable, so this is the primary precedence key.
    """
    try:
        latency = SOURCES[SourceId(source)].latency_seconds
    except (ValueError, KeyError):
        return math.inf
    return math.inf if latency is None else float(latency)


class MatchStateStore:
    def __init__(self, db: LiveDB) -> None:
        self._conn = db.connection

    def upsert(self, match_id: str, state: MatchState) -> bool:
        """Write state for a canonical match, honouring source precedence.

        Returns True if the store changed, False if an existing, more-authoritative
        state was kept. The rule: a more live-capable source always wins; the same
        source always updates; equally-live sources are decided by fetch freshness.
        """
        existing = self._conn.execute(
            "SELECT source, observed_at FROM match_state WHERE match_id=?",
            (match_id,),
        ).fetchone()

        if existing is not None and not self._supersedes(state, existing):
            return False

        self._conn.execute(
            "INSERT INTO match_state "
            "(match_id, status, home_score, away_score, minute, source, "
            " observed_at, is_stale, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(match_id) DO UPDATE SET "
            "  status=excluded.status, home_score=excluded.home_score, "
            "  away_score=excluded.away_score, minute=excluded.minute, "
            "  source=excluded.source, observed_at=excluded.observed_at, "
            "  is_stale=excluded.is_stale, updated_at=excluded.updated_at",
            (
                match_id,
                state.status,
                state.home_score,
                state.away_score,
                state.minute,
                state.source,
                state.observed_at.isoformat(),
                int(state.is_stale),
                datetime.now(UTC).isoformat(),
            ),
        )
        return True

    def _supersedes(self, incoming: MatchState, existing) -> bool:
        if incoming.source == existing["source"]:
            return True  # latest word from the same source always applies
        incoming_latency = _source_latency(incoming.source)
        existing_latency = _source_latency(existing["source"])
        if incoming_latency != existing_latency:
            return incoming_latency < existing_latency  # more live-capable wins
        # Equally live: fresher fetch wins.
        return incoming.observed_at.isoformat() >= existing["observed_at"]

    def get(self, match_id: str) -> MatchState | None:
        row = self._conn.execute(
            "SELECT status, home_score, away_score, minute, source, observed_at, is_stale "
            "FROM match_state WHERE match_id=?",
            (match_id,),
        ).fetchone()
        if row is None:
            return None
        return MatchState(
            status=MatchStatus(row["status"]),
            home_score=row["home_score"],
            away_score=row["away_score"],
            minute=row["minute"],
            source=row["source"],
            observed_at=datetime.fromisoformat(row["observed_at"]),
            is_stale=bool(row["is_stale"]),
        )

    def list_current(self, *, in_play_only: bool = False, limit: int = 200) -> list[MatchView]:
        """Current matches with resolved names, for display. Kickoff order."""
        rows = self._conn.execute(
            "SELECT cm.internal_id, cm.kickoff_utc, "
            "  h.canonical_name AS home, a.canonical_name AS away, "
            "  c.canonical_name AS competition, "
            "  s.status, s.home_score, s.away_score, s.minute, s.source, s.is_stale "
            "FROM match_state s "
            "JOIN canonical_match cm ON cm.internal_id = s.match_id "
            "LEFT JOIN canonical_entity h ON h.internal_id = cm.home_team_id "
            "LEFT JOIN canonical_entity a ON a.internal_id = cm.away_team_id "
            "LEFT JOIN canonical_entity c ON c.internal_id = cm.competition_id "
            "ORDER BY cm.kickoff_utc",
            (),
        ).fetchall()

        views = [
            MatchView(
                match_id=row["internal_id"],
                kickoff_utc=datetime.fromisoformat(row["kickoff_utc"]),
                home=row["home"] or "Unknown",
                away=row["away"] or "Unknown",
                competition=row["competition"] or "Unknown",
                status=MatchStatus(row["status"]),
                home_score=row["home_score"],
                away_score=row["away_score"],
                minute=row["minute"],
                source=row["source"],
                is_stale=bool(row["is_stale"]),
            )
            for row in rows
        ]
        if in_play_only:
            views = [v for v in views if v.status.is_in_play]
        return views[:limit]

    def upcoming(self, *, limit: int = 100) -> list[MatchView]:
        """Not-yet-started matches, soonest kickoff first -- the fixture list.

        A dedicated query rather than filtering `list_current`, whose kickoff-ordered
        limit pushes future fixtures off the end behind past and live matches.
        """
        return [v for v in self.list_current(limit=10_000) if v.status is MatchStatus.NOT_STARTED][
            :limit
        ]

    def recent_finished(self, *, days: int = 7, limit: int = 300) -> list[MatchView]:
        """Concluded matches whose kickoff was within the last `days`, most recent first.

        The everyday fallback when nothing is live: recent full-time results, rather than the
        oldest rows a plain kickoff-ordered list surfaces once deep fixture history is loaded.
        """
        cutoff = datetime.now(UTC) - timedelta(days=days)
        recent = [
            v
            for v in self.list_current(limit=1_000_000)
            if v.status.is_concluded and v.kickoff_utc >= cutoff
        ]
        recent.sort(key=lambda v: v.kickoff_utc, reverse=True)
        return recent[:limit]


@dataclass(frozen=True)
class MatchView:
    """A canonical match plus its current state and resolved names, for display."""

    match_id: str
    kickoff_utc: datetime
    home: str
    away: str
    competition: str
    status: MatchStatus
    home_score: int | None
    away_score: int | None
    minute: str | None
    source: str
    is_stale: bool

    @property
    def score(self) -> str:
        if self.home_score is None or self.away_score is None:
            return "-"
        return f"{self.home_score}-{self.away_score}"
