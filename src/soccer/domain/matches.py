"""Match identity resolution.

A match's identity is its components -- competition, the two teams, and kickoff -- not
any provider's match id, because the sources share none. This resolver composes the
entity resolver: it resolves the competition and both teams to canonical ids, then
recognizes a match as one already seen when the components agree and kickoff falls
within a tolerance window.

Two things learned from the real sources shape this:

* **Cross-source overlap is rare.** football-data.org's twelve competitions and
  TheSportsDB's live feed barely intersect at any moment (opposite sides of the clock,
  different tiers). So the everyday value here is a stable id that collapses the same
  match seen across many polls into one entity; cross-source reconciliation is correct
  when it happens but is not the common case.
* **Timestamp skew could not be measured** (no overlapping fixture was available to
  compare). The tolerance is therefore set from reasoning, not measurement: the same
  (competition, home, away) triple never recurs within a day -- two-legged ties are
  days apart, tournaments never repeat a pairing same-day -- so a several-hour window
  absorbs timezone and rounding artifacts without any risk of merging distinct
  fixtures.

Orientation is matched exactly: A-v-B and B-v-A are treated as different matches. That
is correct for home/away legs and conservative for neutral-venue games a source might
report flipped -- those are left for manual linking rather than risk a wrong merge.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from uuid import uuid4

from soccer.domain.crosswalk import EntityResolver, EntityType
from soccer.storage.live_db import LiveDB

# Kickoff tolerance for treating two observations as the same match. Generous on
# purpose (see module docstring): distinct fixtures for one pairing are always at
# least a day apart, so this cannot cause a false merge.
DEFAULT_TOLERANCE = timedelta(hours=6)


class MatchLinkMethod(StrEnum):
    SOURCE_ID = "source_id"
    """Found or created via the source's own match id. No cross-source inference."""
    COMPONENTS = "components"
    """Linked by (competition, teams, kickoff-within-tolerance). Inferred."""
    MANUAL = "manual"


@dataclass(frozen=True)
class SourceRef:
    """A source's view of a competition or team: its id (if any), name, country."""

    id: str | None
    name: str
    country: str | None = None


@dataclass(frozen=True)
class MatchObservation:
    source: str
    source_match_id: str | None
    competition: SourceRef
    home: SourceRef
    away: SourceRef
    kickoff: datetime


@dataclass(frozen=True)
class ResolvedMatch:
    internal_id: str
    competition_id: str
    home_team_id: str
    away_team_id: str
    kickoff_utc: datetime
    method: MatchLinkMethod
    confidence: float
    component_confidence: float
    """Weakest of the three component links -- a match is only as sure as its parts."""
    created: bool


def _as_utc(when: datetime) -> datetime:
    """Assume naive timestamps are UTC and make it explicit, so all math is UTC."""
    return when.replace(tzinfo=UTC) if when.tzinfo is None else when.astimezone(UTC)


class MatchResolver:
    def __init__(
        self,
        db: LiveDB,
        entities: EntityResolver,
        *,
        tolerance: timedelta = DEFAULT_TOLERANCE,
        id_factory: Callable[[], str] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._conn = db.connection
        self._entities = entities
        self._tolerance = tolerance
        self._new_id = id_factory or (lambda: str(uuid4()))
        self._now = clock or (lambda: datetime.now(UTC))

    def resolve(self, obs: MatchObservation) -> ResolvedMatch:
        kickoff = _as_utc(obs.kickoff)

        # 1. Resolve the components. A match is only as trustworthy as its weakest one.
        competition = self._entities.resolve(
            EntityType.COMPETITION,
            source=obs.source,
            source_id=obs.competition.id,
            name=obs.competition.name,
            country=obs.competition.country,
        )
        home = self._entities.resolve(
            EntityType.TEAM,
            source=obs.source,
            source_id=obs.home.id,
            name=obs.home.name,
            country=obs.home.country,
        )
        away = self._entities.resolve(
            EntityType.TEAM,
            source=obs.source,
            source_id=obs.away.id,
            name=obs.away.name,
            country=obs.away.country,
        )
        component_confidence = min(competition.confidence, home.confidence, away.confidence)

        # No-id sources (openfootball, football-data.co.uk) key on the resolved
        # components + UTC date, so repeat sightings stay idempotent.
        key = obs.source_match_id or self._component_key(
            competition.internal_id, home.internal_id, away.internal_id, kickoff
        )

        existing = self._lookup(obs.source, key)
        if existing is not None:
            self._touch(obs.source, key)
            internal_id, method, confidence = existing
            return self._load(
                internal_id,
                MatchLinkMethod(method),
                confidence,
                component_confidence,
                created=False,
            )

        # 2. Attach to a match another source (or an earlier poll) established, by
        #    components within the kickoff tolerance.
        candidate = self._match_by_components(
            competition.internal_id, home.internal_id, away.internal_id, kickoff
        )
        if candidate is not None:
            self._write(
                obs.source, key, candidate, MatchLinkMethod.COMPONENTS, component_confidence
            )
            return self._load(
                candidate,
                MatchLinkMethod.COMPONENTS,
                component_confidence,
                component_confidence,
                created=False,
            )

        # 3. New match.
        internal_id = self._new_id()
        self._create(
            internal_id,
            competition.internal_id,
            home.internal_id,
            away.internal_id,
            kickoff,
        )
        method = MatchLinkMethod.SOURCE_ID if obs.source_match_id else MatchLinkMethod.COMPONENTS
        confidence = 1.0 if obs.source_match_id else component_confidence
        self._write(obs.source, key, internal_id, method, confidence)
        return self._load(internal_id, method, confidence, component_confidence, created=True)

    def link_manually(self, *, source: str, source_match_id: str, internal_id: str) -> None:
        """Join matches the resolver left separate (e.g. flipped neutral-venue games)."""
        self._write(source, source_match_id, internal_id, MatchLinkMethod.MANUAL, 1.0)

    # --- internals ---------------------------------------------------------

    @staticmethod
    def _component_key(comp_id: str, home_id: str, away_id: str, kickoff: datetime) -> str:
        return f"comp:{comp_id}|h:{home_id}|a:{away_id}|d:{kickoff.date().isoformat()}"

    def _lookup(self, source: str, key: str) -> tuple[str, str, float] | None:
        row = self._conn.execute(
            "SELECT internal_id, method, confidence FROM match_crosswalk "
            "WHERE source=? AND source_match_id=?",
            (source, key),
        ).fetchone()
        return (row["internal_id"], row["method"], row["confidence"]) if row else None

    def _match_by_components(
        self, comp_id: str, home_id: str, away_id: str, kickoff: datetime
    ) -> str | None:
        rows = self._conn.execute(
            "SELECT internal_id, kickoff_utc FROM canonical_match "
            "WHERE competition_id=? AND home_team_id=? AND away_team_id=?",
            (comp_id, home_id, away_id),
        ).fetchall()

        best: tuple[float, str] | None = None
        for row in rows:
            delta = abs(
                (kickoff - _as_utc(datetime.fromisoformat(row["kickoff_utc"]))).total_seconds()
            )
            if delta <= self._tolerance.total_seconds() and (best is None or delta < best[0]):
                best = (delta, row["internal_id"])
        return best[1] if best else None

    def _create(
        self,
        internal_id: str,
        comp_id: str,
        home_id: str,
        away_id: str,
        kickoff: datetime,
    ) -> None:
        self._conn.execute(
            "INSERT INTO canonical_match "
            "(internal_id, competition_id, home_team_id, away_team_id, kickoff_utc, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (internal_id, comp_id, home_id, away_id, kickoff.isoformat(), self._now().isoformat()),
        )

    def _write(
        self,
        source: str,
        key: str,
        internal_id: str,
        method: MatchLinkMethod,
        confidence: float,
    ) -> None:
        now = self._now().isoformat()
        self._conn.execute(
            "INSERT INTO match_crosswalk "
            "(source, source_match_id, internal_id, method, confidence, first_seen, last_seen) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(source, source_match_id) DO UPDATE SET "
            "  internal_id=excluded.internal_id, method=excluded.method, "
            "  confidence=excluded.confidence, last_seen=excluded.last_seen",
            (source, key, internal_id, method, confidence, now, now),
        )

    def _touch(self, source: str, key: str) -> None:
        self._conn.execute(
            "UPDATE match_crosswalk SET last_seen=? WHERE source=? AND source_match_id=?",
            (self._now().isoformat(), source, key),
        )

    def _load(
        self,
        internal_id: str,
        method: MatchLinkMethod,
        confidence: float,
        component_confidence: float,
        *,
        created: bool,
    ) -> ResolvedMatch:
        row = self._conn.execute(
            "SELECT competition_id, home_team_id, away_team_id, kickoff_utc "
            "FROM canonical_match WHERE internal_id=?",
            (internal_id,),
        ).fetchone()
        return ResolvedMatch(
            internal_id=internal_id,
            competition_id=row["competition_id"],
            home_team_id=row["home_team_id"],
            away_team_id=row["away_team_id"],
            kickoff_utc=_as_utc(datetime.fromisoformat(row["kickoff_utc"])),
            method=method,
            confidence=confidence,
            component_confidence=component_confidence,
            created=created,
        )

    def sources_for(self, internal_id: str) -> list[dict]:
        """Every source match id attached to one canonical match."""
        rows = self._conn.execute(
            "SELECT source, source_match_id, method, confidence FROM match_crosswalk "
            "WHERE internal_id=? ORDER BY source",
            (internal_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def match_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM canonical_match").fetchone()[0]
