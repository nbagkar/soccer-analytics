"""Entity resolution and the source crosswalk.

Turns a provider's identifier or bare name into a stable internal id, and records how
that mapping was made. The design commitment, in direct response to the cricket-mcp
"name-based identity" anti-pattern the plan flagged:

* A provider's own stable id is trusted for that provider's records (confidence 1.0).
* Linking one provider's entity to another's by name is an *inference*, recorded as
  such (method "exact_name", lower confidence) -- never a silent merge.
* Cross-source name inference is refused when both sides declare different countries.
  "Arsenal" in England and "Arsenal de Sarandí" in Argentina are not the same club,
  and a normalized-name match would otherwise merge them.
* Nothing is deleted or overwritten on conflict: a wrong link is a visible row with a
  method and confidence, so it can be found and corrected.

Internal ids are opaque (uuid4 by default; injectable for deterministic tests). They
are intentionally not derived from names -- names change and collide, and an id that
encodes a name would bake today's spelling into a permanent key.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from soccer.domain.aliases import AliasStore
from soccer.domain.names import normalize_name
from soccer.storage.live_db import LiveDB


class EntityType(StrEnum):
    COMPETITION = "competition"
    TEAM = "team"


class LinkMethod(StrEnum):
    SOURCE_ID = "source_id"
    """Found or created via the source's own stable identifier. No inference."""
    EXACT_NAME = "exact_name"
    """Linked to a pre-existing entity by normalized name (+country). Inferred."""
    NAME_ONLY = "name_only"
    """A source with no stable id; the normalized name is its only key."""
    MANUAL = "manual"


# Confidence that a crosswalk row maps its source entity to the right canonical one.
# The gap between SOURCE_ID and the name methods is the cost of cross-source name
# inference, which the Köln/Cologne case shows is fallible.
_CONFIDENCE = {
    LinkMethod.SOURCE_ID: 1.0,
    LinkMethod.EXACT_NAME: 0.9,
    LinkMethod.NAME_ONLY: 0.85,
    LinkMethod.MANUAL: 1.0,
}


@dataclass(frozen=True)
class ResolvedEntity:
    internal_id: str
    entity_type: EntityType
    canonical_name: str
    country: str | None
    method: LinkMethod
    confidence: float
    created: bool
    """True if this call created a new canonical entity rather than matching one."""


def _country_compatible(a: str | None, b: str | None) -> bool:
    """Two countries may refer to one entity if they agree, or either is unknown."""
    if a is None or b is None:
        return True
    return a.strip().lower() == b.strip().lower()


class EntityResolver:
    def __init__(
        self,
        db: LiveDB,
        *,
        aliases: AliasStore | None = None,
        id_factory: Callable[[], str] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._db = db
        self._conn = db.connection
        self._aliases = aliases
        self._new_id = id_factory or (lambda: str(uuid4()))
        self._now = clock or (lambda: datetime.now(UTC))

    def resolve(
        self,
        entity_type: EntityType,
        *,
        source: str,
        source_id: str | None,
        name: str,
        country: str | None = None,
    ) -> ResolvedEntity:
        """Resolve a source entity to an internal id, creating or linking as needed.

        `source_id` is the provider's stable identifier when it has one. Sources
        without one (openfootball, football-data.co.uk) pass None, and the normalized
        name becomes the crosswalk key instead.

        A curated alias routes a variant spelling onto its canonical form before
        matching, so "PSV Eindhoven" and "PSV" resolve together. The source's original
        spelling is still recorded in the crosswalk.
        """
        effective_name = name
        if self._aliases is not None:
            aliased = self._aliases.resolve_name(entity_type, name, country)
            if aliased is not None:
                effective_name = aliased

        normalized = normalize_name(effective_name)
        has_id = bool(source_id)
        # No-id sources key the crosswalk on the normalized name so repeat sightings
        # of the same name are idempotent rather than duplicating.
        crosswalk_key = source_id if has_id else f"name:{normalized}"

        existing = self._lookup_crosswalk(entity_type, source, crosswalk_key)
        if existing is not None:
            self._touch(entity_type, source, crosswalk_key)
            internal_id, canonical_name, stored_country, method, confidence = existing
            return ResolvedEntity(
                internal_id=internal_id,
                entity_type=entity_type,
                canonical_name=canonical_name,
                country=stored_country,
                method=LinkMethod(method),
                confidence=confidence,
                created=False,
            )

        if not normalized:
            # An unnameable entity with no id cannot be resolved to anything meaningful.
            raise ValueError(
                f"Cannot resolve {entity_type} from {source}: no id and no usable name ({name!r})"
            )

        # Try to attach to an entity another source already established, by name.
        match = self._match_by_name(entity_type, normalized, country)
        if match is not None:
            internal_id, canonical_name, stored_country = match
            method = LinkMethod.EXACT_NAME if has_id else LinkMethod.NAME_ONLY
            self._write_crosswalk(entity_type, source, crosswalk_key, internal_id, name, method)
            return ResolvedEntity(
                internal_id=internal_id,
                entity_type=entity_type,
                canonical_name=canonical_name,
                country=stored_country,
                method=method,
                confidence=_CONFIDENCE[method],
                created=False,
            )

        # Nothing matched: mint a new canonical entity under its canonical (aliased) name.
        internal_id = self._new_id()
        self._create_entity(internal_id, entity_type, effective_name, normalized, country)
        method = LinkMethod.SOURCE_ID if has_id else LinkMethod.NAME_ONLY
        self._write_crosswalk(entity_type, source, crosswalk_key, internal_id, name, method)
        return ResolvedEntity(
            internal_id=internal_id,
            entity_type=entity_type,
            canonical_name=effective_name,
            country=country,
            method=method,
            confidence=_CONFIDENCE[method],
            created=True,
        )

    def link_manually(
        self,
        entity_type: EntityType,
        *,
        source: str,
        source_id: str,
        internal_id: str,
    ) -> None:
        """Force a crosswalk row, for correcting a mismatch the resolver got wrong.

        The escape hatch that makes conservative automatic matching safe: anything the
        resolver leaves as a duplicate can be joined by hand here.
        """
        row = self._conn.execute(
            "SELECT source_name FROM entity_crosswalk "
            "WHERE entity_type=? AND source=? AND source_entity_id=?",
            (entity_type, source, source_id),
        ).fetchone()
        self._write_crosswalk(
            entity_type,
            source,
            source_id,
            internal_id,
            row["source_name"] if row else None,
            LinkMethod.MANUAL,
        )

    # --- internals ---------------------------------------------------------

    def _lookup_crosswalk(
        self, entity_type: EntityType, source: str, key: str
    ) -> tuple[str, str, str | None, str, float] | None:
        row = self._conn.execute(
            "SELECT x.internal_id, e.canonical_name, e.country, x.method, x.confidence "
            "FROM entity_crosswalk x "
            "JOIN canonical_entity e ON e.internal_id = x.internal_id "
            "WHERE x.entity_type=? AND x.source=? AND x.source_entity_id=?",
            (entity_type, source, key),
        ).fetchone()
        if row is None:
            return None
        return (
            row["internal_id"],
            row["canonical_name"],
            row["country"],
            row["method"],
            row["confidence"],
        )

    def _match_by_name(
        self, entity_type: EntityType, normalized: str, country: str | None
    ) -> tuple[str, str, str | None] | None:
        """Find an existing canonical entity by normalized name and compatible country.

        Prefers an exact country match over a country-unknown one, so a well-specified
        entity is not shadowed by a vaguer namesake.
        """
        rows = self._conn.execute(
            "SELECT internal_id, canonical_name, country FROM canonical_entity "
            "WHERE entity_type=? AND normalized_name=?",
            (entity_type, normalized),
        ).fetchall()

        compatible = [row for row in rows if _country_compatible(row["country"], country)]
        if not compatible:
            return None

        exact = [row for row in compatible if country is not None and row["country"] is not None]
        chosen = exact[0] if exact else compatible[0]
        return (chosen["internal_id"], chosen["canonical_name"], chosen["country"])

    def _create_entity(
        self,
        internal_id: str,
        entity_type: EntityType,
        name: str,
        normalized: str,
        country: str | None,
    ) -> None:
        self._conn.execute(
            "INSERT INTO canonical_entity "
            "(internal_id, entity_type, canonical_name, normalized_name, country, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (internal_id, entity_type, name, normalized, country, self._now().isoformat()),
        )

    def _write_crosswalk(
        self,
        entity_type: EntityType,
        source: str,
        key: str,
        internal_id: str,
        source_name: str | None,
        method: LinkMethod,
    ) -> None:
        now = self._now().isoformat()
        # ON CONFLICT: a manual re-link should repoint and refresh, but must not reset
        # first_seen -- provenance of when we first saw this source entity is kept.
        self._conn.execute(
            "INSERT INTO entity_crosswalk "
            "(entity_type, source, source_entity_id, internal_id, source_name, "
            " method, confidence, first_seen, last_seen) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(entity_type, source, source_entity_id) DO UPDATE SET "
            "  internal_id=excluded.internal_id, "
            "  method=excluded.method, "
            "  confidence=excluded.confidence, "
            "  last_seen=excluded.last_seen",
            (
                entity_type,
                source,
                key,
                internal_id,
                source_name,
                method,
                _CONFIDENCE[method],
                now,
                now,
            ),
        )

    def _touch(self, entity_type: EntityType, source: str, key: str) -> None:
        self._conn.execute(
            "UPDATE entity_crosswalk SET last_seen=? "
            "WHERE entity_type=? AND source=? AND source_entity_id=?",
            (self._now().isoformat(), entity_type, source, key),
        )

    # --- inspection --------------------------------------------------------

    def sources_for(self, entity_type: EntityType, internal_id: str) -> list[dict]:
        """Every source identifier attached to one canonical entity."""
        rows = self._conn.execute(
            "SELECT source, source_entity_id, source_name, method, confidence "
            "FROM entity_crosswalk WHERE entity_type=? AND internal_id=? "
            "ORDER BY source",
            (entity_type, internal_id),
        ).fetchall()
        return [dict(row) for row in rows]

    def entity_count(self, entity_type: EntityType) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM canonical_entity WHERE entity_type=?",
            (entity_type,),
        ).fetchone()[0]
