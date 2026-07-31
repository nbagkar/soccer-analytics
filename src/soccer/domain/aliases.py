"""Curated aliases and duplicate discovery.

Normalization (`domain/names.py`) is deliberately conservative: it bridges "Arsenal FC"
/ "Arsenal" but refuses "PSV" / "PSV Eindhoven" (appended city) and "Köln" / "Cologne"
(translation), because a rule aggressive enough to merge those would also merge genuine
rivals. This module is where a human closes that gap by hand -- declaring specific
equivalences the resolver then applies.

Two halves:

* **AliasStore** -- persisted equivalences the resolver consults, mapping a variant
  spelling onto a chosen canonical form. Applied going forward: a variant seen after
  the alias exists resolves onto the canonical entity.
* **suggest_duplicates** -- surfaces probable splits for review. It finds the
  appended-word class (one name's tokens a subset of another's) that discovery can spot
  reliably; translations cannot be auto-detected and must be aliased from knowledge.

Retroactively merging entities that split *before* an alias existed is deliberately out
of scope here -- the crosswalk short-circuits on a known source id, so an alias only
affects newly-seen entities. With sources that barely overlap there is nothing to
retrofit yet; when there is, merge/replay is the follow-up.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from soccer.domain.names import name_tokens, normalize_name
from soccer.storage.live_db import LiveDB


def _country_key(country: str | None) -> str:
    return normalize_name(country) if country else ""


@dataclass(frozen=True)
class Alias:
    entity_type: str
    alias_name: str
    canonical_name: str
    country: str | None
    note: str | None


class AliasStore:
    def __init__(self, db: LiveDB) -> None:
        self._conn = db.connection

    def add(
        self,
        entity_type: str,
        alias_name: str,
        canonical_name: str,
        *,
        country: str | None = None,
        note: str | None = None,
    ) -> None:
        """Declare that `alias_name` refers to the same entity as `canonical_name`."""
        alias_norm = normalize_name(alias_name)
        canonical_norm = normalize_name(canonical_name)
        if not alias_norm or not canonical_norm:
            raise ValueError("alias and canonical names must be non-empty")
        if alias_norm == canonical_norm:
            raise ValueError("alias and canonical names are already equivalent")

        self._conn.execute(
            "INSERT INTO entity_alias "
            "(entity_type, alias_normalized, country_key, canonical_name, "
            " canonical_normalized, note, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(entity_type, alias_normalized, country_key) DO UPDATE SET "
            "  canonical_name=excluded.canonical_name, "
            "  canonical_normalized=excluded.canonical_normalized, note=excluded.note",
            (
                entity_type,
                alias_norm,
                _country_key(country),
                canonical_name,
                canonical_norm,
                note,
                datetime.now(UTC).isoformat(),
            ),
        )

    def resolve_name(self, entity_type: str, name: str, country: str | None) -> str | None:
        """Canonical name for a variant, or None if no alias applies.

        Prefers a country-specific alias, then a country-agnostic one. A single lookup,
        never chained, so aliases cannot loop.
        """
        alias_norm = normalize_name(name)
        if not alias_norm:
            return None
        for ck in (_country_key(country), ""):
            row = self._conn.execute(
                "SELECT canonical_name FROM entity_alias "
                "WHERE entity_type=? AND alias_normalized=? AND country_key=?",
                (entity_type, alias_norm, ck),
            ).fetchone()
            if row is not None:
                return row["canonical_name"]
        return None

    def all(self, entity_type: str | None = None) -> list[Alias]:
        if entity_type is not None:
            rows = self._conn.execute(
                "SELECT * FROM entity_alias WHERE entity_type=? ORDER BY alias_normalized",
                (entity_type,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM entity_alias ORDER BY entity_type, alias_normalized"
            ).fetchall()
        return [
            Alias(
                entity_type=row["entity_type"],
                alias_name=row["alias_normalized"],
                canonical_name=row["canonical_name"],
                country=row["country_key"] or None,
                note=row["note"],
            )
            for row in rows
        ]


@dataclass(frozen=True)
class DuplicateCandidate:
    entity_type: str
    name_a: str
    name_b: str
    country: str | None
    reason: str

    def alias_command(self) -> str:
        """The `soccer alias add` invocation that would join this pair."""
        longer, shorter = (
            (self.name_a, self.name_b)
            if len(self.name_a) >= len(self.name_b)
            else (self.name_b, self.name_a)
        )
        return f'soccer alias-add "{longer}" "{shorter}"'


def suggest_duplicates(db: LiveDB, entity_type: str = "team") -> list[DuplicateCandidate]:
    """Probable entity splits, for human review.

    Detects the appended-word class: one entity's normalized tokens a proper subset of
    another's, with compatible country (equal or either unknown). This is advisory --
    reserve/second sides ("Atlanta United" vs "Atlanta United II") will surface too, and
    a human decides. Translations (Köln/Cologne) share no tokens and are not detectable
    here; those need an alias added from knowledge.
    """
    rows = db.connection.execute(
        "SELECT canonical_name, normalized_name, country FROM canonical_entity "
        "WHERE entity_type=? ORDER BY normalized_name",
        (entity_type,),
    ).fetchall()

    entities = [
        (row["canonical_name"], name_tokens(row["normalized_name"]), row["country"]) for row in rows
    ]
    candidates: list[DuplicateCandidate] = []
    for i, (name_a, tokens_a, country_a) in enumerate(entities):
        for name_b, tokens_b, country_b in entities[i + 1 :]:
            if not tokens_a or not tokens_b:
                continue
            if country_a and country_b and country_a != country_b:
                continue
            if tokens_a < tokens_b or tokens_b < tokens_a:
                candidates.append(
                    DuplicateCandidate(
                        entity_type=entity_type,
                        name_a=name_a,
                        name_b=name_b,
                        country=country_a or country_b,
                        reason="one name's words are contained in the other",
                    )
                )
    return candidates
