"""Ingestion pipeline: adapters -> resolvers -> live state.

The seam that turns the tested-in-isolation parts into a working system. For each
source it fetches (or serves cached), maps payloads to observations, resolves them to
canonical ids, and upserts current state under the source-precedence rule. Raw
snapshots are already written by the adapters, so this layer never touches the network
store directly -- it only builds the queryable live state on top.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from soccer.domain.aliases import AliasStore
from soccer.domain.crosswalk import EntityResolver, EntityType
from soccer.domain.match_state import MatchStateStore
from soccer.domain.matches import MatchResolver
from soccer.ingest.mappers import map_football_data_match, map_thesportsdb_live
from soccer.sources.football_data_org import FootballDataOrg
from soccer.sources.registry import SourceId
from soccer.sources.thesportsdb import TheSportsDB, parse_live_match
from soccer.storage.live_db import LiveDB
from soccer.storage.raw import RawStore

# Derived tables rebuilt by replay, child-first so foreign keys stay satisfied during
# the wipe. entity_alias is deliberately absent -- aliases are human input, not derived
# from raw, and must survive a rebuild so they can be applied to it.
_DERIVED_TABLES = (
    "match_state",
    "match_crosswalk",
    "canonical_match",
    "entity_crosswalk",
    "canonical_entity",
)


@dataclass
class IngestSummary:
    source: str
    matches_seen: int = 0
    matches_created: int = 0
    """Canonical matches minted this run (first time seen)."""
    state_changed: int = 0
    """States written -- excludes those a more-authoritative source already held."""
    stale: bool = False
    """True if the source served cached data rather than a fresh fetch."""

    def __str__(self) -> str:
        flag = " [STALE]" if self.stale else ""
        return (
            f"{self.source}: {self.matches_seen} seen, "
            f"{self.matches_created} new, {self.state_changed} state updates{flag}"
        )


@dataclass
class ReplaySummary:
    snapshots: int = 0
    records: int = 0
    entities: int = 0
    matches: int = 0

    def __str__(self) -> str:
        return (
            f"replayed {self.snapshots} snapshot(s), {self.records} record(s) "
            f"-> {self.entities} entities, {self.matches} matches"
        )


class IngestPipeline:
    def __init__(self, db: LiveDB) -> None:
        self._db = db
        self._entities = EntityResolver(db, aliases=AliasStore(db))
        self._matches = MatchResolver(db, self._entities)
        self._state = MatchStateStore(db)

    async def ingest_football_data(
        self, adapter: FootballDataOrg, date_from: date, date_to: date
    ) -> IngestSummary:
        summary = IngestSummary(source="football_data_org")
        results = await adapter.matches_over_range(date_from, date_to)

        for result in results:
            if result.is_stale:
                summary.stale = True
            for record in result.payload.get("matches", []):
                observation, state = map_football_data_match(
                    record, observed_at=result.fetched_at, is_stale=result.is_stale
                )
                created, changed = self._apply(observation, state)
                summary.matches_seen += 1
                summary.matches_created += created
                summary.state_changed += changed
        return summary

    async def ingest_thesportsdb(self, adapter: TheSportsDB) -> IngestSummary:
        summary = IngestSummary(source="thesportsdb")
        result = await adapter.livescore()
        summary.stale = result.is_stale

        for live in result.matches:
            observation, state = map_thesportsdb_live(
                live, observed_at=result.fetched_at, is_stale=result.is_stale
            )
            created, changed = self._apply(observation, state)
            summary.matches_seen += 1
            summary.matches_created += created
            summary.state_changed += changed
        return summary

    def replay_from_raw(self, raw: RawStore) -> ReplaySummary:
        """Rebuild all derived state from immutable snapshots, applying current aliases.

        Makes aliases retroactive and re-derives everything after any resolver or mapper
        fix -- the payoff of storing raw responses verbatim. Runs in one transaction, so
        a failure leaves the previous state intact rather than a half-rebuilt database.
        Refuses to wipe if no snapshots exist, so a missing raw store cannot silently
        erase live state.
        """
        total = raw.count_snapshots(SourceId.FOOTBALL_DATA_ORG, "matches") + raw.count_snapshots(
            SourceId.THESPORTSDB, "livescore"
        )
        if total == 0:
            raise ValueError("no raw snapshots to replay from; refusing to wipe state")

        summary = ReplaySummary()
        with self._db.transaction():
            for table in _DERIVED_TABLES:
                # Table names come from the fixed _DERIVED_TABLES constant, not input.
                self._db.connection.execute(f"DELETE FROM {table}")

            # football-data.org matches, oldest snapshot first.
            for snap in raw.iter_snapshots(SourceId.FOOTBALL_DATA_ORG, "matches"):
                summary.snapshots += 1
                for record in snap.payload.get("matches", []):
                    obs, state = map_football_data_match(
                        record, observed_at=snap.fetched_at, is_stale=False
                    )
                    self._apply(obs, state)
                    summary.records += 1

            # TheSportsDB livescore, oldest snapshot first.
            for snap in raw.iter_snapshots(SourceId.THESPORTSDB, "livescore"):
                summary.snapshots += 1
                for row in snap.payload.get("livescore") or []:
                    live = parse_live_match(row)
                    if live is None:
                        continue
                    obs, state = map_thesportsdb_live(
                        live, observed_at=snap.fetched_at, is_stale=False
                    )
                    self._apply(obs, state)
                    summary.records += 1

        summary.entities = self._entities.entity_count(EntityType.TEAM)
        summary.matches = self._matches.match_count()
        return summary

    def _apply(self, observation, state) -> tuple[bool, bool]:
        """Resolve one observation and upsert its state. Returns (created, state_changed)."""
        resolved = self._matches.resolve(observation)
        changed = self._state.upsert(resolved.internal_id, state)
        return resolved.created, changed
