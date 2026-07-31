"""Match resolver tests.

The behaviours that matter: the same match seen repeatedly collapses to one id
(the everyday case), the same fixture from two sources with divergent team names
still collapses (the rare-but-must-work reconciliation case), and distinct fixtures
stay distinct -- including the same pairing on different days and flipped orientation.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from itertools import count

import pytest

from soccer.domain.crosswalk import EntityResolver
from soccer.domain.matches import (
    MatchLinkMethod,
    MatchObservation,
    MatchResolver,
    SourceRef,
)
from soccer.storage.live_db import LiveDB


def sequential_ids(prefix: str) -> Callable[[], str]:
    counter = count(1)
    return lambda: f"{prefix}-{next(counter)}"


@pytest.fixture
def resolver(tmp_path) -> MatchResolver:
    db = LiveDB(tmp_path / "live.sqlite")
    clock = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)
    entities = EntityResolver(db, id_factory=sequential_ids("e"), clock=lambda: clock)
    return MatchResolver(db, entities, id_factory=sequential_ids("m"), clock=lambda: clock)


def fd_match(
    match_id: str,
    home: str,
    away: str,
    kickoff: datetime,
    *,
    source: str = "football_data_org",
    competition: str = "Eredivisie",
    country: str = "Netherlands",
    home_id: str | None = None,
    away_id: str | None = None,
) -> MatchObservation:
    return MatchObservation(
        source=source,
        source_match_id=match_id,
        competition=SourceRef(id="2003", name=competition, country=country),
        home=SourceRef(id=home_id, name=home, country=country),
        away=SourceRef(id=away_id, name=away, country=country),
        kickoff=kickoff,
    )


KICKOFF = datetime(2026, 8, 8, 18, 0, tzinfo=UTC)


class TestSameSourceIdempotence:
    """The everyday case: one match polled many times is one entity."""

    def test_repeat_poll_returns_same_match(self, resolver: MatchResolver) -> None:
        first = resolver.resolve(fd_match("558214", "PSV", "Fortuna Sittard", KICKOFF))
        second = resolver.resolve(fd_match("558214", "PSV", "Fortuna Sittard", KICKOFF))

        assert first.internal_id == second.internal_id
        assert first.created is True
        assert second.created is False
        assert resolver.match_count() == 1

    def test_score_change_does_not_fork_the_match(self, resolver: MatchResolver) -> None:
        # A later poll of the same match id is the same match regardless of state.
        first = resolver.resolve(fd_match("558214", "PSV", "Fortuna Sittard", KICKOFF))
        # kickoff reported a minute later on a subsequent poll -- still one match.
        second = resolver.resolve(
            fd_match("558214", "PSV", "Fortuna Sittard", KICKOFF + timedelta(minutes=1))
        )
        assert first.internal_id == second.internal_id
        assert resolver.match_count() == 1


class TestCrossSourceReconciliation:
    """The rare case that must still work: same fixture, two sources, divergent names."""

    def test_same_fixture_two_sources_one_id(self, resolver: MatchResolver) -> None:
        # English names diverge only by the "FC" affix, which normalization bridges.
        # A source naming teams "Arsenal FC" / "Chelsea FC" with its ids...
        fd = resolver.resolve(
            MatchObservation(
                source="football_data_org",
                source_match_id="558214",
                competition=SourceRef(id="2021", name="Premier League", country="England"),
                home=SourceRef(id="57", name="Arsenal FC", country="England"),
                away=SourceRef(id="61", name="Chelsea FC", country="England"),
                kickoff=KICKOFF,
            )
        )
        # ...and TheSportsDB naming them "Arsenal" / "Chelsea", its own ids, kickoff
        # reported 3 minutes off. Same match.
        other = resolver.resolve(
            MatchObservation(
                source="thesportsdb",
                source_match_id="99001",
                competition=SourceRef(id="4328", name="Premier League", country="England"),
                home=SourceRef(id="133604", name="Arsenal", country="England"),
                away=SourceRef(id="133610", name="Chelsea", country="England"),
                kickoff=KICKOFF + timedelta(minutes=3),
            )
        )
        assert fd.internal_id == other.internal_id
        assert other.created is False
        assert other.method is MatchLinkMethod.COMPONENTS
        assert resolver.match_count() == 1
        assert {r["source"] for r in resolver.sources_for(fd.internal_id)} == {
            "football_data_org",
            "thesportsdb",
        }

    def test_component_name_divergence_blocks_reconciliation(self, resolver: MatchResolver) -> None:
        # A documented limitation, not a bug: "PSV" vs "PSV Eindhoven" diverges beyond
        # what affix-stripping bridges (an appended city, like Köln/Cologne), so the
        # home teams resolve separately and the match does NOT auto-reconcile. This is
        # what an alias layer would eventually fix; until then, manual linking.
        fd = resolver.resolve(fd_match("558214", "PSV", "Fortuna Sittard", KICKOFF))
        other = resolver.resolve(
            MatchObservation(
                source="thesportsdb",
                source_match_id="99001",
                competition=SourceRef(id="2003", name="Eredivisie", country="Netherlands"),
                home=SourceRef(id="1001", name="PSV Eindhoven", country="Netherlands"),
                away=SourceRef(id="1002", name="Fortuna Sittard", country="Netherlands"),
                kickoff=KICKOFF,
            )
        )
        assert fd.internal_id != other.internal_id
        assert resolver.match_count() == 2


class TestDistinctMatchesStayDistinct:
    def test_same_pairing_different_day_is_a_different_match(self, resolver: MatchResolver) -> None:
        # Home and away legs of a tie: same teams, days apart.
        leg1 = resolver.resolve(fd_match("1", "PSV", "Ajax", KICKOFF))
        leg2 = resolver.resolve(fd_match("2", "PSV", "Ajax", KICKOFF + timedelta(days=7)))
        assert leg1.internal_id != leg2.internal_id
        assert resolver.match_count() == 2

    def test_flipped_orientation_is_not_auto_merged(self, resolver: MatchResolver) -> None:
        # A v B and B v A on the same day -- kept separate; manual linking if needed.
        ab = resolver.resolve(fd_match("1", "PSV", "Ajax", KICKOFF))
        ba = resolver.resolve(fd_match("2", "Ajax", "PSV", KICKOFF))
        assert ab.internal_id != ba.internal_id

    def test_kickoff_outside_tolerance_is_a_different_match(self, resolver: MatchResolver) -> None:
        near = resolver.resolve(fd_match("1", "PSV", "Ajax", KICKOFF))
        far = resolver.resolve(
            MatchObservation(
                source="thesportsdb",
                source_match_id="2",
                competition=SourceRef(id="2003", name="Eredivisie", country="Netherlands"),
                home=SourceRef(id=None, name="PSV", country="Netherlands"),
                away=SourceRef(id=None, name="Ajax", country="Netherlands"),
                kickoff=KICKOFF + timedelta(hours=12),  # beyond the 6h window
            )
        )
        assert near.internal_id != far.internal_id


class TestComponentConfidence:
    def test_match_confidence_reflects_weakest_component(self, resolver: MatchResolver) -> None:
        # A cross-source link inherits the component confidence (name-linked = 0.9),
        # never claiming more certainty than its shakiest part.
        resolver.resolve(fd_match("1", "PSV", "Ajax", KICKOFF, home_id="674", away_id="678"))
        linked = resolver.resolve(
            MatchObservation(
                source="thesportsdb",
                source_match_id="2",
                competition=SourceRef(id="4337", name="Eredivisie", country="Netherlands"),
                home=SourceRef(id="1", name="PSV", country="Netherlands"),
                away=SourceRef(id="2", name="Ajax", country="Netherlands"),
                kickoff=KICKOFF,
            )
        )
        assert linked.method is MatchLinkMethod.COMPONENTS
        assert linked.component_confidence < 1.0


class TestNameOnlySource:
    def test_no_match_id_uses_component_key(self, resolver: MatchResolver) -> None:
        # openfootball / football-data.co.uk give no match id.
        first = resolver.resolve(
            MatchObservation(
                source="openfootball",
                source_match_id=None,
                competition=SourceRef(id=None, name="Eredivisie", country="Netherlands"),
                home=SourceRef(id=None, name="PSV", country="Netherlands"),
                away=SourceRef(id=None, name="Ajax", country="Netherlands"),
                kickoff=KICKOFF,
            )
        )
        second = resolver.resolve(
            MatchObservation(
                source="openfootball",
                source_match_id=None,
                competition=SourceRef(id=None, name="Eredivisie", country="Netherlands"),
                home=SourceRef(id=None, name="PSV", country="Netherlands"),
                away=SourceRef(id=None, name="Ajax", country="Netherlands"),
                kickoff=KICKOFF,
            )
        )
        assert first.internal_id == second.internal_id
        assert resolver.match_count() == 1


class TestManualLink:
    def test_manual_link_joins_flipped_neutral_venue_games(self, resolver: MatchResolver) -> None:
        ab = resolver.resolve(
            fd_match("1", "Argentina", "France", KICKOFF, competition="World Cup", country="World")
        )
        ba = resolver.resolve(
            fd_match(
                "2",
                "France",
                "Argentina",
                KICKOFF,
                source="thesportsdb",
                competition="World Cup",
                country="World",
            )
        )
        assert ab.internal_id != ba.internal_id

        resolver.link_manually(
            source="thesportsdb", source_match_id="2", internal_id=ab.internal_id
        )
        relinked = resolver.resolve(
            fd_match(
                "2",
                "France",
                "Argentina",
                KICKOFF,
                source="thesportsdb",
                competition="World Cup",
                country="World",
            )
        )
        assert relinked.internal_id == ab.internal_id
        assert relinked.method is MatchLinkMethod.MANUAL


class TestSchemaMigration:
    def test_database_initializes_at_current_schema(self, tmp_path) -> None:
        # A fresh live.sqlite must carry every table at the current schema version.
        from soccer.storage.live_db import SCHEMA_VERSION

        path = tmp_path / "live.sqlite"
        db = LiveDB(path)
        assert db.connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        for tbl in ("canonical_match", "match_crosswalk", "match_state", "entity_alias"):
            db.connection.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()
        db.close()

    def test_existing_v1_database_migrates_forward(self, tmp_path) -> None:
        # A database written by an earlier schema (only entity tables, user_version=1)
        # must gain the later tables on reopen, without losing its rows.
        import sqlite3

        from soccer.storage.live_db import SCHEMA_VERSION

        path = tmp_path / "live.sqlite"
        conn = sqlite3.connect(str(path))
        conn.executescript(
            "CREATE TABLE canonical_entity ("
            "  internal_id TEXT PRIMARY KEY, entity_type TEXT, canonical_name TEXT, "
            "  normalized_name TEXT, country TEXT, created_at TEXT);"
            "INSERT INTO canonical_entity VALUES "
            "  ('e1','team','Arsenal FC','arsenal','England','2026-07-31');"
            "PRAGMA user_version=1;"
        )
        conn.close()

        db = LiveDB(path)
        assert db.connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        db.connection.execute("SELECT COUNT(*) FROM canonical_match").fetchone()
        db.connection.execute("SELECT COUNT(*) FROM entity_alias").fetchone()
        # Existing data survived the migration.
        preserved = db.connection.execute(
            "SELECT canonical_name FROM canonical_entity WHERE internal_id='e1'"
        ).fetchone()
        assert preserved["canonical_name"] == "Arsenal FC"
        db.close()
