"""Entity resolver tests.

The behaviour that matters: the same club from two sources resolves to ONE internal
id, distinct clubs stay distinct even when their names normalize alike, and every link
records how it was made. Uses a deterministic id factory so assertions can name ids.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from itertools import count

import pytest

from soccer.domain.crosswalk import EntityResolver, EntityType, LinkMethod
from soccer.storage.live_db import LiveDB


def sequential_ids() -> Callable[[], str]:
    counter = count(1)
    return lambda: f"id-{next(counter)}"


@pytest.fixture
def resolver(tmp_path) -> EntityResolver:
    db = LiveDB(tmp_path / "live.sqlite")
    clock = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)
    return EntityResolver(db, id_factory=sequential_ids(), clock=lambda: clock)


TEAM = EntityType.TEAM


class TestCrossSourceLinking:
    """The core promise: one club, two sources, one internal id."""

    def test_same_club_from_two_sources_shares_one_id(self, resolver: EntityResolver) -> None:
        # football-data.org: "Arsenal FC" with its integer id.
        fd = resolver.resolve(
            TEAM,
            source="football_data_org",
            source_id="57",
            name="Arsenal FC",
            country="England",
        )
        # TheSportsDB: "Arsenal" with its own id -- different id space, same club.
        tsdb = resolver.resolve(
            TEAM,
            source="thesportsdb",
            source_id="133604",
            name="Arsenal",
            country="England",
        )

        assert fd.internal_id == tsdb.internal_id
        assert fd.created is True
        assert tsdb.created is False
        # The second link is an inference, and says so.
        assert fd.method is LinkMethod.SOURCE_ID
        assert tsdb.method is LinkMethod.EXACT_NAME
        assert tsdb.confidence < fd.confidence

    def test_one_entity_gathers_all_source_ids(self, resolver: EntityResolver) -> None:
        fd = resolver.resolve(
            TEAM,
            source="football_data_org",
            source_id="57",
            name="Arsenal FC",
            country="England",
        )
        resolver.resolve(
            TEAM,
            source="thesportsdb",
            source_id="133604",
            name="Arsenal",
            country="England",
        )

        crosswalk = resolver.sources_for(TEAM, fd.internal_id)
        assert {row["source"] for row in crosswalk} == {
            "football_data_org",
            "thesportsdb",
        }
        assert resolver.entity_count(TEAM) == 1

    def test_name_only_source_links_to_id_based_entity(self, resolver: EntityResolver) -> None:
        # openfootball / football-data.co.uk have no ids -- only names.
        fd = resolver.resolve(
            TEAM,
            source="football_data_org",
            source_id="57",
            name="Arsenal FC",
            country="England",
        )
        csv = resolver.resolve(
            TEAM,
            source="football_data_co_uk",
            source_id=None,
            name="Arsenal",
            country="England",
        )
        assert csv.internal_id == fd.internal_id
        assert csv.method is LinkMethod.NAME_ONLY


class TestDistinctEntitiesStayDistinct:
    def test_different_countries_block_name_merge(self, resolver: EntityResolver) -> None:
        # Arsenal FC (England) and Arsenal de Sarandí (Argentina) normalize close but
        # are different clubs. Country must keep them apart.
        england = resolver.resolve(
            TEAM,
            source="football_data_org",
            source_id="57",
            name="Arsenal",
            country="England",
        )
        argentina = resolver.resolve(
            TEAM,
            source="some_source",
            source_id="999",
            name="Arsenal",
            country="Argentina",
        )
        assert england.internal_id != argentina.internal_id
        assert resolver.entity_count(TEAM) == 2

    def test_rivals_are_not_merged(self, resolver: EntityResolver) -> None:
        united = resolver.resolve(
            TEAM,
            source="football_data_org",
            source_id="66",
            name="Manchester United FC",
            country="England",
        )
        city = resolver.resolve(
            TEAM,
            source="football_data_org",
            source_id="65",
            name="Manchester City FC",
            country="England",
        )
        assert united.internal_id != city.internal_id


class TestIdempotence:
    def test_repeat_sighting_returns_same_entity_without_creating(
        self, resolver: EntityResolver
    ) -> None:
        first = resolver.resolve(
            TEAM,
            source="thesportsdb",
            source_id="133604",
            name="Arsenal",
            country="England",
        )
        second = resolver.resolve(
            TEAM,
            source="thesportsdb",
            source_id="133604",
            name="Arsenal",
            country="England",
        )
        assert first.internal_id == second.internal_id
        assert second.created is False
        assert resolver.entity_count(TEAM) == 1

    def test_name_only_repeat_is_idempotent(self, resolver: EntityResolver) -> None:
        first = resolver.resolve(TEAM, source="openfootball", source_id=None, name="Liverpool FC")
        second = resolver.resolve(TEAM, source="openfootball", source_id=None, name="Liverpool")
        # Both normalize to "liverpool"; the second must not create a duplicate.
        assert first.internal_id == second.internal_id
        assert resolver.entity_count(TEAM) == 1


class TestManualCorrection:
    def test_manual_link_repoints_a_bad_match(self, resolver: EntityResolver) -> None:
        # Köln and Cologne cannot be matched automatically, so they start separate.
        koln = resolver.resolve(
            TEAM,
            source="football_data_org",
            source_id="1",
            name="1. FC Köln",
            country="Germany",
        )
        cologne = resolver.resolve(
            TEAM,
            source="thesportsdb",
            source_id="2",
            name="FC Cologne",
            country="Germany",
        )
        assert koln.internal_id != cologne.internal_id
        assert resolver.entity_count(TEAM) == 2

        # An operator joins them by hand.
        resolver.link_manually(
            TEAM, source="thesportsdb", source_id="2", internal_id=koln.internal_id
        )
        relinked = resolver.resolve(
            TEAM,
            source="thesportsdb",
            source_id="2",
            name="FC Cologne",
            country="Germany",
        )
        assert relinked.internal_id == koln.internal_id
        assert relinked.method is LinkMethod.MANUAL


class TestUnresolvable:
    def test_no_id_and_empty_name_raises(self, resolver: EntityResolver) -> None:
        with pytest.raises(ValueError, match="no id and no usable name"):
            resolver.resolve(TEAM, source="x", source_id=None, name="...")


class TestPersistence:
    def test_state_survives_reopen(self, tmp_path) -> None:
        path = tmp_path / "live.sqlite"
        clock = datetime(2026, 7, 31, tzinfo=UTC)

        db1 = LiveDB(path)
        r1 = EntityResolver(db1, id_factory=lambda: "fixed-id", clock=lambda: clock)
        r1.resolve(
            TEAM,
            source="football_data_org",
            source_id="57",
            name="Arsenal FC",
            country="England",
        )
        db1.close()

        db2 = LiveDB(path)
        r2 = EntityResolver(db2, clock=lambda: clock)
        again = r2.resolve(
            TEAM,
            source="thesportsdb",
            source_id="133604",
            name="Arsenal",
            country="England",
        )
        assert again.internal_id == "fixed-id"
        assert again.created is False
        db2.close()
