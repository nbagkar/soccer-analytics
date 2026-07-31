"""Alias layer tests.

The point of the alias layer is to bridge exactly the divergences normalization
refuses -- appended cities and translations -- so those are the core cases. Also
checked: aliases do not merge genuinely distinct entities, and discovery surfaces the
detectable class without inventing matches.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from itertools import count

import pytest

from soccer.domain.aliases import AliasStore, suggest_duplicates
from soccer.domain.crosswalk import EntityResolver, EntityType
from soccer.storage.live_db import LiveDB

TEAM = EntityType.TEAM


def sequential_ids() -> Callable[[], str]:
    counter = count(1)
    return lambda: f"id-{next(counter)}"


@pytest.fixture
def db(tmp_path) -> LiveDB:
    return LiveDB(tmp_path / "live.sqlite")


@pytest.fixture
def aliased_resolver(db: LiveDB) -> tuple[EntityResolver, AliasStore]:
    aliases = AliasStore(db)
    clock = datetime(2026, 7, 31, tzinfo=UTC)
    resolver = EntityResolver(db, aliases=aliases, id_factory=sequential_ids(), clock=lambda: clock)
    return resolver, aliases


class TestAliasBridgesDivergence:
    def test_appended_city_resolves_together(
        self, aliased_resolver: tuple[EntityResolver, AliasStore]
    ) -> None:
        resolver, aliases = aliased_resolver
        aliases.add(TEAM, "PSV Eindhoven", "PSV", country="Netherlands")

        # football-data.org calls it "PSV"...
        fd = resolver.resolve(
            TEAM,
            source="football_data_org",
            source_id="674",
            name="PSV",
            country="Netherlands",
        )
        # ...TheSportsDB calls it "PSV Eindhoven" -- the alias routes it to the same id.
        tsdb = resolver.resolve(
            TEAM,
            source="thesportsdb",
            source_id="1001",
            name="PSV Eindhoven",
            country="Netherlands",
        )
        assert fd.internal_id == tsdb.internal_id
        assert resolver.entity_count(TEAM) == 1

    def test_translation_resolves_together(
        self, aliased_resolver: tuple[EntityResolver, AliasStore]
    ) -> None:
        # The case normalization provably cannot bridge.
        resolver, aliases = aliased_resolver
        aliases.add(TEAM, "FC Cologne", "1. FC Köln", country="Germany")

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
        assert koln.internal_id == cologne.internal_id

    def test_alias_order_independent(
        self, aliased_resolver: tuple[EntityResolver, AliasStore]
    ) -> None:
        # The variant may be seen before the canonical spelling.
        resolver, aliases = aliased_resolver
        aliases.add(TEAM, "Bayern Munich", "Bayern München", country="Germany")

        first = resolver.resolve(
            TEAM,
            source="thesportsdb",
            source_id="2",
            name="Bayern Munich",
            country="Germany",
        )
        second = resolver.resolve(
            TEAM,
            source="football_data_org",
            source_id="5",
            name="Bayern München",
            country="Germany",
        )
        assert first.internal_id == second.internal_id
        # The entity carries the canonical spelling, not the variant seen first.
        assert first.canonical_name == "Bayern München"

    def test_crosswalk_keeps_the_original_spelling(
        self, aliased_resolver: tuple[EntityResolver, AliasStore]
    ) -> None:
        resolver, aliases = aliased_resolver
        aliases.add(TEAM, "PSV Eindhoven", "PSV")
        resolved = resolver.resolve(
            TEAM, source="thesportsdb", source_id="1001", name="PSV Eindhoven"
        )
        sources = resolver.sources_for(TEAM, resolved.internal_id)
        assert sources[0]["source_name"] == "PSV Eindhoven"  # what the source said


class TestAliasSafety:
    def test_country_scoped_alias_does_not_apply_elsewhere(
        self, aliased_resolver: tuple[EntityResolver, AliasStore]
    ) -> None:
        resolver, aliases = aliased_resolver
        aliases.add(TEAM, "City", "Manchester City", country="England")
        # A "City" in another country must not be swept into Manchester City.
        resolver.resolve(TEAM, source="s", source_id="1", name="Manchester City", country="England")
        other = resolver.resolve(TEAM, source="s", source_id="2", name="City", country="Australia")
        assert other.canonical_name == "City"
        assert resolver.entity_count(TEAM) == 2

    def test_add_rejects_self_alias(self, db: LiveDB) -> None:
        with pytest.raises(ValueError, match="already equivalent"):
            AliasStore(db).add(TEAM, "Arsenal FC", "Arsenal")  # both normalize to arsenal

    def test_add_is_idempotent(self, db: LiveDB) -> None:
        store = AliasStore(db)
        store.add(TEAM, "PSV Eindhoven", "PSV")
        store.add(TEAM, "PSV Eindhoven", "PSV")
        assert len(store.all(TEAM)) == 1


class TestDiscovery:
    def _seed(self, resolver: EntityResolver, *names: str) -> None:
        for i, name in enumerate(names):
            resolver.resolve(TEAM, source="s", source_id=str(i), name=name)

    def test_finds_appended_word_split(self, db: LiveDB) -> None:
        resolver = EntityResolver(db, id_factory=sequential_ids())
        self._seed(resolver, "PSV", "PSV Eindhoven", "Ajax")

        candidates = suggest_duplicates(db, "team")
        pairs = {frozenset((c.name_a, c.name_b)) for c in candidates}
        assert frozenset(("PSV", "PSV Eindhoven")) in pairs
        assert not any("Ajax" in {c.name_a, c.name_b} for c in candidates)

    def test_does_not_invent_translation_matches(self, db: LiveDB) -> None:
        # Köln and Cologne share no tokens; discovery cannot and must not pair them.
        resolver = EntityResolver(db, id_factory=sequential_ids())
        self._seed(resolver, "1. FC Köln", "FC Cologne")
        assert suggest_duplicates(db, "team") == []

    def test_suggested_command_would_fix_the_split(self, db: LiveDB) -> None:
        resolver = EntityResolver(db, id_factory=sequential_ids())
        self._seed(resolver, "Atletico", "Atletico Madrid")
        [candidate] = suggest_duplicates(db, "team")
        # The suggested command aliases the longer form onto the shorter.
        assert 'alias-add "Atletico Madrid" "Atletico"' in candidate.alias_command()


class TestDefaultResolverUnaffected:
    def test_resolver_without_aliases_still_works(self, db: LiveDB) -> None:
        # Aliases are optional; the resolver must behave exactly as before without one.
        resolver = EntityResolver(db, id_factory=sequential_ids())
        a = resolver.resolve(TEAM, source="s", source_id="1", name="PSV")
        b = resolver.resolve(TEAM, source="s2", source_id="2", name="PSV Eindhoven")
        # No alias -> these stay distinct, as established.
        assert a.internal_id != b.internal_id
