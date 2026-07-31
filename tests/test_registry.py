"""Registry tests.

These exist to stop the codebase quietly re-acquiring the assumptions that the
source verification disproved. Each test below corresponds to a specific claim in
the original project plan that turned out to be false; if one starts failing, it
means someone has re-added a capability a provider does not actually offer for free.
"""

from __future__ import annotations

import pytest

from soccer.sources.registry import (
    SOURCES,
    Capability,
    Source,
    SourceId,
    Trust,
    attributions,
    sources_for,
)


class TestVerifiedRealities:
    """Guards against regressing to the original plan's disproved assumptions."""

    def test_football_data_org_does_not_claim_live(self) -> None:
        # Free tier is documented as "Scores delayed"; live starts at EUR 12/mo.
        source = SOURCES[SourceId.FOOTBALL_DATA_ORG]
        assert not source.supports(Capability.LIVE_SCORES)

    def test_football_data_org_does_not_claim_lineups_or_squads(self) -> None:
        # Lineups, subs, scorers, cards and squads all start at EUR 29/mo.
        source = SOURCES[SourceId.FOOTBALL_DATA_ORG]
        assert not source.supports(Capability.LINEUPS)
        assert not source.supports(Capability.SQUADS)

    def test_openligadb_does_not_claim_live(self) -> None:
        # Empirically disproved: 0 of 306 Bundesliga 2025/26 matches updated during
        # play; median lag 186 minutes.
        source = SOURCES[SourceId.OPENLIGADB]
        assert not source.supports(Capability.LIVE_SCORES)
        assert source.latency_seconds is not None
        assert source.latency_seconds > 3600

    def test_openligadb_flagged_as_retroactively_mutable(self) -> None:
        # Any logged-in user may edit any result for six days after the match.
        assert SOURCES[SourceId.OPENLIGADB].mutable_history

    def test_espn_is_absent(self) -> None:
        # Disney's terms bar automated collection and database building. Omitted
        # entirely rather than shipped behind a flag.
        assert not any("espn" in source_id.value for source_id in SOURCES)

    def test_grey_and_unstable_sources_are_off_by_default(self) -> None:
        assert not SOURCES[SourceId.FPL].enabled_by_default
        assert not SOURCES[SourceId.OPENLIGADB].enabled_by_default

    def test_only_openfootball_is_redistributable(self) -> None:
        # Only CC0 data may be republished verbatim. Everything else is either
        # unlicensed or explicitly restricted.
        redistributable = {s.id for s in SOURCES.values() if s.may_redistribute}
        assert redistributable == {SourceId.OPENFOOTBALL}


class TestCapabilityResolution:
    def test_live_scores_resolve_to_thesportsdb(self) -> None:
        # The only free live source whose competitions we control.
        live = sources_for(Capability.LIVE_SCORES)
        assert [s.id for s in live] == [SourceId.THESPORTSDB]

    def test_disabled_sources_excluded_unless_requested(self) -> None:
        default = {s.id for s in sources_for(Capability.GOAL_EVENTS)}
        everything = {s.id for s in sources_for(Capability.GOAL_EVENTS, include_disabled=True)}
        assert SourceId.FPL not in default
        assert SourceId.FPL in everything

    def test_results_ordered_by_trust_then_freshness(self) -> None:
        results = sources_for(Capability.RESULTS)
        trust_rank = {Trust.PRIMARY: 0, Trust.CORROBORATING: 1, Trust.EXPERIMENTAL: 2}
        ranks = [trust_rank[s.trust] for s in results]
        assert ranks == sorted(ranks)

    def test_unserved_capability_returns_empty_not_error(self) -> None:
        # Nothing free provides live xG. Callers must render a "not available"
        # badge rather than an empty section, so this must not raise.
        assert sources_for(Capability.EXPECTED_GOALS) == []


class TestIntegrity:
    @pytest.mark.parametrize("source", SOURCES.values(), ids=lambda s: s.id.value)
    def test_source_is_self_consistent(self, source: Source) -> None:
        assert source.capabilities, f"{source.id} declares no capabilities"
        assert source.licence, f"{source.id} has no licence recorded"
        # A source claiming live data must be fresh enough to justify it.
        if source.supports(Capability.LIVE_SCORES):
            assert source.latency_seconds is not None
            assert source.latency_seconds <= 120

    @pytest.mark.parametrize("source", SOURCES.values(), ids=lambda s: s.id.value)
    def test_unlicensed_sources_are_not_redistributable(self, source: Source) -> None:
        if "NO EXPLICIT LICENCE" in source.licence or "UNCONFIRMED" in source.licence:
            assert not source.may_redistribute, (
                f"{source.id} has unresolved licensing and must not be redistributed"
            )

    def test_registry_keys_match_source_ids(self) -> None:
        assert all(key == source.id for key, source in SOURCES.items())

    def test_attribution_required_where_terms_demand_it(self) -> None:
        # TheSportsDB requires attribution with a link back.
        assert SOURCES[SourceId.THESPORTSDB].attribution is not None
        assert any("TheSportsDB" in line for line in attributions())
