"""Source registry: what each provider actually gives us, verified 2026-07-31.

This module is deliberately data-heavy. Every capability claim here was verified by
direct fetch, not from documentation or recall -- see docs/source-verification.md for
the evidence behind each entry. The original project plan assumed capabilities that
several of these providers do not offer for free, so the registry is the single place
where reality is recorded and the rest of the codebase reads it.

The important invariant: no feature may depend on a capability a source does not
declare here. If `SourceId.FOOTBALL_DATA_ORG` does not declare `Capability.LINEUPS`,
the match page must render a "not available" badge rather than an empty section.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class SourceId(StrEnum):
    FOOTBALL_DATA_ORG = "football_data_org"
    THESPORTSDB = "thesportsdb"
    OPENLIGADB = "openligadb"
    FOOTBALL_DATA_CO_UK = "football_data_co_uk"
    OPENFOOTBALL = "openfootball"
    FPL = "fpl"


class Capability(StrEnum):
    """What a source can actually deliver. Absence is meaningful."""

    FIXTURES = "fixtures"
    RESULTS = "results"
    STANDINGS = "standings"
    LIVE_SCORES = "live_scores"
    """Genuinely in-match. Delayed results do NOT count -- see `latency_seconds`."""
    GOAL_EVENTS = "goal_events"
    LINEUPS = "lineups"
    SQUADS = "squads"
    MATCH_STATS = "match_stats"
    """Shots, corners, cards -- match-level aggregates, not event streams."""
    BETTING_ODDS = "betting_odds"
    PLAYER_STATS = "player_stats"
    EXPECTED_GOALS = "expected_goals"


class Trust(StrEnum):
    """How much weight this source gets when observations conflict."""

    PRIMARY = "primary"
    CORROBORATING = "corroborating"
    EXPERIMENTAL = "experimental"
    """Undocumented or unstable. Never the sole source for a feature."""


@dataclass(frozen=True)
class Source:
    id: SourceId
    name: str
    base_url: str
    capabilities: frozenset[Capability]
    trust: Trust

    latency_seconds: int | None
    """Typical delay between real-world event and availability. None = static dataset."""

    rate_limit_per_minute: int | None
    licence: str
    attribution: str | None = None

    enabled_by_default: bool = True
    may_redistribute: bool = False
    """Whether we may expose this source's payloads verbatim. Almost always False."""

    mutable_history: bool = False
    """Whether already-published values can change retroactively. Forces re-checks."""

    caveats: tuple[str, ...] = field(default_factory=tuple)

    def supports(self, capability: Capability) -> bool:
        return capability in self.capabilities


SOURCES: dict[SourceId, Source] = {
    SourceId.FOOTBALL_DATA_ORG: Source(
        id=SourceId.FOOTBALL_DATA_ORG,
        name="football-data.org",
        base_url="https://api.football-data.org/v4",
        # Deliberately no LIVE_SCORES: the free tier is documented as "Scores delayed".
        # Livescores begin at the EUR 12/mo tier. Also no LINEUPS/SQUADS/GOAL_EVENTS --
        # those start at EUR 29/mo.
        capabilities=frozenset({Capability.FIXTURES, Capability.RESULTS, Capability.STANDINGS}),
        trust=Trust.PRIMARY,
        latency_seconds=None,
        rate_limit_per_minute=10,
        licence="Free tier, proprietary terms",
        caveats=(
            "10 req/min is the binding constraint. Always poll the date-batched "
            "/matches endpoint; per-match polling exhausts the budget on one matchday.",
            "12 competitions only, no domestic cups, no women's competitions.",
        ),
    ),
    SourceId.THESPORTSDB: Source(
        id=SourceId.THESPORTSDB,
        name="TheSportsDB",
        base_url="https://www.thesportsdb.com/api/v1/json",
        capabilities=frozenset({Capability.LIVE_SCORES, Capability.GOAL_EVENTS}),
        trust=Trust.CORROBORATING,
        latency_seconds=60,
        rate_limit_per_minute=30,
        licence="Proprietary; storage explicitly permitted, resale prohibited",
        attribution="Data from TheSportsDB (https://www.thesportsdb.com)",
        caveats=(
            "Currently the only confirmed free live feed. Verified at ~60s refresh "
            "with match minute.",
            "Free access to livescore.php appears UNDOCUMENTED -- the pricing page "
            "frames livescore as paid. Could be withdrawn without notice; the live "
            "surface must degrade to delayed rather than break.",
            "Bulk endpoints are crippled on free (5 leagues, 15 events/season). "
            "Never use for backfill.",
        ),
    ),
    SourceId.OPENLIGADB: Source(
        id=SourceId.OPENLIGADB,
        name="OpenLigaDB",
        base_url="https://api.openligadb.de",
        # Explicitly NOT live: all 306 Bundesliga 2025/26 matches were checked and
        # none updated during play. Median lag 186 min -- a scheduled post-match import.
        capabilities=frozenset(
            {
                Capability.FIXTURES,
                Capability.RESULTS,
                Capability.STANDINGS,
                Capability.GOAL_EVENTS,
            }
        ),
        trust=Trust.CORROBORATING,
        latency_seconds=5400,
        rate_limit_per_minute=None,
        licence="ODbL (UNCONFIRMED -- licence page would not render)",
        enabled_by_default=False,
        mutable_history=True,
        caveats=(
            "Any logged-in user can edit any result for six days after the match. "
            "Store lastUpdateDateTime and re-check within that window.",
            "Poll getlastchangedate before getmatchdata -- the documented polite path.",
            "League shortcuts are a trap: 816 all-time entries include test junk and "
            "duplicate shortcuts for the same competition, most dead. Pin known-good "
            "codes; never enumerate getavailableleagues blindly.",
            "LICENCE UNRESOLVED. If ODbL genuinely applies it carries share-alike "
            "obligations on derived databases. Confirm before building on this.",
        ),
    ),
    SourceId.FOOTBALL_DATA_CO_UK: Source(
        id=SourceId.FOOTBALL_DATA_CO_UK,
        name="football-data.co.uk",
        base_url="https://www.football-data.co.uk",
        capabilities=frozenset(
            {Capability.RESULTS, Capability.MATCH_STATS, Capability.BETTING_ODDS}
        ),
        trust=Trust.PRIMARY,
        latency_seconds=172_800,
        rate_limit_per_minute=None,
        licence="NO EXPLICIT LICENCE FOUND",
        caveats=(
            "The analytics backbone: 22 divisions, shots/corners/cards back to "
            "2000/01, plus closing odds from 17 bookmakers. No xG.",
            "No copyright grant and no redistribution clause exist. Downloading and "
            "analysing is plainly intended; republishing the CSVs is not granted.",
            "Men's football only.",
        ),
    ),
    SourceId.OPENFOOTBALL: Source(
        id=SourceId.OPENFOOTBALL,
        name="openfootball (Football.TXT)",
        # The per-country .TXT repos, NOT football.json -- the JSON artifact is a stale
        # downstream generation, ~2 months behind with no current season.
        base_url="https://github.com/openfootball",
        capabilities=frozenset({Capability.FIXTURES, Capability.RESULTS}),
        trust=Trust.CORROBORATING,
        latency_seconds=86_400,
        rate_limit_per_minute=None,
        licence="CC0-1.0",
        may_redistribute=True,
        caveats=(
            "Cleanest licence of any source here -- public domain, no attribution "
            "required, redistribution fine.",
            "Requires a Football.TXT parser. That cost buys the current season; "
            "football.json does not have it.",
            "Stops updating out of season. Not a live fallback.",
        ),
    ),
    SourceId.FPL: Source(
        id=SourceId.FPL,
        name="Fantasy Premier League API",
        base_url="https://fantasy.premierleague.com/api",
        capabilities=frozenset(
            {
                Capability.LIVE_SCORES,
                Capability.PLAYER_STATS,
                Capability.EXPECTED_GOALS,
                Capability.GOAL_EVENTS,
            }
        ),
        trust=Trust.EXPERIMENTAL,
        latency_seconds=60,
        rate_limit_per_minute=None,
        licence="Premier League terms; 'creating a database' expressly barred",
        enabled_by_default=False,
        caveats=(
            "Uniquely valuable: free Opta-derived xG/xA, BPS, injury news. Premier League only.",
            "GENUINELY GREY. PL terms contemplate private personal use but bar "
            "'creating a database'. Off by default and surfaced in the UI as such.",
            "Undocumented, no published rate limit. Cache hard, poll gently, never redistribute.",
        ),
    ),
}

# ESPN is deliberately absent. Disney's terms bar automated access and "compiling,
# building, creating or contributing to any collection of data, data set or database"
# -- precisely what this project does. Everything ESPN offers is available from a
# source with better terms, so it is omitted entirely rather than shipped behind a
# flag that would eventually get flipped.


def sources_for(capability: Capability, *, include_disabled: bool = False) -> list[Source]:
    """Sources that can serve a capability, best-trusted and freshest first."""
    trust_order = {Trust.PRIMARY: 0, Trust.CORROBORATING: 1, Trust.EXPERIMENTAL: 2}
    candidates = [
        source
        for source in SOURCES.values()
        if source.supports(capability) and (include_disabled or source.enabled_by_default)
    ]
    return sorted(
        candidates,
        key=lambda s: (trust_order[s.trust], s.latency_seconds or 0),
    )


def attributions() -> list[str]:
    """Attribution strings that must be rendered wherever data is displayed."""
    return sorted(s.attribution for s in SOURCES.values() if s.attribution)
