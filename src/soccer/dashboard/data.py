"""Dashboard data preparation -- pure reads, no Streamlit.

Kept free of any UI import so it is unit-testable and so the Streamlit layer stays a
thin render over these models. Everything here is a read over the live database and the
source registry; the dashboard never touches the network (that is `soccer ingest`'s
job). A view that cannot be filled by the data we actually have is simply not produced
-- there is no Player Hub or shot map here, because nothing ingests that yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from soccer.config import Settings
from soccer.domain.aliases import Alias, AliasStore, DuplicateCandidate, suggest_duplicates
from soccer.domain.match_state import MatchStateStore, MatchView
from soccer.models.elo import EloRating, power_ranking
from soccer.models.poisson import fit_poisson
from soccer.models.simulation import TeamProjection, simulate_season
from soccer.sources.registry import SOURCES, Capability, attributions, sources_for
from soccer.storage.analytics_db import AnalyticsDB, TableRow
from soccer.storage.live_db import LiveDB


@dataclass(frozen=True)
class LiveKpis:
    total: int
    in_play: int
    finished: int
    competitions: int
    sources: int
    last_updated: datetime | None
    any_stale: bool

    @property
    def freshness_label(self) -> str:
        if self.last_updated is None:
            return "never"
        age = datetime.now(UTC) - self.last_updated
        minutes = int(age.total_seconds() // 60)
        if minutes < 1:
            return "just now"
        if minutes < 60:
            return f"{minutes} min ago"
        return f"{minutes // 60}h ago"


@dataclass(frozen=True)
class LiveSnapshot:
    kpis: LiveKpis
    matches: list[MatchView]
    competition_counts: list[tuple[str, int]]
    """(competition, match count), most matches first -- coverage at a glance."""


def live_snapshot(
    db: LiveDB, *, in_play_only: bool = False, competition: str | None = None
) -> LiveSnapshot:
    store = MatchStateStore(db)
    all_views = store.list_current()  # unfiltered, for honest KPIs

    kpis = LiveKpis(
        total=len(all_views),
        in_play=sum(1 for v in all_views if v.status.is_in_play),
        finished=sum(1 for v in all_views if v.status.is_concluded),
        competitions=len({v.competition for v in all_views}),
        sources=len({v.source for v in all_views}),
        last_updated=_last_updated(db),
        any_stale=any(v.is_stale for v in all_views),
    )

    counts: dict[str, int] = {}
    for view in all_views:
        counts[view.competition] = counts.get(view.competition, 0) + 1
    competition_counts = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))

    shown = all_views
    if in_play_only:
        shown = [v for v in shown if v.status.is_in_play]
    if competition:
        shown = [v for v in shown if v.competition == competition]

    return LiveSnapshot(kpis=kpis, matches=shown, competition_counts=competition_counts)


def _last_updated(db: LiveDB) -> datetime | None:
    row = db.connection.execute("SELECT MAX(updated_at) AS t FROM match_state").fetchone()
    return datetime.fromisoformat(row["t"]) if row and row["t"] else None


@dataclass(frozen=True)
class SourceHealth:
    name: str
    enabled: bool
    reason: str | None
    trust: str
    latency_label: str
    is_live: bool
    licence: str
    licence_unresolved: bool


@dataclass(frozen=True)
class CapabilityRow:
    capability: str
    providers: list[str]

    @property
    def available(self) -> bool:
        return bool(self.providers)


@dataclass(frozen=True)
class HealthSnapshot:
    sources: list[SourceHealth]
    coverage: list[CapabilityRow]
    aliases: list[Alias]
    duplicate_suggestions: list[DuplicateCandidate]
    attributions: list[str]


def _latency_label(latency_seconds: int | None) -> tuple[str, bool]:
    """(human label, is_live)."""
    if latency_seconds is None:
        return "static", False
    if latency_seconds <= 120:
        return f"live (~{latency_seconds}s)", True
    if latency_seconds < 86_400:
        return f"delayed (~{latency_seconds // 3600}h)", False
    return f"delayed (~{latency_seconds // 86_400}d)", False


def health_snapshot(settings: Settings, db: LiveDB) -> HealthSnapshot:
    sources: list[SourceHealth] = []
    for source in SOURCES.values():
        enabled = settings.is_enabled(source.id)
        reason = None
        if not enabled:
            reason = "no token" if source.id.value == "football_data_org" else "off by default"
        label, is_live = _latency_label(source.latency_seconds)
        sources.append(
            SourceHealth(
                name=source.name,
                enabled=enabled,
                reason=reason,
                trust=source.trust.value,
                latency_label=label,
                is_live=is_live,
                licence=source.licence,
                licence_unresolved=(
                    "NO EXPLICIT" in source.licence or "UNCONFIRMED" in source.licence
                ),
            )
        )

    coverage = [
        CapabilityRow(
            capability=capability.value,
            providers=[
                s.name
                for s in sources_for(capability, include_disabled=True)
                if settings.is_enabled(s.id)
            ],
        )
        for capability in Capability
    ]

    alias_store = AliasStore(db)
    return HealthSnapshot(
        sources=sources,
        coverage=coverage,
        aliases=alias_store.all(),
        duplicate_suggestions=suggest_duplicates(db, "team"),
        attributions=attributions(),
    )


@dataclass(frozen=True)
class AnalyticsSnapshot:
    season: str
    division: str
    available: list[tuple[str, str, int]]
    """(season, division, match count) pairs loaded, for the selector."""
    table: list[TableRow]
    power: list[EloRating]
    title_odds: list[TeamProjection]
    names: dict[str, str]


def analytics_available(analytics_db: Path) -> list[tuple[str, str, int]]:
    """What (season, division) data exists, for the dashboard selector. [] if none."""
    if not Path(analytics_db).exists():
        return []
    with AnalyticsDB(analytics_db) as adb:
        return adb.seasons_loaded()


def analytics_snapshot(
    analytics_db: Path, season: str, division: str, *, sims: int = 3000, seed: int = 1
) -> AnalyticsSnapshot | None:
    """League table, Elo ranking and title odds for one (season, division).

    Opens its own connection so the caller can cache on plain (path, season, division)
    arguments. Returns None if that slice has no results.
    """
    with AnalyticsDB(analytics_db) as adb:
        available = adb.seasons_loaded()
        table = adb.league_table(season, division)
        outcomes = adb.outcomes_for(season, division)

    if not outcomes:
        return None

    names = {o.home_norm: o.home for o in outcomes} | {o.away_norm: o.away for o in outcomes}
    power = power_ranking(outcomes)
    fixtures = [(o.home_norm, o.away_norm) for o in outcomes]
    projections = simulate_season(
        fit_poisson(outcomes), fixtures, teams=list(names), n_sims=sims, seed=seed
    ).projections

    return AnalyticsSnapshot(
        season=season,
        division=division,
        available=available,
        table=table,
        power=power,
        title_odds=projections,
        names=names,
    )
