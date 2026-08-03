"""Dashboard data preparation -- pure reads, no Streamlit.

Kept free of any UI import so it is unit-testable and so the Streamlit layer stays a
thin render over these models. Everything here is a read over the live database and the
source registry; the dashboard never touches the network (that is `soccer ingest`'s
job). A view that cannot be filled by the data we actually have is simply not produced
-- there is no Player Hub, because nothing ingests player-season data yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from soccer.config import Settings
from soccer.domain.aliases import Alias, AliasStore, DuplicateCandidate, suggest_duplicates
from soccer.domain.match_state import MatchStateStore, MatchView
from soccer.domain.names import normalize_name
from soccer.models.elo import EloRating, power_ranking
from soccer.models.poisson import fit_poisson_shots
from soccer.models.simulation import TeamProjection, simulate_season
from soccer.sources.registry import SOURCES, Capability, attributions, sources_for
from soccer.storage.analytics_db import AnalyticsDB, TableRow, XgRow
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
        fit_poisson_shots(outcomes, alpha=FORECAST_ALPHA, shrinkage=FORECAST_SHRINKAGE),
        fixtures,
        teams=list(names),
        n_sims=sims,
        seed=seed,
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


@dataclass(frozen=True)
class ShotMapData:
    match_id: int
    label: str
    shots: list[dict]
    """Each: team, player, minute, x, y, xg, outcome, is_goal (StatsBomb 120x80 frame)."""
    team_xg: list[XgRow]
    timeline: list[dict]
    """Cumulative-xG points per team: team, minute, cum_xg, is_goal, player (a step chart)."""


def shot_matches(analytics_db: Path) -> list[tuple[int, str, str, str]]:
    """(match_id, label, competition, season) for matches with StatsBomb shots. [] if none."""
    if not Path(analytics_db).exists():
        return []
    with AnalyticsDB(analytics_db) as adb:
        return [
            (mid, label, comp, season)
            for mid, label, comp, season, _date in adb.shot_matches_indexed()
        ]


def _xg_timeline(shots: list[dict]) -> list[dict]:
    """Build per-team cumulative-xG points over match minutes, from a match's shots.

    Each team starts at (0, 0) and steps up by a shot's xG at its minute -- the "xG race"
    that shows how a scoreline compares to the balance of chances.
    """
    teams = sorted({s["team"] for s in shots})
    cumulative = dict.fromkeys(teams, 0.0)
    points = [
        {"team": t, "minute": 0, "cum_xg": 0.0, "is_goal": False, "player": ""} for t in teams
    ]
    for shot in sorted(shots, key=lambda s: s["minute"]):
        cumulative[shot["team"]] += shot["xg"]
        points.append(
            {
                "team": shot["team"],
                "minute": shot["minute"],
                "cum_xg": round(cumulative[shot["team"]], 3),
                "is_goal": shot["is_goal"],
                "player": shot["player"],
            }
        )
    return points


def shot_map(analytics_db: Path, match_id: int) -> ShotMapData | None:
    """Shots, per-team xG and the cumulative-xG timeline for one match. None if absent."""
    with AnalyticsDB(analytics_db) as adb:
        shots = adb.shots_for(match_id)
        if not shots:
            return None
        team_xg = adb.team_xg(match_id)
        label = adb.match_label(match_id) or next(
            (lbl for mid, lbl, _c, _s, _d in adb.shot_matches_indexed() if mid == match_id),
            str(match_id),
        )
    return ShotMapData(
        match_id=match_id,
        label=label,
        shots=shots,
        team_xg=team_xg,
        timeline=_xg_timeline(shots),
    )


def forecast_teams(analytics_db: Path, season: str, division: str) -> list[str]:
    """Display names available to forecast in a season/division, for the selectors."""
    if not Path(analytics_db).exists():
        return []
    with AnalyticsDB(analytics_db) as adb:
        outcomes = adb.outcomes_for(season, division)
    names = {o.home for o in outcomes} | {o.away for o in outcomes}
    return sorted(names)


def team_form(analytics_db: Path, season: str, division: str, *, last_n: int = 5):
    """Per-team form (list[TeamForm]) for a season, hottest first. [] if no results."""
    if not Path(analytics_db).exists():
        return []
    with AnalyticsDB(analytics_db) as adb:
        return adb.team_form(season, division, last_n=last_n)


@dataclass(frozen=True)
class SeasonRecords:
    streaks: list  # TeamStreak, longest active unbeaten first
    biggest_wins: list  # MatchRecord
    highest_scoring: list  # MatchRecord


def season_records(
    analytics_db: Path, season: str, division: str, *, limit: int = 6
) -> SeasonRecords | None:
    """Active streaks and notable results for a season, or None if it has no results."""
    if not Path(analytics_db).exists():
        return None
    with AnalyticsDB(analytics_db) as adb:
        streaks = adb.team_streaks(season, division)
        if not streaks:
            return None
        return SeasonRecords(
            streaks=streaks,
            biggest_wins=adb.biggest_wins(season, division, limit=limit),
            highest_scoring=adb.highest_scoring(season, division, limit=limit),
        )


@dataclass(frozen=True)
class SeasonBriefing:
    season: str
    division: str
    n_sims: int
    top_n: int
    relegation: int
    projections: list  # TeamProjection, team = normalized name
    names: dict  # normalized -> display name


def season_briefing(
    analytics_db: Path,
    season: str,
    division: str,
    *,
    n_sims: int = 10_000,
    top_n: int = 4,
    relegation: int = 3,
    seed: int = 1,
) -> SeasonBriefing | None:
    """Monte Carlo a full season among `season`'s teams, from the recency-weighted model.

    Simulates every team playing every other home and away (a clean round-robin, not the
    real schedule) using the same multi-season Dixon-Coles fit the match forecasts use --
    a pre-season projection of title / top-N / relegation odds and expected points. None
    if the season has no results.
    """
    from soccer.models.dixon_coles import fit_dixon_coles
    from soccer.models.simulation import simulate_season

    with AnalyticsDB(analytics_db) as adb:
        anchor = adb.outcomes_for(season, division)
        window = adb.recent_outcomes_through(division, season, n_seasons=FORECAST_SEASONS)
    if not anchor:
        return None

    names = {o.home_norm: o.home for o in anchor} | {o.away_norm: o.away for o in anchor}
    teams = sorted(names)
    model = fit_dixon_coles(window, time_decay=_decay(FORECAST_HALF_LIFE_DAYS))
    fixtures = [(home, away) for home in teams for away in teams if home != away]
    result = simulate_season(
        model, fixtures, teams=teams, n_sims=n_sims, top_n=top_n, relegation=relegation, seed=seed
    )
    return SeasonBriefing(
        season=season,
        division=division,
        n_sims=n_sims,
        top_n=top_n,
        relegation=relegation,
        projections=result.projections,
        names=names,
    )


# Recency-aware forecasting: fit on the last few seasons and re-fit as new results land,
# so the model tracks current form instead of one frozen old season. Time-decay was the
# obvious next knob, but a walk-forward backtest measured it DOWN skill monotonically
# (+2.7% none, +2.4% at 140d, +1.6% at 90d on E0), so it is deliberately off -- the win is
# the multi-season window plus self-updating, not down-weighting. Kept as a tunable, not a
# default. (See docs / `soccer backtest --half-life`.)
FORECAST_SEASONS = 3
FORECAST_HALF_LIFE_DAYS = 0  # 0 = no time-decay (measured best); >0 halves weight every N days
# Shots-on-target blend for the standalone forecast model: pseudo-goals = a*goals + (1-a)*SoT-xG.
# Backtest across the top leagues put the model's log-loss gap to the closing line at roughly
# half the goals-only model's (best around 0.25); matches without shot data fall back to goals.
FORECAST_ALPHA = 0.25
# Pseudo-match shrinkage toward league average, so a side with only a few games (a newly
# promoted team early in the season) is regularised rather than taking an extreme value from
# one result. Measured: k=3 slightly improves overall log loss AND rescues promoted teams'
# early games (E0 that slice 1.33 -> 0.90, near the market's 0.86).
FORECAST_SHRINKAGE = 3.0


def _decay(half_life_days: float) -> float:
    """xi (per day) for fit_dixon_coles, from a half-life: weight halves every N days."""
    import math

    return math.log(2) / half_life_days if half_life_days > 0 else 0.0


def forecast_slate(
    analytics_db: Path,
    season: str,
    division: str,
    home: str,
    away: str,
    *,
    mle: bool = True,
    weighted: bool = True,
):
    """Full market slate for a matchup, or None if a team is unknown.

    Default fit is the shots-on-target-blended Poisson (steadier than goals-only, ~half the
    log-loss gap to the market). `mle` switches to Dixon-Coles instead. `weighted` fits over
    the last few seasons up to `season` so recent form dominates -- the self-updating fit.
    """
    from soccer.domain.names import normalize_name
    from soccer.models.dixon_coles import fit_dixon_coles
    from soccer.models.markets import compute_markets
    from soccer.models.poisson import fit_poisson_shots

    with AnalyticsDB(analytics_db) as adb:
        if weighted:
            outcomes = adb.recent_outcomes_through(division, season, n_seasons=FORECAST_SEASONS)
        else:
            outcomes = adb.outcomes_for(season, division)
    if not outcomes:
        return None
    names = {o.home_norm: o.home for o in outcomes} | {o.away_norm: o.away for o in outcomes}
    if mle:
        decay = _decay(FORECAST_HALF_LIFE_DAYS) if weighted else 0.0
        model = fit_dixon_coles(outcomes, time_decay=decay)
    else:
        model = fit_poisson_shots(outcomes, alpha=FORECAST_ALPHA, shrinkage=FORECAST_SHRINKAGE)
    hn, an = normalize_name(home), normalize_name(away)
    if hn not in model.strengths or an not in model.strengths:
        return None
    lam, mu = model.expected_goals(hn, an)
    return compute_markets(names.get(hn, home), names.get(an, away), lam, mu, model.rho)


def market_edge(analytics_db: Path, season: str, division: str, *, model: str = "poisson"):
    """Closing-line-value backtest for a slice, or None if no odds are loaded.

    Uses the fast ratio-method (poisson) model by default so it can run interactively;
    the CLI `soccer value` offers the slower Dixon-Coles fit. Returns a ValueReport.
    """
    with AnalyticsDB(analytics_db) as adb:
        covered, _ = adb.odds_coverage(season, division)
        if covered == 0:
            return None
        rows = adb.outcomes_with_odds(season, division)
    from soccer.models.value import value_backtest

    try:
        return value_backtest(rows, model=model, min_history=60)
    except ValueError:
        return None


def forecast_report(analytics_db: Path, division: str, *, n_seasons: int = 6, model: str = "shots"):
    """Model-vs-market-vs-blend scorecard for a division's recent odds-bearing seasons.

    The honest measurement instrument: walk-forward scores the model, the vig-free closing
    line, and their log-opinion blend on log loss / RPS / Brier, plus a blend-weight curve
    (its optimum near 0 means the market dominates), calibration, and the biggest
    model-vs-market disagreements. Returns a ForecastReport, or None if no odds are loaded.
    """
    from soccer.models.evaluation import evaluate_forecasts
    from soccer.sources.football_data_co_uk import season_sort_key

    with AnalyticsDB(analytics_db) as adb:
        seasons = sorted(
            {s for s, d, _n in adb.seasons_loaded() if d == division},
            key=season_sort_key,
            reverse=True,
        )[:n_seasons]
        rows = [r for s in seasons for r in adb.outcomes_with_odds(s, division)]
    if not rows:
        return None
    return evaluate_forecasts(
        rows, model=model, alpha=FORECAST_ALPHA, shrinkage=FORECAST_SHRINKAGE, min_history=60
    )


def player_board(analytics_db: Path, *, top: int = 25, min_shots: int = 3, order: str = "xg"):
    """Player leaderboard (list[PlayerRow]) from ingested shots. [] if none."""
    if not Path(analytics_db).exists():
        return []
    with AnalyticsDB(analytics_db) as adb:
        if adb.player_count() == 0:
            return []
        return adb.player_leaderboard(limit=top, min_shots=min_shots, order=order)


def player_profiles(
    analytics_db: Path,
    *,
    top: int = 30,
    min_minutes: int = 180,
    order: str = "contributions",
    competition: str | None = None,
    season: str | None = None,
):
    """Full player profiles (list[PlayerProfile]) from ingested events. [] if none loaded."""
    if not Path(analytics_db).exists():
        return []
    with AnalyticsDB(analytics_db) as adb:
        if adb.player_stats_count() == 0:
            return []
        return adb.player_profiles(
            limit=top,
            min_minutes=min_minutes,
            order=order,
            competition=competition,
            season=season,
        )


def player_profile(
    analytics_db: Path, player: str, *, competition: str | None = None, season: str | None = None
):
    """One player's full profile (PlayerProfile) or None."""
    if not Path(analytics_db).exists():
        return None
    with AnalyticsDB(analytics_db) as adb:
        return adb.player_profile(player, competition=competition, season=season)


def player_competitions(analytics_db: Path) -> list[tuple[str, int]]:
    """(competition, match count) for competitions with player stats. [] if none."""
    if not Path(analytics_db).exists():
        return []
    with AnalyticsDB(analytics_db) as adb:
        return adb.competitions_loaded()


def player_seasons(analytics_db: Path, competition: str) -> list[tuple[str, int]]:
    """(season, match count) within one competition, newest first. [] if none."""
    if not Path(analytics_db).exists():
        return []
    with AnalyticsDB(analytics_db) as adb:
        return adb.competition_seasons(competition)


def has_player_events(analytics_db: Path) -> bool:
    """Whether any full-event player stats are loaded -- gates the rich Players view."""
    if not Path(analytics_db).exists():
        return False
    with AnalyticsDB(analytics_db) as adb:
        return adb.player_stats_count() > 0


@dataclass(frozen=True)
class MetricPercentile:
    category: str
    label: str
    value: float
    """Per-90 value (or a raw ratio like pass %), as displayed."""
    percentile: float  # 0..100 within the qualifying pool


# The scouting fingerprint: metrics grouped by phase of play, each a per-90 rate unless
# flagged otherwise. (category, label, attribute, is_per90).
_PERCENTILE_METRICS = [
    ("Attacking", "Non-pen xG", "npxg", True),
    ("Attacking", "Goals", "goals", True),
    ("Attacking", "Shots", "shots", True),
    ("Attacking", "xA", "xa", True),
    ("Attacking", "Key passes", "key_passes", True),
    ("Possession", "Passes", "passes", True),
    ("Possession", "Pass %", "pass_pct", False),
    ("Possession", "Prog. passes", "progressive_passes", True),
    ("Possession", "Prog. carries", "progressive_carries", True),
    ("Possession", "Dribbles", "dribbles_completed", True),
    ("Defending", "Tackles", "tackles", True),
    ("Defending", "Interceptions", "interceptions", True),
    ("Defending", "Blocks", "blocks", True),
    ("Defending", "Clearances", "clearances", True),
    ("Defending", "Recoveries", "ball_recoveries", True),
    ("Defending", "Pressures", "pressures", True),
]


def player_percentiles(
    analytics_db: Path,
    player: str,
    *,
    min_minutes: int = 200,
    competition: str | None = None,
    season: str | None = None,
) -> list[MetricPercentile]:
    """A player's per-90 rates and their percentile rank within the qualifying pool.

    The FBref-style scouting fingerprint: for each metric, what fraction of players
    (with at least `min_minutes`) this player is at or above. [] if the player is not in
    the pool. `competition`/`season` keep the comparison within one league season, where
    percentiles are far more meaningful than across a mix of tournaments and eras.
    """
    profiles = player_profiles(
        analytics_db,
        top=100_000,
        min_minutes=min_minutes,
        order="minutes",
        competition=competition,
        season=season,
    )
    target = next((p for p in profiles if p.player == player), None)
    if target is None or not profiles:
        return []

    def value_of(profile, attr: str, is_per90: bool) -> float:
        raw = getattr(profile, attr)
        return profile.per90(raw) if is_per90 else raw

    out: list[MetricPercentile] = []
    for category, label, attr, is_per90 in _PERCENTILE_METRICS:
        target_value = value_of(target, attr, is_per90)
        pool = [value_of(p, attr, is_per90) for p in profiles]
        pct = 100.0 * sum(1 for v in pool if v <= target_value) / len(pool)
        out.append(MetricPercentile(category, label, round(target_value, 2), round(pct)))
    return out


# football-data.org competition names -> football-data.co.uk division codes, for
# forecasting upcoming fixtures with the model fitted on that league's history.
COMPETITION_TO_DIVISION = {
    "Premier League": "E0",
    "Championship": "E1",
    "Primera Division": "SP1",
    "La Liga": "SP1",
    "Serie A": "I1",
    "Bundesliga": "D1",
    "Ligue 1": "F1",
    "Eredivisie": "N1",
    "Primeira Liga": "P1",
    "Campeonato Brasileiro Série A": "BRA",
    "Campeonato Brasileiro Serie A": "BRA",
}

# football-data.org (verbose) fixture names -> football-data.co.uk (terse) model names,
# for the handful of clubs whose short names are not a token-subset of the verbose ones
# ("Athletic Club" vs "Ath Bilbao"). Keyed and valued by normalized name; only entries
# that hit a real team in the loaded model actually apply, so a stale one is inert.
_FDCOUK_ALIASES_RAW = {
    # Spain (SP1)
    "Athletic Club": "Ath Bilbao",
    "Club Atlético de Madrid": "Ath Madrid",
    "Atlético de Madrid": "Ath Madrid",
    "RCD Espanyol de Barcelona": "Espanol",
    "RCD Espanyol": "Espanol",
    # Portugal (P1)
    "Sporting Clube de Portugal": "Sp Lisbon",
    "Sporting CP": "Sp Lisbon",
    "Sporting Clube de Braga": "Sp Braga",
    "SC Braga": "Sp Braga",
    "Vitória SC": "Guimaraes",
    # Netherlands (N1)
    "Fortuna Sittard": "For Sittard",
    "NEC": "Nijmegen",
    "NEC Nijmegen": "Nijmegen",
    # England (E1, Championship)
    "Queens Park Rangers FC": "QPR",
    "West Bromwich Albion FC": "West Brom",
    # Brazil (BRA) -- verbose club prefixes the short file drops
    "CA Mineiro": "Atletico-MG",
    "Clube Atlético Mineiro": "Atletico-MG",
    "CA Paranaense": "Athletico-PR",
    "Botafogo FR": "Botafogo RJ",
    "CR Flamengo": "Flamengo RJ",
}
FDCOUK_ALIASES: dict[str, str] = {normalize_name(k): v for k, v in _FDCOUK_ALIASES_RAW.items()}


@dataclass(frozen=True)
class FixtureForecast:
    kickoff_utc: datetime
    competition: str
    home: str
    away: str
    slate: object | None  # MarketSlate, or None when no model covers the matchup


def fixture_forecasts(
    live_db: Path, analytics_db: Path, *, limit: int = 60
) -> list[FixtureForecast]:
    """Upcoming fixtures (from the live DB), each forecast via its league's model.

    A fixture is forecast only when its competition maps to loaded history and both team
    names resolve into that model; otherwise it is listed without a forecast, honestly.
    Each division's model is fit on its most-recent loaded season -- a preseason
    projection (European "2526", Brazil's calendar-year "2026", ... resolved per league).
    """
    from soccer.domain.names import normalize_name
    from soccer.models.markets import compute_markets

    def resolve(name: str, model) -> str | None:
        """Match a fixture team name to a model team: exact, curated alias, then fuzzy.

        Bridges verbose football-data.org names ("GD Estoril Praia") to the terser
        football-data.co.uk model names ("Estoril"): a token-subset match handles most,
        and a small curated alias map covers clubs whose short name shares no token with
        the verbose one ("Athletic Club" -> "Ath Bilbao").
        """
        n = normalize_name(name)
        if n in model.strengths:
            return n
        aliased = FDCOUK_ALIASES.get(n)
        if aliased and normalize_name(aliased) in model.strengths:
            return normalize_name(aliased)
        tokens = set(n.split())
        subset = [t for t in model.strengths if set(t.split()) < tokens or tokens < set(t.split())]
        return subset[0] if len(subset) == 1 else None  # unique match only, else skip

    if not Path(live_db).exists():
        return []
    with LiveDB(live_db) as db:
        ups = MatchStateStore(db).upcoming(limit=limit)

    models: dict[str, object] = {}

    def model_for(division: str):
        if division not in models:
            model = None
            if Path(analytics_db).exists():
                with AnalyticsDB(analytics_db) as adb:
                    season = adb.latest_season(division)
                    outcomes = (
                        adb.recent_outcomes_through(division, season, n_seasons=FORECAST_SEASONS)
                        if season
                        else []
                    )
                if outcomes:
                    # Recency window (last few seasons), fit on the shots-on-target blend --
                    # measured to roughly halve the goals-only model's gap to the market.
                    model = fit_poisson_shots(
                        outcomes, alpha=FORECAST_ALPHA, shrinkage=FORECAST_SHRINKAGE
                    )
            models[division] = model
        return models[division]

    out: list[FixtureForecast] = []
    for v in ups:
        slate = None
        division = COMPETITION_TO_DIVISION.get(v.competition)
        if division:
            model = model_for(division)
            if model:
                hn, an = resolve(v.home, model), resolve(v.away, model)
                if hn and an:
                    lam, mu = model.expected_goals(hn, an)
                    slate = compute_markets(v.home, v.away, lam, mu, model.rho)
        out.append(FixtureForecast(v.kickoff_utc, v.competition, v.home, v.away, slate))
    return out
