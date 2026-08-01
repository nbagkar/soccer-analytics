"""MCP tool implementations -- pure functions, no MCP import.

Each returns a JSON-able dict (never raises for expected failures -- a missing database
or unknown team comes back as ``{"error": ...}``), so it is unit-testable and the server
in ``server.py`` stays a thin set of decorated wrappers. Everything here reuses the same
storage and models the CLI and dashboard use; the MCP surface adds no new logic, only a
conversational entry point.
"""

from __future__ import annotations

from typing import Any

from soccer.config import Settings
from soccer.domain.match_state import MatchStateStore
from soccer.domain.names import normalize_name
from soccer.models.elo import compute_ratings, power_ranking
from soccer.models.poisson import fit_poisson
from soccer.models.simulation import simulate_season
from soccer.sources.registry import SOURCES, Capability, sources_for
from soccer.storage.analytics_db import AnalyticsDB
from soccer.storage.live_db import LiveDB


def _load(settings: Settings, season: str, division: str) -> dict[str, Any]:
    """Results + display names for a slice, or an ``error`` dict."""
    if not settings.analytics_db.exists():
        return {"error": "No historical data. Run `soccer ingest-history` first."}
    with AnalyticsDB(settings.analytics_db) as adb:
        outcomes = adb.outcomes_for(season, division)
    if not outcomes:
        return {"error": f"No results loaded for {season}/{division}."}
    names = {o.home_norm: o.home for o in outcomes} | {o.away_norm: o.away for o in outcomes}
    return {"outcomes": outcomes, "names": names}


def live_matches(settings: Settings, *, in_play_only: bool = True, limit: int = 50) -> dict:
    if not settings.live_db.exists():
        return {"error": "No live data. Run `soccer ingest` first."}
    with LiveDB(settings.live_db) as db:
        views = MatchStateStore(db).list_current(in_play_only=in_play_only, limit=limit)
    return {
        "count": len(views),
        "matches": [
            {
                "home": v.home,
                "away": v.away,
                "score": v.score,
                "status": v.status.value,
                "minute": v.minute,
                "competition": v.competition,
                "kickoff_utc": v.kickoff_utc.isoformat(),
                "source": v.source,
                "stale": v.is_stale,
            }
            for v in views
        ],
    }


def league_table(settings: Settings, season: str, division: str) -> dict:
    if not settings.analytics_db.exists():
        return {"error": "No historical data. Run `soccer ingest-history` first."}
    with AnalyticsDB(settings.analytics_db) as adb:
        rows = adb.league_table(season, division)
    if not rows:
        return {"error": f"No results loaded for {season}/{division}."}
    return {
        "season": season,
        "division": division,
        "table": [
            {
                "position": r.position,
                "team": r.team,
                "played": r.played,
                "won": r.won,
                "drawn": r.drawn,
                "lost": r.lost,
                "goals_for": r.goals_for,
                "goals_against": r.goals_against,
                "goal_difference": r.goal_difference,
                "points": r.points,
            }
            for r in rows
        ],
    }


def power_rankings(settings: Settings, season: str, division: str, top: int = 0) -> dict:
    loaded = _load(settings, season, division)
    if "error" in loaded:
        return loaded
    ranking = power_ranking(loaded["outcomes"])
    if top > 0:
        ranking = ranking[:top]
    names = loaded["names"]
    return {
        "season": season,
        "division": division,
        "rankings": [
            {"position": r.position, "team": names.get(r.team, r.team), "elo": round(r.rating)}
            for r in ranking
        ],
    }


def forecast_match(settings: Settings, season: str, division: str, home: str, away: str) -> dict:
    loaded = _load(settings, season, division)
    if "error" in loaded:
        return loaded
    outcomes, names = loaded["outcomes"], loaded["names"]
    model = fit_poisson(outcomes)

    home_norm, away_norm = normalize_name(home), normalize_name(away)
    for original, norm in ((home, home_norm), (away, away_norm)):
        if norm not in model.strengths:
            return {
                "error": f"Unknown team '{original}' in {season}/{division}.",
                "available_teams": sorted(names.values()),
            }

    fc = model.forecast(home_norm, away_norm)
    ratings = compute_ratings(outcomes)
    return {
        "home": names[home_norm],
        "away": names[away_norm],
        "expected_goals": {
            "home": round(fc.home_expected, 2),
            "away": round(fc.away_expected, 2),
        },
        "probabilities": {
            "home_win": round(fc.prob_home, 3),
            "draw": round(fc.prob_draw, 3),
            "away_win": round(fc.prob_away, 3),
        },
        "likely_scores": [
            {"score": f"{x}-{y}", "probability": round(p, 3)} for x, y, p in fc.top_scores
        ],
        "elo": {"home": round(ratings[home_norm]), "away": round(ratings[away_norm])},
    }


def simulate_league(settings: Settings, season: str, division: str, sims: int = 5000) -> dict:
    loaded = _load(settings, season, division)
    if "error" in loaded:
        return loaded
    outcomes, names = loaded["outcomes"], loaded["names"]
    fixtures = [(o.home_norm, o.away_norm) for o in outcomes]
    result = simulate_season(
        fit_poisson(outcomes), fixtures, teams=list(names), n_sims=sims, seed=1
    )
    return {
        "season": season,
        "division": division,
        "simulations": sims,
        "note": "Full-season replay from the fitted strengths.",
        "projections": [
            {
                "team": names.get(p.team, p.team),
                "title_pct": round(p.title_pct, 3),
                "top4_pct": round(p.top_pct, 3),
                "relegation_pct": round(p.relegation_pct, 3),
                "expected_points": round(p.expected_points, 1),
            }
            for p in result.projections
        ],
    }


def search_teams(settings: Settings, query: str, season: str, division: str) -> dict:
    loaded = _load(settings, season, division)
    if "error" in loaded:
        return loaded
    q = normalize_name(query)
    matches = sorted({name for norm, name in loaded["names"].items() if q and q in norm})
    return {"query": query, "matches": matches}


def data_health(settings: Settings) -> dict:
    sources = []
    for source in SOURCES.values():
        sources.append(
            {
                "name": source.name,
                "enabled": settings.is_enabled(source.id),
                "trust": source.trust.value,
                "live": source.latency_seconds is not None and source.latency_seconds <= 120,
                "licence": source.licence,
            }
        )
    coverage = {
        capability.value: [
            s.name
            for s in sources_for(capability, include_disabled=True)
            if settings.is_enabled(s.id)
        ]
        for capability in Capability
    }
    return {"sources": sources, "capability_coverage": coverage}


def available_data(settings: Settings) -> dict:
    live_count = 0
    if settings.live_db.exists():
        with LiveDB(settings.live_db) as db:
            live_count = db.connection.execute("SELECT COUNT(*) FROM canonical_match").fetchone()[0]
    history: list[dict] = []
    if settings.analytics_db.exists():
        with AnalyticsDB(settings.analytics_db) as adb:
            history = [
                {"season": s, "division": d, "matches": n} for s, d, n in adb.seasons_loaded()
            ]
    return {"live_matches": live_count, "history": history}
