"""In-dashboard data actions -- so a non-technical user never needs the terminal.

These are the only place the dashboard *writes*: thin wrappers over the same source
adapters the CLI uses, driven by buttons on the Home page. Each returns a short,
human-readable result string (or raises, which the page shows as an error). Kept separate
from the read-only `data.py` so the read path stays obviously side-effect-free.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from soccer.config import Settings
from soccer.ingest.pipeline import IngestPipeline
from soccer.sources.football_data_co_uk import (
    NEW_LEAGUE_CODES,
    FootballDataCoUk,
    division_name,
)
from soccer.storage.analytics_db import AnalyticsDB
from soccer.storage.live_db import LiveDB
from soccer.storage.raw import RawStore

# Friendly league name -> football-data.co.uk division code, for the "add a league" picker.
LEAGUE_CHOICES = {
    "Premier League (England)": "E0",
    "Championship (England)": "E1",
    "League One (England)": "E2",
    "League Two (England)": "E3",
    "La Liga (Spain)": "SP1",
    "La Liga 2 (Spain)": "SP2",
    "Bundesliga (Germany)": "D1",
    "2. Bundesliga (Germany)": "D2",
    "Serie A (Italy)": "I1",
    "Serie B (Italy)": "I2",
    "Ligue 1 (France)": "F1",
    "Ligue 2 (France)": "F2",
    "MLS (USA)": "USA",
}

# The set a fresh install downloads itself on first launch, so it "just works" without any
# clicks -- fast, token-free league results that power tables, forecasts, trends, records
# and the season oracle. Kept to the leagues actually wanted.
STARTER_LEAGUES = ["E0", "E1", "E2", "E3", "SP1", "SP2", "D1", "D2", "I1", "I2", "F1", "F2", "USA"]

# Friendly player-data pack -> (StatsBomb competition_id, season_id, approx match count).
EVENT_PACKS = {
    "2022 World Cup (64 matches)": (43, 106, 64),
    "Premier League 2015/16 (380 matches)": (2, 27, 380),
    "La Liga — Barcelona 2020/21 (35 matches)": (11, 90, 35),
    "Champions League finals (recent)": (16, 27, 1),
}


def data_status(settings: Settings) -> dict:
    """Counts for the Home page: leagues, history matches, player competitions, fixtures."""
    status = {"leagues": 0, "history_matches": 0, "player_competitions": 0, "upcoming": 0}
    if settings.analytics_db.exists():
        with AnalyticsDB(settings.analytics_db) as adb:
            loaded = adb.seasons_loaded()
            status["leagues"] = len({d for _s, d, _n in loaded})
            status["history_matches"] = sum(n for _s, _d, n in loaded)
            status["player_competitions"] = len(adb.competitions_loaded())
    if settings.live_db.exists():
        from soccer.domain.match_state import MatchStateStore

        with LiveDB(settings.live_db) as db:
            status["upcoming"] = len(MatchStateStore(db).upcoming(limit=1000))
    return status


def refresh_scores(settings: Settings) -> str:
    """Pull the latest live/recent scores from TheSportsDB (works with the free key)."""
    settings.ensure_dirs()
    raw = RawStore(settings.raw_dir)

    async def run() -> str:
        from soccer.sources.thesportsdb import TheSportsDB

        with LiveDB(settings.live_db) as db:
            async with TheSportsDB(
                raw,
                api_key=settings.thesportsdb_key,
                rate_limit_per_minute=settings.thesportsdb_rpm,
            ) as tsdb:
                return str(await IngestPipeline(db).ingest_thesportsdb(tsdb))

    return asyncio.run(run())


def update_fixtures(settings: Settings) -> str:
    """Pull upcoming fixtures from football-data.org (needs a free token)."""
    if not settings.football_data_org_token:
        return (
            "To load fixtures, add a free football-data.org token to your `.env` "
            "(SOCCER_FOOTBALL_DATA_ORG_TOKEN). Live scores work without one."
        )
    settings.ensure_dirs()
    raw = RawStore(settings.raw_dir)

    async def run() -> str:
        from soccer.sources.football_data_org import FootballDataOrg

        today = datetime.now(UTC).date()
        with LiveDB(settings.live_db) as db:
            async with FootballDataOrg(
                settings.football_data_org_token,
                raw,
                rate_limit_per_minute=settings.football_data_org_rpm,
            ) as fd:
                summary = await IngestPipeline(db).ingest_football_data(
                    fd, today, today + timedelta(days=10)
                )
        return str(summary)

    return asyncio.run(run())


def _recent_seasons(n: int = 3) -> list[str]:
    today = datetime.now(UTC).date()
    start = today.year if today.month >= 7 else today.year - 1
    return [f"{(start - i) % 100:02d}{(start - i + 1) % 100:02d}" for i in range(n)]


def add_league_history(settings: Settings, division: str, *, seasons: int = 3) -> str:
    """Download recent seasons of a league's results (football-data.co.uk) into the store."""
    settings.ensure_dirs()
    raw = RawStore(settings.raw_dir)
    total = 0
    with FootballDataCoUk(raw) as source, AnalyticsDB(settings.analytics_db) as adb:
        if division in NEW_LEAGUE_CODES:
            results = source.fetch_new_league(division, recent_seasons=seasons)
            if results:
                adb.load_results(results)
                total += len(results)
        else:
            for season in _recent_seasons(seasons):
                results = source.fetch_division(season, division)
                if results:
                    adb.load_results(results)
                    total += len(results)
    if not total:
        return f"No data available yet for {division_name(division)}."
    return f"Added {total} matches for {division_name(division)}."


# football-data.co.uk carries the major European leagues back to 1993/94 (its earliest); 34
# reaches that from the current season, deeper history meaning richer tables, forecasts, form
# and records. (Shot/event data is separately capped by StatsBomb's catalog.)
FULL_HISTORY_SEASONS = 34


def load_full_history(
    settings: Settings, *, seasons: int = FULL_HISTORY_SEASONS, on_progress: Callable | None = None
) -> str:
    """Backfill up to `seasons` of results for every starter league, idempotently.

    Each (season, division) unit is replaced on load, so re-running never duplicates. A
    season a league did not yet have simply 404s and is skipped -- so this reaches as far
    back as each league's data actually goes.
    """
    settings.ensure_dirs()
    raw = RawStore(settings.raw_dir)
    total = 0
    with FootballDataCoUk(raw) as source, AnalyticsDB(settings.analytics_db) as adb:
        for i, division in enumerate(STARTER_LEAGUES, 1):
            if division in NEW_LEAGUE_CODES:
                results = source.fetch_new_league(division, recent_seasons=seasons)
                if results:
                    adb.load_results(results)
                    total += len(results)
            else:
                for season in _recent_seasons(seasons):
                    results = source.fetch_division(season, division)
                    if results:
                        adb.load_results(results)
                        total += len(results)
            if on_progress:
                on_progress(i, len(STARTER_LEAGUES))
    return f"Loaded {total} matches of history across {len(STARTER_LEAGUES)} leagues."


def remove_league(settings: Settings, division: str) -> str:
    """Delete a league's results so it no longer appears in the dashboard."""
    if not settings.analytics_db.exists():
        return "Nothing to remove."
    with AnalyticsDB(settings.analytics_db) as adb:
        removed = adb.delete_division(division)
    return f"Removed {division_name(division)} ({removed} matches)."


def starter_setup(settings: Settings, *, on_progress: Callable | None = None) -> str:
    """Download the first-launch starter set so a fresh install populates itself.

    Token-free league results for a handful of major leagues (plus fixtures if a
    football-data.org token is configured). Fetched live from the sources -- their terms
    forbid redistribution, so nothing can ship bundled; this is the permitted use.
    """
    settings.ensure_dirs()
    raw = RawStore(settings.raw_dir)
    seasons = _recent_seasons(3)
    total = 0
    with FootballDataCoUk(raw) as source, AnalyticsDB(settings.analytics_db) as adb:
        for i, division in enumerate(STARTER_LEAGUES, 1):
            for season in seasons:
                results = source.fetch_division(season, division)
                if results:
                    adb.load_results(results)
                    total += len(results)
            if on_progress:
                on_progress(i, len(STARTER_LEAGUES))
    if settings.football_data_org_token:
        import contextlib

        with contextlib.suppress(Exception):  # fixtures are a bonus; never block setup
            update_fixtures(settings)
    return f"Loaded {total} matches across {len(STARTER_LEAGUES)} leagues — you're ready to go!"


def load_event_pack(
    settings: Settings, competition_id: int, season_id: int, *, on_progress: Callable | None = None
) -> str:
    """Ingest a StatsBomb competition's events (shots + full player stats) with progress."""
    settings.ensure_dirs()
    raw = RawStore(settings.raw_dir)
    from soccer.sources.registry import SourceId
    from soccer.sources.statsbomb import (
        StatsBomb,
        parse_match_meta,
        parse_player_stats,
        parse_shots,
    )
    from soccer.storage.analytics_db import MatchMeta

    with StatsBomb(raw) as sb, AnalyticsDB(settings.analytics_db) as adb:
        listing = sb.matches(competition_id, season_id)
        match_ids = [m["match_id"] for m in listing]
        metas = [MatchMeta(**parse_match_meta(m)) for m in listing if m.get("match_id")]
        if metas:
            adb.load_match_meta(metas)
        shots_total = players_total = 0
        for i, match_id in enumerate(match_ids, 1):
            snapshot = raw.latest(SourceId.STATSBOMB, f"events_{match_id}")
            events = snapshot.payload if snapshot else sb.fetch_events(match_id)
            if events:
                shots = parse_shots(events, match_id)
                stats = parse_player_stats(events, match_id)
                if shots:
                    adb.load_shots(shots)
                    shots_total += len(shots)
                if stats:
                    adb.load_player_stats(stats)
                    players_total += len(stats)
            if on_progress:
                on_progress(i, len(match_ids))
    return (
        f"Loaded {shots_total} shots and {players_total} player rows from {len(match_ids)} matches."
    )


def load_all_events(settings: Settings, *, on_progress: Callable | None = None) -> str:
    """Load every curated player-data pack in one go, so nothing is picked by hand.

    Non-commercial/personal use of StatsBomb open data is within its terms, so a single
    press pulls the lot -- World Cup, a full Premier League and La Liga season, and the
    Champions League finals.
    """
    packs = list(EVENT_PACKS.values())
    for i, (competition_id, season_id, _approx) in enumerate(packs, 1):
        load_event_pack(settings, competition_id, season_id)
        if on_progress:
            on_progress(i, len(packs))
    return f"Loaded {len(packs)} player datasets — the marquee names are ready."
