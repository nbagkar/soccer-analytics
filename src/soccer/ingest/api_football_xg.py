"""Backfill team xG per match from API-Football, a little each day, until a season is done.

Why: current-season team xG is the one thing a paid API-Football plan would add that could
improve the model. The free plan can't serve the current season, but it does serve 2022-2024
-- so a past season can be backfilled for free and tested in the Scorecard harness before
anyone pays. The free plan also refuses the multi-fixture `ids` parameter, so each match's
statistics cost one request: about 380 requests for a top-flight season, or ~4 days of a
100/day quota.

The backfill is resumable and idempotent: it fetches the season's fixture list once (cached
for a week), skips matches already stored, re-reads any statistics already in the raw store
without spending quota, and stops cleanly when the day's budget for jobs is gone. A match
whose statistics carry no `expected_goals` is stored with NULL xG rather than skipped, so it
isn't re-requested every day.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, timedelta
from typing import Any

from soccer.ingest.quota import QuotaExhaustedError
from soccer.sources.api_football import ApiFootball, ApiFootballError
from soccer.sources.registry import SourceId
from soccer.storage.analytics_db import AnalyticsDB, TeamMatchXg

PROVIDER = SourceId.API_FOOTBALL.value
_FINISHED = {"FT", "AET", "PEN"}
# A finished season's fixture list doesn't change; re-read it from the raw store for a week.
FIXTURE_LIST_MAX_AGE = timedelta(days=7)
# Statistics for a finished match never change; any stored answer is good forever.
STATS_MAX_AGE = timedelta(days=36_500)


def season_code(api_season: int) -> str:
    """API-Football's start year -> football-data.co.uk's code: 2024 -> "2425"."""
    return f"{api_season % 100:02d}{(api_season + 1) % 100:02d}"


@dataclass(frozen=True)
class BackfillResult:
    division: str
    season: str
    finished: int  # finished matches in the season
    stored: int  # matches with a stored row (xG or a recorded absence)
    with_xg: int
    bridged: int  # rows whose both club names matched the results table
    requests: int  # spent this run (cached reads are free)
    stopped_for_quota: bool

    @property
    def complete(self) -> bool:
        return self.stored >= self.finished

    def summary(self) -> str:
        state = "complete" if self.complete else "in progress"
        why = " (stopped: daily budget for jobs used up)" if self.stopped_for_quota else ""
        return (
            f"{self.division} {self.season}: {self.stored}/{self.finished} matches stored, "
            f"{self.with_xg} with xG, {self.bridged} matched to results; {self.requests} "
            f"requests this run; {state}{why}"
        )


def _xg_by_team(stats: Iterable[Any]) -> dict[int, float | None]:
    out: dict[int, float | None] = {}
    for entry in stats:
        team_id = ((entry or {}).get("team") or {}).get("id")
        if team_id is None:
            continue
        value = next(
            (
                s.get("value")
                for s in entry.get("statistics") or []
                if str(s.get("type", "")).lower() == "expected_goals"
            ),
            None,
        )
        try:
            out[int(team_id)] = float(value) if value not in (None, "") else None
        except (TypeError, ValueError):
            out[int(team_id)] = None
    return out


def backfill_team_xg(
    api: ApiFootball,
    adb: AnalyticsDB,
    *,
    league_id: int,
    api_season: int,
    division: str,
    use_reserve: bool = False,
    max_requests: int | None = None,
) -> BackfillResult:
    """Store team xG for as many unfetched finished matches as today's budget allows."""
    from soccer.dashboard.data import match_team  # the one audited cross-source name bridge

    season = season_code(api_season)
    requests = 0
    stopped = False

    listing = api.get(
        "/fixtures",
        {"league": league_id, "season": api_season},
        max_age=FIXTURE_LIST_MAX_AGE,
        use_reserve=use_reserve,
    )
    requests += 0 if listing.from_cache else 1
    finished = [
        f
        for f in listing.response
        if ((f.get("fixture") or {}).get("status") or {}).get("short") in _FINISHED
    ]
    candidates = {o.home_norm for o in adb.outcomes_for(season, division)} | {
        o.away_norm for o in adb.outcomes_for(season, division)
    }
    done = adb.team_xg_fixture_ids(PROVIDER, league_id, season)

    for f in sorted(finished, key=lambda f: (f.get("fixture") or {}).get("date") or ""):
        fixture = f.get("fixture") or {}
        fid = int(fixture["id"])
        if fid in done:
            continue
        if max_requests is not None and requests >= max_requests:
            break
        try:
            r = api.get(
                "/fixtures/statistics",
                {"fixture": fid},
                max_age=STATS_MAX_AGE,
                use_reserve=use_reserve,
            )
        except QuotaExhaustedError:
            stopped = True
            break
        except ApiFootballError:
            requests += 1
            continue  # reported by the counts below; retried on the next run
        requests += 0 if r.from_cache else 1
        teams = f.get("teams") or {}
        home, away = teams.get("home") or {}, teams.get("away") or {}
        xg = _xg_by_team(r.response)
        kickoff = str(fixture.get("date") or "")[:10]
        adb.upsert_team_xg(
            [
                TeamMatchXg(
                    provider=PROVIDER,
                    fixture_id=fid,
                    league_id=league_id,
                    season=season,
                    division=division,
                    match_date=date.fromisoformat(kickoff),
                    home=str(home.get("name")),
                    away=str(away.get("name")),
                    home_norm=match_team(str(home.get("name")), candidates),
                    away_norm=match_team(str(away.get("name")), candidates),
                    home_xg=xg.get(int(home["id"])) if home.get("id") is not None else None,
                    away_xg=xg.get(int(away["id"])) if away.get("id") is not None else None,
                    fetched_at=r.fetched_at.astimezone(UTC).replace(tzinfo=None),
                )
            ]
        )

    rows = adb.team_xg_for(division, season)
    rows = [r for r in rows if r.league_id == league_id]
    return BackfillResult(
        division=division,
        season=season,
        finished=len(finished),
        stored=len(rows),
        with_xg=sum(1 for r in rows if r.home_xg is not None and r.away_xg is not None),
        bridged=sum(1 for r in rows if r.home_norm and r.away_norm),
        requests=requests,
        stopped_for_quota=stopped,
    )


# What `soccer serve` backfills while API-Football is configured: one league-season at a time,
# in order, each until complete. Premier League first (the most-viewed league).
BACKFILL_PLAN: list[tuple[int, int, str]] = [(39, 2024, "E0")]


def run_backfill_plan(api: ApiFootball, adb: AnalyticsDB) -> str:
    """Advance the first incomplete league-season in BACKFILL_PLAN with today's job budget."""
    for league_id, api_season, division in BACKFILL_PLAN:
        try:
            result = backfill_team_xg(
                api, adb, league_id=league_id, api_season=api_season, division=division
            )
        except QuotaExhaustedError as exc:
            return f"team xG backfill waiting for tomorrow's budget ({exc})"
        if not result.complete:
            return result.summary()
    return "team xG backfill: every planned season is complete"
