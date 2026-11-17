"""What does this API-Football key actually get? A coverage matrix from real responses.

For each priority competition and one season, the probe asks every endpoint the upgrade
plan needs -- about ten requests per competition, inside the free plan's 100/day -- and
classifies the answer:

- ``ok``      data came back (with a count, and notes such as "xG present")
- ``empty``   the request succeeded but returned nothing (e.g. no odds for a past round)
- ``denied``  the plan refuses it (an `errors.plan` message) -- the decisive finding
- ``error``   any other failure, with the server's message
- ``skipped`` not asked (no finished fixture to inspect, season denied, or out of budget)

Nothing here is inferred from documentation: each cell is one real response, and every
response is snapshotted in the raw store.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

from soccer.ingest.quota import QuotaExhaustedError
from soccer.sources.api_football import PRIORITY_LEAGUES, ApiFootball, ApiFootballError
from soccer.sources.errors import SourceUnavailableError

OK, EMPTY, DENIED, ERROR, SKIPPED = "ok", "empty", "denied", "error", "skipped"

ENDPOINTS = (
    "season",
    "fixtures",
    "statistics",
    "events",
    "lineups",
    "player match stats",
    "player season stats",
    "injuries",
    "odds",
    "standings",
)
REQUESTS_PER_LEAGUE = len(ENDPOINTS)
_FINISHED = {"FT", "AET", "PEN"}


@dataclass(frozen=True)
class Cell:
    status: str
    results: int = 0
    note: str = ""


@dataclass
class CoverageReport:
    season: int
    ran_at: datetime
    account: dict[str, Any] = field(default_factory=dict)
    declared: dict[int, dict[str, Any]] = field(default_factory=dict)
    """Per league: the coverage flags API-Football itself declares for the season."""
    cells: dict[int, dict[str, Cell]] = field(default_factory=dict)
    requests_used: int = 0


def _classify(exc: Exception) -> Cell:
    if isinstance(exc, ApiFootballError):
        message = (
            "; ".join(f"{k}: {v}" for k, v in (exc.errors or {}).items())
            if isinstance(exc.errors, dict)
            else str(exc.errors)
        )
        return Cell(DENIED if exc.is_plan_restriction else ERROR, 0, message[:200])
    return Cell(ERROR, 0, f"{type(exc).__name__}: {exc}"[:200])


def _stat_types(items: list[Any]) -> set[str]:
    types: set[str] = set()
    for team in items:
        for stat in (team or {}).get("statistics") or []:
            if isinstance(stat, dict) and stat.get("type"):
                types.add(str(stat["type"]).lower())
    return types


def _keys_deep(obj: Any, depth: int = 4) -> set[str]:
    if depth == 0:
        return set()
    if isinstance(obj, dict):
        out = {str(k).lower() for k in obj}
        for v in obj.values():
            out |= _keys_deep(v, depth - 1)
        return out
    if isinstance(obj, list):
        out = set()
        for v in obj[:3]:
            out |= _keys_deep(v, depth - 1)
        return out
    return set()


def probe_coverage(
    api: ApiFootball,
    *,
    season: int,
    leagues: list[int] | None = None,
    today: date | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> CoverageReport:
    """Ask each endpoint once per league; never raises for a single failed cell."""
    today = today or datetime.now(UTC).date()
    report = CoverageReport(season=season, ran_at=datetime.now(UTC))
    say = on_progress or (lambda _m: None)

    def ask(endpoint: str, params: dict[str, Any]) -> tuple[Cell, list[Any]]:
        try:
            r = api.get(endpoint, params, use_reserve=True)
        except QuotaExhaustedError:
            raise
        except (ApiFootballError, SourceUnavailableError, OSError) as exc:
            report.requests_used += 1
            return _classify(exc), []
        except Exception as exc:  # httpx transport/status errors
            report.requests_used += 1
            return _classify(exc), []
        report.requests_used += 1
        return Cell(OK if r.results else EMPTY, r.results), r.response

    try:
        status_cell, status = ask("/status", {})
        if status and isinstance(status[0], dict):
            acct = status[0]
            report.account = {
                "plan": (acct.get("subscription") or {}).get("plan"),
                "active": (acct.get("subscription") or {}).get("active"),
                "requests_today": (acct.get("requests") or {}).get("current"),
                "daily_limit": (acct.get("requests") or {}).get("limit_day"),
            }
        elif status_cell.status != OK:
            report.account = {"error": status_cell.note}

        for league in leagues or list(PRIORITY_LEAGUES):
            name = PRIORITY_LEAGUES.get(league, (str(league), str(league)))[1]
            say(f"{name}: season {season}")
            cells: dict[str, Cell] = {}
            report.cells[league] = cells

            cell, items = ask("/leagues", {"id": league, "season": season})
            seasons = (
                items[0].get("seasons") if items and isinstance(items[0], dict) else []
            ) or []
            this = next((s for s in seasons if s.get("year") == season), None)
            if this is not None:
                report.declared[league] = this.get("coverage") or {}
                cell = Cell(OK, 1, f"{this.get('start')} to {this.get('end')}")
            elif cell.status == OK:
                cell = Cell(EMPTY, 0, "season not listed")
            cells["season"] = cell
            if cell.status == DENIED:
                # Every other query names the same season; asking again only spends quota.
                for ep in ENDPOINTS[1:]:
                    cells[ep] = Cell(SKIPPED, 0, "season refused by the plan")
                continue

            base = {"league": league, "season": season}
            # A finished season's last fortnight, else the fortnight up to today.
            end = today
            if this is not None and this.get("end"):
                end = min(today, date.fromisoformat(str(this["end"])))
            window = {"from": (end - timedelta(days=14)).isoformat(), "to": end.isoformat()}
            cells["fixtures"], fixtures = ask("/fixtures", base | window)
            finished = [
                f
                for f in fixtures
                if ((f.get("fixture") or {}).get("status") or {}).get("short") in _FINISHED
            ]
            if finished:
                latest = max(finished, key=lambda f: (f.get("fixture") or {}).get("date") or "")
                fid = (latest.get("fixture") or {}).get("id")
                teams = latest.get("teams") or {}
                cells["fixtures"] = Cell(
                    OK,
                    len(fixtures),
                    f"inspecting {((teams.get('home') or {}).get('name'))} v "
                    f"{((teams.get('away') or {}).get('name'))} ({fid})",
                )
                c, st = ask("/fixtures/statistics", {"fixture": fid})
                types = _stat_types(st)
                cells["statistics"] = Cell(
                    c.status,
                    c.results,
                    c.note or ("xG present" if "expected_goals" in types else "no xG field"),
                )
                cells["events"], _ = ask("/fixtures/events", {"fixture": fid})
                cells["lineups"], _ = ask("/fixtures/lineups", {"fixture": fid})
                c, pl = ask("/fixtures/players", {"fixture": fid})
                keys = _keys_deep(pl, 9)
                cells["player match stats"] = Cell(
                    c.status,
                    c.results,
                    c.note
                    or ", ".join(
                        n
                        for n, k in (
                            ("rating", "rating"),
                            ("key passes", "key"),
                            ("xG", "expected"),
                        )
                        if k in keys
                    )
                    or "basic counts",
                )
            else:
                for ep in ("statistics", "events", "lineups", "player match stats"):
                    cells[ep] = Cell(SKIPPED, 0, "no finished fixture in the 14-day window")

            c, players = ask("/players", base | {"page": 1})
            keys = _keys_deep(players, 9)
            cells["player season stats"] = Cell(
                c.status,
                c.results,
                c.note or ("xG present" if "expected" in keys else "no xG / progressive fields"),
            )
            cells["injuries"], _ = ask("/injuries", base)
            c, odds = ask("/odds", base)
            bets = {
                str(b.get("name"))
                for o in odds[:5]
                for bm in (o.get("bookmakers") or [])[:3]
                for b in (bm.get("bets") or [])
            }
            cells["odds"] = Cell(
                c.status,
                c.results,
                c.note or ("BTTS priced" if any("both teams" in b.lower() for b in bets) else ""),
            )
            cells["standings"], _ = ask("/standings", base)
    except QuotaExhaustedError as exc:
        say(f"stopped: {exc}")
        for league in leagues or list(PRIORITY_LEAGUES):
            cells = report.cells.setdefault(league, {})
            for ep in ENDPOINTS:
                cells.setdefault(ep, Cell(SKIPPED, 0, "daily quota reached"))
    return report


_MARK = {OK: "✅", EMPTY: "◻️", DENIED: "⛔", ERROR: "⚠️", SKIPPED: "·"}


def coverage_markdown(report: CoverageReport) -> str:
    """The matrix as Markdown, for docs/api-football-coverage.md."""
    leagues = list(report.cells)
    names = [PRIORITY_LEAGUES.get(lg, (str(lg), str(lg)))[1] for lg in leagues]
    lines = [
        "# API-Football coverage — measured",
        "",
        f"Probed {report.ran_at:%Y-%m-%d %H:%M} UTC, season **{report.season}** "
        f"({report.season}/{(report.season + 1) % 100:02d}), {report.requests_used} requests. "
        "Generated by `soccer probe-api-football`; every cell is one real response.",
        "",
    ]
    if report.account:
        lines += ["Account: " + ", ".join(f"{k} = {v}" for k, v in report.account.items()), ""]
    lines += [
        "✅ data · ◻️ empty but allowed · ⛔ refused by the plan · ⚠️ other error · · not asked",
        "",
        "| Endpoint | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
    ]
    for ep in ENDPOINTS:
        row = []
        for lg in leagues:
            c = report.cells[lg].get(ep, Cell(SKIPPED))
            row.append(f"{_MARK[c.status]} {c.results if c.status == OK else ''}".strip())
        lines.append(f"| {ep} | " + " | ".join(row) + " |")
    lines += ["", "## Notes", ""]
    for lg, name in zip(leagues, names, strict=True):
        notes = [
            f"{ep}: {c.note}"
            for ep, c in report.cells[lg].items()
            if c.note and c.note != "season refused by the plan"
        ]
        if notes:
            lines.append(f"**{name}** — " + "; ".join(notes))
            lines.append("")
    if report.declared:
        lines += ["## Coverage flags API-Football declares for the season", ""]
        for lg, cov in report.declared.items():
            name = PRIORITY_LEAGUES.get(lg, (str(lg), str(lg)))[1]
            flat = {**{f"fixtures.{k}": v for k, v in (cov.get("fixtures") or {}).items()}}
            flat |= {k: v for k, v in cov.items() if k != "fixtures"}
            lines.append(f"- **{name}**: " + ", ".join(f"{k}={v}" for k, v in sorted(flat.items())))
        lines.append("")
    return "\n".join(lines)
