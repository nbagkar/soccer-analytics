"""Data self-checks over the analytics store -- quality built in, not inspected in later.

The "2026/2027" season-ordering bug passed 595 tests because the fixtures only used one
season format; it surfaced only when someone noticed Austria's table showed 2012/13. These
checks run against the REAL loaded data (`soccer check`, `soccer doctor`, the weekly
review), so that class of defect announces itself instead of waiting to be spotted.

Errors are states that are wrong by definition. Warnings are patterns that are usually a
data problem (a club split across two spellings) but can be legitimate (play-off sides with
only a few end-of-season games) -- worth a look, not a stop.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from soccer.sources.football_data_co_uk import division_name, season_sort_key
from soccer.storage.analytics_db import AnalyticsDB

ERROR = "error"
WARNING = "warning"


@dataclass(frozen=True)
class Finding:
    level: str
    check: str
    division: str
    detail: str


def run_checks(adb: AnalyticsDB, *, today: date | None = None) -> list[Finding]:
    """Every finding across all loaded divisions, errors first."""
    con = adb._con  # same storage layer; read-only queries
    today = today or date.today()
    findings: list[Finding] = []

    divisions = [r[0] for r in con.execute("SELECT DISTINCT division FROM results").fetchall()]
    for division in sorted(divisions):
        name = division_name(division)
        seasons = con.execute(
            "SELECT season, MAX(match_date), COUNT(*) FROM results WHERE division = ? "
            "GROUP BY season",
            [division],
        ).fetchall()
        ordered = sorted(seasons, key=lambda r: season_sort_key(r[0]))
        latest, latest_last_match = ordered[-1][0], ordered[-1][1]
        newest = max(seasons, key=lambda r: r[1])
        if latest_last_match < newest[1]:  # by date, so a tie isn't flagged
            findings.append(
                Finding(
                    ERROR,
                    "season order",
                    division,
                    f"{name}: latest season sorts as {latest!r}, but the newest match is in "
                    f"{newest[0]!r} -- season codes are being ordered wrongly",
                )
            )

        teams = _team_games(adb, division, latest)
        if len(ordered) > 1:
            previous = _team_games(adb, division, ordered[-2][0])
            if len(teams) > len(previous) + 2:
                findings.append(
                    Finding(
                        WARNING,
                        "team count",
                        division,
                        f"{name} {latest}: {len(teams)} teams vs {len(previous)} the season "
                        "before -- a club may be split across two spellings",
                    )
                )
        counts = sorted(teams.values())
        median = counts[len(counts) // 2] if counts else 0
        if median >= 6:  # too early in a season to judge otherwise
            thin = sorted(t for t, n in teams.items() if n < 0.5 * median)
            if thin:
                findings.append(
                    Finding(
                        WARNING,
                        "thin team",
                        division,
                        f"{name} {latest}: {', '.join(thin)} played under half the median "
                        f"({median}) -- a split name, or play-off sides",
                    )
                )

    for division, season, d, home, away, n in con.execute(
        "SELECT division, season, match_date, home, away, COUNT(*) FROM results "
        "GROUP BY division, season, match_date, home_norm, away_norm, home, away "
        "HAVING COUNT(*) > 1"
    ).fetchall():
        findings.append(
            Finding(ERROR, "duplicate", division, f"{home} v {away} on {d} ({season}) x{n}")
        )
    for division, d, home, away in con.execute(
        "SELECT division, match_date, home, away FROM results WHERE match_date > ?", [today]
    ).fetchall():
        findings.append(
            Finding(ERROR, "future result", division, f"{home} v {away} dated {d}, after today")
        )
    for division, d, home in con.execute(
        "SELECT division, match_date, home FROM results WHERE home_norm = away_norm"
    ).fetchall():
        findings.append(Finding(ERROR, "self match", division, f"{home} v {home} on {d}"))

    return sorted(findings, key=lambda f: (f.level != ERROR, f.division, f.check))


def _team_games(adb: AnalyticsDB, division: str, season: str) -> dict[str, int]:
    rows = adb._con.execute(
        "SELECT team, COUNT(*) FROM ("
        "  SELECT home_norm AS team FROM results WHERE division = ? AND season = ?"
        "  UNION ALL"
        "  SELECT away_norm FROM results WHERE division = ? AND season = ?"
        ") GROUP BY team",
        [division, season, division, season],
    ).fetchall()
    return {team: n for team, n in rows}
