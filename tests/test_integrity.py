"""Data self-checks: each catches the defect it exists for, and clean data passes."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

from typer.testing import CliRunner

from soccer.sources.football_data_co_uk import MatchResult
from soccer.storage import integrity
from soccer.storage.analytics_db import AnalyticsDB
from soccer.storage.integrity import ERROR, WARNING, run_checks
from tests.test_dashboard_data import seed_results

TODAY = date(2026, 10, 5)
TEAMS = ["Arsenal", "Chelsea", "Fulham", "Brentford"]


def _row(season: str, division: str, d: date, home: str, away: str) -> MatchResult:
    from soccer.domain.names import normalize_name

    return MatchResult(
        season=season,
        division=division,
        match_date=d,
        home=home,
        away=away,
        home_norm=normalize_name(home),
        away_norm=normalize_name(away),
        fthg=1,
        ftag=0,
        ftr="H",
        hthg=None,
        htag=None,
        home_shots=None,
        away_shots=None,
        home_shots_target=None,
        away_shots_target=None,
        home_corners=None,
        away_corners=None,
        home_yellows=None,
        away_yellows=None,
        home_reds=None,
        away_reds=None,
        referee=None,
    )


def _checks(path):
    with AnalyticsDB(path) as adb:
        return run_checks(adb, today=TODAY)


def test_clean_data_passes(tmp_path) -> None:
    path = tmp_path / "a.duckdb"
    seed_results(path, division="E0", teams=TEAMS, season="2425")
    seed_results(path, division="E0", teams=TEAMS, season="2526")
    assert _checks(path) == []


def test_catches_the_season_ordering_bug(tmp_path, monkeypatch) -> None:
    # Replay the real defect: "2026/2027" codes all scored 0 under the old sort key.
    path = tmp_path / "a.duckdb"
    with AnalyticsDB(path) as adb:
        adb.load_results(
            [
                _row("2012/2013", "AUT", date(2013, 5, 1), "Rapid", "Austria Vienna"),
                _row("2026/2027", "AUT", date(2026, 9, 20), "Rapid", "Sturm"),
            ]
        )
    assert _checks(path) == []  # fixed key: fine
    monkeypatch.setattr(integrity, "season_sort_key", lambda s: (s.isdigit() and int(s)) or 0)
    found = _checks(path)
    assert [(f.level, f.check) for f in found] == [(ERROR, "season order")]


def test_duplicates_future_results_and_self_matches_are_errors(tmp_path) -> None:
    path = tmp_path / "a.duckdb"
    dup = _row("2526", "E0", date(2026, 1, 3), "Arsenal", "Chelsea")
    with AnalyticsDB(path) as adb:
        adb.load_results(
            [
                dup,
                dup,
                _row("2526", "E0", date(2026, 12, 1), "Fulham", "Brentford"),
                replace(_row("2526", "E0", date(2026, 1, 4), "Fulham", "Fulham")),
            ]
        )
    checks = {f.check for f in _checks(path) if f.level == ERROR}
    assert {"duplicate", "future result", "self match"} <= checks


def test_a_split_club_name_warns(tmp_path) -> None:
    path = tmp_path / "a.duckdb"
    seed_results(path, division="E0", teams=TEAMS, season="2425")
    seed_results(path, division="E0", teams=TEAMS, season="2526")
    # The same club arriving under three new spellings -> more teams than last season.
    with AnalyticsDB(path) as adb:
        rows = [
            _row("2526", "E0", date(2026, 2, d), alias, "Chelsea")
            for d, alias in enumerate(["Arsenal FC London", "The Arsenal", "Arsenal Gunners"], 1)
        ]
        adb.load_results([*rows, *_existing(adb, "2526")])
    found = _checks(path)
    assert any(f.level == WARNING and f.check == "team count" for f in found)


def _existing(adb: AnalyticsDB, season: str) -> list[MatchResult]:
    return [
        _row(season, "E0", r.match_date, r.home, r.away) for r in adb.outcomes_for(season, "E0")
    ]


def test_check_command_exits_nonzero_on_errors(tmp_path, monkeypatch) -> None:
    import soccer.config as config
    from soccer.cli.main import app

    dup = _row("2526", "E0", date(2026, 1, 3), "Arsenal", "Chelsea")
    with AnalyticsDB(tmp_path / "analytics.duckdb") as adb:
        adb.load_results([dup, dup])
    monkeypatch.setenv("SOCCER_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(config, "_settings", None)
    result = CliRunner().invoke(app, ["check"])
    monkeypatch.setattr(config, "_settings", None)
    assert result.exit_code == 1
    assert "duplicate" in result.output
