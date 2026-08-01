"""DuckDB analytics store tests -- load idempotence and the computed league table."""

from __future__ import annotations

from datetime import date

import pytest

from soccer.sources.football_data_co_uk import MatchResult
from soccer.storage.analytics_db import AnalyticsDB


def result(
    home: str, away: str, hg: int, ag: int, *, season="2526", division="E0", day=1
) -> MatchResult:
    from soccer.domain.names import normalize_name

    return MatchResult(
        season=season,
        division=division,
        match_date=date(2026, 1, day),
        home=home,
        away=away,
        home_norm=normalize_name(home),
        away_norm=normalize_name(away),
        fthg=hg,
        ftag=ag,
        ftr="H" if hg > ag else "A" if ag > hg else "D",
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


@pytest.fixture
def adb(tmp_path) -> AnalyticsDB:
    return AnalyticsDB(tmp_path / "analytics.duckdb")


class TestLoad:
    def test_load_counts(self, adb: AnalyticsDB) -> None:
        assert adb.load_results([result("A", "B", 1, 0), result("C", "D", 2, 2)]) == 2
        assert adb.result_count() == 2

    def test_reload_replaces_unit_not_duplicates(self, adb: AnalyticsDB) -> None:
        adb.load_results([result("A", "B", 1, 0)])
        adb.load_results([result("A", "B", 1, 0), result("C", "D", 0, 1)])
        # Same (season, division) unit replaced, not appended.
        assert adb.result_count("2526", "E0") == 2

    def test_different_units_coexist(self, adb: AnalyticsDB) -> None:
        adb.load_results([result("A", "B", 1, 0, division="E0")])
        adb.load_results([result("X", "Y", 1, 0, division="E1")])
        assert adb.result_count() == 2
        assert adb.result_count(division="E0") == 1

    def test_empty_load_is_noop(self, adb: AnalyticsDB) -> None:
        assert adb.load_results([]) == 0


class TestLeagueTable:
    def test_table_computation(self, adb: AnalyticsDB) -> None:
        # A 3-team round: A beats B, A beats C, B draws C.
        adb.load_results(
            [
                result("A", "B", 2, 0, day=1),
                result("A", "C", 1, 0, day=2),
                result("B", "C", 1, 1, day=3),
            ]
        )
        table = adb.league_table("2526", "E0")
        top = table[0]
        assert top.team == "A"
        assert (top.played, top.won, top.drawn, top.lost) == (2, 2, 0, 0)
        assert top.points == 6
        assert top.goals_for == 3 and top.goals_against == 0

    def test_table_ordering_by_points_then_gd(self, adb: AnalyticsDB) -> None:
        adb.load_results(
            [
                result("A", "B", 5, 0, day=1),  # A: +5, 3pts
                result("C", "D", 1, 0, day=2),  # C: +1, 3pts
            ]
        )
        table = adb.league_table("2526", "E0")
        # A and C both on 3 points; A ahead on goal difference.
        assert table[0].team == "A"
        assert table[1].team == "C"

    def test_points_are_internally_consistent(self, adb: AnalyticsDB) -> None:
        adb.load_results(
            [
                result("A", "B", 2, 0),
                result("B", "C", 1, 1),
                result("C", "A", 0, 3),
            ]
        )
        for row in adb.league_table("2526", "E0"):
            assert row.won * 3 + row.drawn == row.points

    def test_missing_season_returns_empty(self, adb: AnalyticsDB) -> None:
        assert adb.league_table("9999", "ZZ") == []


class TestInventory:
    def test_seasons_loaded(self, adb: AnalyticsDB) -> None:
        adb.load_results([result("A", "B", 1, 0, season="2526", division="E0")])
        adb.load_results([result("X", "Y", 1, 0, season="2425", division="E0")])
        loaded = adb.seasons_loaded()
        assert ("2526", "E0", 1) in loaded
        assert loaded[0][0] == "2526"  # newest season first
