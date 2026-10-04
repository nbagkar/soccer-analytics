"""CLI season defaults resolve from the loaded data, never from a hardcoded code."""

from __future__ import annotations

import pytest
import typer

from soccer.cli.main import _resolve_season
from tests.test_dashboard_data import seed_results

TEAMS = ["Arsenal", "Chelsea", "Fulham", "Brentford"]  # 12 matches per seeded season


@pytest.fixture
def store(tmp_path, monkeypatch):
    import soccer.config as config

    monkeypatch.setenv("SOCCER_DATA_DIR", str(tmp_path))
    config._settings = None
    yield tmp_path / "analytics.duckdb"
    config._settings = None


def _seed_partial(path, season: str, keep: int) -> None:
    """A season still in progress: only its first `keep` matches are in."""
    from soccer.storage.analytics_db import AnalyticsDB

    seed_results(path, division="E0", teams=TEAMS, season=season)
    with AnalyticsDB(path) as adb:
        adb._con.execute(
            "DELETE FROM results WHERE season = ? AND rowid NOT IN "
            "(SELECT rowid FROM results WHERE season = ? ORDER BY match_date LIMIT ?)",
            [season, season, keep],
        )


class TestResolveSeason:
    def test_explicit_season_wins(self, store) -> None:
        assert _resolve_season("2122", "E0") == "2122"

    def test_latest_loaded_is_chronological_not_lexical(self, store) -> None:
        for season in ("9900", "2526", "2627"):
            seed_results(store, division="E0", teams=TEAMS, season=season)
        assert _resolve_season(None, "E0") == "2627"  # "9900" sorts last lexically

    def test_complete_skips_a_season_still_in_progress(self, store) -> None:
        seed_results(store, division="E0", teams=TEAMS, season="2526")
        _seed_partial(store, "2627", keep=3)
        assert _resolve_season(None, "E0") == "2627"
        assert _resolve_season(None, "E0", complete=True) == "2526"

    def test_complete_keeps_a_finished_latest_season(self, store) -> None:
        for season in ("2425", "2526"):
            seed_results(store, division="E0", teams=TEAMS, season=season)
        assert _resolve_season(None, "E0", complete=True) == "2526"

    def test_unknown_division_exits_cleanly(self, store) -> None:
        seed_results(store, division="E0", teams=TEAMS)
        with pytest.raises(typer.Exit):
            _resolve_season(None, "XX")

    def test_no_store_exits_cleanly(self, store) -> None:
        with pytest.raises(typer.Exit):
            _resolve_season(None, "E0")
