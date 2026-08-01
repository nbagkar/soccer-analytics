"""MCP tool tests.

The MCP server is a thin set of wrappers; the logic worth testing is `mcp/tools.py` --
that each tool returns a sensible JSON-able dict and that expected failures come back as
a structured ``error`` rather than an exception (an LLM client must get a clean message,
never a crash).
"""

from __future__ import annotations

from datetime import date

import pytest

from soccer.config import Settings
from soccer.mcp import tools
from soccer.sources.football_data_co_uk import MatchResult
from soccer.storage.analytics_db import AnalyticsDB


def seed_history(path) -> None:
    from soccer.domain.names import normalize_name

    def r(home, away, hg, ag, day):
        return MatchResult(
            season="2526",
            division="E0",
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

    teams = ["Arsenal", "Chelsea", "Fulham", "Brentford"]
    rows, day = [], 1
    for i, h in enumerate(teams):
        for a in teams[i + 1 :]:
            hg, ag = (3, 0) if h == "Arsenal" else (0, 2) if h == "Brentford" else (1, 1)
            rows.append(r(h, a, hg, ag, day))
            rows.append(r(a, h, ag, hg, day + 1))
            day += 2
    with AnalyticsDB(path) as adb:
        adb.load_results(rows)


@pytest.fixture
def settings(tmp_path):
    seed_history(tmp_path / "analytics.duckdb")
    return Settings(data_dir=tmp_path)


class TestHistoricalTools:
    def test_league_table(self, settings) -> None:
        out = tools.league_table(settings, "2526", "E0")
        assert "error" not in out
        assert len(out["table"]) == 4
        assert out["table"][0]["team"] == "Arsenal"

    def test_power_rankings(self, settings) -> None:
        out = tools.power_rankings(settings, "2526", "E0")
        assert out["rankings"][0]["team"] == "Arsenal"
        assert isinstance(out["rankings"][0]["elo"], int)

    def test_forecast_match(self, settings) -> None:
        out = tools.forecast_match(settings, "2526", "E0", "Arsenal", "Brentford")
        assert "error" not in out
        p = out["probabilities"]
        assert p["home_win"] + p["draw"] + p["away_win"] == pytest.approx(1.0, abs=1e-6)
        assert p["home_win"] > p["away_win"]  # strong home team favoured
        assert len(out["likely_scores"]) == 5

    def test_forecast_unknown_team_returns_structured_error(self, settings) -> None:
        out = tools.forecast_match(settings, "2526", "E0", "Arsenal", "Nobody FC")
        assert "error" in out
        assert "available_teams" in out  # helps the caller recover

    def test_simulate_league(self, settings) -> None:
        out = tools.simulate_league(settings, "2526", "E0", sims=500)
        assert sum(p["title_pct"] for p in out["projections"]) == pytest.approx(1.0, abs=1e-6)
        assert out["projections"][0]["team"] == "Arsenal"

    def test_search_teams(self, settings) -> None:
        out = tools.search_teams(settings, "ars", "2526", "E0")
        assert "Arsenal" in out["matches"]


class TestErrorPaths:
    def test_missing_history_is_structured_error(self, tmp_path) -> None:
        # No analytics DB at all.
        empty = Settings(data_dir=tmp_path / "empty")
        out = tools.league_table(empty, "2526", "E0")
        assert "error" in out
        assert "ingest-history" in out["error"]

    def test_missing_live_is_structured_error(self, tmp_path) -> None:
        empty = Settings(data_dir=tmp_path / "empty")
        out = tools.live_matches(empty)
        assert "error" in out


class TestMetaTools:
    def test_data_health(self, settings) -> None:
        out = tools.data_health(settings)
        names = {s["name"] for s in out["sources"]}
        assert "TheSportsDB" in names
        assert "expected_goals" in out["capability_coverage"]

    def test_available_data_lists_history(self, settings) -> None:
        out = tools.available_data(settings)
        assert any(h["season"] == "2526" and h["division"] == "E0" for h in out["history"])
