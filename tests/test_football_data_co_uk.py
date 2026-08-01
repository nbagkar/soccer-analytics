"""football-data.co.uk parser and adapter tests.

The parser is the risk: real files carry mixed date formats, blank trailing lines,
abandoned matches with no score, and fewer stat columns in older seasons. It must skip
the unusable without corrupting the good.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import httpx
import pytest

from soccer.sources.football_data_co_uk import (
    FootballDataCoUk,
    parse_new_league_csv,
    parse_results_csv,
)
from soccer.storage.raw import RawStore

# A compact CSV in the real shape: a full row, a modern date, an abandoned row (no
# score), an older DD/MM/YY date, and a trailing blank line. Trimmed to the columns
# under test.
CSV = """Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HTHG,HTAG,HS,AS,HST,AST,Referee
E0,15/08/2025,20:00,Liverpool,Bournemouth,4,2,H,1,0,19,10,7,4,M Oliver
E0,16/08/2025,15:00,Aston Villa,Newcastle,0,0,D,0,0,3,16,1,5,A Taylor
E0,17/08/2025,14:00,Abandoned,Match,,,,,,,,,,
E0,20/05/26,15:00,Man United,Arsenal,1,3,A,0,2,8,14,3,6,C Kavanagh

"""


class TestParser:
    def test_parses_good_rows(self) -> None:
        results = parse_results_csv(CSV, season="2526", division="E0")
        assert len(results) == 3  # abandoned + blank line dropped

    def test_extracts_core_fields(self) -> None:
        first = parse_results_csv(CSV, season="2526", division="E0")[0]
        assert first.home == "Liverpool"
        assert first.away == "Bournemouth"
        assert first.match_date == date(2025, 8, 15)
        assert (first.fthg, first.ftag, first.ftr) == (4, 2, "H")
        assert (first.hthg, first.htag) == (1, 0)
        assert (first.home_shots, first.away_shots) == (19, 10)
        assert first.referee == "M Oliver"

    def test_normalizes_names_for_joining(self) -> None:
        # "Man United" normalizes to "man united" -- note this will NOT match
        # football-data.org's "Manchester United FC" without an alias, by design.
        man_u = parse_results_csv(CSV, season="2526", division="E0")[2]
        assert man_u.home == "Man United"
        assert man_u.home_norm == "man united"

    def test_handles_two_digit_year(self) -> None:
        man_u = parse_results_csv(CSV, season="2526", division="E0")[2]
        assert man_u.match_date == date(2026, 5, 20)

    def test_abandoned_match_skipped(self) -> None:
        teams = {r.home for r in parse_results_csv(CSV, season="2526", division="E0")}
        assert "Abandoned" not in teams

    def test_empty_csv_yields_nothing(self) -> None:
        assert (
            parse_results_csv("Div,Date,HomeTeam,AwayTeam,FTHG,FTAG\n", season="x", division="y")
            == []
        )

    def test_result_derived_when_ftr_missing(self) -> None:
        csv = (
            "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"
            "01/01/2026,A,B,2,1,\n"  # FTR blank -> derive H
        )
        assert parse_results_csv(csv, season="2526", division="E0")[0].ftr == "H"


# The "new leagues" schema: one file, every season, Country/League/Season/Date/Home/
# Away/HG/AG/Res and odds -- no shot or card columns. A future row without a score too.
NEW_CSV = """Country,League,Season,Date,Time,Home,Away,HG,AG,Res,PSCH,PSCD,PSCA
Brazil,Serie A,2024,19/05/2024,22:30,Palmeiras,Santos,2,0,H,1.75,3.8,5.2
Brazil,Serie A,2025,20/05/2025,22:30,Flamengo RJ,Vasco,1,1,D,2.1,3.4,3.6
Brazil,Serie A,2026,21/05/2026,22:30,Corinthians,Gremio,,,,1.9,3.3,4.1
"""


class TestNewLeagueParser:
    def test_parses_new_schema_fields(self) -> None:
        rows = parse_new_league_csv(NEW_CSV, division="BRA")
        # The 2026 row has no score yet -> dropped; two usable results remain.
        assert len(rows) == 2
        first = rows[0]
        assert (first.home, first.away) == ("Palmeiras", "Santos")
        assert (first.fthg, first.ftag, first.ftr) == (2, 0, "H")
        assert first.season == "2024"
        assert first.division == "BRA"
        assert first.home_norm == "palmeiras"
        # No shot/card columns in this schema.
        assert first.home_shots is None and first.home_yellows is None

    def test_recent_seasons_keeps_only_newest(self) -> None:
        rows = parse_new_league_csv(NEW_CSV, division="BRA", recent_seasons=1)
        # Only 2026 season kept -- but its one row has no score, so nothing survives.
        assert rows == []
        rows2 = parse_new_league_csv(NEW_CSV, division="BRA", recent_seasons=2)
        # 2025 and 2026 kept; only 2025's row has a score.
        assert {r.season for r in rows2} == {"2025"}


class TestClosingOdds:
    def test_prefers_pinnacle_closing(self) -> None:
        # Both Pinnacle closing (PSC*) and Bet365 (B365*) present -> PSC wins.
        csv = (
            "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A,PSCH,PSCD,PSCA\n"
            "01/01/2026,A,B,2,1,H,2.0,3.0,4.0,1.85,3.6,4.5\n"
        )
        r = parse_results_csv(csv, season="2526", division="E0")[0]
        assert (r.close_home_odds, r.close_draw_odds, r.close_away_odds) == (1.85, 3.6, 4.5)

    def test_falls_back_to_bet365_when_no_closing(self) -> None:
        csv = (
            "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A\n"
            "01/01/2026,A,B,2,1,H,2.0,3.0,4.0\n"
        )
        r = parse_results_csv(csv, season="2526", division="E0")[0]
        assert (r.close_home_odds, r.close_draw_odds, r.close_away_odds) == (2.0, 3.0, 4.0)

    def test_missing_odds_are_none(self) -> None:
        csv = "Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n01/01/2026,A,B,2,1,H\n"
        r = parse_results_csv(csv, season="2526", division="E0")[0]
        assert r.close_home_odds is None

    def test_new_league_odds_parsed(self) -> None:
        # The BRA schema carries PSCH/PSCD/PSCA closing odds too.
        rows = parse_new_league_csv(NEW_CSV, division="BRA")
        assert rows[0].close_home_odds == 1.75  # from the sample row's PSCH


class TestAdapter:
    @pytest.fixture
    def raw(self, tmp_path: Path) -> RawStore:
        return RawStore(tmp_path / "raw")

    def make_adapter(self, raw: RawStore, handler: httpx.MockTransport) -> FootballDataCoUk:
        return FootballDataCoUk(raw, client=httpx.Client(transport=handler))

    def test_fetch_parses_and_snapshots(self, raw: RawStore) -> None:
        transport = httpx.MockTransport(lambda request: httpx.Response(200, text=CSV))
        with self.make_adapter(raw, transport) as adapter:
            results = adapter.fetch_division("2526", "E0")
        assert len(results) == 3
        # Raw CSV snapshotted for later re-parse.
        from soccer.sources.registry import SourceId

        assert raw.latest(SourceId.FOOTBALL_DATA_CO_UK, "2526_E0") is not None

    def test_404_returns_empty_not_error(self, raw: RawStore) -> None:
        # Sweeping many (season, division) combos, some just do not exist.
        transport = httpx.MockTransport(lambda request: httpx.Response(404, text="Not found"))
        with self.make_adapter(raw, transport) as adapter:
            assert adapter.fetch_division("9999", "ZZ") == []

    def test_server_error_raises(self, raw: RawStore) -> None:
        transport = httpx.MockTransport(lambda request: httpx.Response(500, text="boom"))
        with (
            self.make_adapter(raw, transport) as adapter,
            pytest.raises(httpx.HTTPStatusError),
        ):
            adapter.fetch_division("2526", "E0")
