"""Market intelligence tests: early/closing odds parsing, upcoming lines (parse, fetch, store),
the line-movement analysis, and the offline re-parse that backfills early odds."""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from soccer.config import Settings
from soccer.market.implied import implied_probabilities, overround
from soccer.market.movement import MovementRecord, line_movement
from soccer.models.evaluation import evaluate_forecasts
from soccer.sources.football_data_co_uk import (
    FootballDataCoUk,
    MatchResult,
    UpcomingOdds,
    parse_new_league_csv,
    parse_results_csv,
    parse_upcoming_csv,
)
from soccer.sources.registry import SourceId
from soccer.storage.analytics_db import AnalyticsDB
from soccer.storage.raw import RawStore

# Pinnacle early + close on row 1; only market-average closing on row 2 (Pinnacle blank);
# only pre-match Bet365 on row 3 (an old season with no closing columns at all).
RESULTS_CSV = (
    "﻿Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,PSH,PSD,PSA,AvgH,AvgD,AvgA,"
    "B365H,B365D,B365A,PSCH,PSCD,PSCA,AvgCH,AvgCD,AvgCA\n"
    "E0,15/08/2025,Liverpool,Bournemouth,4,2,H,1.28,6.56,9.07,1.27,6.2,9.5,"
    "1.25,6.0,10,1.29,6.55,9.75,1.28,6.3,9.6\n"
    "E0,16/08/2025,Aston Villa,Newcastle,0,0,D,,,,2.7,3.4,2.6,2.6,3.4,2.7,,,,2.8,3.3,2.55\n"
    "E0,17/08/2025,Man United,Arsenal,1,3,A,,,,,,,3.1,3.4,2.3,,,,,,\n"
)

FIXTURES_CSV = (
    "﻿Div,Date,Time,HomeTeam,AwayTeam,Referee,B365H,B365D,B365A,PSH,PSD,PSA,"
    "AvgH,AvgD,AvgA,Avg>2.5,Avg<2.5\n"
    "E0,21/11/2026,15:00,Arsenal,Chelsea,,2.0,3.5,3.8,,,,1.95,3.6,3.9,1.8,2.0\n"
    "SP1,22/11/2026,20:00,Barcelona,Getafe,,1.2,7,12,1.22,7.1,13,,,,,\n"
    "E1,22/11/2026,12:30,Leeds,Hull,,,,,,,,,,,,\n"  # no price -> not an observation
)
NEW_FIXTURES_CSV = (
    "﻿Country,League,Date,Time,Home,Away,PSH,PSD,PSA,AvgH,AvgD,AvgA\n"
    "Brazil,Serie A,21/11/2026,21:30,Mirassol,Atletico-MG,,,,1.94,3.47,3.66\n"
    "Switzerland,Super League,22/11/2026,15:00,Basel,Servette,1.8,3.8,4.2,,,\n"  # not loaded
)


class TestEarlyAndClosingOdds:
    def test_early_line_comes_from_the_same_book_as_the_close(self) -> None:
        rows = parse_results_csv(RESULTS_CSV, season="2526", division="E0")
        pin, avg, _old = rows
        assert pin.close_odds_book == "PSC"
        assert (pin.close_home_odds, pin.early_home_odds) == (1.29, 1.28)
        # Pinnacle missing: the close is the market average, so the early line must be the
        # market average's too -- never Pinnacle's or Bet365's.
        assert avg.close_odds_book == "AvgC"
        assert (avg.early_home_odds, avg.early_draw_odds, avg.early_away_odds) == (2.7, 3.4, 2.6)

    def test_pre_match_fallback_has_no_early_line(self) -> None:
        old = parse_results_csv(RESULTS_CSV, season="2526", division="E0")[2]
        assert old.close_odds_book == "B365"  # a pre-match line standing in for the close
        assert old.close_home_odds == 3.1
        assert old.early_home_odds is None  # nothing to compare it with

    def test_new_league_files_record_the_book(self) -> None:
        text = (
            "Country,League,Season,Date,Time,Home,Away,HG,AG,Res,PSCH,PSCD,PSCA,AvgCH,AvgCD,AvgCA\n"
            "Brazil,Serie A,2026,19/05/2026,22:30,Palmeiras,Santos,1,1,D,,,,1.7,3.6,5.0\n"
        )
        (row,) = parse_new_league_csv(text, division="BRA")
        assert row.close_odds_book == "AvgC"
        assert row.early_home_odds is None


class TestUpcomingParser:
    def test_european_schema_prefers_pinnacle_then_average(self) -> None:
        lines = parse_upcoming_csv(FIXTURES_CSV)
        by_home = {o.home: o for o in lines}
        assert set(by_home) == {"Arsenal", "Barcelona"}  # the unpriced Leeds row is skipped
        ars = by_home["Arsenal"]
        assert (ars.division, ars.match_date, ars.kickoff) == ("E0", date(2026, 11, 21), "15:00")
        assert (ars.book, ars.home_odds, ars.over25_odds) == ("Avg", 1.95, 1.8)
        assert by_home["Barcelona"].book == "PS"
        assert by_home["Barcelona"].over25_odds is None

    def test_new_league_schema_maps_country_and_skips_unknown(self) -> None:
        (line,) = parse_upcoming_csv(NEW_FIXTURES_CSV)
        assert (line.division, line.home_norm, line.book) == ("BRA", "mirassol", "Avg")


class TestUpcomingFetch:
    def _adapter(self, tmp_path: Path, handler: object) -> FootballDataCoUk:
        client = httpx.Client(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]
        return FootballDataCoUk(RawStore(tmp_path / "raw"), client=client)

    def test_fetches_both_files_and_snapshots_them(self, tmp_path: Path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            new = "new_league" in request.url.path
            return httpx.Response(200, text=NEW_FIXTURES_CSV if new else FIXTURES_CSV)

        with self._adapter(tmp_path, handler) as source:
            lines = source.fetch_upcoming_odds()
        assert {o.division for o in lines} == {"E0", "SP1", "BRA"}
        raw = RawStore(tmp_path / "raw")
        assert raw.latest(SourceId.FOOTBALL_DATA_CO_UK, "fixtures") is not None
        assert raw.latest(SourceId.FOOTBALL_DATA_CO_UK, "new_league_fixtures") is not None

    def test_a_failed_download_raises_rather_than_returning_no_odds(self, tmp_path: Path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="down")

        with self._adapter(tmp_path, handler) as source, pytest.raises(httpx.HTTPStatusError):
            source.fetch_upcoming_odds()


def _line(**kw: object) -> UpcomingOdds:
    base: dict[str, object] = {
        "division": "E0",
        "match_date": date(2026, 11, 21),
        "kickoff": "15:00",
        "home": "Arsenal",
        "away": "Chelsea",
        "home_norm": "arsenal",
        "away_norm": "chelsea",
        "book": "Avg",
        "home_odds": 2.0,
        "draw_odds": 3.5,
        "away_odds": 3.8,
    }
    base.update(kw)
    return UpcomingOdds(**base)  # type: ignore[arg-type]


class TestOddsStore:
    def test_only_a_changed_price_adds_an_observation(self, tmp_path: Path) -> None:
        t0 = datetime(2026, 11, 18, 9)
        with AnalyticsDB(tmp_path / "a.duckdb") as adb:
            assert adb.load_upcoming_odds([_line()], t0) == 1
            assert adb.load_upcoming_odds([_line()], t0 + timedelta(hours=6)) == 0
            assert adb.load_upcoming_odds([_line(home_odds=1.9)], t0 + timedelta(days=1)) == 1
            (latest,) = adb.upcoming_odds(date(2026, 11, 20))
            assert latest.home_odds == 1.9 and latest.first_home_odds == 2.0
            assert latest.observations == 2
            assert latest.first_observed_at == t0
            assert adb.latest_odds_observation() == t0 + timedelta(days=1)

    def test_upcoming_excludes_matches_before_the_date(self, tmp_path: Path) -> None:
        with AnalyticsDB(tmp_path / "a.duckdb") as adb:
            adb.load_upcoming_odds(
                [_line(), _line(match_date=date(2026, 11, 14), home_norm="x")],
                datetime(2026, 11, 13),
            )
            assert [o.match_date for o in adb.upcoming_odds(date(2026, 11, 16))] == [
                date(2026, 11, 21)
            ]


def _result(i: int, **kw: object) -> MatchResult:
    base: dict[str, object] = {
        "season": "2526",
        "division": "E0",
        "match_date": date(2025, 8, 1) + timedelta(days=i),
        "home": f"H{i}",
        "away": f"A{i}",
        "home_norm": f"h{i}",
        "away_norm": f"a{i}",
        "fthg": 1,
        "ftag": 0,
        "ftr": "H",
        "hthg": None,
        "htag": None,
        "home_shots": None,
        "away_shots": None,
        "home_shots_target": None,
        "away_shots_target": None,
        "home_corners": None,
        "away_corners": None,
        "home_yellows": None,
        "away_yellows": None,
        "home_reds": None,
        "away_reds": None,
        "referee": None,
    }
    base.update(kw)
    return MatchResult(**base)  # type: ignore[arg-type]


class TestLoadResultsSchema:
    def test_odds_absent_from_the_first_hundred_rows_still_load(self, tmp_path: Path) -> None:
        # Regression: polars inferred dtypes from 100 rows, so a float appearing later in an
        # all-None column failed the whole load (an older season concatenated before a newer).
        rows = [_result(i) for i in range(150)]
        rows[140] = replace(rows[140], early_home_odds=1.5, close_odds_book="PSC")
        with AnalyticsDB(tmp_path / "a.duckdb") as adb:
            assert adb.load_results(rows) == 150
            got = adb._con.execute(
                "SELECT count(early_home_odds), count(close_odds_book) FROM results"
            ).fetchone()
        assert got == (1, 1)


class TestImplied:
    def test_overround_and_devig(self) -> None:
        assert overround(2.0, 4.0, 4.0) == pytest.approx(0.0)
        p = implied_probabilities(1.9, 3.8, 3.8)
        assert sum(p) == pytest.approx(1.0)
        assert p[0] == pytest.approx(0.5)


def _records(toward: bool, n: int = 60) -> list[MovementRecord]:
    out = []
    for i in range(n):
        gap = 0.06 if i % 2 else -0.06
        move = gap / 2 if toward else -gap / 2
        early = (0.45, 0.27, 0.28)
        out.append(
            MovementRecord(
                model=(0.45 + gap, 0.27, 0.28 - gap),
                early=early,
                close=(0.45 + move, 0.27, 0.28 - move),
                actual=i % 3,
            )
        )
    return out


class TestLineMovement:
    def test_close_moving_toward_the_model_is_detected(self) -> None:
        lm = line_movement(_records(toward=True))
        assert lm is not None
        assert lm.toward_model == 1.0 and lm.moves_toward_model
        assert lm.slope == pytest.approx(0.5)  # the close took half the model's gap
        assert lm.mean_move == pytest.approx(0.03)

    def test_close_moving_away_from_the_model(self) -> None:
        lm = line_movement(_records(toward=False))
        assert lm is not None
        assert lm.toward_model == 0.0 and not lm.moves_toward_model
        assert lm.slope == pytest.approx(-0.5)

    def test_too_few_matches(self) -> None:
        assert line_movement(_records(toward=True, n=10)) is None


def _synthetic_rows_with_early(start: date = date(2025, 1, 1)) -> list:  # type: ignore[type-arg]
    from tests.test_evaluation import _synthetic_rows

    rows = _synthetic_rows(start)
    # Early line: the close shaded toward even money, so early -> close is a real move.
    return [
        replace(
            r,
            early_home_odds=round((r.close_home_odds or 2) * 0.97 + 0.08, 2),
            early_draw_odds=r.close_draw_odds,
            early_away_odds=round((r.close_away_odds or 2) * 0.97 + 0.08, 2),
        )
        for r in rows
    ]


class TestEvaluateLineMovement:
    def test_report_carries_line_movement_when_early_odds_exist(self) -> None:
        report = evaluate_forecasts(_synthetic_rows_with_early(), min_history=20)
        assert report is not None and report.line_movement is not None
        lm = report.line_movement
        assert 30 <= lm.n <= report.n
        assert math.isfinite(lm.slope) and 0.0 <= lm.best_weight_vs_early <= 1.0

    def test_no_early_odds_no_movement(self) -> None:
        from tests.test_evaluation import _synthetic_rows

        report = evaluate_forecasts(_synthetic_rows(), min_history=20)
        assert report is not None and report.line_movement is None

    def test_lag_limits_what_the_early_line_model_may_know(self) -> None:
        rows = _synthetic_rows_with_early()
        # A lag longer than the whole sample leaves no matches to fit on: no comparison at
        # all, rather than one quietly built on results the early line could not have known.
        assert evaluate_forecasts(rows, min_history=20, early_lag_days=10_000).line_movement is None  # type: ignore[union-attr]
        short = evaluate_forecasts(rows, min_history=20, early_lag_days=2)
        long = evaluate_forecasts(rows, min_history=20, early_lag_days=30)
        assert short is not None and long is not None
        assert short.line_movement is not None and long.line_movement is not None
        # A longer lag means a later warm-up (fewer matches compared) -- the cutoff is real.
        assert long.line_movement.n < short.line_movement.n


class TestReparseFromRaw:
    def test_backfills_early_odds_for_loaded_slices_only(self, tmp_path: Path) -> None:
        settings = Settings(data_dir=tmp_path)
        settings.ensure_dirs()
        raw = RawStore(settings.raw_dir)
        raw.write(SourceId.FOOTBALL_DATA_CO_UK, "2526_E0", RESULTS_CSV)
        raw.write(SourceId.FOOTBALL_DATA_CO_UK, "2425_E0", RESULTS_CSV)  # not loaded
        old = [
            replace(r, early_home_odds=None, early_draw_odds=None, early_away_odds=None)
            for r in parse_results_csv(RESULTS_CSV, season="2526", division="E0")
        ]
        with AnalyticsDB(settings.analytics_db) as adb:
            adb.load_results(old)

        from soccer.dashboard.actions import reparse_results_from_raw

        assert "3 results across 1 slice" in reparse_results_from_raw(settings)
        with AnalyticsDB(settings.analytics_db) as adb:
            assert adb.seasons_loaded() == [("2526", "E0", 3)]
            (row,) = [r for r in adb.outcomes_with_odds("2526", "E0") if r.home == "Liverpool"]
        assert row.early_home_odds == 1.28 and row.close_odds_book == "PSC"
