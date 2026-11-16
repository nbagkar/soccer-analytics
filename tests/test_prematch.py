"""Pre-match odds for upcoming fixtures: parsed, stored, matched, shown and graded.

The failure paths matter most here: a fixture must never pick up another match's price (the
reversed fixture, the wrong division, an ambiguous name), a failed fetch must not wipe odds
already stored, and the model's own forecast must survive untouched beside the odds.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import ClassVar

import httpx
import pytest

from soccer.domain.match_state import MatchStatus
from soccer.domain.prematch import (
    KEEP_DAYS,
    UpcomingOddsStore,
    find_line,
    line_from,
    vig_free,
)
from soccer.sources.football_data_co_uk import (
    FIXTURES_URL,
    FootballDataCoUk,
    UpcomingOdds,
    parse_fixtures_csv,
)
from soccer.sources.registry import SourceId
from soccer.storage.live_db import LiveDB
from soccer.storage.raw import RawStore
from tests.test_dashboard_data import add_match, seed_results

HEADER = (
    "﻿Div,Date,Time,HomeTeam,AwayTeam,Referee,B365H,B365D,B365A,AvgH,AvgD,AvgA,"
    "B365>2.5,B365<2.5,Avg>2.5,Avg<2.5,AvgCH,AvgCD,AvgCA"
)
CSV = "\n".join(
    [
        HEADER,
        "E0,21/11/2026,15:00,Arsenal,Chelsea,,1.80,3.80,4.50,1.83,3.75,4.40,1.70,2.15,1.72,2.12,,,",
        # market average missing -> falls back to Bet365, and to its over/under
        "E0,21/11/2026,15:00,Fulham,Brentford,,2.40,3.30,3.00,,,,1.95,1.90,,,,,",
        # no price at all -> nothing to offer
        "E1,21/11/2026,15:00,Leeds,Burnley,,,,,,,,,,,,,,",
        # a broken price (odds of 1.0 or below) is not a price
        "E1,22/11/2026,12:30,Stoke,Hull,,1.00,3.4,2.9,0.5,3.4,2.9,,,,,,,",
        ",,,,,,,,,,,,,,,,,,",
    ]
)
NOW = datetime(2026, 11, 20, 9, 0, tzinfo=UTC)
KICKOFF = datetime(2026, 11, 21, 15, 0, tzinfo=UTC)


def _odds(
    home: str = "Arsenal",
    away: str = "Chelsea",
    *,
    division: str = "E0",
    day: date | None = None,
    prices: tuple[float, float, float] = (1.83, 3.75, 4.40),
    over_under: tuple[float, float] | None = (1.72, 2.12),
) -> UpcomingOdds:
    from soccer.domain.names import normalize_name

    return UpcomingOdds(
        division=division,
        match_date=day or KICKOFF.date(),
        home=home,
        away=away,
        home_norm=normalize_name(home),
        away_norm=normalize_name(away),
        book="avg",
        home_odds=prices[0],
        draw_odds=prices[1],
        away_odds=prices[2],
        over25_odds=over_under[0] if over_under else None,
        under25_odds=over_under[1] if over_under else None,
    )


class TestParseFixturesCsv:
    def test_keeps_only_priced_fixtures(self) -> None:
        rows = parse_fixtures_csv(CSV)
        assert [(r.division, r.home, r.away) for r in rows] == [
            ("E0", "Arsenal", "Chelsea"),
            ("E0", "Fulham", "Brentford"),
        ]

    def test_prefers_the_market_average_then_bet365(self) -> None:
        avg, b365 = parse_fixtures_csv(CSV)
        assert (avg.book, avg.home_odds, avg.over25_odds) == ("avg", 1.83, 1.72)
        assert (b365.book, b365.home_odds, b365.over25_odds) == ("b365", 2.40, 1.95)

    def test_reads_the_bom_and_the_date(self) -> None:
        first = parse_fixtures_csv(CSV)[0]
        assert first.match_date == date(2026, 11, 21)
        assert first.home_norm == "arsenal"

    def test_empty_file_yields_nothing(self) -> None:
        assert parse_fixtures_csv(HEADER + "\n") == []


class TestFetchUpcomingOdds:
    def test_fetches_the_fixtures_file_and_snapshots_it(self, tmp_path) -> None:
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url))
            return httpx.Response(200, text=CSV)

        raw = RawStore(tmp_path / "raw")
        client = httpx.Client(transport=httpx.MockTransport(handler))
        with FootballDataCoUk(raw, client=client) as source:
            rows = source.fetch_upcoming_odds()
        assert seen == [FIXTURES_URL]
        assert len(rows) == 2
        assert raw.latest(SourceId.FOOTBALL_DATA_CO_UK, "fixtures") is not None

    def test_a_failed_fetch_raises_rather_than_looking_empty(self, tmp_path) -> None:
        client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(503)))
        with (
            FootballDataCoUk(RawStore(tmp_path / "raw"), client=client) as source,
            pytest.raises(httpx.HTTPStatusError),
        ):
            source.fetch_upcoming_odds()


class TestVigFree:
    def test_divides_the_margin_out(self) -> None:
        probs = vig_free(1.83, 3.75, 4.40)
        assert sum(probs) == pytest.approx(1.0)
        assert probs[0] > probs[1] > probs[2]

    def test_line_carries_the_over_under(self) -> None:
        line = line_from(_odds(), NOW)
        assert line.p_over25 == pytest.approx((1 / 1.72) / (1 / 1.72 + 1 / 2.12))
        assert line_from(_odds(over_under=None), NOW).p_over25 is None


class TestUpcomingOddsStore:
    def test_round_trips(self, tmp_path) -> None:
        with LiveDB(tmp_path / "live.sqlite") as db:
            store = UpcomingOddsStore(db)
            assert store.replace([_odds()], now=NOW) == 1
            ((row, fetched),) = store.all()
        assert row == _odds()
        assert fetched == NOW

    def test_a_new_price_overwrites_the_old_one(self, tmp_path) -> None:
        with LiveDB(tmp_path / "live.sqlite") as db:
            store = UpcomingOddsStore(db)
            store.replace([_odds()], now=NOW)
            store.replace([_odds(prices=(2.0, 3.5, 3.9))], now=NOW + timedelta(hours=6))
            ((row, fetched),) = store.all()
        assert row.home_odds == 2.0
        assert fetched == NOW + timedelta(hours=6)

    def test_a_fixture_missing_from_the_next_file_keeps_its_price(self, tmp_path) -> None:
        # The file rolls on to the next round before a Monday game of this one is played.
        with LiveDB(tmp_path / "live.sqlite") as db:
            store = UpcomingOddsStore(db)
            store.replace([_odds()], now=NOW)
            store.replace([_odds("Fulham", "Brentford")], now=NOW + timedelta(days=1))
            assert store.count() == 2

    def test_old_fixtures_are_dropped(self, tmp_path) -> None:
        with LiveDB(tmp_path / "live.sqlite") as db:
            store = UpcomingOddsStore(db)
            store.replace([_odds()], now=NOW)
            store.replace([], now=datetime.combine(KICKOFF.date(), datetime.min.time(), UTC))
            assert store.count() == 1  # match day itself: kept
            long_after = KICKOFF + timedelta(days=KEEP_DAYS + 1)
            store.replace([], now=long_after)
            assert store.count() == 0

    def test_an_existing_store_gains_the_table(self, tmp_path) -> None:
        path = tmp_path / "live.sqlite"
        with LiveDB(path) as db:
            db.connection.execute("DROP TABLE upcoming_odds")
            db.connection.execute("PRAGMA user_version=11")
        with LiveDB(path) as db:  # a v11 store upgrades in place
            assert UpcomingOddsStore(db).count() == 0


class TestFindLine:
    def _stored(self, *rows: UpcomingOdds) -> list[tuple[UpcomingOdds, datetime]]:
        return [(r, NOW) for r in rows]

    def _find(self, stored, **kw):
        args = {"division": "E0", "kickoff_utc": KICKOFF, "home": "Arsenal", "away": "Chelsea"}
        return find_line(stored, **(args | kw))

    def test_matches_the_fixture(self) -> None:
        line = self._find(self._stored(_odds()))
        assert line is not None
        assert line.probs == pytest.approx(vig_free(1.83, 3.75, 4.40))

    def test_a_local_date_a_day_off_still_matches(self) -> None:
        late = datetime(2026, 11, 22, 0, 30, tzinfo=UTC)  # 21st in local time
        assert self._find(self._stored(_odds()), kickoff_utc=late) is not None

    def test_the_reversed_fixture_is_not_a_match(self) -> None:
        assert self._find(self._stored(_odds("Chelsea", "Arsenal"))) is None

    def test_another_division_is_not_a_match(self) -> None:
        assert self._find(self._stored(_odds(division="E1"))) is None

    def test_a_different_week_is_not_a_match(self) -> None:
        stale = _odds(day=KICKOFF.date() - timedelta(days=7))
        assert self._find(self._stored(stale)) is None

    def test_an_unresolved_side_is_not_a_match(self) -> None:
        assert self._find(self._stored(_odds()), away="Chelsea Women") is None

    def test_two_equally_near_candidates_are_refused(self) -> None:
        day = KICKOFF.date()
        stored = self._stored(
            _odds(day=day - timedelta(days=1)), _odds(day=day + timedelta(days=1))
        )
        assert self._find(stored) is None

    def test_uses_the_apps_name_matcher(self) -> None:
        from soccer.dashboard.data import match_team

        stored = self._stored(_odds("Man United", "Wolves"))
        line = self._find(
            stored,
            home="Manchester United FC",
            away="Wolverhampton Wanderers FC",
            matcher=match_team,
        )
        assert line is not None


def _seed(tmp_path, *, odds: list[UpcomingOdds] | None = None, away: str = "Chelsea FC"):
    """E0 history, one upcoming Arsenal fixture tomorrow, and optionally stored odds."""
    analytics, live = tmp_path / "analytics.duckdb", tmp_path / "live.sqlite"
    teams = ["Arsenal", "Chelsea", "Fulham", "Brentford", "Everton", "Wolves", "Burnley", "Leeds"]
    seed_results(analytics, division="E0", teams=teams)
    kickoff = datetime.now(UTC) + timedelta(days=1)
    with LiveDB(live) as db:
        add_match(
            db,
            match_id="fx1",
            home="Arsenal FC",
            away=away,
            competition="Premier League",
            status=MatchStatus.NOT_STARTED,
            observed_at=kickoff,
        )
        if odds is not None:
            UpcomingOddsStore(db).replace(odds)
    return analytics, live, kickoff


class TestFixtureForecasts:
    def test_without_odds_the_model_leads(self, tmp_path) -> None:
        from soccer.dashboard.data import fixture_forecasts

        analytics, live, _ = _seed(tmp_path)
        (f,) = fixture_forecasts(live, analytics)
        assert f.prematch is None and f.market is None
        assert not f.from_odds
        assert f.headline is f.slate is not None

    def test_with_odds_the_odds_lead_and_the_model_is_kept(self, tmp_path) -> None:
        from soccer.dashboard.data import fixture_forecasts

        analytics, live, kickoff = _seed(tmp_path)
        (before,) = fixture_forecasts(live, analytics)
        with LiveDB(live) as db:
            UpcomingOddsStore(db).replace([_odds(day=kickoff.date())])
        (f,) = fixture_forecasts(live, analytics)
        assert f.from_odds and f.headline is f.market
        # 1X2 is the vig-free price exactly, not a fit to it
        assert [m.probability for m in f.market.result] == pytest.approx(vig_free(1.83, 3.75, 4.40))
        assert [m.name for m in f.market.result] == [f.home, "Draw", f.away]
        # the model's own forecast is untouched by the odds
        assert f.slate is not None and before.slate is not None
        assert [m.probability for m in f.slate.result] == pytest.approx(
            [m.probability for m in before.slate.result]
        )

    def test_the_implied_markets_agree_with_the_prices(self, tmp_path) -> None:
        from soccer.dashboard.data import fixture_forecasts

        analytics, live, kickoff = _seed(tmp_path)
        with LiveDB(live) as db:
            UpcomingOddsStore(db).replace([_odds(day=kickoff.date())])
        (f,) = fixture_forecasts(live, analytics)
        over = next(o for o in f.market.over_under if o.line == 2.5).over
        assert over == pytest.approx(f.prematch.p_over25, abs=0.02)
        assert f.market.home_expected > f.market.away_expected  # the favourite scores more
        assert sum(m.probability for m in f.market.double_chance) == pytest.approx(2.0)

    def test_someone_elses_odds_do_not_attach(self, tmp_path) -> None:
        from soccer.dashboard.data import fixture_forecasts

        analytics, live, kickoff = _seed(tmp_path)
        with LiveDB(live) as db:
            UpcomingOddsStore(db).replace(
                [
                    _odds("Chelsea", "Arsenal", day=kickoff.date()),  # reversed
                    _odds("Arsenal", "Chelsea", division="E1", day=kickoff.date()),
                    _odds("Arsenal", "Fulham", day=kickoff.date()),
                ]
            )
        (f,) = fixture_forecasts(live, analytics)
        assert f.prematch is None and f.headline is f.slate

    def test_odds_cover_a_fixture_the_model_cannot(self, tmp_path) -> None:
        from soccer.dashboard.data import fixture_forecasts, ledger_rows

        # Sunderland have no history in the seeded model: no model forecast, but a price.
        analytics, live, kickoff = _seed(tmp_path, away="Sunderland AFC")
        with LiveDB(live) as db:
            UpcomingOddsStore(db).replace([_odds("Arsenal", "Sunderland", day=kickoff.date())])
        (f,) = fixture_forecasts(live, analytics)
        assert f.slate is None and f.headline is not None and f.from_odds
        assert ledger_rows([f]) == []  # the ledger is the MODEL's record: nothing to grade

    def test_the_switch_puts_the_model_back_in_the_lead(self, tmp_path, monkeypatch) -> None:
        from soccer.dashboard import data

        analytics, live, kickoff = _seed(tmp_path)
        with LiveDB(live) as db:
            UpcomingOddsStore(db).replace([_odds(day=kickoff.date())])
        (f,) = data.fixture_forecasts(live, analytics)
        monkeypatch.setattr(data, "PREMATCH_ODDS_HEADLINE", False)
        assert not f.from_odds and f.headline is f.slate

    def test_the_setting_does_not_start_a_new_model_version(self, monkeypatch) -> None:
        from soccer.dashboard import data

        version = data.forecast_model_version()
        monkeypatch.setattr(data, "PREMATCH_ODDS_HEADLINE", False)
        assert data.forecast_model_version() == version


class TestLedger:
    def _published(self, **kw):
        from tests.test_ledger import _fc

        return _fc(**kw)

    def _record(self, path, forecast, *, hours_before: float = 48):
        from soccer.domain.ledger import ForecastLedger, LedgerDB
        from tests.test_ledger import KICKOFF as LEDGER_KICKOFF

        with LedgerDB(path) as db:
            ledger = ForecastLedger(db)
            outcome = ledger.record([forecast], now=LEDGER_KICKOFF - timedelta(hours=hours_before))
            return outcome, ledger.entries()

    def test_records_the_odds_beside_the_model(self, tmp_path) -> None:
        from dataclasses import replace

        f = replace(self._published(), prematch=(0.52, 0.26, 0.22), prematch_book="avg")
        _, (entry,) = self._record(tmp_path / "ledger.sqlite", f)
        assert entry.prematch == pytest.approx((0.52, 0.26, 0.22))
        assert entry.prematch_book == "avg"
        assert entry.last.probs == pytest.approx((0.5, 0.3, 0.2))  # the model's, unchanged

    def test_moving_odds_update_the_line_but_are_not_a_model_revision(self, tmp_path) -> None:
        from dataclasses import replace

        path = tmp_path / "ledger.sqlite"
        self._record(path, replace(self._published(), prematch=(0.52, 0.26, 0.22)))
        outcome, (entry,) = self._record(
            path, replace(self._published(), prematch=(0.55, 0.25, 0.20)), hours_before=24
        )
        assert outcome.updated == 1
        assert entry.prematch == pytest.approx((0.55, 0.25, 0.20))
        assert entry.revisions == 1

    def test_a_recomputation_without_odds_keeps_the_line(self, tmp_path) -> None:
        from dataclasses import replace

        path = tmp_path / "ledger.sqlite"
        self._record(path, replace(self._published(), prematch=(0.52, 0.26, 0.22)))
        outcome, (entry,) = self._record(path, self._published(), hours_before=24)
        assert outcome.unchanged == 1
        assert entry.prematch == pytest.approx((0.52, 0.26, 0.22))

    def test_no_odds_are_written_after_kickoff(self, tmp_path) -> None:
        from dataclasses import replace

        path = tmp_path / "ledger.sqlite"
        self._record(path, replace(self._published(), prematch=(0.52, 0.26, 0.22)))
        _, (entry,) = self._record(
            path, replace(self._published(), prematch=(0.9, 0.05, 0.05)), hours_before=-1
        )
        assert entry.prematch == pytest.approx((0.52, 0.26, 0.22))

    def test_a_ledger_from_before_the_columns_upgrades_in_place(self, tmp_path) -> None:
        import sqlite3

        from soccer.domain.ledger import _ADDED_COLUMNS, ForecastLedger, LedgerDB

        path = tmp_path / "ledger.sqlite"
        self._record(path, self._published())
        con = sqlite3.connect(path)
        for column, _ddl in _ADDED_COLUMNS:
            con.execute(f"ALTER TABLE forecast DROP COLUMN {column}")
        con.commit()
        con.close()
        with LedgerDB(path) as db:
            (entry,) = ForecastLedger(db).entries()
        assert entry.prematch is None and entry.last.probs == pytest.approx((0.5, 0.3, 0.2))

    def test_grading_scores_the_odds_on_the_matches_that_had_them(self, tmp_path) -> None:
        from dataclasses import replace

        from soccer.domain.ledger import ForecastLedger, LedgerDB, grade, verdict
        from tests.test_ledger import KICKOFF as LEDGER_KICKOFF
        from tests.test_ledger import _result

        with LedgerDB(tmp_path / "ledger.sqlite") as db:
            ledger = ForecastLedger(db)
            ledger.record(
                [
                    replace(self._published(), prematch=(0.6, 0.25, 0.15), prematch_book="avg"),
                    self._published(match_id="m2", home="Fulham", away="Brentford"),
                ],
                now=LEDGER_KICKOFF - timedelta(days=1),
            )
            entries = ledger.entries()
        record = grade(
            entries,
            {"E0": [_result(), _result("Fulham", "Brentford")]},
            now=LEDGER_KICKOFF + timedelta(days=3),
        )
        s = record.summary
        assert s is not None and s.n == 2 and s.n_with_prematch == 1
        assert s.prematch is not None and s.model_on_prematch is not None
        # the home side won: the odds (60%) beat the model (50%) on that one match
        assert s.prematch.log_loss < s.model_on_prematch.log_loss
        assert "pre-match odds" in verdict(record)

    def test_fixture_forecasts_reach_the_ledger_with_their_odds(self, tmp_path) -> None:
        from soccer.dashboard.data import fixture_forecasts, record_fixture_forecasts
        from soccer.domain.ledger import ForecastLedger, LedgerDB

        analytics, live, kickoff = _seed(tmp_path, odds=None)
        with LiveDB(live) as db:
            UpcomingOddsStore(db).replace([_odds(day=kickoff.date())])
        ledger_db = tmp_path / "ledger.sqlite"
        record_fixture_forecasts(ledger_db, fixture_forecasts(live, analytics))
        with LedgerDB(ledger_db) as db:
            (entry,) = ForecastLedger(db).entries()
        assert entry.prematch == pytest.approx(vig_free(1.83, 3.75, 4.40), abs=1e-6)
        assert entry.last.probs != pytest.approx(entry.prematch)  # the model's own forecast


class _FakeSource:
    rows: ClassVar[list[UpcomingOdds]] = []
    fail = False

    def __init__(self, _raw) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def fetch_upcoming_odds(self) -> list[UpcomingOdds]:
        if self.fail:
            raise httpx.ConnectError("offline")
        return self.rows


class TestRefresh:
    def _settings(self, tmp_path):
        from soccer.config import Settings

        return Settings(data_dir=tmp_path, football_data_org_token=None, _env_file=None)

    def test_update_stores_the_odds_and_stamps_the_log(self, tmp_path, monkeypatch) -> None:
        from soccer.dashboard import actions
        from soccer.dashboard.data import data_freshness

        settings = self._settings(tmp_path)
        monkeypatch.setattr(actions, "FootballDataCoUk", _FakeSource)
        monkeypatch.setattr(_FakeSource, "rows", [_odds(day=date.today())])
        monkeypatch.setattr(_FakeSource, "fail", False)
        assert data_freshness(settings).odds_stale
        message = actions.update_upcoming_odds(settings)
        assert "1 fixtures" in message
        assert not data_freshness(settings).odds_stale
        with LiveDB(settings.live_db) as db:
            assert UpcomingOddsStore(db).count() == 1

    def test_a_failed_fetch_keeps_the_stored_odds_and_is_reported(
        self, tmp_path, monkeypatch
    ) -> None:
        from soccer.dashboard import actions
        from soccer.dashboard.data import data_freshness

        settings = self._settings(tmp_path)
        monkeypatch.setattr(actions, "FootballDataCoUk", _FakeSource)
        monkeypatch.setattr(_FakeSource, "rows", [_odds(day=date.today())])
        monkeypatch.setattr(_FakeSource, "fail", False)
        actions.update_upcoming_odds(settings)
        monkeypatch.setattr(_FakeSource, "fail", True)
        with pytest.raises(httpx.ConnectError):
            actions.update_upcoming_odds(settings)
        with LiveDB(settings.live_db) as db:
            assert UpcomingOddsStore(db).count() == 1
        fresh = data_freshness(settings)
        assert fresh.odds_stale and "ConnectError" in (fresh.last_error or "")

    def test_auto_refresh_pulls_odds_only_when_there_are_fixtures_to_price(self, tmp_path) -> None:
        from soccer.dashboard import autorefresh

        settings = self._settings(tmp_path)
        assert "odds" not in [n for n, _ in autorefresh._jobs(settings)]
        with LiveDB(settings.live_db) as db:
            add_match(
                db,
                match_id="fx1",
                home="Arsenal",
                away="Chelsea",
                competition="Premier League",
                status=MatchStatus.NOT_STARTED,
            )
        names = [n for n, _ in autorefresh._jobs(settings)]
        assert names == ["results", "odds"]


class TestAssistantNote:
    def test_says_when_the_numbers_are_the_odds(self, tmp_path) -> None:
        from soccer.dashboard.assistant import answer

        analytics, live, kickoff = _seed(tmp_path)
        plain = answer("upcoming fixtures", analytics, live_db=live).text
        assert "pre-match odds" not in plain
        with LiveDB(live) as db:
            UpcomingOddsStore(db).replace([_odds(day=kickoff.date())])
        priced = answer("upcoming fixtures", analytics, live_db=live).text
        assert "pre-match odds" in priced
