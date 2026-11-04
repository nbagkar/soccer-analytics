"""Forecast ledger: recording (idempotent, frozen at kickoff, first and last kept), grading,
the result-matching edge cases, and the surfaces (data hook, CLI, assistant). No network."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta

import pytest
from typer.testing import CliRunner

from soccer.domain.ledger import (
    SMALL_SAMPLE,
    UNMATCHED_AFTER,
    FinalResult,
    ForecastLedger,
    LedgerDB,
    LedgerEntry,
    LiveOutcome,
    PublishedForecast,
    find_result,
    fixture_key,
    grade,
    summarize,
    verdict,
)
from soccer.domain.names import normalize_name

NOW = datetime(2026, 11, 4, 12, 0, tzinfo=UTC)
KICKOFF = NOW + timedelta(days=2)


def _fc(
    *,
    match_id: str | None = "m1",
    home: str = "Arsenal",
    away: str = "Chelsea",
    kickoff: datetime = KICKOFF,
    probs: tuple[float, float, float] = (0.5, 0.3, 0.2),
    version: str = "league-aaaa",
    division: str | None = "E0",
) -> PublishedForecast:
    return PublishedForecast(
        match_id=match_id,
        competition="Premier League",
        division=division,
        kickoff_utc=kickoff,
        home=home,
        away=away,
        home_norm=normalize_name(home),
        away_norm=normalize_name(away),
        p_home=probs[0],
        p_draw=probs[1],
        p_away=probs[2],
        home_xg=1.6,
        away_xg=1.0,
        model_version=version,
    )


def _ledger(tmp_path) -> ForecastLedger:
    return ForecastLedger(LedgerDB(tmp_path / "ledger.sqlite"))


class TestRecording:
    def test_first_insert_sets_first_and_last(self, tmp_path) -> None:
        ledger = _ledger(tmp_path)
        out = ledger.record([_fc()], now=NOW)
        assert out.inserted == 1
        (e,) = ledger.entries()
        assert e.first.probs == e.last.probs == (0.5, 0.3, 0.2)
        assert e.first.recorded_at == e.last.recorded_at == NOW
        assert e.fixture_key == "match:m1" and e.revisions == 1
        assert e.first.model_version == "league-aaaa" and e.first.team_news is False

    def test_recording_the_same_forecast_again_writes_nothing(self, tmp_path) -> None:
        ledger = _ledger(tmp_path)
        ledger.record([_fc()], now=NOW)
        before = ledger.entries()
        out = ledger.record([_fc()], now=NOW + timedelta(hours=3))
        assert (out.inserted, out.updated, out.unchanged) == (0, 0, 1)
        assert ledger.entries() == before
        assert ledger.count() == 1

    def test_a_changed_forecast_updates_last_and_keeps_first(self, tmp_path) -> None:
        ledger = _ledger(tmp_path)
        ledger.record([_fc()], now=NOW)
        later = NOW + timedelta(days=1)
        ledger.record([_fc(probs=(0.4, 0.3, 0.3), version="league-bbbb")], now=later)
        (e,) = ledger.entries()
        assert e.first.probs == (0.5, 0.3, 0.2) and e.first.recorded_at == NOW
        assert e.last.probs == (0.4, 0.3, 0.3) and e.last.recorded_at == later
        assert e.last.model_version == "league-bbbb" and e.revisions == 2

    def test_nothing_is_written_after_kickoff(self, tmp_path) -> None:
        ledger = _ledger(tmp_path)
        ledger.record([_fc()], now=NOW)
        after = KICKOFF + timedelta(minutes=1)
        out = ledger.record([_fc(probs=(0.1, 0.1, 0.8))], now=after)
        assert out.skipped_kicked_off == 1 and out.updated == 0
        assert ledger.entries()[0].last.probs == (0.5, 0.3, 0.2)

    def test_frozen_row_ignores_a_late_kickoff_move(self, tmp_path) -> None:
        # The stored kickoff has passed: even a feed now claiming a later kickoff can't amend it.
        ledger = _ledger(tmp_path)
        ledger.record([_fc()], now=NOW)
        after = KICKOFF + timedelta(hours=1)
        moved = _fc(kickoff=KICKOFF + timedelta(days=3), probs=(0.2, 0.2, 0.6))
        out = ledger.record([moved], now=after)
        assert out.skipped_kicked_off == 1
        (e,) = ledger.entries()
        assert e.kickoff_utc == KICKOFF and e.last.probs == (0.5, 0.3, 0.2)

    def test_forecast_for_a_fixture_already_kicked_off_is_never_inserted(self, tmp_path) -> None:
        ledger = _ledger(tmp_path)
        out = ledger.record([_fc(kickoff=NOW - timedelta(minutes=5))], now=NOW)
        assert out.skipped_kicked_off == 1 and ledger.count() == 0

    def test_a_rescheduled_kickoff_before_it_is_reached_is_followed(self, tmp_path) -> None:
        ledger = _ledger(tmp_path)
        ledger.record([_fc()], now=NOW)
        new_kickoff = KICKOFF + timedelta(days=7)
        out = ledger.record([_fc(kickoff=new_kickoff)], now=NOW + timedelta(hours=1))
        assert out.updated == 1
        (e,) = ledger.entries()
        assert e.kickoff_utc == new_kickoff and e.revisions == 1  # same forecast, moved
        # ... and a write between the old and new kickoff is still allowed.
        out = ledger.record([_fc(kickoff=new_kickoff, probs=(0.3, 0.3, 0.4))], now=KICKOFF)
        assert out.updated == 1

    def test_identity_without_a_match_id_is_division_date_and_teams(self, tmp_path) -> None:
        a = _fc(match_id=None)
        b = _fc(match_id=None, kickoff=KICKOFF + timedelta(hours=2))  # same day, new time
        assert a.fixture_key == b.fixture_key == (f"E0|{KICKOFF.date()}|arsenal|chelsea")
        assert fixture_key(None, "E0", KICKOFF, "Arsenal", "Chelsea") == a.fixture_key
        ledger = _ledger(tmp_path)
        ledger.record([a], now=NOW)
        ledger.record([b], now=NOW)
        assert ledger.count() == 1

    def test_ledger_is_its_own_file(self, tmp_path) -> None:
        from soccer.config import Settings

        settings = Settings(data_dir=tmp_path)
        assert settings.ledger_db not in (settings.live_db, settings.analytics_db)
        assert settings.ledger_db.parent == settings.analytics_db.parent


def _entry(**kw) -> LedgerEntry:
    ledger_db = kw.pop("_db")
    ledger = ForecastLedger(LedgerDB(ledger_db))
    ledger.record([_fc(**kw)], now=kw.get("kickoff", KICKOFF) - timedelta(days=1))
    return ledger.entries()[-1]


def _result(
    home: str = "Arsenal",
    away: str = "Chelsea",
    day: date | None = None,
    score: tuple[int, int] = (2, 1),
    odds: tuple[float, float, float] | None = (1.8, 3.8, 4.5),
) -> FinalResult:
    return FinalResult(
        match_date=day or KICKOFF.date(),
        home=home,
        away=away,
        home_norm=normalize_name(home),
        away_norm=normalize_name(away),
        home_goals=score[0],
        away_goals=score[1],
        closing_odds=odds,
    )


class TestMatching:
    def test_same_teams_same_day(self, tmp_path) -> None:
        e = _entry(_db=tmp_path / "l.sqlite")
        assert find_result(e, [_result()]) == _result()

    def test_result_dated_a_day_off_still_matches(self, tmp_path) -> None:
        e = _entry(_db=tmp_path / "l.sqlite")
        r = _result(day=KICKOFF.date() - timedelta(days=1))
        assert find_result(e, [r]) == r

    def test_two_days_off_does_not_match(self, tmp_path) -> None:
        e = _entry(_db=tmp_path / "l.sqlite")
        assert find_result(e, [_result(day=KICKOFF.date() + timedelta(days=2))]) is None

    def test_reversed_fixture_is_not_a_match(self, tmp_path) -> None:
        e = _entry(_db=tmp_path / "l.sqlite")
        assert find_result(e, [_result(home="Chelsea", away="Arsenal")]) is None

    def test_nearest_date_wins_and_a_tie_is_refused(self, tmp_path) -> None:
        e = _entry(_db=tmp_path / "l.sqlite")
        exact, before = _result(), _result(day=KICKOFF.date() - timedelta(days=1), score=(0, 0))
        assert find_result(e, [before, exact]) == exact
        after = _result(day=KICKOFF.date() + timedelta(days=1), score=(1, 1))
        assert find_result(e, [before, after]) is None

    def test_names_bridge_through_the_matcher(self, tmp_path) -> None:
        from soccer.dashboard.data import match_team

        # A football-data.org spelling against football-data.co.uk's terse one.
        e = _entry(_db=tmp_path / "l.sqlite", home="Nottingham Forest FC", away="Arsenal FC")
        r = _result(home="Nott'm Forest", away="Arsenal")
        assert find_result(e, [r]) is None  # exact-only can't bridge it
        assert find_result(e, [r], match_team) == r


class TestGrading:
    def test_scores_model_market_and_base_rates(self, tmp_path) -> None:
        e = _entry(_db=tmp_path / "l.sqlite")
        rec = grade(
            [e],
            {"E0": [_result()]},
            base_rates={"E0": (0.45, 0.25, 0.30)},
            now=KICKOFF + timedelta(days=1),
        )
        assert rec.n == 1 and rec.awaiting == rec.upcoming == 0
        g = rec.graded[0]
        assert g.actual == 0 and g.base == (0.45, 0.25, 0.30)
        assert g.market is not None and sum(g.market) == pytest.approx(1.0)
        s = rec.summary
        assert s is not None
        import math

        assert s.model.log_loss == pytest.approx(-math.log(0.5))
        assert s.baseline.log_loss == pytest.approx(-math.log(0.45))
        assert s.market is not None and s.n_with_odds == 1
        assert s.gap == pytest.approx(-math.log(0.5) + math.log(g.market[0]))
        assert rec.is_small and "too few" in verdict(rec)

    def test_grades_the_last_forecast_and_reports_the_first(self, tmp_path) -> None:
        ledger = _ledger(tmp_path)
        ledger.record([_fc(probs=(0.5, 0.3, 0.2))], now=NOW)
        ledger.record([_fc(probs=(0.7, 0.2, 0.1))], now=NOW + timedelta(days=1))
        rec = grade(ledger.entries(), {"E0": [_result()]}, now=KICKOFF + timedelta(hours=3))
        s = rec.summary
        assert s is not None
        assert rec.graded[0].model == (0.7, 0.2, 0.1)
        assert s.model.log_loss < s.first.log_loss

    def test_no_odds_means_no_market_comparison(self, tmp_path) -> None:
        e = _entry(_db=tmp_path / "l.sqlite")
        rec = grade([e], {"E0": [_result(odds=None)]}, now=KICKOFF + timedelta(days=1))
        s = rec.summary
        assert s is not None and s.market is None and s.gap is None and s.n_with_odds == 0

    def test_awaiting_then_unmatched_when_no_result_arrives(self, tmp_path) -> None:
        e = _entry(_db=tmp_path / "l.sqlite")
        assert grade([e], {}, now=KICKOFF + timedelta(days=1)).awaiting == 1
        late = KICKOFF + UNMATCHED_AFTER + timedelta(days=1)
        assert grade([e], {}, now=late).unmatched == 1
        assert grade([e], {}, now=NOW).upcoming == 1

    def test_live_state_grades_a_result_not_yet_in_the_history(self, tmp_path) -> None:
        e = _entry(_db=tmp_path / "l.sqlite")
        live = {"m1": LiveOutcome("finished", 0, 3)}
        rec = grade([e], {}, live=live, now=KICKOFF + timedelta(hours=3))
        assert rec.n == 1 and rec.graded[0].actual == 2
        assert rec.graded[0].result.source == "live" and rec.graded[0].market is None

    def test_history_beats_live_state(self, tmp_path) -> None:
        e = _entry(_db=tmp_path / "l.sqlite")
        live = {"m1": LiveOutcome("finished", 0, 3)}
        rec = grade([e], {"E0": [_result()]}, live=live, now=KICKOFF + timedelta(days=1))
        assert rec.graded[0].result.source == "results" and rec.graded[0].actual == 0

    @pytest.mark.parametrize("status", ["postponed", "cancelled", "awarded"])
    def test_called_off_matches_are_void(self, tmp_path, status) -> None:
        e = _entry(_db=tmp_path / "l.sqlite")
        live = {"m1": LiveOutcome(status, None, None)}
        rec = grade([e], {}, live=live, now=KICKOFF + timedelta(hours=3))
        assert rec.void == 1 and rec.n == 0

    def test_in_play_live_state_is_still_awaiting(self, tmp_path) -> None:
        e = _entry(_db=tmp_path / "l.sqlite")
        live = {"m1": LiveOutcome("second_half", 1, 0)}
        assert grade([e], {}, live=live, now=KICKOFF + timedelta(hours=1)).awaiting == 1

    def test_by_version_splits_the_record(self, tmp_path) -> None:
        db = tmp_path / "l.sqlite"
        a = _entry(_db=db, match_id="a", version="league-old")
        b = _entry(_db=db, match_id="b", home="Fulham", away="Brentford", version="league-new")
        results = {"E0": [_result(), _result(home="Fulham", away="Brentford", score=(0, 0))]}
        rec = grade([a, b], results, now=KICKOFF + timedelta(days=1))
        versions = dict(rec.by_version())
        assert set(versions) == {"league-old", "league-new"}
        assert versions["league-old"].n == versions["league-new"].n == 1

    def test_gap_standard_error_and_large_sample_verdict(self, tmp_path) -> None:
        db = tmp_path / "l.sqlite"
        base = _entry(_db=db)
        graded = grade([base], {"E0": [_result()]}, now=KICKOFF + timedelta(days=1)).graded
        many = graded * (SMALL_SAMPLE + 5)
        s = summarize(many)
        assert s is not None and s.n == SMALL_SAMPLE + 5
        assert s.gap_se == pytest.approx(0.0)  # identical matches: no spread in the gap
        rec = replace(
            grade([base], {"E0": [_result()]}, now=KICKOFF + timedelta(days=1)), graded=many
        )
        assert not rec.is_small and "too few" not in verdict(rec)

    def test_empty_ledger_verdict_is_honest(self) -> None:
        rec = grade([], {}, now=NOW)
        assert rec.summary is None and "No forecasts recorded yet" in verdict(rec)


# --- The dashboard hook: fixture forecasts -> ledger -> graded against DuckDB ------------


def _seed_store(tmp_path):
    """E0 history plus one upcoming fixture in the live store."""
    from soccer.domain.match_state import MatchStatus
    from soccer.storage.live_db import LiveDB
    from tests.test_dashboard_data import add_match, seed_results

    analytics, live = tmp_path / "analytics.duckdb", tmp_path / "live.sqlite"
    teams = ["Arsenal", "Chelsea", "Fulham", "Brentford", "Everton", "Wolves", "Burnley", "Leeds"]
    seed_results(analytics, division="E0", teams=teams)
    with LiveDB(live) as db:
        add_match(
            db,
            match_id="fx1",
            home="Arsenal FC",
            away="Chelsea FC",
            competition="Premier League",
            status=MatchStatus.NOT_STARTED,
        )
    return analytics, live


class TestDashboardHook:
    def test_fixture_forecasts_are_recorded_once(self, tmp_path) -> None:
        from soccer.dashboard.data import (
            fixture_forecasts,
            forecast_model_version,
            record_fixture_forecasts,
        )

        analytics, live = _seed_store(tmp_path)
        forecasts = fixture_forecasts(live, analytics, limit=50)
        assert forecasts and forecasts[0].slate is not None
        assert forecasts[0].division == "E0" and forecasts[0].home_norm == "arsenal"
        ledger_db = tmp_path / "ledger.sqlite"
        first = record_fixture_forecasts(ledger_db, forecasts)
        again = record_fixture_forecasts(ledger_db, forecasts)
        assert first is not None and first.inserted == 1
        assert again is not None and again.inserted == 0 and again.unchanged == 1
        with LedgerDB(ledger_db) as db:
            (e,) = ForecastLedger(db).entries()
        assert e.match_id == forecasts[0].match_id and e.division == "E0"
        assert e.home == "Arsenal" and e.last.model_version == forecast_model_version()
        assert sum(e.last.probs) == pytest.approx(1.0, abs=1e-4)

    def test_recording_does_not_touch_the_data_stores(self, tmp_path) -> None:
        from soccer.dashboard.data import fixture_forecasts, record_fixture_forecasts

        analytics, live = _seed_store(tmp_path)
        forecasts = fixture_forecasts(live, analytics, limit=50)
        before = (analytics.stat().st_mtime_ns, live.stat().st_mtime_ns)
        record_fixture_forecasts(tmp_path / "ledger.sqlite", forecasts)
        assert (analytics.stat().st_mtime_ns, live.stat().st_mtime_ns) == before

    def test_a_ledger_failure_never_raises(self, tmp_path) -> None:
        from soccer.dashboard.data import fixture_forecasts, record_fixture_forecasts

        analytics, live = _seed_store(tmp_path)
        forecasts = fixture_forecasts(live, analytics, limit=50)
        blocker = tmp_path / "not-a-dir"
        blocker.write_text("x")
        assert record_fixture_forecasts(blocker / "ledger.sqlite", forecasts) is None

    def test_model_version_changes_with_a_forecast_constant(self, monkeypatch) -> None:
        from soccer.dashboard import data

        before = data.forecast_model_version()
        assert before.startswith("league-") and data.forecast_model_version(cup=True) != before
        monkeypatch.setattr(data, "FORECAST_SPREAD", data.FORECAST_SPREAD + 0.01)
        assert data.forecast_model_version() != before

    def test_live_record_grades_against_duckdb_results(self, tmp_path) -> None:
        from soccer.dashboard.data import live_record
        from soccer.sources.football_data_co_uk import MatchResult
        from soccer.storage.analytics_db import AnalyticsDB

        analytics, live = _seed_store(tmp_path)
        ledger_db = tmp_path / "ledger.sqlite"
        played = datetime.now(UTC) - timedelta(days=2)
        with LedgerDB(ledger_db) as db:
            ForecastLedger(db).record(
                [
                    _fc(match_id=None, kickoff=played, home="Arsenal FC", away="Chelsea FC"),
                    _fc(match_id=None, kickoff=played, home="Fulham", away="Brentford"),
                ],
                now=played - timedelta(days=1),
            )

        def result(home: str, away: str, hg: int, ag: int, odds: bool) -> MatchResult:
            return MatchResult(
                season="2627",
                division="E0",
                match_date=played.date(),
                home=home,
                away=away,
                home_norm=normalize_name(home),
                away_norm=normalize_name(away),
                fthg=hg,
                ftag=ag,
                ftr="H" if hg > ag else "A" if ag > hg else "D",
                **dict.fromkeys(
                    (
                        "hthg",
                        "htag",
                        "home_shots",
                        "away_shots",
                        "home_shots_target",
                        "away_shots_target",
                        "home_corners",
                        "away_corners",
                        "home_yellows",
                        "away_yellows",
                        "home_reds",
                        "away_reds",
                        "referee",
                    )
                ),
                close_home_odds=2.0 if odds else None,
                close_draw_odds=3.5 if odds else None,
                close_away_odds=3.8 if odds else None,
            )

        with AnalyticsDB(analytics) as adb:
            adb.load_results(
                [
                    result("Arsenal", "Chelsea", 1, 1, True),
                    result("Fulham", "Brentford", 0, 2, False),
                ]
            )
        rec = live_record(ledger_db, analytics, live)
        assert rec.n == 2
        s = rec.summary
        assert s is not None and s.n_with_odds == 1
        # Base rates come from the seeded history (all before the ledger's matches).
        assert rec.graded[0].base != (1 / 3, 1 / 3, 1 / 3)

    def test_live_record_without_a_ledger_is_empty(self, tmp_path) -> None:
        from soccer.dashboard.data import live_record

        rec = live_record(tmp_path / "none.sqlite", tmp_path / "a.duckdb", None)
        assert rec.recorded == 0 and rec.summary is None


# --- Surfaces: CLI and assistant -------------------------------------------------------


def _cli(tmp_path, monkeypatch, *args: str):
    import soccer.config as config
    from soccer.cli.main import app

    monkeypatch.setenv("SOCCER_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SOCCER_ENV_FILE", str(tmp_path / "none.env"))
    monkeypatch.setattr(config, "_settings", None)
    try:
        return CliRunner().invoke(app, ["ledger", *args])
    finally:
        monkeypatch.setattr(config, "_settings", None)


class TestSurfaces:
    def test_cli_without_a_ledger_says_so(self, tmp_path, monkeypatch) -> None:
        result = _cli(tmp_path, monkeypatch)
        assert result.exit_code == 1 and "No forecasts recorded yet" in result.output

    def test_cli_reports_recorded_and_graded(self, tmp_path, monkeypatch) -> None:
        analytics, live = _seed_store(tmp_path)
        from soccer.dashboard.data import fixture_forecasts, record_fixture_forecasts

        record_fixture_forecasts(
            tmp_path / "ledger.sqlite", fixture_forecasts(live, analytics, limit=50)
        )
        result = _cli(tmp_path, monkeypatch)
        assert result.exit_code == 0, result.output
        out = " ".join(result.output.split())  # rich wraps at the terminal width
        assert "1 forecasts recorded" in out and "1 still to play" in out

    def test_assistant_answers_from_the_live_record(self, tmp_path) -> None:
        from soccer.dashboard.assistant import answer

        analytics, live = _seed_store(tmp_path)
        reply = answer("how accurate have the predictions been?", analytics, live)
        assert reply.intent == "track_record"
        assert "Live record" in reply.text and "No forecasts recorded yet" in reply.text
        # Nothing graded yet, so it adds the backtest's (here: no-odds) answer.
        assert "odds" in reply.text.lower()

    def test_backtest_questions_still_go_to_the_model_intent(self, tmp_path) -> None:
        from soccer.dashboard.assistant import answer

        analytics, live = _seed_store(tmp_path)
        assert answer("how accurate is your model", analytics, live).intent == "model"
