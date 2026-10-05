"""Data freshness: the refresh log, staleness rules, the shared results refresh, and the
dashboard's background auto-refresh. No network -- sources and jobs are faked."""

from __future__ import annotations

import threading
import time
from datetime import UTC, date, datetime, timedelta
from typing import ClassVar

import pytest

from soccer.config import Settings
from soccer.dashboard import actions, autorefresh
from soccer.dashboard.data import data_freshness
from soccer.domain.freshness import FIXTURES, RESULTS, STALE_AFTER, RefreshLog, freshness
from soccer.storage.live_db import LiveDB
from tests.test_dashboard_data import seed_results

NOW = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)


def _settings(tmp_path, **overrides) -> Settings:
    return Settings(data_dir=tmp_path, _env_file=None, **overrides)


def _store_injuries(db: LiveDB, *, fetched: datetime) -> None:
    from soccer.domain.availability import AvailabilityStore, PlayerAvailability

    AvailabilityStore(db).replace_source(
        "fpl",
        [
            PlayerAvailability(
                source="fpl",
                team="Arsenal",
                team_norm="arsenal",
                player="Saka",
                full_name="Bukayo Saka",
                status="i",
                availability="Injured",
                chance=0,
                news="Hamstring",
                news_added=None,
                fetched_at=fetched.isoformat(),
            )
        ],
    )


class TestRefreshLog:
    def test_mark_overwrites_the_jobs_single_row(self, tmp_path) -> None:
        log = RefreshLog(LiveDB(tmp_path / "live.sqlite"))
        log.mark(RESULTS, ok=True, message="first", at=NOW - timedelta(days=1))
        log.mark(RESULTS, ok=True, message="second", at=NOW)
        run = log.last(RESULTS)
        assert run is not None and run.message == "second" and run.ran_at == NOW

    def test_last_success_ignores_a_failed_latest_run(self, tmp_path) -> None:
        log = RefreshLog(LiveDB(tmp_path / "live.sqlite"))
        log.mark(RESULTS, ok=False, message="boom", at=NOW)
        assert log.last_success(RESULTS) is None
        assert log.last(FIXTURES) is None

    def test_rejects_unknown_job(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="unknown refresh job"):
            RefreshLog(LiveDB(tmp_path / "live.sqlite")).mark("lineups", ok=True)


class TestStaleness:
    def _fresh(self, tmp_path, checked_ago: timedelta | None, *, ok: bool = True):
        db = LiveDB(tmp_path / "live.sqlite")
        if checked_ago is not None:
            RefreshLog(db).mark(RESULTS, ok=ok, message="m", at=NOW - checked_ago)
        return freshness(db, date(2026, 9, 22), now=NOW)

    def test_never_checked_is_stale(self, tmp_path) -> None:
        f = self._fresh(tmp_path, None)
        assert f.is_stale and f.checked_label == "never"

    def test_checked_within_a_day_is_fresh_even_if_last_match_is_old(self, tmp_path) -> None:
        # International break: newest result is 12 days old, but we did check -- no nagging.
        f = self._fresh(tmp_path, timedelta(hours=3))
        assert not f.is_stale and f.checked_label == "3h ago"
        assert f.results_through == date(2026, 9, 22)

    def test_checked_over_a_day_ago_is_stale(self, tmp_path) -> None:
        f = self._fresh(tmp_path, STALE_AFTER + timedelta(days=2))
        assert f.is_stale and f.checked_label == "3 days ago"

    def test_failed_refresh_stays_stale_and_reports_the_error(self, tmp_path) -> None:
        f = self._fresh(tmp_path, timedelta(minutes=5), ok=False)
        assert f.is_stale and f.last_error == "m"


class TestInjuryAndSquadStaleness:
    def test_injury_news_age_comes_from_the_data_itself(self, tmp_path) -> None:
        db = LiveDB(tmp_path / "live.sqlite")
        _store_injuries(db, fetched=NOW - timedelta(days=11))
        f = freshness(db, None, now=NOW)
        assert f.injuries_as_of == NOW - timedelta(days=11) and f.injuries_stale

    def test_recent_injury_news_is_fresh(self, tmp_path) -> None:
        db = LiveDB(tmp_path / "live.sqlite")
        _store_injuries(db, fetched=NOW - timedelta(hours=2))
        assert not freshness(db, None, now=NOW).injuries_stale

    def test_squads_go_stale_after_a_week_not_a_day(self, tmp_path) -> None:
        db = LiveDB(tmp_path / "live.sqlite")
        assert not freshness(db, None, squads_as_of=date(2026, 9, 30), now=NOW).squads_stale
        assert freshness(db, None, squads_as_of=date(2026, 9, 23), now=NOW).squads_stale
        assert freshness(db, None, now=NOW).squads_stale  # never fetched


class TestLoggedRefresh:
    def test_success_is_recorded(self, tmp_path) -> None:
        settings = _settings(tmp_path)
        assert actions._logged(settings, RESULTS, lambda: "refreshed 10") == "refreshed 10"
        with LiveDB(settings.live_db) as db:
            run = RefreshLog(db).last(RESULTS)
        assert run is not None and run.ok and run.message == "refreshed 10"

    def test_failure_is_recorded_and_reraised(self, tmp_path) -> None:
        settings = _settings(tmp_path)

        def boom() -> str:
            raise ConnectionError("offline")

        with pytest.raises(ConnectionError):
            actions._logged(settings, FIXTURES, boom)
        with LiveDB(settings.live_db) as db:
            run = RefreshLog(db).last(FIXTURES)
        assert run is not None and not run.ok and "offline" in (run.message or "")


class _FakeSource:
    """Stands in for FootballDataCoUk: records what was asked, returns no new rows."""

    calls: ClassVar[list[str]] = []

    def __init__(self, raw) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def fetch_division(self, season, division):
        _FakeSource.calls.append(f"{season}/{division}")
        return []

    def fetch_new_league(self, code):
        _FakeSource.calls.append(code)
        return []


class TestRefreshResults:
    def test_refreshes_every_loaded_league_and_stamps_the_log(self, tmp_path, monkeypatch) -> None:
        settings = _settings(tmp_path)
        seed_results(settings.analytics_db, division="E0", teams=["Arsenal", "Chelsea", "Fulham"])
        seed_results(settings.analytics_db, division="SP1", teams=["Real Madrid", "Barcelona"])
        _FakeSource.calls = []
        monkeypatch.setattr(actions, "FootballDataCoUk", _FakeSource)

        message = actions.refresh_results(settings)

        assert "no new results" in message
        assert sorted(c.split("/")[1] for c in _FakeSource.calls) == ["E0", "SP1"]
        assert not data_freshness(settings).is_stale  # the check itself counts

    def test_no_store_yet_is_a_quiet_skip(self, tmp_path) -> None:
        assert "skipped" in actions.refresh_results(_settings(tmp_path))


@pytest.fixture
def fresh_autorefresh(monkeypatch):
    """Reset the module's process-wide state so tests don't leak into each other."""
    monkeypatch.setattr(autorefresh, "_last_attempt", None)
    monkeypatch.setattr(autorefresh, "_warmed", True)  # no stray warm-up threads in tests
    autorefresh._running.clear()
    yield
    autorefresh._running.clear()


class TestAutoRefresh:
    def _wire(self, monkeypatch, settings) -> threading.Event:
        done = threading.Event()

        def fake_job(s: Settings) -> str:
            return actions._logged(s, RESULTS, lambda: "fake refresh")

        def fake_jobs(s: Settings):
            return [("results", fake_job)] if data_freshness(s).is_stale else []

        def run_then_signal(s):
            try:
                original_run(s)
            finally:
                done.set()

        original_run = autorefresh._run
        monkeypatch.setattr(autorefresh, "_jobs", fake_jobs)
        monkeypatch.setattr(autorefresh, "_run", run_then_signal)
        seed_results(settings.analytics_db, division="E0", teams=["Arsenal", "Chelsea", "Fulham"])
        return done

    def test_stale_store_refreshes_in_the_background(
        self, tmp_path, monkeypatch, fresh_autorefresh
    ) -> None:
        settings = _settings(tmp_path)
        done = self._wire(monkeypatch, settings)
        assert data_freshness(settings).is_stale
        assert autorefresh.start_if_stale(settings)
        assert done.wait(10)
        assert not autorefresh.is_running()
        assert not data_freshness(settings).is_stale

    def test_fresh_store_does_nothing(self, tmp_path, monkeypatch, fresh_autorefresh) -> None:
        settings = _settings(tmp_path)
        self._wire(monkeypatch, settings)
        with LiveDB(settings.live_db) as db:
            RefreshLog(db).mark(RESULTS, ok=True, message="recent")
        assert not autorefresh.start_if_stale(settings)

    def test_disabled_does_nothing(self, tmp_path, monkeypatch, fresh_autorefresh) -> None:
        settings = _settings(tmp_path, auto_refresh=False)
        self._wire(monkeypatch, settings)
        assert not autorefresh.start_if_stale(settings)

    def test_one_refresh_at_a_time_and_a_cooldown_after(
        self, tmp_path, monkeypatch, fresh_autorefresh
    ) -> None:
        settings = _settings(tmp_path)
        self._wire(monkeypatch, settings)
        autorefresh._running.set()  # another session's refresh is mid-flight
        assert not autorefresh.start_if_stale(settings)
        autorefresh._running.clear()
        # A just-failed attempt must not re-fire on the very next click.
        monkeypatch.setattr(autorefresh, "_last_attempt", time.monotonic())
        assert not autorefresh.start_if_stale(settings)


class TestAutoRefreshJobs:
    def test_only_stale_feeds_are_refreshed(self, tmp_path) -> None:
        settings = _settings(tmp_path, football_data_org_token="test-token", enable_fpl=True)
        names = [n for n, _ in autorefresh._jobs(settings)]
        assert names == ["results", "fixtures", "injuries", "squads"]  # nothing fetched yet
        with LiveDB(settings.live_db) as db:
            RefreshLog(db).mark(FIXTURES, ok=True, message="recent")
            RefreshLog(db).mark(RESULTS, ok=True, message="recent")
            _store_injuries(db, fetched=datetime.now(UTC))
        assert [n for n, _ in autorefresh._jobs(settings)] == ["squads"]

    def test_injuries_need_fpl_enabled(self, tmp_path) -> None:
        settings = _settings(tmp_path, enable_fpl=False)
        assert "injuries" not in [n for n, _ in autorefresh._jobs(settings)]

    def test_no_token_means_results_only(self, tmp_path) -> None:
        settings = _settings(tmp_path, football_data_org_token=None)
        assert [n for n, _ in autorefresh._jobs(settings)] == ["results"]

    def test_a_wedged_refresh_stops_counting_as_running(
        self, monkeypatch, fresh_autorefresh
    ) -> None:
        autorefresh._running.set()
        started = time.monotonic() - autorefresh.GIVE_UP_AFTER_SECONDS - 1
        monkeypatch.setattr(autorefresh, "_last_attempt", started)
        assert not autorefresh.is_running()


class TestDashboardSettings:
    """The dashboard runs from the package dir; it must still read the project's .env."""

    def test_env_file_var_is_honoured_from_any_directory(self, tmp_path, monkeypatch) -> None:
        import soccer.config as config

        env = tmp_path / "project" / ".env"
        env.parent.mkdir()
        env.write_text("SOCCER_FOOTBALL_DATA_ORG_TOKEN=abc\nSOCCER_ENABLE_FPL=true\n")
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        monkeypatch.delenv("SOCCER_FOOTBALL_DATA_ORG_TOKEN", raising=False)
        monkeypatch.delenv("SOCCER_ENABLE_FPL", raising=False)
        monkeypatch.setenv(config.ENV_FILE_VAR, str(env))
        monkeypatch.setattr(config, "_settings", None)
        settings = config.get_settings()
        monkeypatch.setattr(config, "_settings", None)
        assert settings.football_data_org_token == "abc" and settings.enable_fpl

    def test_dashboard_command_passes_the_env_file(self, tmp_path, monkeypatch) -> None:
        import subprocess

        from typer.testing import CliRunner

        import soccer.config as config
        from soccer.cli.main import app

        (tmp_path / ".env").write_text("SOCCER_LOG_LEVEL=INFO\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(config, "_settings", None)
        seen: dict[str, str] = {}

        def fake_run(cmd, *, cwd, env, check):
            seen.update(env)
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = CliRunner().invoke(app, ["dashboard"])
        monkeypatch.setattr(config, "_settings", None)
        assert result.exit_code == 0, result.output
        assert seen[config.ENV_FILE_VAR] == str((tmp_path / ".env").resolve())


class TestHeadlineLeagueDate:
    def test_premier_league_date_is_reported_separately(self, tmp_path) -> None:
        settings = _settings(tmp_path)
        seed_results(settings.analytics_db, division="E0", teams=["Arsenal", "Chelsea"])
        seed_results(
            settings.analytics_db, division="USA", teams=["LA Galaxy", "Seattle"], season="2026"
        )
        fresh = data_freshness(settings)
        assert fresh.headline_through is not None and fresh.results_through is not None
        assert fresh.headline_through <= fresh.results_through


class TestCachedUntilDataChanges:
    def test_repeat_calls_hit_the_cache_until_the_file_changes(self, tmp_path) -> None:
        import os

        from soccer.dashboard.data import _cached_until_data_changes

        db = tmp_path / "store.db"
        db.write_text("v1")
        calls: list[int] = []

        @_cached_until_data_changes()
        def expensive(path, division: str) -> int:
            calls.append(1)
            return len(calls)

        assert expensive(db, "E0") == 1
        assert expensive(db, "E0") == 1  # cached
        assert expensive(db, "SP1") == 2  # different args -> own entry
        stat = db.stat()
        os.utime(db, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))  # a refresh wrote it
        assert expensive(db, "E0") == 3


class TestLiveAutoRefresh:
    @pytest.fixture(autouse=True)
    def _reset(self, monkeypatch):
        monkeypatch.setattr(autorefresh, "_last_live_attempt", None)

    def _fake_scores(self, monkeypatch, calls: list[str], *, fail: bool = False) -> None:
        def fake(s: Settings) -> str:
            calls.append("live")
            if fail:
                return actions._logged(s, "live", lambda: (_ for _ in ()).throw(OSError("down")))
            return actions._logged(s, "live", lambda: "thesportsdb: 3 seen")

        monkeypatch.setattr(actions, "refresh_scores", fake)

    def test_stale_live_scores_refresh_when_viewed(self, tmp_path, monkeypatch) -> None:
        settings, calls = _settings(tmp_path), []
        self._fake_scores(monkeypatch, calls)
        assert autorefresh.refresh_live_if_stale(settings)
        assert calls == ["live"]
        assert not data_freshness(settings).live_stale

    def test_fresh_live_scores_are_left_alone(self, tmp_path, monkeypatch) -> None:
        settings, calls = _settings(tmp_path), []
        self._fake_scores(monkeypatch, calls)
        with LiveDB(settings.live_db) as db:
            RefreshLog(db).mark("live", ok=True, message="m", at=datetime.now(UTC))
        assert not autorefresh.refresh_live_if_stale(settings)
        assert calls == []

    def test_a_failure_backs_off_instead_of_retrying_every_click(
        self, tmp_path, monkeypatch
    ) -> None:
        settings, calls = _settings(tmp_path), []
        self._fake_scores(monkeypatch, calls, fail=True)
        assert not autorefresh.refresh_live_if_stale(settings)
        assert not autorefresh.refresh_live_if_stale(settings)  # within the cooldown
        assert calls == ["live"]
        assert data_freshness(settings).last_error is not None  # visible, not silent

    def test_disabled_never_fetches(self, tmp_path, monkeypatch) -> None:
        calls: list[str] = []
        self._fake_scores(monkeypatch, calls)
        assert not autorefresh.refresh_live_if_stale(_settings(tmp_path, auto_refresh=False))
        assert calls == []


class TestWarmCaches:
    def test_warm_up_fills_the_cache_the_predictions_page_hits(self, tmp_path) -> None:
        from soccer.dashboard.data import fixture_forecasts, forecast_report

        settings = _settings(tmp_path)
        seed_results(settings.analytics_db, division="E0", teams=["Arsenal", "Chelsea", "Fulham"])
        LiveDB(settings.live_db).close()
        autorefresh.warm_caches(settings)
        # Same arguments as app._cached_fixture_forecasts / _cached_forecast_report use:
        first = fixture_forecasts(settings.live_db, settings.analytics_db, limit=5000)
        assert fixture_forecasts(settings.live_db, settings.analytics_db, limit=5000) is first
        report = forecast_report(settings.analytics_db, "E0", n_seasons=6)
        assert forecast_report(settings.analytics_db, "E0", n_seasons=6) is report

    def test_fresh_start_warms_once_in_the_background(
        self, tmp_path, monkeypatch, fresh_autorefresh
    ) -> None:
        settings = _settings(tmp_path)
        seed_results(settings.analytics_db, division="E0", teams=["Arsenal", "Chelsea"])
        with LiveDB(settings.live_db) as db:
            RefreshLog(db).mark(RESULTS, ok=True, message="fresh")
        warmed = threading.Event()
        monkeypatch.setattr(autorefresh, "_warmed", False)
        monkeypatch.setattr(autorefresh, "warm_caches", lambda _s: warmed.set())
        assert not autorefresh.start_if_stale(settings)  # nothing stale to refresh...
        assert warmed.wait(5)  # ...but the cold caches still get warmed
        warmed.clear()
        autorefresh.start_if_stale(settings)
        assert not warmed.wait(0.3)  # only once per process
