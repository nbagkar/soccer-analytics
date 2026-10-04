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
            RefreshLog(LiveDB(tmp_path / "live.sqlite")).mark("squads", ok=True)


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
    autorefresh._running.clear()
    yield
    autorefresh._running.clear()


class TestAutoRefresh:
    def _wire(self, monkeypatch, settings) -> threading.Event:
        done = threading.Event()

        def fake_job(s: Settings) -> str:
            return actions._logged(s, RESULTS, lambda: "fake refresh")

        def fake_jobs(_s):
            return [("results", fake_job)]

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
    def test_fixtures_only_when_they_are_stale_too(self, tmp_path) -> None:
        settings = _settings(tmp_path, football_data_org_token="test-token")
        assert [n for n, _ in autorefresh._jobs(settings)] == ["results", "fixtures"]
        with LiveDB(settings.live_db) as db:
            RefreshLog(db).mark(FIXTURES, ok=True, message="recent")
        assert [n for n, _ in autorefresh._jobs(settings)] == ["results"]

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
