"""Local usage log tests: recording, the summary report, and the assistant intent tags."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from typer.testing import CliRunner

from soccer.config import Settings
from soccer.dashboard.assistant import answer, intent_names
from soccer.domain.usage import ACTION, ASK, FALLBACK_INTENT, PAGE, UsageDB, UsageLog, track
from soccer.storage.live_db import LiveDB
from tests.test_dashboard_data import seed_results

NOW = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)


def _log(tmp_path) -> UsageLog:
    return UsageLog(UsageDB(tmp_path / "usage.sqlite"))


class TestUsageLog:
    def test_counts_rank_most_used_first_with_last_use(self, tmp_path) -> None:
        log = _log(tmp_path)
        for days_ago in (1, 2, 3):
            log.record(PAGE, "Predictor", at=NOW - timedelta(days=days_ago))
        log.record(PAGE, "Home", at=NOW - timedelta(days=5))
        pages = log.summary(NOW - timedelta(days=30)).for_kind(PAGE)
        assert [(c.name, c.count) for c in pages] == [("Predictor", 3), ("Home", 1)]
        assert pages[0].last_at == NOW - timedelta(days=1)

    def test_window_excludes_old_events_but_tracking_since_does_not(self, tmp_path) -> None:
        log = _log(tmp_path)
        log.record(PAGE, "Records", at=NOW - timedelta(days=60))
        log.record(PAGE, "Home", at=NOW - timedelta(days=1))
        summary = log.summary(NOW - timedelta(days=30))
        assert [c.name for c in summary.for_kind(PAGE)] == ["Home"]
        assert summary.tracking_since == NOW - timedelta(days=60)

    def test_unused_lists_known_names_never_seen_in_order(self, tmp_path) -> None:
        log = _log(tmp_path)
        log.record(PAGE, "Team", at=NOW)
        summary = log.summary(NOW - timedelta(days=1))
        assert summary.unused(PAGE, ["Home", "Team", "Records"]) == ["Home", "Records"]

    def test_unanswered_questions_newest_first(self, tmp_path) -> None:
        log = _log(tmp_path)
        log.record(ASK, FALLBACK_INTENT, "who won the 1966 world cup", at=NOW - timedelta(days=2))
        log.record(ASK, FALLBACK_INTENT, "best pundit?", at=NOW - timedelta(days=1))
        log.record(ASK, "standings", at=NOW)  # answered: no text kept
        summary = log.summary(NOW - timedelta(days=30))
        assert [q for _, q in summary.unanswered] == ["best pundit?", "who won the 1966 world cup"]

    def test_rejects_unknown_kind(self, tmp_path) -> None:
        with pytest.raises(ValueError, match="unknown usage kind"):
            _log(tmp_path).record("click", "x")

    def test_empty_log_has_no_tracking_start(self, tmp_path) -> None:
        summary = _log(tmp_path).summary(NOW - timedelta(days=30))
        assert summary.counts == [] and summary.tracking_since is None


class TestTrack:
    def test_logging_a_page_view_leaves_the_live_store_untouched(self, tmp_path) -> None:
        # The live store's mtime keys the dashboard's data caches; a page view must not move
        # it, or browsing throws away seconds of cached forecasts on every click.
        settings = Settings(data_dir=tmp_path, _env_file=None)
        LiveDB(settings.live_db).close()
        before = settings.live_db.stat().st_mtime_ns
        track(settings, PAGE, "Home")
        assert settings.live_db.stat().st_mtime_ns == before
        assert settings.usage_db.exists()

    def test_records_when_enabled(self, tmp_path) -> None:
        settings = Settings(data_dir=tmp_path, _env_file=None)
        track(settings, ACTION, "refresh_scores")
        with UsageDB(settings.usage_db) as db:
            rows = db.connection.execute("SELECT kind, name FROM usage_event").fetchall()
        assert [(r["kind"], r["name"]) for r in rows] == [(ACTION, "refresh_scores")]

    def test_disabled_writes_nothing(self, tmp_path) -> None:
        settings = Settings(data_dir=tmp_path, usage_tracking=False, _env_file=None)
        track(settings, PAGE, "Home")
        assert not settings.usage_db.exists()

    def test_storage_failure_never_raises(self, tmp_path) -> None:
        # usage_db path is a directory -> sqlite cannot open it; the page must not break.
        settings = Settings(data_dir=tmp_path, _env_file=None)
        settings.usage_db.mkdir(parents=True)
        track(settings, PAGE, "Home")


class TestAssistantIntentTags:
    def _seed(self, tmp_path):
        path = tmp_path / "analytics.duckdb"
        seed_results(path, division="E0", teams=["Arsenal", "Chelsea", "Fulham", "Brentford"])
        return path

    def test_answered_question_carries_its_intent(self, tmp_path) -> None:
        reply = answer("who is top of the premier league?", self._seed(tmp_path))
        assert reply.intent == "standings"

    def test_unanswerable_question_is_tagged_fallback(self, tmp_path) -> None:
        reply = answer("what's the weather like on mars", self._seed(tmp_path))
        assert reply.intent == FALLBACK_INTENT

    def test_no_data_and_empty_question_are_tagged(self, tmp_path) -> None:
        assert answer("anything", tmp_path / "missing.duckdb").intent == "no_data"
        assert answer("   ", tmp_path / "missing.duckdb").intent == "help"

    def test_intent_names_cover_every_handler_once(self) -> None:
        names = intent_names()
        assert len(names) == len(set(names))
        assert {"forecast", "standings", "team", FALLBACK_INTENT} <= set(names)


class TestUsageCommand:
    def _run(self, tmp_path, monkeypatch, *args: str):
        import soccer.config as config
        from soccer.cli.main import app

        monkeypatch.setenv("SOCCER_DATA_DIR", str(tmp_path))
        config._settings = None
        try:
            return CliRunner().invoke(app, ["usage", *args])
        finally:
            config._settings = None

    def test_reports_usage_unused_pages_and_unanswered(self, tmp_path, monkeypatch) -> None:
        log = _log(tmp_path)
        now = datetime.now(UTC)
        log.record(PAGE, "Predictor", at=now - timedelta(days=20))
        log.record(PAGE, "Predictor", at=now)
        log.record(ASK, "forecast", at=now)
        log.record(ASK, FALLBACK_INTENT, "who is the best manager ever", at=now)
        result = self._run(tmp_path, monkeypatch, "--days", "30")
        assert result.exit_code == 0, result.output
        assert "Predictions" in result.output  # pages shown by their sidebar label
        assert "Never visited:" in result.output and "Records" in result.output
        assert "Never asked:" in result.output and "standings" in result.output
        assert "who is the best manager ever" in result.output
        assert "too early" not in result.output  # 20 days of history is enough

    def test_flags_thin_history_as_provisional(self, tmp_path, monkeypatch) -> None:
        _log(tmp_path).record(PAGE, "Home", at=datetime.now(UTC) - timedelta(days=2))
        result = self._run(tmp_path, monkeypatch)
        assert result.exit_code == 0, result.output
        assert "too early" in result.output

    def test_no_database_yet(self, tmp_path, monkeypatch) -> None:
        result = self._run(tmp_path, monkeypatch)
        assert result.exit_code == 1
        assert "No usage recorded yet" in result.output
