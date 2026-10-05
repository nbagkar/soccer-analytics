"""`soccer serve` runs the same refreshes as the dashboard, and skips fresh injury/squad data."""

from __future__ import annotations

from datetime import UTC, datetime

from typer.testing import CliRunner

from soccer.dashboard import actions
from soccer.domain.freshness import RefreshLog
from soccer.storage.live_db import LiveDB


def _run_once(tmp_path, monkeypatch, *, token: str | None, fpl: bool) -> tuple[str, list[str]]:
    import soccer.config as config
    from soccer.cli.main import app

    called: list[str] = []
    for name in (
        "refresh_scores",
        "update_fixtures",
        "refresh_results",
        "update_availability",
        "update_squads",
    ):
        monkeypatch.setattr(
            actions, name, lambda _s, _n=name, **_kw: called.append(_n) or f"{_n} ok"
        )
    monkeypatch.setenv("SOCCER_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SOCCER_ENV_FILE", str(tmp_path / "none.env"))
    if token:
        monkeypatch.setenv("SOCCER_FOOTBALL_DATA_ORG_TOKEN", token)
    else:
        monkeypatch.delenv("SOCCER_FOOTBALL_DATA_ORG_TOKEN", raising=False)
    monkeypatch.setenv("SOCCER_ENABLE_FPL", "true" if fpl else "false")
    monkeypatch.setattr(config, "_settings", None)
    try:
        result = CliRunner().invoke(app, ["serve", "--once"])
    finally:
        monkeypatch.setattr(config, "_settings", None)
    assert result.exit_code == 0, result.output
    return result.output, called


def test_every_job_delegates_to_the_shared_actions(tmp_path, monkeypatch) -> None:
    _out, called = _run_once(tmp_path, monkeypatch, token="t", fpl=True)
    assert set(called) == {
        "refresh_scores",
        "update_fixtures",
        "refresh_results",
        "update_availability",
        "update_squads",
    }


def test_fresh_injury_news_and_missing_credentials_are_skipped(tmp_path, monkeypatch) -> None:
    from tests.test_freshness import _store_injuries

    with LiveDB(tmp_path / "live.sqlite") as db:
        _store_injuries(db, fetched=datetime.now(UTC))
        RefreshLog(db).mark("results", ok=True, message="x")
    out, called = _run_once(tmp_path, monkeypatch, token=None, fpl=True)
    assert "update_availability" not in called  # fetched just now
    assert "update_fixtures" not in called and "update_squads" not in called  # no token
    assert "fresh, skipped" in out
