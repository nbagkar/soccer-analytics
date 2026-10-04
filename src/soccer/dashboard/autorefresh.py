"""Refresh results and fixtures in the background when the dashboard opens on stale data.

Just-in-time instead of push-a-button: nothing refreshed current-season results unless
`soccer serve` happened to be running, so forecasts silently aged. Opening the dashboard is
the moment fresh data is actually needed, so that's when this pulls it -- in a daemon
thread, so the page renders immediately on the data already there, and the next
interaction picks up the refreshed store (cached loaders key on the DB's mtime).

One refresh per process at a time, however many browser sessions open at once. Plain
Python, no Streamlit: the thread never touches `st.*`, which is not thread-safe.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

from soccer.config import Settings
from soccer.dashboard.data import data_freshness

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_running = threading.Event()
# After an attempt (success or not), wait this long before trying again -- so an offline
# machine or a failing source doesn't re-fire the refresh on every click.
RETRY_AFTER_SECONDS = 30 * 60
# A refresh still "running" after this long is presumed wedged (football-data.org's 429
# back-off can stretch a run to many minutes): stop advertising it, and let a new one start.
GIVE_UP_AFTER_SECONDS = 20 * 60
_last_attempt: float | None = None


def is_running() -> bool:
    if not _running.is_set():
        return False
    if _last_attempt is not None and time.monotonic() - _last_attempt > GIVE_UP_AFTER_SECONDS:
        logger.warning("auto-refresh exceeded %ss; treating it as finished", GIVE_UP_AFTER_SECONDS)
        _running.clear()
        return False
    return True


def _jobs(settings: Settings) -> list[tuple[str, Callable[[Settings], str]]]:
    """Results always (football-data.co.uk, unmetered); fixtures only when they are stale too,
    since football-data.org's free tier is rate-limited and fixtures change slowly."""
    from soccer.dashboard import actions

    jobs: list[tuple[str, Callable[[Settings], str]]] = [("results", actions.refresh_results)]
    if settings.football_data_org_token and data_freshness(settings).fixtures_stale:
        jobs.append(("fixtures", actions.update_fixtures))
    return jobs


def _run(settings: Settings) -> None:
    try:
        for name, job in _jobs(settings):
            try:
                logger.info("auto-refresh %s: %s", name, job(settings))
            except Exception:  # recorded in refresh_log by the action; keep going
                logger.exception("auto-refresh %s failed", name)
    finally:
        _running.clear()


def start_if_stale(settings: Settings) -> bool:
    """Start a background refresh if results are stale and none is running. True if started."""
    if not settings.auto_refresh or not settings.analytics_db.exists():
        return False
    global _last_attempt
    with _lock:
        recent = (
            _last_attempt is not None and time.monotonic() - _last_attempt < RETRY_AFTER_SECONDS
        )
        if is_running() or recent or not data_freshness(settings).is_stale:
            return False
        _running.set()
        _last_attempt = time.monotonic()
    threading.Thread(target=_run, args=(settings,), name="soccer-autorefresh", daemon=True).start()
    return True
