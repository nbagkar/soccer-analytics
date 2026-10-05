"""Refresh stale data in the background when the dashboard opens: results, fixtures, injury
news and squads, each only when it is stale.

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
    """Only the feeds that are actually stale, and only those this install can fetch.

    Results and injuries go stale in a day (matches are played; team news moves daily), squads
    in a week. Fixtures and squads share football-data.org's rate-limited free tier, so they
    are skipped when fresh rather than re-pulled alongside every results refresh.
    """
    from soccer.dashboard import actions

    fresh = data_freshness(settings)
    token = settings.football_data_org_token
    jobs: list[tuple[str, Callable[[Settings], str]]] = []
    if fresh.is_stale:
        jobs.append(("results", actions.refresh_results))
    if token and fresh.fixtures_stale:
        jobs.append(("fixtures", actions.update_fixtures))
    if settings.enable_fpl and fresh.injuries_stale:
        jobs.append(("injuries", actions.update_availability))
    if token and fresh.squads_stale:
        jobs.append(("squads", actions.update_squads))
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


# Live scores are refreshed inline (one cheap request) when someone actually looks at them;
# after an attempt, don't retry for a few minutes so an offline machine doesn't stall every
# click on a timeout.
LIVE_RETRY_AFTER_SECONDS = 5 * 60
_last_live_attempt: float | None = None


def refresh_live_if_stale(settings: Settings) -> bool:
    """Refresh live scores now if they are over 30 minutes old. True if a refresh succeeded."""
    global _last_live_attempt
    if not settings.auto_refresh or not data_freshness(settings).live_stale:
        return False
    now = time.monotonic()
    if _last_live_attempt is not None and now - _last_live_attempt < LIVE_RETRY_AFTER_SECONDS:
        return False
    _last_live_attempt = now
    from soccer.dashboard import actions

    try:
        logger.info("live auto-refresh: %s", actions.refresh_scores(settings))
    except Exception:  # recorded in refresh_log by the action; show what's cached
        logger.exception("live auto-refresh failed")
        return False
    return True


def start_if_stale(settings: Settings) -> bool:
    """Start a background refresh if any feed is stale and none is running. True if started."""
    if not settings.auto_refresh or not settings.analytics_db.exists():
        return False
    global _last_attempt
    with _lock:
        recent = (
            _last_attempt is not None and time.monotonic() - _last_attempt < RETRY_AFTER_SECONDS
        )
        if is_running() or recent or not _jobs(settings):
            return False
        _running.set()
        _last_attempt = time.monotonic()
    threading.Thread(target=_run, args=(settings,), name="soccer-autorefresh", daemon=True).start()
    return True
