"""A small cadence scheduler for unattended ingestion.

The plan wants different data on different cadences -- live scores every minute, fixtures
hourly, housekeeping daily. This models that as a set of jobs with intervals, kept
deliberately separate from the serve loop so the *logic* (what is due when, what happens
when a job fails) is pure and testable while the daemon stays a thin wrapper.

The failure rule matters: a job that raises is caught, reported, and its clock advanced
like any other, so one dead source degrades to a logged warning instead of killing the
loop or hammering the source on every tick.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta


@dataclass
class Job:
    name: str
    interval: timedelta
    run: Callable[[], str]
    """Performs the work and returns a one-line summary."""
    last_run: datetime | None = None

    def is_due(self, now: datetime) -> bool:
        return self.last_run is None or (now - self.last_run) >= self.interval


@dataclass(frozen=True)
class JobResult:
    name: str
    message: str
    ok: bool


@dataclass
class Scheduler:
    jobs: list[Job]
    clock: Callable[[], datetime] = field(default=lambda: datetime.now(UTC))

    def due(self, now: datetime | None = None) -> list[Job]:
        now = now or self.clock()
        return [job for job in self.jobs if job.is_due(now)]

    def run_due(self, now: datetime | None = None) -> list[JobResult]:
        """Run every due job, isolating failures. Each job's clock advances regardless,
        so a failing job waits its full interval before retrying rather than hammering.
        """
        now = now or self.clock()
        results: list[JobResult] = []
        for job in self.due(now):
            try:
                message = job.run()
                results.append(JobResult(job.name, message, ok=True))
            except Exception as exc:
                results.append(JobResult(job.name, f"{type(exc).__name__}: {exc}", ok=False))
            finally:
                job.last_run = now
        return results

    def seconds_until_next(self, now: datetime | None = None) -> float:
        """How long until the next job is due -- lets the loop sleep instead of spin."""
        now = now or self.clock()
        waits = [
            0.0
            if job.last_run is None
            else max(0.0, (job.last_run + job.interval - now).total_seconds())
            for job in self.jobs
        ]
        return min(waits) if waits else 0.0
