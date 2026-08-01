"""Scheduler tests.

The loop is a thin daemon; the logic worth testing is here -- what is due when, and that
a failing job is isolated (caught, reported, its clock advanced) so it neither kills the
loop nor hammers the source. A fake clock makes cadence deterministic.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from soccer.ingest.scheduler import Job, Scheduler


class Clock:
    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


def counting_job(name: str, interval_s: int, calls: list[str]):
    return Job(name, timedelta(seconds=interval_s), lambda: (calls.append(name), "ok")[1])


class TestDueness:
    def test_all_jobs_due_initially(self) -> None:
        clock = Clock(datetime(2026, 8, 1, tzinfo=UTC))
        sched = Scheduler([Job("a", timedelta(seconds=60), lambda: "x")], clock=clock)
        assert len(sched.due()) == 1

    def test_not_due_until_interval_elapses(self) -> None:
        clock = Clock(datetime(2026, 8, 1, tzinfo=UTC))
        calls: list[str] = []
        sched = Scheduler([counting_job("a", 60, calls)], clock=clock)

        sched.run_due()  # runs once
        assert calls == ["a"]

        clock.advance(30)
        assert sched.due() == []  # not yet
        sched.run_due()
        assert calls == ["a"]  # still once

        clock.advance(31)  # 61s total since last run
        sched.run_due()
        assert calls == ["a", "a"]

    def test_different_cadences(self) -> None:
        clock = Clock(datetime(2026, 8, 1, tzinfo=UTC))
        calls: list[str] = []
        sched = Scheduler(
            [counting_job("fast", 10, calls), counting_job("slow", 100, calls)], clock=clock
        )
        for _ in range(11):
            sched.run_due()
            clock.advance(10)
        # fast ran ~11 times, slow ~2 (at t=0 and t=100).
        assert calls.count("fast") >= 10
        assert 1 <= calls.count("slow") <= 2


class TestFailureIsolation:
    def test_failing_job_does_not_stop_others(self) -> None:
        clock = Clock(datetime(2026, 8, 1, tzinfo=UTC))
        calls: list[str] = []

        def boom() -> str:
            raise RuntimeError("source down")

        sched = Scheduler(
            [Job("bad", timedelta(seconds=60), boom), counting_job("good", 60, calls)],
            clock=clock,
        )
        results = sched.run_due()

        assert calls == ["good"]  # good ran despite bad failing
        by = {r.name: r for r in results}
        assert by["bad"].ok is False
        assert "source down" in by["bad"].message
        assert by["good"].ok is True

    def test_failed_job_waits_full_interval_not_hammered(self) -> None:
        clock = Clock(datetime(2026, 8, 1, tzinfo=UTC))
        attempts = {"n": 0}

        def flaky() -> str:
            attempts["n"] += 1
            raise RuntimeError("nope")

        sched = Scheduler([Job("f", timedelta(seconds=60), flaky)], clock=clock)
        sched.run_due()
        clock.advance(30)
        sched.run_due()  # not due yet -> no second attempt
        assert attempts["n"] == 1


class TestSleepHint:
    def test_seconds_until_next_after_run(self) -> None:
        clock = Clock(datetime(2026, 8, 1, tzinfo=UTC))
        sched = Scheduler([Job("a", timedelta(seconds=60), lambda: "x")], clock=clock)
        sched.run_due()
        clock.advance(20)
        assert sched.seconds_until_next() == 40.0

    def test_zero_when_a_job_is_due(self) -> None:
        clock = Clock(datetime(2026, 8, 1, tzinfo=UTC))
        sched = Scheduler([Job("a", timedelta(seconds=60), lambda: "x")], clock=clock)
        assert sched.seconds_until_next() == 0.0  # never run -> due now
