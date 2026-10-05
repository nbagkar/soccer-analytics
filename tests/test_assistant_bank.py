"""The assistant's quality score on the fixed question bank (tests/assistant_bank.py).

A ratchet: each floor is the best score recorded so far, so a change that makes the
assistant answer fewer bank questions correctly fails here. Raise a floor when a fix lifts
the score. HOLDOUT failures are deliberately not printed -- it is scored, never studied.
"""

from __future__ import annotations

from tests.assistant_bank import DEV, HOLDOUT, build_store, report, score

# Baseline 2026-10-05, before any fix: DEV 21/45, HOLDOUT 5/20.
DEV_FLOOR = 21
HOLDOUT_FLOOR = 5


def test_question_bank_score_never_drops(tmp_path) -> None:
    analytics, live = build_store(tmp_path)
    dev = score(DEV, analytics, live)
    holdout = score(HOLDOUT, analytics, live)
    dev_passed = sum(r.ok for r in dev)
    holdout_passed = sum(r.ok for r in holdout)
    print(report("DEV", dev))
    print(f"HOLDOUT: {holdout_passed}/{len(holdout)}")
    assert dev_passed >= DEV_FLOOR, report("DEV", dev)
    assert holdout_passed >= HOLDOUT_FLOOR, f"HOLDOUT {holdout_passed}/{len(holdout)}"
