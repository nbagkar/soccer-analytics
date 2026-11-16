"""Line movement: how a bookmaker's early price moves to its close, and whether the model saw
it coming.

football-data.co.uk records each bookmaker's line twice: early (collected Friday for a
weekend match, Tuesday for midweek) and at the close. The close is the sharper number --
it has absorbed team news and the money that moved it. That gives the one test of a model
that does not need a betting record:

    When the model disagrees with the early line, does the close move toward the model?

If it does, the model knew something the early market had not priced yet -- the model's
"closing-line value". If the moves are a coin flip, its disagreements are noise. The model
used here is fit only on matches played at least two days before kickoff, so it cannot
have seen anything the early line had not (see ``EARLY_LAG_DAYS``).

Analysis only: a positive result is evidence of information, not a betting system.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from soccer.market.implied import Probs

_EPS = 1e-15

# The model behind the early-line comparison is fit only on matches at least this many days
# before kickoff. Early lines are collected the day before the previous round's last games
# at the latest (Friday for Saturday/Sunday, Tuesday for Wednesday), so a two-day lag keeps
# the model from knowing a result the early market could not have.
EARLY_LAG_DAYS = 2

# A disagreement this large (on one outcome, in probability) counts as the model "taking a
# view" in the direction test; smaller gaps are within rounding of the book's margin.
VIEW_THRESHOLD = 0.03


@dataclass(frozen=True)
class MovementRecord:
    model: Probs  # the lagged model (EARLY_LAG_DAYS), vig-free
    early: Probs  # the bookmaker's early line, vig-free
    close: Probs  # the same bookmaker's close, vig-free
    actual: int  # 0 home / 1 draw / 2 away


@dataclass(frozen=True)
class LineMovement:
    n: int
    early_log_loss: float
    close_log_loss: float
    model_log_loss: float
    mean_move: float
    """Average total-variation distance early -> close (0.03 = three points of probability)."""
    views: int
    """Matches where the model's largest gap to the early line was at least VIEW_THRESHOLD."""
    toward_model: float
    """Of those views where the line moved, the share that moved toward the model."""
    toward_model_se: float
    slope: float
    """Least-squares slope of the move (close - early) on the gap (model - early), home and
    away outcomes pooled: the fraction of the model's disagreement the market closed. 0 means
    the moves ignore the model; 1 means the close lands on the model."""
    best_weight_vs_early: float
    """Model weight in the best log-opinion blend with the EARLY line (0 = early line alone)."""
    blend_vs_early_log_loss: float

    @property
    def model_beats_early(self) -> bool:
        return self.model_log_loss < self.early_log_loss

    @property
    def moves_toward_model(self) -> bool:
        """True only when the direction share clears 50% by two standard errors."""
        return self.views > 0 and self.toward_model - 2 * self.toward_model_se > 0.5


def _ll(p: Probs, actual: int) -> float:
    return -math.log(max(p[actual], _EPS))


def _blend(model_p: Probs, market_p: Probs, weight: float) -> Probs:
    raw = [
        (max(m, _EPS) ** weight) * (max(k, _EPS) ** (1.0 - weight))
        for m, k in zip(model_p, market_p, strict=True)
    ]
    total = sum(raw)
    return (raw[0] / total, raw[1] / total, raw[2] / total)


def line_movement(records: list[MovementRecord], *, min_n: int = 30) -> LineMovement | None:
    """Summarize early -> close movement against the model. None below ``min_n`` matches."""
    n = len(records)
    if n < min_n:
        return None

    def mean_ll(pick: str) -> float:
        return sum(_ll(getattr(r, pick), r.actual) for r in records) / n

    moves = [0.5 * sum(abs(c - e) for c, e in zip(r.close, r.early, strict=True)) for r in records]

    views = toward = 0
    xs: list[float] = []
    ys: list[float] = []
    for r in records:
        gaps = [m - e for m, e in zip(r.model, r.early, strict=True)]
        shifts = [c - e for c, e in zip(r.close, r.early, strict=True)]
        for i in (0, 2):  # draw is determined by the other two; pooling it double-counts
            xs.append(gaps[i])
            ys.append(shifts[i])
        i = max(range(3), key=lambda k: abs(gaps[k]))
        if abs(gaps[i]) >= VIEW_THRESHOLD and shifts[i] != 0:
            views += 1
            toward += (gaps[i] > 0) == (shifts[i] > 0)
    share = toward / views if views else 0.5
    se = math.sqrt(share * (1 - share) / views) if views else 0.0
    sxx = sum(x * x for x in xs)
    slope = sum(x * y for x, y in zip(xs, ys, strict=True)) / sxx if sxx else 0.0

    weights = [s / 20 for s in range(21)]
    curve = [
        (w, sum(_ll(_blend(r.model, r.early, w), r.actual) for r in records) / n) for w in weights
    ]
    best_w, best_ll = min(curve, key=lambda p: p[1])

    return LineMovement(
        n=n,
        early_log_loss=mean_ll("early"),
        close_log_loss=mean_ll("close"),
        model_log_loss=mean_ll("model"),
        mean_move=sum(moves) / n,
        views=views,
        toward_model=share,
        toward_model_se=se,
        slope=slope,
        best_weight_vs_early=best_w,
        blend_vs_early_log_loss=best_ll,
    )
