"""Walk-forward backtesting and calibration.

The plan is explicit that forecasts must be judged by calibration and log loss, not
headline accuracy -- so this measures exactly that, honestly. It walks a season in date
order, and for each match (after a warmup) fits the Poisson model on only the matches
already played, predicts, and scores the prediction against what actually happened. No
match ever informs its own forecast.

Two things make the numbers meaningful:
* A **baseline**: the season's outcome base rates (home/draw/away frequencies) predicted
  constantly. Beating it means the model's team-specific information adds real skill
  beyond simply knowing home advantage exists.
* A **calibration table**: are matches the model calls 70%-home actually won by the home
  side ~70% of the time? Discrimination (log loss) and calibration are different virtues;
  a model can rank well yet be over-confident.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Protocol

from soccer.models.poisson import Outcome, fit_poisson

_EPS = 1e-15  # clamp probabilities out of log's singularity


class DatedOutcome(Outcome, Protocol):
    """An Outcome that also carries a date, so the walk can order matches in time."""

    match_date: date


@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    mean_predicted: float
    observed_rate: float
    count: int


@dataclass(frozen=True)
class BacktestResult:
    n_predictions: int
    log_loss: float
    brier: float
    baseline_log_loss: float
    baseline_brier: float
    calibration: list[CalibrationBin]

    @property
    def log_loss_skill(self) -> float:
        """Fractional improvement over the base-rate baseline; >0 means the model helps."""
        if self.baseline_log_loss == 0:
            return 0.0
        return (self.baseline_log_loss - self.log_loss) / self.baseline_log_loss


def _outcome_index(home_goals: int, away_goals: int) -> int:
    return 0 if home_goals > away_goals else 1 if home_goals == away_goals else 2


def _log_loss(probs: tuple[float, float, float], actual: int) -> float:
    return -math.log(max(probs[actual], _EPS))


def _brier(probs: tuple[float, float, float], actual: int) -> float:
    return sum((p - (1.0 if i == actual else 0.0)) ** 2 for i, p in enumerate(probs))


def _calibration(
    predictions: list[tuple[tuple[float, float, float], int]], bins: int = 10
) -> list[CalibrationBin]:
    """Calibration of the home-win probability, in equal-width bins."""
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for probs, actual in predictions:
        p_home = probs[0]
        idx = min(int(p_home * bins), bins - 1)
        buckets[idx].append((p_home, actual == 0))

    result: list[CalibrationBin] = []
    for i, bucket in enumerate(buckets):
        if not bucket:
            continue
        result.append(
            CalibrationBin(
                lower=i / bins,
                upper=(i + 1) / bins,
                mean_predicted=sum(p for p, _ in bucket) / len(bucket),
                observed_rate=sum(1 for _, hit in bucket if hit) / len(bucket),
                count=len(bucket),
            )
        )
    return result


def backtest_poisson(
    outcomes: list[DatedOutcome], *, min_history: int = 60, calibration_bins: int = 10
) -> BacktestResult:
    """Walk-forward backtest of the Poisson forecast over one set of results.

    `min_history` matches are used to warm up before the first prediction, and a match
    is only predicted once both its teams have appeared (the model cannot rate an unseen
    team). Refits on the growing history before each prediction -- O(n^2) but tiny at
    league-season scale.
    """
    chronological = sorted(outcomes, key=lambda o: o.match_date)
    played: list[Outcome] = []
    seen: set[str] = set()
    predictions: list[tuple[tuple[float, float, float], int]] = []

    for o in chronological:
        if len(played) >= min_history and o.home_norm in seen and o.away_norm in seen:
            fc = fit_poisson(played).forecast(o.home_norm, o.away_norm)
            probs = (fc.prob_home, fc.prob_draw, fc.prob_away)
            predictions.append((probs, _outcome_index(o.fthg, o.ftag)))
        played.append(o)
        seen.update((o.home_norm, o.away_norm))

    if not predictions:
        raise ValueError(f"no predictions made -- need more than {min_history} matches of history")

    n = len(predictions)
    # Base-rate baseline: predict the observed H/D/A frequencies for every match.
    counts = [0, 0, 0]
    for _, actual in predictions:
        counts[actual] += 1
    base = (counts[0] / n, counts[1] / n, counts[2] / n)

    return BacktestResult(
        n_predictions=n,
        log_loss=sum(_log_loss(p, a) for p, a in predictions) / n,
        brier=sum(_brier(p, a) for p, a in predictions) / n,
        baseline_log_loss=sum(_log_loss(base, a) for _, a in predictions) / n,
        baseline_brier=sum(_brier(base, a) for _, a in predictions) / n,
        calibration=_calibration(predictions, calibration_bins),
    )
