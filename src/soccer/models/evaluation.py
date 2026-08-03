"""Forecast evaluation: proper scoring, market comparison, and model-market blending.

The honest backtest finding is that the model loses to the closing line on log loss. This
module turns that into something actionable. On the same walk-forward predictions it scores
the model, the vig-free market, and their blend -- with RPS (the standard ordered-outcome
metric for 1X2, kinder to confident near-misses than log loss) alongside log loss and Brier.
It also surfaces where model and market most disagree, because that divergence is the only
place an edge can live.

No leakage: each match is predicted from a model fit on only earlier matches. The blend
weight is reported across the whole 0..1 grid (not merely its argmin), so a gain reads as a
robust trend rather than an overfit point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from soccer.models.backtest import CalibrationBin, _calibration, _outcome_index
from soccer.models.value import implied_probabilities

_EPS = 1e-15
Probs = tuple[float, float, float]


def rps(probs: Probs, actual: int) -> float:
    """Ranked Probability Score for ordered outcomes home<draw<away. 0 = perfect, 1 = worst.

    Rewards being close on the ordinal scale (calling a draw when the away team wins is a
    smaller error than calling a home win). Constantinou & Fenton (2012) argue it is the
    right metric for 1X2 forecasts, where log loss over-penalises confident near-misses.
    """
    cum_p = cum_o = total = 0.0
    for i in range(2):  # r-1 = 2 partial sums for 3 ordered outcomes
        cum_p += probs[i]
        cum_o += 1.0 if actual == i else 0.0
        total += (cum_p - cum_o) ** 2
    return total / 2.0


def _log_loss(probs: Probs, actual: int) -> float:
    return -math.log(max(probs[actual], _EPS))


def _brier(probs: Probs, actual: int) -> float:
    return sum((p - (1.0 if i == actual else 0.0)) ** 2 for i, p in enumerate(probs))


def blend_probs(model_p: Probs, market_p: Probs, weight: float) -> Probs:
    """Log-opinion pool of two probability vectors. weight=1 -> pure model, 0 -> pure market.

    A geometric (log) pool multiplies the opinions and renormalises -- the natural, sharper,
    better-calibrated combination for probabilities when both inputs carry information.
    """
    raw = [
        (max(m, _EPS) ** weight) * (max(k, _EPS) ** (1.0 - weight))
        for m, k in zip(model_p, market_p, strict=True)
    ]
    total = sum(raw)
    return (raw[0] / total, raw[1] / total, raw[2] / total)


@dataclass(frozen=True)
class Score:
    log_loss: float
    rps: float
    brier: float


@dataclass(frozen=True)
class BlendPoint:
    weight: float  # model weight: 0 = pure market, 1 = pure model
    log_loss: float
    rps: float


@dataclass(frozen=True)
class Divergence:
    home: str
    away: str
    model: Probs
    market: Probs
    actual: int  # 0 home / 1 draw / 2 away
    distance: float  # total-variation distance between model and market


@dataclass(frozen=True)
class ForecastReport:
    n: int
    model: Score
    market: Score
    baseline: Score
    blend_curve: list[BlendPoint]
    best_weight: float
    blend: Score  # blend scored at best_weight
    model_calibration: list[CalibrationBin]
    blend_calibration: list[CalibrationBin]
    divergences: list[Divergence]

    @property
    def blend_beats_market(self) -> bool:
        return self.blend.log_loss < self.market.log_loss


def _score(preds: list[tuple[Probs, int]]) -> Score:
    n = len(preds)
    return Score(
        log_loss=sum(_log_loss(p, a) for p, a in preds) / n,
        rps=sum(rps(p, a) for p, a in preds) / n,
        brier=sum(_brier(p, a) for p, a in preds) / n,
    )


def evaluate_forecasts(
    rows: list,
    *,
    model: str = "poisson",
    min_history: int = 60,
    weight_steps: int = 21,
    top_divergences: int = 12,
) -> ForecastReport | None:
    """Walk a slice, scoring model vs market vs blend vs baseline. None if too little data.

    Each match (after warmup, both teams seen, odds present) is predicted from a model fit
    on only the matches already played, then compared to the vig-free closing line.
    """
    from soccer.models.dixon_coles import fit_dixon_coles
    from soccer.models.poisson import fit_poisson

    def fit(played: list):
        return fit_poisson(played) if model == "poisson" else fit_dixon_coles(played)

    chronological = sorted(rows, key=lambda o: o.match_date)
    played: list = []
    seen: set[str] = set()
    records: list[tuple[Probs, Probs, int, object]] = []  # (model_p, market_p, actual, row)

    for o in chronological:
        eligible = (
            len(played) >= min_history
            and o.home_norm in seen
            and o.away_norm in seen
            and o.has_odds
        )
        if eligible:
            fc = fit(played).forecast(o.home_norm, o.away_norm)
            model_p = (fc.prob_home, fc.prob_draw, fc.prob_away)
            market_p = implied_probabilities(
                o.close_home_odds, o.close_draw_odds, o.close_away_odds
            )
            records.append((model_p, market_p, _outcome_index(o.fthg, o.ftag), o))
        played.append(o)
        seen.update((o.home_norm, o.away_norm))

    if len(records) < 30:
        return None

    n = len(records)
    model_preds = [(m, a) for m, _k, a, _o in records]
    market_preds = [(k, a) for _m, k, a, _o in records]

    counts = [0, 0, 0]
    for _m, _k, a, _o in records:
        counts[a] += 1
    base = (counts[0] / n, counts[1] / n, counts[2] / n)
    baseline_preds = [(base, a) for _m, _k, a, _o in records]

    # Blend-weight curve (0 = pure market ... 1 = pure model), scored by log loss + RPS.
    curve = []
    for s in range(weight_steps):
        w = s / (weight_steps - 1)
        sc = _score([(blend_probs(m, k, w), a) for m, k, a, _o in records])
        curve.append(BlendPoint(weight=w, log_loss=sc.log_loss, rps=sc.rps))
    best = min(curve, key=lambda p: p.log_loss)
    best_blend = [(blend_probs(m, k, best.weight), a) for m, k, a, _o in records]

    divergences = sorted(
        (
            Divergence(
                home=o.home,
                away=o.away,
                model=m,
                market=k,
                actual=a,
                distance=0.5 * sum(abs(mi - ki) for mi, ki in zip(m, k, strict=True)),
            )
            for m, k, a, o in records
        ),
        key=lambda d: d.distance,
        reverse=True,
    )[:top_divergences]

    return ForecastReport(
        n=n,
        model=_score(model_preds),
        market=_score(market_preds),
        baseline=_score(baseline_preds),
        blend_curve=curve,
        best_weight=best.weight,
        blend=_score(best_blend),
        model_calibration=_calibration(model_preds),
        blend_calibration=_calibration(best_blend),
        divergences=divergences,
    )
