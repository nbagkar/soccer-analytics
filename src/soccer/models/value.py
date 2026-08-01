"""Market-edge analysis: does the model actually beat the closing line?

The honest counterpart to a "value bets" feature. There is no free source of odds for
*upcoming* matches (football-data.org's free tier returns none), so rather than fake a
live edge, this measures a real, verifiable question against history: taking the closing
1X2 odds already in the football-data.co.uk files -- with Pinnacle's close, the sharpest
public line, preferred -- would betting the model's positive-EV selections have made
money, and is the model even better calibrated than the vig-free market?

The expected answer is humbling: the closing line is very hard to beat, so yields hover
around the bookmaker margin and the market usually has the lower log loss. Surfacing that
plainly is the point -- it stops the forecasts from being mistaken for a money printer.
The same primitives (`implied_probabilities`, `expected_value`, `kelly_fraction`) power
the dashboard's what-if calculator, where the user supplies the odds they can actually
get.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

_EPS = 1e-15


def implied_probabilities(
    home_odds: float, draw_odds: float, away_odds: float
) -> tuple[float, float, float]:
    """Vig-free 1X2 probabilities implied by decimal odds (normalized to sum to 1).

    Dividing out the overround (the bookmaker's margin) via simple normalization -- the
    standard first-order de-vig. The result is what the market 'thinks' before its cut.
    """
    raw = (1.0 / home_odds, 1.0 / draw_odds, 1.0 / away_odds)
    total = sum(raw)
    return (raw[0] / total, raw[1] / total, raw[2] / total)


def overround(home_odds: float, draw_odds: float, away_odds: float) -> float:
    """The bookmaker margin: how far the implied probabilities sum above 1 (e.g. 0.05)."""
    return (1.0 / home_odds + 1.0 / draw_odds + 1.0 / away_odds) - 1.0


def expected_value(prob: float, decimal_odds: float) -> float:
    """EV per 1 unit staked at these odds given the model probability: prob*odds - 1."""
    return prob * decimal_odds - 1.0


def kelly_fraction(prob: float, decimal_odds: float) -> float:
    """Kelly-optimal stake fraction of bankroll, floored at 0 (never bet a negative edge)."""
    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    return max(0.0, (prob * b - (1.0 - prob)) / b)


@dataclass(frozen=True)
class ValueReport:
    n_matches: int
    """Matches predicted with odds present (the eligible universe)."""
    n_bets: int
    staked: float
    returned: float
    model_log_loss: float
    market_log_loss: float
    baseline_log_loss: float

    @property
    def profit(self) -> float:
        return self.returned - self.staked

    @property
    def yield_pct(self) -> float:
        """Profit as a percentage of total staked -- the headline betting result."""
        return 100.0 * self.profit / self.staked if self.staked else 0.0

    @property
    def beats_market(self) -> bool:
        """Whether the model's log loss is below the vig-free market's -- a real edge."""
        return self.model_log_loss < self.market_log_loss

    @property
    def market_edge(self) -> float:
        """How much better (─) or worse (+) the model's log loss is vs the market."""
        return self.model_log_loss - self.market_log_loss


def _outcome_index(home_goals: int, away_goals: int) -> int:
    return 0 if home_goals > away_goals else 1 if home_goals == away_goals else 2


def _ll(prob: float) -> float:
    return -math.log(max(prob, _EPS))


def value_backtest(
    rows: list,
    *,
    model: str = "dixon_coles",
    min_history: int = 60,
    edge_threshold: float = 0.0,
    time_decay: float = 0.0,
    prior: list | None = None,
) -> ValueReport:
    """Walk a season in date order, betting the model's positive-EV picks at closing odds.

    For each match (after a warmup, both teams seen, odds present) the model is fit on
    only the matches already played, then its 1X2 probabilities are compared to the
    closing line. A 1-unit bet goes on every outcome whose EV exceeds `edge_threshold`.
    Reports betting yield plus model vs market vs base-rate log loss. No match informs its
    own prediction.
    """
    from soccer.models.dixon_coles import fit_dixon_coles
    from soccer.models.poisson import fit_poisson

    def fit(played: list):
        if model == "poisson":
            return fit_poisson(played)
        return fit_dixon_coles(played, time_decay=time_decay)

    chronological = sorted(rows, key=lambda o: o.match_date)
    played: list = list(prior or [])
    seen: set[str] = {t for o in played for t in (o.home_norm, o.away_norm)}

    scored: list[tuple[tuple[float, float, float], tuple[float, float, float], int]] = []
    staked = returned = 0.0
    n_bets = 0

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
            odds = (o.close_home_odds, o.close_draw_odds, o.close_away_odds)
            market_p = implied_probabilities(*odds)
            actual = _outcome_index(o.fthg, o.ftag)
            scored.append((model_p, market_p, actual))
            for i in range(3):
                if expected_value(model_p[i], odds[i]) > edge_threshold:
                    staked += 1.0
                    n_bets += 1
                    if i == actual:
                        returned += odds[i]
        played.append(o)
        seen.update((o.home_norm, o.away_norm))

    if not scored:
        raise ValueError("no predictions with odds -- need more history or an odds-bearing slice")

    n = len(scored)
    counts = [0, 0, 0]
    for _, _, actual in scored:
        counts[actual] += 1
    base = (counts[0] / n, counts[1] / n, counts[2] / n)

    return ValueReport(
        n_matches=n,
        n_bets=n_bets,
        staked=staked,
        returned=returned,
        model_log_loss=sum(_ll(mp[a]) for mp, _, a in scored) / n,
        market_log_loss=sum(_ll(kp[a]) for _, kp, a in scored) / n,
        baseline_log_loss=sum(_ll(base[a]) for _, _, a in scored) / n,
    )
