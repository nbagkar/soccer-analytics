"""Backtest of an accumulator strategy: does combining the model's picks pay off?

The dashboard's Accumulator calculator (dashboard/app.py) prices whatever legs a user
picks honestly, but honest pricing of an arbitrary bet says nothing about whether a
*strategy* -- e.g. "every week, parlay the model's two most confident Premier League
picks" -- would have actually made money. That's a different, harder question, and this
project's rule is to measure it rather than assume (see value.py's same discipline for
single bets). Walk-forward, no leakage: each match is predicted from a model fit on only
the matches already played.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from soccer.models.backtest import _outcome_index
from soccer.storage.analytics_db import OddsRow


@dataclass(frozen=True)
class ParlayBacktestResult:
    n_bets: int
    """Weeks with enough eligible matches to place a full-size accumulator."""
    legs_per_bet: int
    staked: float
    returned: float
    hits: int
    """Accumulators that won outright (every leg correct)."""
    straight_staked: float
    """The same legs, same weeks, bet individually instead of combined."""
    straight_returned: float

    @property
    def yield_pct(self) -> float:
        return 100.0 * (self.returned - self.staked) / self.staked if self.staked else 0.0

    @property
    def straight_yield_pct(self) -> float:
        if not self.straight_staked:
            return 0.0
        return 100.0 * (self.straight_returned - self.straight_staked) / self.straight_staked

    @property
    def hit_rate(self) -> float:
        return self.hits / self.n_bets if self.n_bets else 0.0


def backtest_accumulator(
    rows: list[OddsRow],
    *,
    legs_per_bet: int = 2,
    model: str = "shots",
    alpha: float = 0.25,
    shrinkage: float = 3.0,
    time_decay: float = 0.0,
    min_history: int = 60,
) -> ParlayBacktestResult | None:
    """Walk forward, one accumulator per calendar week.

    Each week, take the model's `legs_per_bet` most confident 1X2 picks (by predicted
    probability, whichever side) among that week's eligible matches, combine them into one
    accumulator priced at the product of each leg's own closing odds -- the standard way a
    straight accumulator is priced, and exactly how the dashboard's Accumulator calculator
    treats it. Weeks with fewer than `legs_per_bet` eligible matches place no bet. The same
    chosen legs are also tallied bet individually ("straight"), so the strategy can be
    judged against the direct alternative rather than a guess. None if too little data to
    fit at all.
    """
    from soccer.models.dixon_coles import fit_dixon_coles
    from soccer.models.poisson import fit_poisson, fit_poisson_shots

    def fit(played: list[OddsRow]) -> object:
        if model == "shots":
            return fit_poisson_shots(
                played, alpha=alpha, shrinkage=shrinkage, time_decay=time_decay
            )
        if model == "dixon_coles":
            return fit_dixon_coles(played)
        return fit_poisson(played)

    chronological = sorted(rows, key=lambda o: o.match_date)
    played: list[OddsRow] = []
    seen: set[str] = set()

    # ISO (year, week) -> that week's eligible picks, in the order matches were played.
    by_week: dict[tuple[int, int], list[tuple[float, float, bool]]] = defaultdict(list)

    for o in chronological:
        eligible = (
            len(played) >= min_history
            and o.home_norm in seen
            and o.away_norm in seen
            and o.has_odds
        )
        if eligible:
            # `eligible` already required `o.has_odds` -- all three are genuinely not None.
            assert o.close_home_odds is not None
            assert o.close_draw_odds is not None
            assert o.close_away_odds is not None
            fc = fit(played).forecast(o.home_norm, o.away_norm)  # type: ignore[attr-defined]
            probs = (fc.prob_home, fc.prob_draw, fc.prob_away)
            odds_row = (o.close_home_odds, o.close_draw_odds, o.close_away_odds)
            pick = max(range(3), key=lambda i: probs[i])
            actual = _outcome_index(o.fthg, o.ftag)
            iso_year, iso_week, _weekday = o.match_date.isocalendar()
            by_week[(iso_year, iso_week)].append((probs[pick], odds_row[pick], pick == actual))
        played.append(o)
        seen.update((o.home_norm, o.away_norm))

    staked = returned = straight_staked = straight_returned = 0.0
    n_bets = hits = 0

    for legs in by_week.values():
        if len(legs) < legs_per_bet:
            continue
        legs.sort(key=lambda t: t[0], reverse=True)  # most confident first
        chosen = legs[:legs_per_bet]

        n_bets += 1
        staked += 1.0
        won = all(correct for _p, _odds, correct in chosen)
        if won:
            hits += 1
            combined_odds = 1.0
            for _p, odds, _correct in chosen:
                combined_odds *= odds
            returned += combined_odds

        for _p, odds, correct in chosen:
            straight_staked += 1.0
            if correct:
                straight_returned += odds

    if n_bets == 0:
        return None

    return ParlayBacktestResult(
        n_bets=n_bets,
        legs_per_bet=legs_per_bet,
        staked=staked,
        returned=returned,
        hits=hits,
        straight_staked=straight_staked,
        straight_returned=straight_returned,
    )
