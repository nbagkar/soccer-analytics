"""Poisson scoreline model (Dixon-Coles-style).

Models each team's goals as Poisson with multiplicative attack/defence strengths and a
home-advantage baseline carried by the league's home/away scoring averages. Strengths
are fit by the ratio method -- no optimiser, no scipy -- which is honest about what this
is: the Maher independent-Poisson model that Dixon & Coles (1997) extend. The one DC
touch included is the low-score correlation correction (`rho`), a closed-form tweak that
nudges 0-0/1-0/0-1/1-1 toward observed frequencies.

Deliberately NOT here (documented as the refinement): maximum-likelihood fitting of all
parameters jointly, and time-decay weighting of older matches. Those need an optimiser;
the ratio method is a principled, dependency-free starting point that a full season of
results supports well.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

DEFAULT_RHO = -0.13  # typical Dixon-Coles low-score correlation
MAX_GOALS = 10  # scoreline grid ceiling; P(>10 goals) is negligible


class Outcome(Protocol):
    """Structural type the models consume -- MatchResult satisfies it."""

    home_norm: str
    away_norm: str
    fthg: int
    ftag: int


@dataclass(frozen=True)
class TeamStrength:
    attack: float
    defence: float


@dataclass(frozen=True)
class MatchForecast:
    home: str
    away: str
    home_expected: float
    away_expected: float
    prob_home: float
    prob_draw: float
    prob_away: float
    top_scores: list[tuple[int, int, float]]
    """Most likely exact scorelines: (home_goals, away_goals, probability)."""


def _poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * lam**k / math.factorial(k)


def _dc_tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """Dixon-Coles low-score adjustment; 1.0 outside the four affected cells."""
    if x == 0 and y == 0:
        return 1.0 - lam * mu * rho
    if x == 0 and y == 1:
        return 1.0 + lam * rho
    if x == 1 and y == 0:
        return 1.0 + mu * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


class PoissonModel:
    def __init__(
        self,
        strengths: dict[str, TeamStrength],
        home_avg: float,
        away_avg: float,
        rho: float = DEFAULT_RHO,
    ) -> None:
        self.strengths = strengths
        self.home_avg = home_avg
        self.away_avg = away_avg
        self.rho = rho

    @property
    def teams(self) -> list[str]:
        return sorted(self.strengths)

    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        h, a = self.strengths.get(home), self.strengths.get(away)
        if h is None or a is None:
            raise KeyError(f"Unknown team(s): {home if h is None else away}")
        return (
            self.home_avg * h.attack * a.defence,
            self.away_avg * a.attack * h.defence,
        )

    def forecast(self, home: str, away: str, *, top_n: int = 5) -> MatchForecast:
        lam, mu = self.expected_goals(home, away)
        return scoreline_forecast(home, away, lam, mu, self.rho, top_n=top_n)


def score_grid(lam: float, mu: float, rho: float) -> dict[tuple[int, int], float]:
    """Normalized probability of every scoreline (home_goals, away_goals).

    The full joint distribution the forecast is built from. Every betting market -- 1X2,
    over/under, both-teams-to-score, correct score -- is a sum over cells of this grid,
    so exposing it lets `models/markets.py` derive them all without re-deriving the model.
    """
    grid: dict[tuple[int, int], float] = {}
    for x in range(MAX_GOALS + 1):
        for y in range(MAX_GOALS + 1):
            p = _poisson_pmf(x, lam) * _poisson_pmf(y, mu) * _dc_tau(x, y, lam, mu, rho)
            grid[(x, y)] = max(p, 0.0)  # tau can push tiny cells slightly negative
    total = sum(grid.values())
    return {k: v / total for k, v in grid.items()}  # renormalize after tau


def scoreline_forecast(
    home: str, away: str, lam: float, mu: float, rho: float, *, top_n: int = 5
) -> MatchForecast:
    """Outcome probabilities and likely scores from two goal rates and the DC rho.

    Shared by the ratio-method Poisson model and the Dixon-Coles MLE model so both turn
    (lambda, mu, rho) into a forecast identically.
    """
    grid = score_grid(lam, mu, rho)

    prob_home = sum(p for (x, y), p in grid.items() if x > y)
    prob_draw = sum(p for (x, y), p in grid.items() if x == y)
    prob_away = sum(p for (x, y), p in grid.items() if x < y)

    top = sorted(grid.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return MatchForecast(
        home=home,
        away=away,
        home_expected=lam,
        away_expected=mu,
        prob_home=prob_home,
        prob_draw=prob_draw,
        prob_away=prob_away,
        top_scores=[(x, y, p) for (x, y), p in top],
    )


def fit_poisson(outcomes: list[Outcome], *, rho: float = DEFAULT_RHO) -> PoissonModel:
    """Fit team strengths by the ratio method over a set of results.

    attack/defence are each a single multiplicative strength per team (using all of a
    team's games, home and away, for stability); the home advantage lives in the
    separate home/away league averages, so an average matchup predicts exactly the
    league's typical home and away scorelines.
    """
    if not outcomes:
        raise ValueError("cannot fit a model with no results")

    scored: dict[str, int] = {}
    conceded: dict[str, int] = {}
    games: dict[str, int] = {}
    home_goals = away_goals = 0

    for o in outcomes:
        home_goals += o.fthg
        away_goals += o.ftag
        for team, gf, ga in ((o.home_norm, o.fthg, o.ftag), (o.away_norm, o.ftag, o.fthg)):
            scored[team] = scored.get(team, 0) + gf
            conceded[team] = conceded.get(team, 0) + ga
            games[team] = games.get(team, 0) + 1

    matches = len(outcomes)
    overall = (home_goals + away_goals) / (2 * matches)  # mean goals per team-game

    strengths = {
        team: TeamStrength(
            attack=(scored[team] / games[team]) / overall,
            defence=(conceded[team] / games[team]) / overall,
        )
        for team in games
    }
    return PoissonModel(
        strengths=strengths,
        home_avg=home_goals / matches,
        away_avg=away_goals / matches,
        rho=rho,
    )
