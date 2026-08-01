"""Elo ratings.

A single strength number per team, updated after each match in date order, with a home
advantage and an optional margin-of-victory multiplier. Elo is a two-outcome model, so
it is used here for power rankings and a single "expected result" number (win plus half
the draw), not for a full win/draw/loss split -- the Poisson model owns that, because
deriving draws from Elo needs an ad-hoc extra model this deliberately avoids.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol


class EloOutcome(Protocol):
    home_norm: str
    away_norm: str
    fthg: int
    ftag: int
    match_date: date


@dataclass(frozen=True)
class EloConfig:
    k: float = 20.0
    home_advantage: float = 65.0  # Elo points added to the home side
    initial: float = 1500.0
    margin_scaling: bool = True


# Shared default so callers need not construct one, and so it is not a call in a
# function's argument defaults.
_DEFAULT_CONFIG = EloConfig()


@dataclass(frozen=True)
class EloRating:
    position: int
    team: str
    rating: float
    played: int


def _margin_multiplier(goal_diff: int, *, enabled: bool) -> float:
    """Down-weight nothing, up-weight blowouts (FiveThirtyEight-style)."""
    if not enabled:
        return 1.0
    g = abs(goal_diff)
    if g <= 1:
        return 1.0
    if g == 2:
        return 1.5
    return (11 + g) / 8.0


def expected_score(home_rating: float, away_rating: float, config: EloConfig) -> float:
    """Home team's expected result in [0,1] -- P(win) + 0.5*P(draw), home advantage in."""
    diff = home_rating + config.home_advantage - away_rating
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


def compute_ratings(
    outcomes: list[EloOutcome], config: EloConfig = _DEFAULT_CONFIG
) -> dict[str, float]:
    """Final Elo per team. Processes matches in date order regardless of input order."""
    ratings: dict[str, float] = {}
    for o in sorted(outcomes, key=lambda m: m.match_date):
        rh = ratings.setdefault(o.home_norm, config.initial)
        ra = ratings.setdefault(o.away_norm, config.initial)

        actual_home = 1.0 if o.fthg > o.ftag else 0.5 if o.fthg == o.ftag else 0.0
        exp_home = expected_score(rh, ra, config)
        mult = _margin_multiplier(o.fthg - o.ftag, enabled=config.margin_scaling)
        delta = config.k * mult * (actual_home - exp_home)

        ratings[o.home_norm] = rh + delta
        ratings[o.away_norm] = ra - delta
    return ratings


def power_ranking(
    outcomes: list[EloOutcome], config: EloConfig = _DEFAULT_CONFIG
) -> list[EloRating]:
    """Teams ranked by final Elo, highest first."""
    ratings = compute_ratings(outcomes, config)
    played: dict[str, int] = {}
    for o in outcomes:
        played[o.home_norm] = played.get(o.home_norm, 0) + 1
        played[o.away_norm] = played.get(o.away_norm, 0) + 1

    ordered = sorted(ratings.items(), key=lambda kv: kv[1], reverse=True)
    return [
        EloRating(position=i + 1, team=team, rating=rating, played=played.get(team, 0))
        for i, (team, rating) in enumerate(ordered)
    ]
