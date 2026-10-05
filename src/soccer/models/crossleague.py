"""Cross-league forecasts for European cup ties (e.g. the Champions League).

Domestic models rate each club only against its own league -- an attack of 1.3 in the
Premier League and 1.3 in the Eredivisie are not the same thing -- so a cup tie between
leagues had no forecast at all. This keeps each club's domestic attack/defence (the
production fit) and adds one log-strength per league, plus the competition's own home and
away scoring rates, all fitted by Poisson maximum likelihood on past cup results:

    lambda_home = H * exp(S_home_league - S_away_league) * attack_home * defence_away
    mu_away     = A * exp(S_away_league - S_home_league) * attack_away * defence_home

League strengths are ridge-shrunk toward 0 (the average league), because most leagues
appear in only a few dozen cup matches. With `league_effects=False` every S is 0, which is
the "just use domestic ratings" baseline the fitted model has to beat.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class ClubRating:
    division: str
    attack: float
    defence: float


@dataclass(frozen=True)
class CupMatch:
    home: ClubRating
    away: ClubRating
    home_goals: int
    away_goals: int


@dataclass(frozen=True)
class CrossLeagueModel:
    home_rate: float
    away_rate: float
    league_strength: dict[str, float] = field(default_factory=dict)

    def expected_goals(self, home: ClubRating, away: ClubRating) -> tuple[float, float]:
        s = self.league_strength.get(home.division, 0.0) - self.league_strength.get(
            away.division, 0.0
        )
        return (
            self.home_rate * math.exp(s) * home.attack * away.defence,
            self.away_rate * math.exp(-s) * away.attack * home.defence,
        )


def fit_cross_league(
    matches: Sequence[CupMatch], *, ridge: float = 2.0, league_effects: bool = True
) -> CrossLeagueModel:
    """Fit H, A and (optionally) one strength per league by penalized Poisson likelihood."""
    if not matches:
        raise ValueError("cannot fit a cross-league model with no matches")
    leagues = sorted({m.home.division for m in matches} | {m.away.division for m in matches})
    col = {lg: i for i, lg in enumerate(leagues)}
    n = len(matches)
    hi = np.array([col[m.home.division] for m in matches])
    ai = np.array([col[m.away.division] for m in matches])
    base_h = np.log([m.home.attack * m.away.defence for m in matches])
    base_a = np.log([m.away.attack * m.home.defence for m in matches])
    gh = np.array([m.home_goals for m in matches], dtype=float)
    ga = np.array([m.away_goals for m in matches], dtype=float)

    def unpack(x: np.ndarray) -> tuple[float, float, np.ndarray]:
        s = x[2:] if league_effects else np.zeros(len(leagues))
        return float(x[0]), float(x[1]), s

    def nll(x: np.ndarray) -> float:
        log_h, log_a, s = unpack(x)
        eta_h = log_h + s[hi] - s[ai] + base_h
        eta_a = log_a + s[ai] - s[hi] + base_a
        # Poisson NLL up to constants: exp(eta) - y * eta
        value = float(np.sum(np.exp(eta_h) - gh * eta_h + np.exp(eta_a) - ga * eta_a))
        return value + ridge * float(np.sum(s**2))

    start = np.zeros(2 + (len(leagues) if league_effects else 0))
    start[0] = math.log(max(gh.mean(), 0.1))
    start[1] = math.log(max(ga.mean(), 0.1))
    result = minimize(nll, start, method="L-BFGS-B")
    log_h, log_a, s = unpack(result.x)
    # Only differences matter; centre on the matches' average league for readability.
    s = s - float(np.mean(np.concatenate([s[hi], s[ai]]))) if n else s
    return CrossLeagueModel(
        home_rate=math.exp(log_h),
        away_rate=math.exp(log_a),
        league_strength={lg: float(s[col[lg]]) for lg in leagues} if league_effects else {},
    )
