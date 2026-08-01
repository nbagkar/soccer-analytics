"""Dixon-Coles model, fitted by maximum likelihood.

The proper version of the forecast the backtest measured as only mildly skilful and
over-confident: attack/defence strengths, a home advantage and the low-score correlation
`rho` are all fitted jointly by maximising the (optionally time-decayed) likelihood of
the observed scorelines, rather than estimated by the ratio method. Time decay is
Dixon & Coles's own idea -- down-weight older matches so the model tracks current form.

Interface mirrors `PoissonModel` (`strengths`, `expected_goals`, `forecast`) so it drops
into the forecast, simulation and backtest machinery unchanged. The negative
log-likelihood is vectorised with numpy so a single fit is well under a second, which
keeps the walk-forward backtest (a fit before every match) tolerable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

from soccer.models.poisson import MatchForecast, Outcome, scoreline_forecast

_RHO_BOUND = 0.2  # keep the low-score correction from driving any cell non-positive


@dataclass(frozen=True)
class DatedOutcome(Outcome):  # structural: MatchResult / ResultRow satisfy it
    match_date: object


class DixonColesModel:
    def __init__(
        self,
        attack: dict[str, float],
        defence: dict[str, float],
        home_advantage: float,
        rho: float,
    ) -> None:
        self._attack = attack
        self._defence = defence
        self.home_advantage = home_advantage
        self.rho = rho
        # Keyed by team so callers can membership-check exactly as with PoissonModel.
        self.strengths = attack

    @property
    def teams(self) -> list[str]:
        return sorted(self._attack)

    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        if home not in self._attack or away not in self._attack:
            missing = home if home not in self._attack else away
            raise KeyError(f"Unknown team: {missing}")
        lam = math.exp(self._attack[home] - self._defence[away] + self.home_advantage)
        mu = math.exp(self._attack[away] - self._defence[home])
        return lam, mu

    def forecast(self, home: str, away: str, *, top_n: int = 5) -> MatchForecast:
        lam, mu = self.expected_goals(home, away)
        return scoreline_forecast(home, away, lam, mu, self.rho, top_n=top_n)


def fit_dixon_coles(
    outcomes: list[Outcome], *, time_decay: float = 0.0, max_iter: int = 300
) -> DixonColesModel:
    """Fit attack/defence/home/rho by maximum likelihood.

    `time_decay` (xi, per day) down-weights older matches: weight = exp(-xi * age_days).
    0 weights every match equally. Attack of the first team is fixed to 0 to remove the
    attack/defence translation invariance; predictions are unaffected by the choice.
    """
    if not outcomes:
        raise ValueError("cannot fit a model with no results")
    teams = sorted({t for o in outcomes for t in (o.home_norm, o.away_norm)})
    n = len(teams)
    if n < 2:
        raise ValueError("need at least two teams")
    idx = {t: i for i, t in enumerate(teams)}

    home_i = np.array([idx[o.home_norm] for o in outcomes])
    away_i = np.array([idx[o.away_norm] for o in outcomes])
    hg = np.array([o.fthg for o in outcomes], dtype=float)
    ag = np.array([o.ftag for o in outcomes], dtype=float)

    if time_decay > 0 and all(hasattr(o, "match_date") for o in outcomes):
        days = np.array([o.match_date.toordinal() for o in outcomes], dtype=float)  # type: ignore[attr-defined]
        weights = np.exp(-time_decay * (days.max() - days))
    else:
        weights = np.ones(len(outcomes))

    lg_hg, lg_ag = gammaln(hg + 1), gammaln(ag + 1)
    is00 = (hg == 0) & (ag == 0)
    is01 = (hg == 0) & (ag == 1)
    is10 = (hg == 1) & (ag == 0)
    is11 = (hg == 1) & (ag == 1)

    def unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
        attack = np.empty(n)
        attack[0] = 0.0
        attack[1:] = x[: n - 1]
        defence = x[n - 1 : 2 * n - 1]
        return attack, defence, x[2 * n - 1], x[2 * n]

    def neg_log_likelihood(x: np.ndarray) -> float:
        attack, defence, home, rho = unpack(x)
        lam = np.exp(attack[home_i] - defence[away_i] + home)
        mu = np.exp(attack[away_i] - defence[home_i])
        tau = np.ones_like(lam)
        tau[is00] = 1.0 - lam[is00] * mu[is00] * rho
        tau[is01] = 1.0 + lam[is01] * rho
        tau[is10] = 1.0 + mu[is10] * rho
        tau[is11] = 1.0 - rho
        tau = np.clip(tau, 1e-10, None)
        ll = weights * (hg * np.log(lam) - lam - lg_hg + ag * np.log(mu) - mu - lg_ag + np.log(tau))
        return -float(ll.sum())

    x0 = np.concatenate([np.zeros(n - 1), np.zeros(n), [0.25], [-0.1]])
    bounds = (
        [(-3.0, 3.0)] * (n - 1)  # attack
        + [(-3.0, 3.0)] * n  # defence
        + [(-1.0, 1.0)]  # home advantage
        + [(-_RHO_BOUND, _RHO_BOUND)]  # rho
    )
    result = minimize(
        neg_log_likelihood,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iter},
    )
    attack, defence, home, rho = unpack(result.x)
    return DixonColesModel(
        attack={t: float(attack[idx[t]]) for t in teams},
        defence={t: float(defence[idx[t]]) for t in teams},
        home_advantage=float(home),
        rho=float(rho),
    )
