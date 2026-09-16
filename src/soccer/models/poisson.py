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
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

DEFAULT_RHO = -0.13  # typical Dixon-Coles low-score correlation
MAX_GOALS = 10  # scoreline grid ceiling; P(>10 goals) is negligible


class Outcome(Protocol):
    """Structural type the models consume -- MatchResult satisfies it.

    Read-only properties, not plain attributes: every concrete Outcome (MatchResult,
    ResultRow, OddsRow, ...) is a frozen dataclass, and a Protocol's plain attributes are
    implicitly settable, which a frozen dataclass's fields structurally are not -- that
    mismatch is invisible at runtime (nothing ever assigns through the protocol) but fails
    strict-mode structural matching. Properties declare the read-only contract explicitly.
    """

    @property
    def home_norm(self) -> str: ...
    @property
    def away_norm(self) -> str: ...
    @property
    def fthg(self) -> int: ...
    @property
    def ftag(self) -> int: ...


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
        home_boost: dict[str, float] | None = None,
    ) -> None:
        self.strengths = strengths
        self.home_avg = home_avg
        self.away_avg = away_avg
        self.rho = rho
        self.home_boost = home_boost or {}
        """Per-team multiplier on `home_avg` when that team is playing at home -- how much
        MORE (or less) than the league-typical home advantage this specific team gets.
        League average is 1.0; empty/absent means every team uses the flat league-wide home
        advantage, which is the historical (and still default) behaviour. See
        `fit_poisson_shots`'s `home_shrinkage` for how this is estimated."""

    @property
    def teams(self) -> list[str]:
        return sorted(self.strengths)

    def add_team(self, name: str, attack: float, defence: float) -> None:
        """Register an extra team the fit never saw (e.g. a newly promoted side) with an
        explicit attack/defence prior, so it can be included in a simulation."""
        self.strengths[name] = TeamStrength(attack=attack, defence=defence)

    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        h, a = self.strengths.get(home), self.strengths.get(away)
        if h is None or a is None:
            raise KeyError(f"Unknown team(s): {home if h is None else away}")
        return (
            self.home_avg * self.home_boost.get(home, 1.0) * h.attack * a.defence,
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


def fit_poisson(outcomes: Sequence[Outcome], *, rho: float = DEFAULT_RHO) -> PoissonModel:
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


def fit_poisson_shots(
    outcomes: Sequence[Outcome],
    *,
    alpha: float = 0.5,
    rho: float = DEFAULT_RHO,
    shrinkage: float = 0.0,
    home_shrinkage: float = 0.0,
    time_decay: float = 0.0,
) -> PoissonModel:
    """Fit strengths on a shrinkage blend of goals and shots-on-target expected goals.

    Finishing is noisy; shots on target are a steadier read on how many chances a team
    actually creates and concedes. Each team-game is credited with
    ``alpha*goals + (1-alpha)*(SoT * league_conversion)`` where the conversion is the
    league's goals-per-SoT -- so the pseudo-goals preserve the league's exact goal total
    (strengths stay on the real goal scale) while shifting credit toward chance quality.
    ``alpha=1`` recovers the goals-only model; ``alpha=0`` is pure SoT expected goals. A
    match missing shot data falls back to its actual scoreline.

    ``shrinkage`` (a pseudo-match count) pulls each team's strength toward the league
    average of 1.0, so a side with only a handful of games -- a newly promoted team early
    in the season -- is regularised toward average instead of taking an extreme value from
    one lucky result. It fades as real games accumulate.

    The home/away league averages stay on actual goals, so forecasts remain goal-scaled.

    ``home_shrinkage`` (also a pseudo-match count, 0 = off) fits a per-team home-advantage
    multiplier on top of the flat league one: for each team, compare what it actually
    scored at home to what the flat-home-advantage model would have expected against its
    ACTUAL home opponents (so a tough home slate isn't mistaken for a weak home boost),
    then shrink that ratio toward 1.0 (no team effect) by this many pseudo home-games --
    same mechanic as `shrinkage`, applied to home advantage instead of attack/defence.

    ``time_decay`` (xi per day, 0 = off) down-weights older matches by
    ``exp(-xi * age_days)``, age measured from the most recent match in `outcomes` -- same
    weighting `fit_dixon_coles` uses, applied here to the ratio method's sums instead of an
    MLE. Requires every outcome to carry `match_date`; silently ignored (all games weighted
    equally) otherwise, same defensive fallback `fit_dixon_coles` uses.
    """
    if not outcomes:
        raise ValueError("cannot fit a model with no results")

    if time_decay > 0 and all(hasattr(o, "match_date") for o in outcomes):
        max_date = max(o.match_date for o in outcomes)  # type: ignore[attr-defined]
        weights = [
            math.exp(-time_decay * (max_date - o.match_date).days)  # type: ignore[attr-defined]
            for o in outcomes
        ]
    else:
        weights = [1.0] * len(outcomes)

    tot_goals = tot_sot = 0.0
    for o, w in zip(outcomes, weights, strict=True):
        hst, ast = getattr(o, "home_shots_target", None), getattr(o, "away_shots_target", None)
        if hst is not None and ast is not None:
            tot_goals += w * (o.fthg + o.ftag)
            tot_sot += w * (hst + ast)
    conv = tot_goals / tot_sot if tot_sot else 0.0

    scored: dict[str, float] = {}
    conceded: dict[str, float] = {}
    games: dict[str, float] = {}
    home_goals = away_goals = 0.0

    for o, w in zip(outcomes, weights, strict=True):
        home_goals += w * o.fthg
        away_goals += w * o.ftag
        hst, ast = getattr(o, "home_shots_target", None), getattr(o, "away_shots_target", None)
        if conv and hst is not None and ast is not None:
            h_val = alpha * o.fthg + (1 - alpha) * hst * conv
            a_val = alpha * o.ftag + (1 - alpha) * ast * conv
        else:  # no shot data for this match -> trust the scoreline
            h_val, a_val = float(o.fthg), float(o.ftag)
        for team, gf, ga in ((o.home_norm, h_val, a_val), (o.away_norm, a_val, h_val)):
            scored[team] = scored.get(team, 0.0) + w * gf
            conceded[team] = conceded.get(team, 0.0) + w * ga
            games[team] = games.get(team, 0.0) + w

    matches = sum(weights)
    overall = sum(scored.values()) / (2 * matches)  # mean pseudo-goals per team-game

    def _strength(total: float, n: float) -> float:
        # per-game rate relative to the league, shrunk toward 1.0 by `shrinkage` pseudo-games
        rate = (total / n) / overall
        return (n * rate + shrinkage) / (n + shrinkage) if shrinkage else rate

    strengths = {
        team: TeamStrength(
            attack=_strength(scored[team], games[team]),
            defence=_strength(conceded[team], games[team]),
        )
        for team in games
    }
    home_avg = home_goals / matches
    away_avg = away_goals / matches

    home_boost: dict[str, float] | None = None
    if home_shrinkage:
        observed_home: dict[str, float] = {}
        expected_home: dict[str, float] = {}
        n_home: dict[str, float] = {}
        for o, w in zip(outcomes, weights, strict=True):
            hst = getattr(o, "home_shots_target", None)
            ast = getattr(o, "away_shots_target", None)
            h_val = (
                alpha * o.fthg + (1 - alpha) * hst * conv
                if conv and hst is not None and ast is not None
                else float(o.fthg)
            )
            team = o.home_norm
            observed_home[team] = observed_home.get(team, 0.0) + w * h_val
            expected_home[team] = expected_home.get(team, 0.0) + w * (
                home_avg * strengths[team].attack * strengths[o.away_norm].defence
            )
            n_home[team] = n_home.get(team, 0.0) + w

        home_boost = {}
        for team, n in n_home.items():
            raw = observed_home[team] / expected_home[team] if expected_home[team] else 1.0
            home_boost[team] = (n * raw + home_shrinkage) / (n + home_shrinkage)

    return PoissonModel(
        strengths=strengths,
        home_avg=home_avg,
        away_avg=away_avg,
        rho=rho,
        home_boost=home_boost,
    )


def fit_poisson_xg(
    outcomes: Sequence[Outcome],
    *,
    alpha: float = 0.5,
    rho: float = DEFAULT_RHO,
    shrinkage: float = 0.0,
    time_decay: float = 0.0,
) -> PoissonModel:
    """Fit strengths on a shrinkage blend of goals and StatsBomb's own shot-quality xG.

    Same shape as `fit_poisson_shots`, but the second signal is per-shot xG (a chance's
    modelled scoring probability, StatsBomb's `statsbomb_xg`) rather than a shots-on-target
    count -- a direct chance-quality read instead of a chance-quantity proxy. xG is already
    goal-scaled, so the pseudo-goal blend needs no conversion factor:
    ``alpha*goals + (1-alpha)*xg``. ``alpha=1`` recovers the goals-only model; ``alpha=0`` is
    pure xG. A match missing xg data (StatsBomb events aren't loaded for it) falls back to
    its actual scoreline, same degrade-gracefully contract as the shots blend.

    ``shrinkage`` and ``time_decay`` behave exactly as in `fit_poisson_shots` (pseudo-match
    shrinkage toward the league average; exponential down-weighting of older matches by
    age in days). No `home_shrinkage` here -- that was a separate, already-tested experiment
    (see `fit_poisson_shots`) and isn't part of what this variant is measuring.
    """
    if not outcomes:
        raise ValueError("cannot fit a model with no results")

    if time_decay > 0 and all(hasattr(o, "match_date") for o in outcomes):
        max_date = max(o.match_date for o in outcomes)  # type: ignore[attr-defined]
        weights = [
            math.exp(-time_decay * (max_date - o.match_date).days)  # type: ignore[attr-defined]
            for o in outcomes
        ]
    else:
        weights = [1.0] * len(outcomes)

    scored: dict[str, float] = {}
    conceded: dict[str, float] = {}
    games: dict[str, float] = {}
    home_goals = away_goals = 0.0

    for o, w in zip(outcomes, weights, strict=True):
        home_goals += w * o.fthg
        away_goals += w * o.ftag
        hxg, axg = getattr(o, "home_xg", None), getattr(o, "away_xg", None)
        if hxg is not None and axg is not None:
            h_val = alpha * o.fthg + (1 - alpha) * hxg
            a_val = alpha * o.ftag + (1 - alpha) * axg
        else:  # no xg data for this match -> trust the scoreline
            h_val, a_val = float(o.fthg), float(o.ftag)
        for team, gf, ga in ((o.home_norm, h_val, a_val), (o.away_norm, a_val, h_val)):
            scored[team] = scored.get(team, 0.0) + w * gf
            conceded[team] = conceded.get(team, 0.0) + w * ga
            games[team] = games.get(team, 0.0) + w

    matches = sum(weights)
    overall = sum(scored.values()) / (2 * matches)  # mean pseudo-goals per team-game

    def _strength(total: float, n: float) -> float:
        rate = (total / n) / overall
        return (n * rate + shrinkage) / (n + shrinkage) if shrinkage else rate

    strengths = {
        team: TeamStrength(
            attack=_strength(scored[team], games[team]),
            defence=_strength(conceded[team], games[team]),
        )
        for team in games
    }
    return PoissonModel(
        strengths=strengths,
        home_avg=home_goals / matches,
        away_avg=away_goals / matches,
        rho=rho,
    )
