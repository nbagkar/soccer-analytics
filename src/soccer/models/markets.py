"""Betting-market slate derived from the forecast scoreline grid.

The forecast already computes the full joint distribution of scorelines; every market is
a sum over cells of that grid. This turns it into the markets people actually think in --
1X2, double chance, over/under at several lines, both-teams-to-score, clean sheets,
win-to-nil, the total-goals distribution and correct scores -- each as a probability and
a fair decimal price (1/p, no bookmaker margin). It adds no model, only reads more out of
the one already fitted.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from soccer.models.poisson import score_grid

OVER_UNDER_LINES = (0.5, 1.5, 2.5, 3.5, 4.5)


@dataclass(frozen=True)
class Market:
    name: str
    probability: float

    @property
    def fair_odds(self) -> float:
        return 1.0 / self.probability if self.probability > 0 else float("inf")


@dataclass(frozen=True)
class OverUnder:
    line: float
    over: float
    under: float


@dataclass(frozen=True)
class MarketSlate:
    home: str
    away: str
    home_expected: float
    away_expected: float
    result: list[Market]  # home / draw / away
    double_chance: list[Market]  # 1X / 12 / X2
    over_under: list[OverUnder]
    btts: list[Market]  # yes / no
    clean_sheet: list[Market]  # home / away
    win_to_nil: list[Market]  # home / away
    total_goals: list[Market]  # 0,1,2,3,4,5+
    correct_scores: list[tuple[int, int, float]]
    grid: dict[tuple[int, int], float]
    """The full scoreline distribution this slate was built from -- every market above is a
    sum over it. Kept on the slate so a same-match combo (two-plus legs at once, e.g. "home
    win AND over 2.5") can be scored as the TRUE joint probability instead of the wrong
    shortcut of multiplying the two markets' standalone probabilities, which ignores the
    correlation between them (see `combo_probability`)."""

    @property
    def most_likely_score(self) -> tuple[int, int, float]:
        return self.correct_scores[0]


def compute_markets(
    home: str, away: str, lam: float, mu: float, rho: float, *, top_scores: int = 8
) -> MarketSlate:
    grid = score_grid(lam, mu, rho)

    def p(predicate: Callable[..., bool]) -> float:
        return sum(prob for (x, y), prob in grid.items() if predicate(x, y))

    home_win = p(lambda x, y: x > y)
    draw = p(lambda x, y: x == y)
    away_win = p(lambda x, y: x < y)

    over_under = [
        OverUnder(
            line=line,
            over=p(lambda x, y, line=line: x + y > line),
            under=p(lambda x, y, line=line: x + y < line),
        )
        for line in OVER_UNDER_LINES
    ]

    btts_yes = p(lambda x, y: x >= 1 and y >= 1)
    total_goals = [Market(str(n), p(lambda x, y, n=n: x + y == n)) for n in range(5)]
    total_goals.append(Market("5+", p(lambda x, y: x + y >= 5)))

    correct = sorted(grid.items(), key=lambda kv: kv[1], reverse=True)[:top_scores]

    return MarketSlate(
        home=home,
        away=away,
        home_expected=lam,
        away_expected=mu,
        result=[Market(home, home_win), Market("Draw", draw), Market(away, away_win)],
        double_chance=[
            Market(f"{home} or Draw", home_win + draw),
            Market(f"{home} or {away}", home_win + away_win),
            Market(f"Draw or {away}", draw + away_win),
        ],
        over_under=over_under,
        btts=[Market("Yes", btts_yes), Market("No", 1.0 - btts_yes)],
        clean_sheet=[
            Market(home, p(lambda x, y: y == 0)),
            Market(away, p(lambda x, y: x == 0)),
        ],
        win_to_nil=[
            Market(home, p(lambda x, y: x > y and y == 0)),
            Market(away, p(lambda x, y: y > x and x == 0)),
        ],
        total_goals=total_goals,
        correct_scores=[(x, y, prob) for (x, y), prob in correct],
        grid=grid,
    )


LegPredicate = Callable[[int, int], bool]


def same_match_legs(home: str, away: str) -> dict[str, LegPredicate]:
    """Named same-match selections, each a predicate on a (home_goals, away_goals) scoreline.

    Mirrors the markets `compute_markets` already builds, so a single-leg "combo" reproduces
    exactly the same probability shown elsewhere on the slate. The point of naming these as
    predicates rather than precomputed probabilities is `combo_probability` below: two or
    more of these picked together must be scored jointly, not multiplied.
    """
    legs: dict[str, LegPredicate] = {
        home: lambda x, y: x > y,
        "Draw": lambda x, y: x == y,
        away: lambda x, y: x < y,
        f"{home} or Draw (1X)": lambda x, y: x >= y,
        f"{home} or {away} (12)": lambda x, y: x != y,
        f"Draw or {away} (X2)": lambda x, y: x <= y,
        "BTTS: Yes": lambda x, y: x >= 1 and y >= 1,
        "BTTS: No": lambda x, y: x == 0 or y == 0,
        f"{home} clean sheet": lambda x, y: y == 0,
        f"{away} clean sheet": lambda x, y: x == 0,
        f"{home} win to nil": lambda x, y: x > y and y == 0,
        f"{away} win to nil": lambda x, y: y > x and x == 0,
    }
    for line in OVER_UNDER_LINES:
        legs[f"Over {line}"] = _over(line)
        legs[f"Under {line}"] = _under(line)
    return legs


def _over(line: float) -> LegPredicate:
    return lambda x, y: x + y > line


def _under(line: float) -> LegPredicate:
    return lambda x, y: x + y < line


def combo_probability(grid: dict[tuple[int, int], float], legs: list[LegPredicate]) -> float:
    """True joint probability of every leg holding at once, from the scoreline grid.

    Not a product of the legs' standalone probabilities -- same-match outcomes are
    correlated (e.g. a home win makes "over 2.5" more likely than it is unconditionally,
    since a 2-0 or 3-1 is a more common route to a home win than a 1-0), so multiplying
    would misstate the true combined odds. This sums the grid over cells where every
    predicate holds, which is exact given the model's own joint distribution.
    """
    return sum(prob for (x, y), prob in grid.items() if all(leg(x, y) for leg in legs))
