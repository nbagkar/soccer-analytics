"""Betting-market slate derived from the forecast scoreline grid.

The forecast already computes the full joint distribution of scorelines; every market is
a sum over cells of that grid. This turns it into the markets people actually think in --
1X2, double chance, over/under at several lines, both-teams-to-score, clean sheets,
win-to-nil, the total-goals distribution and correct scores -- each as a probability and
a fair decimal price (1/p, no bookmaker margin). It adds no model, only reads more out of
the one already fitted.
"""

from __future__ import annotations

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

    @property
    def most_likely_score(self) -> tuple[int, int, float]:
        return self.correct_scores[0]


def compute_markets(
    home: str, away: str, lam: float, mu: float, rho: float, *, top_scores: int = 8
) -> MarketSlate:
    grid = score_grid(lam, mu, rho)

    def p(predicate) -> float:
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
    )
