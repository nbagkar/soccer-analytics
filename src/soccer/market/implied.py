"""Decimal odds -> probabilities."""

from __future__ import annotations

Probs = tuple[float, float, float]


def overround(home_odds: float, draw_odds: float, away_odds: float) -> float:
    """The bookmaker's margin on a 1X2 line: sum of implied probabilities minus 1."""
    return 1.0 / home_odds + 1.0 / draw_odds + 1.0 / away_odds - 1.0


def implied_probabilities(home_odds: float, draw_odds: float, away_odds: float) -> Probs:
    """Vig-free 1X2 probabilities implied by decimal odds (normalized to sum to 1).

    Dividing out the overround (the bookmaker's margin) via simple normalization -- the
    standard first-order de-vig. The result is what the market 'thinks' before its cut.
    """
    raw = (1.0 / home_odds, 1.0 / draw_odds, 1.0 / away_odds)
    total = sum(raw)
    return (raw[0] / total, raw[1] / total, raw[2] / total)
