"""How well each played match's closing prices invert into market-implied expected goals.

Every rating input of the live forecast is `market_expected_goals`: a match's vig-free 1X2
(plus the over/under 2.5 line where present) inverted through the Dixon-Coles score grid.
For prices with no exact solution the inversion falls back to scipy's closest fit. This
counts, per division, how often that happens and how far the fitted rates miss the prices,
split by how heavy the favourite was -- written to test whether the fallback explains the
Scottish Premiership's regression under opponent-adjusted ratings (it does not; see the
FORECAST_OPPONENT_ADJUST note in `soccer/dashboard/data.py`).

Usage:
    python scripts/market_inversion_diagnostics.py [--db data/analytics.duckdb]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

DIVISIONS = (
    *("E0", "E1", "E2", "E3", "SC0", "D1", "D2", "I1", "I2"),
    *("SP1", "SP2", "F1", "F2", "N1", "B1", "P1", "T1", "G1"),
)


def main() -> None:
    import scipy.optimize

    from soccer.models import poisson
    from soccer.storage.analytics_db import AnalyticsDB

    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/analytics.duckdb")
    ns = ap.parse_args()

    calls = {"n": 0}
    real_ls = scipy.optimize.least_squares

    def counting_ls(*a: Any, **k: Any) -> Any:
        calls["n"] += 1
        return real_ls(*a, **k)

    scipy.optimize.least_squares = counting_ls  # the inversions import it lazily

    print(
        f"{'div':4s} {'rows':>6s} {'fav>=.80':>8s} {'fallback':>8s} {'fb|fav80':>8s} "
        f"{'|res| mean':>10s} {'|res| p99':>9s} {'res fav80':>9s} {'fav80 dSup':>10s}"
    )
    with AnalyticsDB(Path(ns.db)) as adb:
        loaded = adb.seasons_loaded()
        for code in DIVISIONS:
            seasons = sorted({s for s, d, _n in loaded if d == code})
            stats: dict[str, list[float]] = defaultdict(list)
            for s in seasons:
                for r in adb.outcomes_with_odds(s, code):
                    odds = (r.close_home_odds, r.close_draw_odds, r.close_away_odds)
                    if any(x is None or x <= 1.0 for x in odds):
                        continue
                    inv = [1.0 / x for x in odds]
                    tot = sum(inv)
                    ph, pa = round(inv[0] / tot, 3), round(inv[2] / tot, 3)
                    over, under = r.close_over25_odds, r.close_under25_odds
                    has_ou = over is not None and under is not None and over > 1 and under > 1
                    before = calls["n"]
                    if has_ou:
                        po = round((1 / over) / (1 / over + 1 / under), 3)
                        lam, mu = poisson.implied_goal_rates_with_total.__wrapped__(ph, pa, po)
                    else:
                        lam, mu = poisson.implied_goal_rates.__wrapped__(ph, pa)
                    fb = calls["n"] > before
                    f = poisson._fast_outcome_probs(lam, mu, poisson.DEFAULT_RHO)
                    res = abs(f[0] - ph) + abs(f[1] - pa)  # 1X2 miss only (O/U is overdetermined)
                    fav = max(ph, pa) >= 0.80
                    stats["fb"].append(fb)
                    stats["fav"].append(fav)
                    stats["res"].append(res)
                    # signed supremacy miss for the favourite: + = model under-prices favourite
                    stats["dsup"].append((ph - f[0]) if ph >= pa else (pa - f[1]))
            if not stats["fb"]:
                continue
            fb_a = np.array(stats["fb"], bool)
            fav_a = np.array(stats["fav"], bool)
            res_a = np.array(stats["res"])
            dsup = np.array(stats["dsup"])
            print(
                f"{code:4s} {len(fb_a):6d} {fav_a.mean():8.3%} {fb_a.mean():8.3%} "
                f"{(fb_a[fav_a].mean() if fav_a.any() else 0):8.1%} "
                f"{res_a.mean():10.5f} {np.quantile(res_a, 0.99):9.5f} "
                f"{(res_a[fav_a].mean() if fav_a.any() else 0):9.5f} "
                f"{(dsup[fav_a].mean() if fav_a.any() else 0):+10.5f}"
            )


if __name__ == "__main__":
    main()
