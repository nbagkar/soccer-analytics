"""Walk-forward predictions of the LIVE forecast model, per match, for the Phase C lab.

For each division and each match date from 2016/17 on: fit `_forecast_fit` on matches played
before that date (the production ~3-season window), forecast every match that day, and record
lam/mu/rho, model 1X2, vig-free close and early 1X2, the result, and both teams' rest days
(days since their previous LEAGUE match; NaN at a season's first match). Read-only on the store.
"""

from __future__ import annotations

import pickle
import sys
import time
from datetime import timedelta
from pathlib import Path

from soccer.config import get_settings
from soccer.dashboard.data import _forecast_fit
from soccer.market.implied import implied_probabilities
from soccer.models.markets import compute_markets
from soccer.sources.football_data_co_uk import season_sort_key
from soccer.storage.analytics_db import AnalyticsDB

OUT = Path(__file__).with_name("preds")
OUT.mkdir(exist_ok=True)
WINDOW = timedelta(days=3 * 365)


def build(division: str) -> list[dict]:
    with AnalyticsDB(get_settings().analytics_db) as adb:
        seasons = sorted(
            {s for s, d, _n in adb.seasons_loaded() if d == division}, key=season_sort_key
        )
        rows = [r for s in seasons for r in adb.outcomes_with_odds(s, division)]
        season_of = {}
        for s in seasons:
            for r in adb.outcomes_with_odds(s, division):
                season_of[(r.match_date, r.home_norm, r.away_norm)] = s
    rows.sort(key=lambda r: (r.match_date, r.home_norm))
    last_played: dict[str, object] = {}
    rest: dict[tuple, tuple[float, float]] = {}
    for r in rows:
        def gap(team: str) -> float:
            prev = last_played.get(team)
            if prev is None:
                return float("nan")
            days = (r.match_date - prev).days  # type: ignore[operator]
            return float(days) if days <= 30 else float("nan")  # >30 = a break / new season

        rest[(r.match_date, r.home_norm, r.away_norm)] = (gap(r.home_norm), gap(r.away_norm))
        last_played[r.home_norm] = r.match_date
        last_played[r.away_norm] = r.match_date

    out = []
    dates = sorted({r.match_date for r in rows})
    by_date: dict = {}
    for r in rows:
        by_date.setdefault(r.match_date, []).append(r)
    start = 0
    for d in dates:
        key_season = season_of[(d, by_date[d][0].home_norm, by_date[d][0].away_norm)]
        if season_sort_key(key_season) < 2016:
            continue
        train = [r for r in rows if d - WINDOW <= r.match_date < d]
        if len(train) < 150:
            continue
        model = _forecast_fit(train)
        for r in by_date[d]:
            if not r.has_odds or r.home_norm not in model.strengths or r.away_norm not in model.strengths:
                continue
            lam, mu = model.expected_goals(r.home_norm, r.away_norm)
            s = compute_markets(r.home, r.away, lam, mu, model.rho)
            rh, ra = rest[(r.match_date, r.home_norm, r.away_norm)]
            out.append(
                {
                    "division": division,
                    "season": season_of[(r.match_date, r.home_norm, r.away_norm)],
                    "date": r.match_date,
                    "home": r.home_norm,
                    "away": r.away_norm,
                    "lam": lam,
                    "mu": mu,
                    "rho": model.rho,
                    "model": tuple(m.probability for m in s.result),
                    "close": implied_probabilities(r.close_home_odds, r.close_draw_odds, r.close_away_odds),
                    "early": implied_probabilities(r.early_home_odds, r.early_draw_odds, r.early_away_odds)
                    if r.has_early_odds
                    else None,
                    "fthg": r.fthg,
                    "ftag": r.ftag,
                    "rest_h": rh,
                    "rest_a": ra,
                }
            )
        start += 1
    return out


if __name__ == "__main__":
    for division in sys.argv[1:]:
        t = time.time()
        preds = build(division)
        (OUT / f"{division}.pkl").write_bytes(pickle.dumps(preds))
        print(division, len(preds), f"{time.time() - t:.0f}s", flush=True)
