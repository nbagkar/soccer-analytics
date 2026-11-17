"""Phase C candidates, walk-forward by season, scored by the gap to the closing line.

A) Rest-regime goal-space factors: per regime (break / home short / away short / both short /
   normal), home and away goals vs expected on PRIOR seasons of the tuning leagues, shrunk
   toward 1, multiplied into lam/mu; probabilities rebuilt from the score grid.
B) Gradient boosting stacked on the model (HistGradientBoostingClassifier), trained on prior
   seasons of the tuning leagues; features = log lam, log mu, + rest (B) or without (B0).
Confirm leagues use parameters trained ONLY on the tuning leagues' prior seasons.
"""

from __future__ import annotations

import math
import sys
import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from analyze import CONFIRM, TUNE, ll, load, outcome
from soccer.models.markets import score_grid
from soccer.sources.football_data_co_uk import season_sort_key

warnings.filterwarnings("ignore")
SHRINK = float(sys.argv[1]) if len(sys.argv) > 1 else 300.0


def regime(r):
    h, a = r["rest_h"], r["rest_a"]
    hb = h != h or h >= 10
    ab = a != a or a >= 10
    if hb and ab:
        return "break"
    hs, as_ = (h == h and h <= 3), (a == a and a <= 3)
    if hs and as_:
        return "both_short"
    if hs:
        return "home_short"
    if as_:
        return "away_short"
    return "normal"


def probs(lam, mu, rho):
    g = score_grid(lam, mu, rho)
    h = sum(p for (x, y), p in g.items() if x > y)
    d = sum(p for (x, y), p in g.items() if x == y)
    return (h, d, 1 - h - d)


def factors(train):
    agg = {}
    for r in train:
        a = agg.setdefault(regime(r), [0.0, 0.0, 0.0, 0.0])
        a[0] += r["fthg"]; a[1] += r["lam"]; a[2] += r["ftag"]; a[3] += r["mu"]
    return {k: ((v[0] + SHRINK) / (v[1] + SHRINK), (v[2] + SHRINK) / (v[3] + SHRINK)) for k, v in agg.items()}


def feats(r, rest):
    base = [math.log(r["lam"]), math.log(r["mu"])]
    if rest:
        base += [r["rest_h"], r["rest_a"]]
    return base


def main():
    rows = load(TUNE + CONFIRM)
    seasons = sorted({r["season"] for r in rows}, key=season_sort_key)
    res = {}  # (division) -> lists of (base, A, B, B0, close) losses
    for s in seasons[2:]:  # need two prior seasons to fit on
        train = [r for r in rows if r["division"] in TUNE and season_sort_key(r["season"]) < season_sort_key(s)]
        test = [r for r in rows if r["season"] == s]
        if not test or len(train) < 2000:
            continue
        f = factors(train)
        X = np.array([feats(r, True) for r in train]); y = np.array([outcome(r) for r in train])
        X0 = np.array([feats(r, False) for r in train])
        kw = dict(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0, early_stopping=True, random_state=0)
        gb = HistGradientBoostingClassifier(**kw).fit(X, y)
        gb0 = HistGradientBoostingClassifier(**kw).fit(X0, y)
        pb = gb.predict_proba(np.array([feats(r, True) for r in test]))
        pb0 = gb0.predict_proba(np.array([feats(r, False) for r in test]))
        for r, b, b0 in zip(test, pb, pb0, strict=True):
            a = outcome(r)
            fh, fa = f.get(regime(r), (1.0, 1.0))
            pa = probs(r["lam"] * fh, r["mu"] * fa, r["rho"])
            res.setdefault(r["division"], []).append(
                (ll(r["model"], a), ll(pa, a), ll(tuple(b), a), ll(tuple(b0), a), ll(r["close"], a))
            )
    print(f"shrink={SHRINK}  (gap = forecaster log loss - closing line; lower is better)")
    print(f"{'div':5}{'n':>6}{'base gap':>10}{'A rest':>10}{'B gbm+rest':>12}{'B0 gbm':>10}")
    tot = {"tune": [], "confirm": []}
    for d in TUNE + CONFIRM:
        v = np.array(res.get(d, []))
        if not len(v):
            continue
        g = v[:, :4].mean(0) - v[:, 4].mean()
        tot["tune" if d in TUNE else "confirm"].append(v)
        print(f"{d:5}{len(v):6}{g[0]:+10.4f}{g[1]:+10.4f}{g[2]:+12.4f}{g[3]:+10.4f}")
    for k, vs in tot.items():
        v = np.vstack(vs)
        g = v[:, :4].mean(0) - v[:, 4].mean()
        diff = v[:, 1] - v[:, 0]
        se = diff.std() / math.sqrt(len(diff))
        better = sum(1 for x in vs if x[:, 1].mean() < x[:, 0].mean())
        print(f"{k:8} n={len(v)} base {g[0]:+.4f}  A {g[1]:+.4f} (A-base {diff.mean():+.5f} ± {se:.5f}, {better}/{len(vs)} leagues better)  B {g[2]:+.4f}  B0 {g[3]:+.4f}")


if __name__ == "__main__":
    main()
