from __future__ import annotations

import math
import pickle
import sys
from pathlib import Path

import numpy as np

from soccer.models.markets import compute_markets
from soccer.sources.football_data_co_uk import season_sort_key

P = Path(__file__).with_name("preds")
TUNE = ["E0", "SP1", "D1", "I1", "F1"]
CONFIRM = ["E1", "N1", "P1", "I2", "SP2", "D2", "F2", "B1", "T1"]


def load(divs):
    return [r for d in divs for r in pickle.loads((P / f"{d}.pkl").read_bytes())]


def outcome(r):
    return 0 if r["fthg"] > r["ftag"] else 2 if r["fthg"] < r["ftag"] else 1


def ll(p, a):
    return -math.log(max(p[a], 1e-15))


def gap_table(rows, key="model"):
    out = {}
    for d in sorted({r["division"] for r in rows}):
        rs = [r for r in rows if r["division"] == d]
        m = np.mean([ll(r[key], outcome(r)) for r in rs])
        c = np.mean([ll(r["close"], outcome(r)) for r in rs])
        out[d] = (len(rs), m, c, m - c)
    return out


def bucket(x):
    if x != x:
        return "none"
    return "<=3" if x <= 3 else "4-5" if x <= 5 else "6-9" if x <= 9 else "10+"


if __name__ == "__main__":
    rows = load(TUNE + CONFIRM)
    print("sanity: gap to close, live model")
    for d, (n, m, c, g) in gap_table(rows).items():
        print(f"  {d:4} n={n:5} model={m:.4f} close={c:.4f} gap={g:+.4f}")
    print("\nactual/expected goals by own rest (tuning leagues), and by opponent rest")
    tune = [r for r in rows if r["division"] in TUNE]
    for label, own, opp in (("home", "rest_h", "rest_a"), ("away", "rest_a", "rest_h")):
        goals = "fthg" if label == "home" else "ftag"
        exp = "lam" if label == "home" else "mu"
        for which, col in (("own", own), ("opp", opp)):
            agg = {}
            for r in tune:
                b = bucket(r[col])
                a = agg.setdefault(b, [0, 0.0, 0])
                a[0] += r[goals]; a[1] += r[exp]; a[2] += 1
            print(f"  {label} goals by {which} rest: " + ", ".join(
                f"{b}: {a[0]/a[1]:.3f} (n={a[2]})" for b, a in sorted(agg.items())))
