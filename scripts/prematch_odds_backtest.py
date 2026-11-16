"""Would pre-match odds for upcoming fixtures make the forecasts stronger? A backtest.

football-data.co.uk publishes `fixtures.csv`: the coming round's fixtures with pre-match 1X2
odds (Bet365 and the market average, collected a day or more before kickoff). The season
files hold the same pre-match columns next to the closing ones from 2019/20, so the question
can be answered on history: score the live model, the vig-free pre-match line, and their
log-opinion blend against the vig-free closing line -- the same benchmark and walk-forward as
`forecast_experiment.py` (three-season window, refit every matchday).

The blend weight is fitted on the tune seasons (2019/20-2020/21) only and scored on the
2021/22+ holdout. Only matches with a true closing triple AND a pre-match triple are scored
(the store falls back to pre-match odds when a closing price is missing; those would compare
the pre-match line with itself).

Measured 2026-11-16, 18 divisions, holdout 2021/22-2026/27 (34,891 matches with the
market-average pre-match line), log-loss gap to the closing line / KL to it:
    live model alone          +0.01088 +/- 0.00074   0.00981
    pre-match average alone   +0.00408 +/- 0.00041   0.00307
    blend (10% model by KL)   +0.00407 +/- 0.00041   0.00306   (vs pre-match: -0.00000 +/- 0.00007)
Bet365's pre-match line: +0.00434 alone, +0.00428 blended at 15% model (KL -0.00006 +/-
0.00001). The pre-match line is closer to the close than the model in all 18 divisions (KL
about a third of the model's); blending the model in adds nothing measurable on log loss.
Not wired into the live forecasts.

Needs the raw snapshots and analytics store from `soccer ingest-history` (see
`forecast_experiment.py`).

Usage:
    python scripts/prematch_odds_backtest.py
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import forecast_experiment as fe

CLOSING = (("PSCH", "PSCD", "PSCA"), ("AvgCH", "AvgCD", "AvgCA"), ("B365CH", "B365CD", "B365CA"))
PREMATCH = {
    "avg": ("AvgH", "AvgD", "AvgA"),  # in fixtures.csv
    "b365": ("B365H", "B365D", "B365A"),  # in fixtures.csv
    "pinnacle": ("PSH", "PSD", "PSA"),  # NOT in fixtures.csv -- reference only
}
FIRST_SEASON = "1920"  # closing columns start here
EPS = 1e-15


def _date(raw: str) -> date | None:
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(raw.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _triple(row: dict[str, str], keys: tuple[str, str, str]) -> tuple[float, ...] | None:
    try:
        odds = tuple(float((row.get(k) or "").strip()) for k in keys)
    except ValueError:
        return None
    if any(o <= 1.0 for o in odds):
        return None
    inv = [1.0 / o for o in odds]
    total = sum(inv)
    return tuple(x / total for x in inv)


def prematch_lines(raw_dir: Path) -> dict[tuple[str, int, str, str], dict[str, tuple[float, ...]]]:
    """(division, ordinal date, home_norm, away_norm) -> {source: vig-free 1X2}, for rows
    that also carry a true closing triple. Latest snapshot per (season, division)."""
    from soccer.domain.names import normalize_name
    from soccer.sources.football_data_co_uk import season_sort_key

    out: dict[tuple[str, int, str, str], dict[str, tuple[float, ...]]] = {}
    for folder in sorted(raw_dir.iterdir()):
        season, _, division = folder.name.partition("_")
        if not division or season_sort_key(season) < season_sort_key(FIRST_SEASON):
            continue
        snapshots = sorted(folder.rglob("*.json.gz"))
        if not snapshots:
            continue
        with gzip.open(snapshots[-1]) as fh:
            payload = json.load(fh)["payload"]
        for row in csv.DictReader(io.StringIO(payload.lstrip("﻿"))):
            day = _date(row.get("Date") or "")
            if day is None or not any(_triple(row, k) for k in CLOSING):
                continue
            lines = {name: t for name, keys in PREMATCH.items() if (t := _triple(row, keys))}
            if lines:
                key = (
                    division,
                    day.toordinal(),
                    normalize_name(row.get("HomeTeam") or ""),
                    normalize_name(row.get("AwayTeam") or ""),
                )
                out[key] = lines
    return out


def blend(model: np.ndarray, market: np.ndarray, w: float) -> np.ndarray:
    """Log-opinion pool, weight `w` on the model (same form as evaluation.blend_probs)."""
    logp = w * np.log(np.maximum(model, EPS)) + (1 - w) * np.log(np.maximum(market, EPS))
    p = np.exp(logp)
    return p / p.sum(axis=1, keepdims=True)


def ll(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return -np.log(np.maximum(p[np.arange(len(y)), y], EPS))


def kl(close: np.ndarray, p: np.ndarray) -> np.ndarray:
    return (close * np.log(close / np.maximum(p, EPS))).sum(axis=1)


def report(name: str, gap: np.ndarray, dist: np.ndarray) -> None:
    se = math.sqrt(len(gap))
    print(
        f"  {name:34s} gap to close {gap.mean():+.5f} +/- {gap.std(ddof=1) / se:.5f}   "
        f"KL to close {dist.mean():.5f} +/- {dist.std(ddof=1) / se:.5f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/analytics.duckdb")
    ap.add_argument("--raw", default="data/raw/football_data_co_uk")
    ap.add_argument("--holdout-from", default="2122")
    ns = ap.parse_args()
    from soccer.sources.football_data_co_uk import season_sort_key

    cfg = fe.live_config()
    codes = tuple(fe.DIVISIONS)
    res = fe.run_configs(Path(ns.db), [cfg], codes, workers=4)[cfg.label]
    divs = fe._load_cached(Path(ns.db), codes)
    lines = prematch_lines(Path(ns.raw))

    recs: list[tuple[Any, ...]] = []  # (division, season, y, model p, close p, {source: p})
    scored = 0
    for code in codes:
        div = divs[code]
        for r in res.get(code, []):
            if season_sort_key(r[0]) < season_sort_key(FIRST_SEASON):
                continue
            scored += 1
            i = r[5]
            key = (code, int(div.day[i]), div.teams[div.home[i]], div.teams[div.away[i]])
            pre = lines.get(key)
            if pre is None:
                continue
            y = 0 if div.hg[i] > div.ag[i] else 1 if div.hg[i] == div.ag[i] else 2
            recs.append((code, r[0], y, (r[6], r[7]), div.p_mkt[i], pre))
    print(f"{len(recs)} of {scored} scored matches since 2019/20 carry pre-match + closing odds")

    hk = season_sort_key(ns.holdout_from)
    for source in PREMATCH:
        rows = [r for r in recs if source in r[5]]
        y = np.array([r[2] for r in rows])
        lam = np.array([r[3][0] for r in rows])
        mu = np.array([r[3][1] for r in rows])
        model = fe.probs(lam, mu, cfg.rho)
        close = np.array([r[4] for r in rows])
        pre = np.array([r[5][source] for r in rows])
        hold = np.array([season_sort_key(r[1]) >= hk for r in rows])
        tune = ~hold
        base = ll(close, y)

        weights = np.linspace(0.0, 1.0, 41)
        tune_ll = [ll(blend(model[tune], pre[tune], w), y[tune]).mean() for w in weights]
        tune_kl = [kl(close[tune], blend(model[tune], pre[tune], w)).mean() for w in weights]
        w_ll = float(weights[int(np.argmin(tune_ll))])
        w_kl = float(weights[int(np.argmin(tune_kl))])

        note = "" if source != "pinnacle" else "  (not in fixtures.csv; reference)"
        print(f"\n== pre-match source: {source}{note}")
        print(f"  tune n={tune.sum()}, holdout n={hold.sum()} (2021/22+)")
        print(f"  blend weight on model, fitted on tune: {w_ll:.3f} by log loss, {w_kl:.3f} by KL")
        report("model alone (live)", (ll(model, y) - base)[hold], kl(close, model)[hold])
        report("pre-match line alone", (ll(pre, y) - base)[hold], kl(close, pre)[hold])
        for label, w in (("blend, log-loss weight", w_ll), ("blend, KL weight", w_kl)):
            b = blend(model, pre, w)
            report(f"{label} ({w:.3f})", (ll(b, y) - base)[hold], kl(close, b)[hold])
        b = blend(model, pre, w_kl)
        d = (ll(b, y) - ll(pre, y))[hold]
        dk = (kl(close, b) - kl(close, pre))[hold]
        n = math.sqrt(hold.sum())
        print(
            f"  blend (KL weight) vs pre-match alone, paired: log loss {d.mean():+.5f} +/- "
            f"{d.std(ddof=1) / n:.5f}, KL {dk.mean():+.5f} +/- {dk.std(ddof=1) / n:.5f}"
        )
        if source == "avg":
            print("  per division, holdout KL to close: model -> pre-match -> blend")
            code_of = np.array([r[0] for r in rows])
            for code in codes:
                m = hold & (code_of == code)
                if m.sum() < 30:
                    continue
                print(
                    f"    {code:4s} n={m.sum():5d}  {kl(close, model)[m].mean():.5f} -> "
                    f"{kl(close, pre)[m].mean():.5f} -> {kl(close, b)[m].mean():.5f}"
                )


if __name__ == "__main__":
    main()
