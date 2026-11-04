"""Refit NEWCOMER_RATING_MAP on the live ratings and score it with the season backtest.

`NEWCOMER_RATING_MAP` (`soccer/dashboard/data.py`) rates a club new to a division from its
rating in the division it came from: log(new) = c + b * log(old), per direction ("down" =
relegated, "up" = promoted), attack and defence. This script

  fit       collects every move between adjacent divisions in DIVISION_LADDERS and fits the
            map by least squares, exactly as the live map was fit: the OLD rating is the live
            forecast fit (`_forecast_fit`) of the division the club left, on the three seasons
            through the move; the NEW rating is a single-season fit of the division it joined.
            Both in uncalibrated log units (log / FORECAST_SPREAD), as `_newcomer_priors`
            applies them. Moves into seasons before `--holdout-from` are the fit; the later
            ones are scored out of sample (mean squared error of the predicted log rating).
  backtest  runs `season_backtest` (the `soccer backtest-season` code path) with a candidate
            map patched in, per division, reporting the pre-season / 25% checkpoints for all
            20 seasons and for the tune (< holdout) and holdout seasons separately.

Needs the history loaded from 1995/96 (the fit uses every move since):
    soccer ingest-history --divisions E0,E1,E2,E3,D1,D2,I1,I2,SP1,SP2,F1,F2 --seasons 9596,...

Usage:
    python scripts/newcomer_map_experiment.py fit --source-half-life 60 --target-half-life 0
    python scripts/newcomer_map_experiment.py backtest --maps maps.json --divisions E1,E2

Results 2026-11-04 -- see the NEWCOMER_RATING_MAP note in `soccer/dashboard/data.py`.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

Map = dict[str, tuple[tuple[float, float], tuple[float, float]]]


def _season_codes(adb: Any, division: str) -> list[str]:
    from soccer.sources.football_data_co_uk import season_sort_key

    return sorted({s for s, d, _n in adb.seasons_loaded() if d == division}, key=season_sort_key)


def collect_moves(
    db: Path, source_half_life: float, target_half_life: float
) -> list[dict[str, Any]]:
    """Every club that moved between adjacent divisions of a ladder, with its old and new
    log ratings (uncalibrated units)."""
    from soccer.dashboard import data
    from soccer.storage.analytics_db import AnalyticsDB

    moves: list[dict[str, Any]] = []
    with AnalyticsDB(db) as adb:
        src_cache: dict[tuple[str, str], Any] = {}
        tgt_cache: dict[tuple[str, str], Any] = {}

        def source(div: str, season: str) -> Any:
            if (div, season) not in src_cache:
                rows = adb.recent_outcomes_through(div, season, n_seasons=data.FORECAST_SEASONS)
                src_cache[div, season] = data._forecast_fit(rows, half_life_days=source_half_life)
            return src_cache[div, season]

        def target(div: str, season: str) -> Any:
            if (div, season) not in tgt_cache:
                rows = adb.outcomes_for(season, div)
                tgt_cache[div, season] = data._forecast_fit(rows, half_life_days=target_half_life)
            return tgt_cache[div, season]

        for ladder in data.DIVISION_LADDERS:
            for upper, lower in itertools.pairwise(ladder):
                common = [s for s in _season_codes(adb, upper) if s in _season_codes(adb, lower)]
                for s0, s1 in itertools.pairwise(common):
                    clubs = {
                        (d, s): {o.home_norm for o in adb.outcomes_for(s, d)}
                        for d in (upper, lower)
                        for s in (s0, s1)
                    }
                    for direction, frm, to in (("down", upper, lower), ("up", lower, upper)):
                        moved = clubs[frm, s0] & clubs[to, s1]
                        if not moved:
                            continue
                        old, new = source(frm, s0), target(to, s1)
                        for t in sorted(moved):
                            if t not in old.strengths or t not in new.strengths:
                                continue
                            o, n = old.strengths[t], new.strengths[t]
                            sp = data.FORECAST_SPREAD
                            moves.append(
                                {
                                    "direction": direction,
                                    "to": to,
                                    "top": to == ladder[0],
                                    "season": s1,
                                    "team": t,
                                    "old": (math.log(o.attack) / sp, math.log(o.defence) / sp),
                                    "new": (math.log(n.attack) / sp, math.log(n.defence) / sp),
                                }
                            )
    return moves


def fit_map(moves: list[dict[str, Any]], *, intercept_only: bool = False) -> Map:
    """Least squares, per direction and for attack (index 0) and defence (index 1)."""
    out: Map = {}
    for direction in ("down", "up"):
        sel = [m for m in moves if m["direction"] == direction]
        pair = []
        for k in (0, 1):
            x = np.array([m["old"][k] for m in sel])
            y = np.array([m["new"][k] for m in sel])
            if intercept_only:
                b, c = 0.0, float(y.mean())
            else:
                b, c = (float(v) for v in np.polyfit(x, y, 1))
            pair.append((round(b, 3), round(c, 3)))
        out[direction] = (pair[0], pair[1])
    return out


def map_mse(moves: list[dict[str, Any]], mp: Map, direction: str) -> tuple[float, int]:
    sel = [m for m in moves if m["direction"] == direction]
    if not sel:
        return float("nan"), 0
    err = 0.0
    for m in sel:
        for k in (0, 1):
            b, c = mp[direction][k]
            err += (c + b * m["old"][k] - m["new"][k]) ** 2
    return err / (2 * len(sel)), len(sel)


def cmd_fit(args: argparse.Namespace) -> None:
    from soccer.dashboard import data
    from soccer.sources.football_data_co_uk import season_sort_key

    moves = collect_moves(Path(args.db), args.source_half_life, args.target_half_life)
    cut = season_sort_key(args.holdout_from)
    tune = [m for m in moves if season_sort_key(m["season"]) < cut]
    hold = [m for m in moves if season_sort_key(m["season"]) >= cut]
    fits = {
        "live": data.NEWCOMER_RATING_MAP,
        "refit": fit_map(tune),
        "refit_all": fit_map(moves),
        "intercept": fit_map(tune, intercept_only=True),
        "refit_top_up": fit_map([m for m in tune if m["top"] or m["direction"] == "down"]),
    }
    print(
        f"moves: {len(moves)} (tune {len(tune)}, holdout {len(hold)}); "
        f"down {sum(m['direction'] == 'down' for m in moves)}, "
        f"up {sum(m['direction'] == 'up' for m in moves)}"
    )
    for direction in ("down", "up"):
        for k, name in ((0, "attack"), (1, "defence")):
            vals = [m["new"][k] for m in tune if m["direction"] == direction]
            print(f"  {direction} {name}: mean new rating {math.exp(np.mean(vals)):.3f}")
    for label, mp in fits.items():
        cells = []
        for direction in ("down", "up"):
            t = map_mse(tune, mp, direction)[0]
            h = map_mse(hold, mp, direction)[0]
            cells.append(f"{direction} tune {t:.4f} hold {h:.4f}")
        print(f"{label:13s} {json.dumps(mp)}")
        print(f"{'':13s} {'; '.join(cells)}")
    if args.out:
        Path(args.out).write_text(json.dumps({k: v for k, v in fits.items()}, indent=1))


# ---------------------------------------------------------------- backtest


def _variant_priors(mp: Map, source_half_life: float, up_into_top: bool, top_map: Map | None):
    """A drop-in `_newcomer_priors` with the map and switches of one variant."""
    from soccer.dashboard import data

    def priors(adb: Any, division: str, last_season: str, teams: list[str]) -> Any:
        ladder = next((lad for lad in data.DIVISION_LADDERS if division in lad), None)
        if ladder is None:
            return {}
        staying = {o.home_norm for o in adb.outcomes_for(last_season, division)}
        newcomers = {t for t in teams if t not in staying}
        level = ladder.index(division)
        out: dict[str, tuple[float, float]] = {}
        for direction, other_level in (("down", level - 1), ("up", level + 1)):
            if not newcomers or not 0 <= other_level < len(ladder):
                continue
            use = mp
            if direction == "up" and level == 0:
                if not up_into_top:
                    continue
                use = top_map or mp
            other = ladder[other_level]
            moved = newcomers & {o.home_norm for o in adb.outcomes_for(last_season, other)}
            if not moved:
                continue
            (b_att, c_att), (b_def, c_def) = use[direction]
            source = data._forecast_fit(
                adb.recent_outcomes_through(other, last_season, n_seasons=data.FORECAST_SEASONS),
                half_life_days=source_half_life,
            )
            sp = data.FORECAST_SPREAD
            for t in sorted(moved):
                s = source.strengths.get(t)
                if s is None:
                    continue
                la, ld = (math.log(v) / sp for v in (s.attack, s.defence))
                out[t] = (
                    math.exp(sp * (c_att + b_att * la)),
                    math.exp(sp * (c_def + b_def * ld)),
                )
            newcomers -= moved
        return out

    return priors


def _run_one(job: dict[str, Any]) -> dict[str, Any]:
    from soccer.dashboard import data
    from soccer.storage.analytics_db import AnalyticsDB

    work = Path(tempfile.mkdtemp(prefix="nm_"))
    db = work / "analytics.duckdb"
    shutil.copy(job["db"], db)
    v = job["variant"]
    if v.get("map") is not None:
        mp = {k: (tuple(a), tuple(b)) for k, (a, b) in v["map"].items()}
        top = v.get("top_map")
        top_map = {k: (tuple(a), tuple(b)) for k, (a, b) in top.items()} if top else None
        data._newcomer_priors = _variant_priors(  # type: ignore[assignment]
            mp, v.get("source_half_life", data.FORECAST_TIME_DECAY_DAYS), v["up_top"], top_map
        )
    elif v.get("no_priors"):
        data._newcomer_priors = lambda *a, **k: {}  # type: ignore[assignment]
    with AnalyticsDB(db) as adb:
        codes = _season_codes(adb, job["division"])
    from soccer.sources.football_data_co_uk import season_sort_key

    cut = season_sort_key(job["holdout_from"])
    scored = codes  # season_backtest itself limits these to the last n_seasons (+history)
    splits = {
        "all": None,
        "tune": tuple(c for c in scored if season_sort_key(c) < cut),
        "hold": tuple(c for c in scored if season_sort_key(c) >= cut),
    }
    out: dict[str, Any] = {"division": job["division"], "variant": v["name"]}
    for split, only in splits.items():
        rep = data.season_backtest.__wrapped__(  # type: ignore[attr-defined]
            db,
            job["division"],
            n_seasons=job["n_seasons"],
            checkpoints=(0.0, 0.25),
            n_sims=job["sims"],
            score_only=only,
        )
        if rep is None:
            continue
        out[split] = {
            f"{cp.checkpoint:g}": {
                "skill": cp.skill,
                "ll": sum(cp.model[e].log_loss for e in ("title", "top", "relegation")),
                "rmse": cp.points_rmse,
                "n": cp.n_team_seasons,
            }
            for cp in rep.checkpoints
        }
    shutil.rmtree(work, ignore_errors=True)
    return out


def cmd_backtest(args: argparse.Namespace) -> None:
    variants = json.loads(Path(args.variants).read_text())
    jobs = [
        {
            "db": args.db,
            "division": d,
            "variant": v,
            "n_seasons": args.seasons,
            "sims": args.sims,
            "holdout_from": args.holdout_from,
        }
        for d in args.divisions.split(",")
        for v in variants
    ]
    results = []
    with ProcessPoolExecutor(args.workers) as pool:
        for r in pool.map(_run_one, jobs):
            results.append(r)
            cells = []
            for split in ("all", "tune", "hold"):
                if split in r:
                    p, q = r[split]["0"], r[split]["0.25"]
                    cells.append(
                        f"{split} pre {p['skill']:+.1%} ({p['rmse']:.2f}) "
                        f"25% {q['skill']:+.1%} ({q['rmse']:.2f})"
                    )
            print(f"{r['division']:4s} {r['variant']:14s} " + " | ".join(cells), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default="data/analytics.duckdb")
    ap.add_argument("--holdout-from", default="2122")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--source-half-life", type=float, default=60.0)
    f.add_argument("--target-half-life", type=float, default=0.0)
    f.add_argument("--out")
    b = sub.add_parser("backtest")
    b.add_argument("--variants", required=True, help="JSON list of {name, map, up_top, ...}")
    b.add_argument("--divisions", default="E1,E2,E3,SP2,D2,I2,F2,E0,SP1,D1,I1,F1")
    b.add_argument("--seasons", type=int, default=20)
    b.add_argument("--sims", type=int, default=4000)
    b.add_argument("--workers", type=int, default=4)
    b.add_argument("--out")
    args = ap.parse_args(argv)
    {"fit": cmd_fit, "backtest": cmd_backtest}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
