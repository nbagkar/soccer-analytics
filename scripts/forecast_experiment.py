"""Walk-forward forecast experiment: the live rating model vs variants, gap to the closing line.

Mirrors the dashboard's live forecast (`soccer.dashboard.data._forecast_fit`): ratings fit on
the last three seasons (`FORECAST_SEASONS`) up to the match, refit every matchday, then the
goal-space recalibration. Each eligible match (both teams in the window, closing 1X2 present)
is scored on log loss against the vig-free closing line; the headline number is the mean gap
``model - market`` (lower is better; 0 would match the market). Paired comparisons also
report the change in KL(closing line || model), a far lower-variance read of whether a
variant moved toward the market -- use it before blaming one division's log loss on the model.

The fit is a vectorized re-implementation of `fit_poisson_shots` + `calibrate_model` so the
whole sweep runs in minutes; `--verify` checks it against the real `_forecast_fit` first.

Needs the history loaded first (18 divisions; the first season of each window is history):
    soccer ingest-history --divisions E0,E1,E2,E3,SC0,D1,D2,I1,I2,SP1,SP2,F1,F2,N1,B1,P1,T1,G1 \
        --seasons 1011,1112,...,2627

Usage:
    python scripts/forecast_experiment.py --verify          # harness == live `_forecast_fit`
    python scripts/forecast_experiment.py --variants baseline,live
    python scripts/forecast_experiment.py --variants adjusted --grid '{"spread": [1.1, 1.14]}'

Presets: ``baseline`` is the live model as of 2026-10 (plain ratio fit, 120d); ``live`` reads
the dashboard's current constants. Tuning picks by the tune seasons (before
``--holdout-from``) only; the holdout is reported, never selected on. Results 2026-11-04 --
see the FORECAST_OPPONENT_ADJUST note in `soccer/dashboard/data.py`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

DIVISIONS = (
    *("E0", "E1", "E2", "E3", "SC0", "D1", "D2", "I1", "I2"),
    *("SP1", "SP2", "F1", "F2", "N1", "B1", "P1", "T1", "G1"),
)
WINDOW_SEASONS = 3  # FORECAST_SEASONS
MAX_GOALS = 10


@dataclass(frozen=True)
class Config:
    alpha: float = 0.25  # FORECAST_ALPHA
    shrinkage: float = 0.5  # FORECAST_SHRINKAGE
    half_life: float = 120.0  # FORECAST_TIME_DECAY_DAYS
    market_weight: float = 1.0  # FORECAST_MARKET_WEIGHT
    spread: float = 1.08  # FORECAST_SPREAD
    home_shift: float = 0.02  # FORECAST_HOME_SHIFT
    rho: float = -0.09  # FORECAST_RHO
    adjusted: bool = False  # opponent-adjusted (iterative) fit instead of plain ratio
    iterations: int = 12
    pin_attack_only: bool = False  # rescale attack, compensate defence (exact MLE fixed point)
    label: str = "baseline"


# ---------------------------------------------------------------- data


@dataclass
class Division:
    code: str
    seasons: list[str]  # chronological
    season_idx: np.ndarray  # per row, index into `seasons`
    day: np.ndarray  # ordinal date
    home: np.ndarray  # team index
    away: np.ndarray
    hg: np.ndarray
    ag: np.ndarray
    hst: np.ndarray  # NaN when missing
    ast: np.ndarray
    mh: np.ndarray  # market-implied xG, NaN when no odds
    ma: np.ndarray
    p_mkt: np.ndarray  # (n, 3) vig-free closing 1X2, NaN when no odds
    teams: list[str] = field(default_factory=list)


def load(db_path: Path, codes: tuple[str, ...]) -> dict[str, Division]:
    from soccer.models.poisson import market_expected_goals
    from soccer.models.value import implied_probabilities
    from soccer.sources.football_data_co_uk import season_sort_key
    from soccer.storage.analytics_db import AnalyticsDB

    out: dict[str, Division] = {}
    with AnalyticsDB(db_path) as adb:
        loaded = adb.seasons_loaded()
        for code in codes:
            seasons = sorted({s for s, d, _n in loaded if d == code}, key=season_sort_key)
            rows = [(i, r) for i, s in enumerate(seasons) for r in adb.outcomes_with_odds(s, code)]
            if not rows:
                continue
            rows.sort(key=lambda t: (t[1].match_date, t[0]))
            teams = sorted({t for _i, r in rows for t in (r.home_norm, r.away_norm)})
            tix = {t: k for k, t in enumerate(teams)}
            n = len(rows)
            mh, ma = np.full(n, np.nan), np.full(n, np.nan)
            pm = np.full((n, 3), np.nan)
            for k, (_i, r) in enumerate(rows):
                m = market_expected_goals(r)
                if m is not None:
                    mh[k], ma[k] = m
                if r.has_odds:
                    pm[k] = implied_probabilities(
                        r.close_home_odds, r.close_draw_odds, r.close_away_odds
                    )

            def col(attr: str, rows: list[Any] = rows) -> np.ndarray:
                values = (getattr(r, attr) for _i, r in rows)
                return np.array([np.nan if v is None else v for v in values], dtype=float)

            out[code] = Division(
                code=code,
                seasons=seasons,
                season_idx=np.array([i for i, _r in rows]),
                day=np.array([r.match_date.toordinal() for _i, r in rows]),
                home=np.array([tix[r.home_norm] for _i, r in rows]),
                away=np.array([tix[r.away_norm] for _i, r in rows]),
                hg=np.array([r.fthg for _i, r in rows], dtype=float),
                ag=np.array([r.ftag for _i, r in rows], dtype=float),
                hst=col("home_shots_target"),
                ast=col("away_shots_target"),
                mh=mh,
                ma=ma,
                p_mkt=pm,
                teams=teams,
            )
    return out


# ---------------------------------------------------------------- model


def fit(div: Division, idx: np.ndarray, cfg: Config) -> tuple[np.ndarray, np.ndarray, float, float]:
    """(attack, defence, home_avg, away_avg) over rows `idx`, calibrated. NaN = team unseen."""
    nt = len(div.teams)
    day, h, a = div.day[idx], div.home[idx], div.away[idx]
    hg, ag = div.hg[idx], div.ag[idx]
    hst, ast = div.hst[idx], div.ast[idx]
    mh, ma = div.mh[idx], div.ma[idx]
    xi = math.log(2) / cfg.half_life if cfg.half_life > 0 else 0.0
    w = np.exp(-xi * (day.max() - day)) if xi else np.ones(len(idx))

    shots = ~(np.isnan(hst) | np.isnan(ast))
    tot_sot = float((w * (hst + ast))[shots].sum())
    conv = float((w * (hg + ag))[shots].sum()) / tot_sot if tot_sot else 0.0
    if conv:
        hv = np.where(shots, cfg.alpha * hg + (1 - cfg.alpha) * np.nan_to_num(hst) * conv, hg)
        av = np.where(shots, cfg.alpha * ag + (1 - cfg.alpha) * np.nan_to_num(ast) * conv, ag)
    else:
        hv, av = hg.copy(), ag.copy()
    if cfg.market_weight:
        mk = ~np.isnan(mh)
        mw = cfg.market_weight
        hv = np.where(mk, (1 - mw) * hv + mw * np.nan_to_num(mh), hv)
        av = np.where(mk, (1 - mw) * av + mw * np.nan_to_num(ma), av)

    total_w = w.sum()
    home_avg, away_avg = float((w * hg).sum() / total_w), float((w * ag).sum() / total_w)
    games = np.bincount(h, w, nt) + np.bincount(a, w, nt)
    scored = np.bincount(h, w * hv, nt) + np.bincount(a, w * av, nt)
    conceded = np.bincount(h, w * av, nt) + np.bincount(a, w * hv, nt)
    seen = games > 0
    k = cfg.shrinkage

    def shrink(rate: np.ndarray) -> np.ndarray:
        return (games * rate + k) / (games + k) if k else rate

    with np.errstate(invalid="ignore", divide="ignore"):
        if not cfg.adjusted:
            overall = scored.sum() / (2 * total_w)
            att = shrink(scored / games / overall)
            dfn = shrink(conceded / games / overall)
        else:
            # Iterative proportional fitting (Maher's Poisson MLE equations): a team's attack
            # is what it scored over what an average attack would have scored against the
            # defences it actually faced, at the venues it actually played -- so neither a
            # hard schedule nor an uneven home/away split is mistaken for weakness.
            ph = float((w * hv).sum() / total_w)  # pseudo-goal venue means
            pa = float((w * av).sum() / total_w)
            att = np.ones(nt)
            dfn = np.ones(nt)
            for _ in range(cfg.iterations):
                exp_s = np.bincount(h, w * ph * dfn[a], nt) + np.bincount(a, w * pa * dfn[h], nt)
                att = shrink(scored / exp_s)
                exp_c = np.bincount(h, w * pa * att[a], nt) + np.bincount(a, w * ph * att[h], nt)
                dfn = shrink(conceded / exp_c)
                # pin the scale: games-weighted mean attack = 1 (and defence = 1, or compensate)
                c = np.nansum(games * att) / games[seen].sum()
                att = att / c
                if cfg.pin_attack_only:
                    dfn = dfn * c
                else:
                    dfn = dfn / (np.nansum(games * dfn) / games[seen].sum())
    att[~seen] = np.nan
    dfn[~seen] = np.nan
    # calibrate_model
    att, dfn = att**cfg.spread, dfn**cfg.spread
    return att, dfn, home_avg * math.exp(cfg.home_shift), away_avg * math.exp(-cfg.home_shift)


_K = np.arange(MAX_GOALS + 1)
_LF = np.array([math.lgamma(i + 1) for i in _K])
_DIFF = np.subtract.outer(_K, _K)


def probs(lam: np.ndarray, mu: np.ndarray, rho: float) -> np.ndarray:
    """(n, 3) home/draw/away on the Dixon-Coles grid (same as `score_grid`)."""
    px = np.exp(np.log(lam)[:, None] * _K - lam[:, None] - _LF)
    py = np.exp(np.log(mu)[:, None] * _K - mu[:, None] - _LF)
    g = px[:, :, None] * py[:, None, :]
    g[:, 0, 0] *= 1.0 - lam * mu * rho
    g[:, 0, 1] *= 1.0 + lam * rho
    g[:, 1, 0] *= 1.0 + mu * rho
    g[:, 1, 1] *= 1.0 - rho
    np.maximum(g, 0.0, out=g)
    g /= g.sum(axis=(1, 2), keepdims=True)
    return np.stack(
        [g[:, _DIFF > 0].sum(1), g[:, _DIFF == 0].sum(1), g[:, _DIFF < 0].sum(1)], axis=1
    )


# ---------------------------------------------------------------- walk-forward


def walk(
    div: Division, cfg: Config, first_test_season: int = 2
) -> list[tuple[str, float, float, float, int]]:
    """Per scored match: (season, model log loss, market log loss, KL to market, thin flag).

    ``KL to market`` is KL(closing line || model) for the match: the gap the model would be
    expected to post if the closing line were the true probabilities. Unlike the log-loss gap
    it does not depend on which result happened, so its standard error is ~5-8x smaller --
    see the Scotland note in `main`'s per-division readout.
    """
    out: list[tuple[str, float, float, float, int]] = []
    actual = np.where(div.hg > div.ag, 0, np.where(div.hg == div.ag, 1, 2))
    days = div.day
    for s in range(first_test_season, len(div.seasons)):
        in_season = np.flatnonzero(div.season_idx == s)
        for d in np.unique(days[in_season]):
            test = in_season[days[in_season] == d]
            test = test[~np.isnan(div.p_mkt[test, 0])]
            if not len(test):
                continue
            train = np.flatnonzero(
                (div.season_idx >= s - (WINDOW_SEASONS - 1)) & (div.season_idx <= s) & (days < d)
            )
            if len(train) < 60:
                continue
            att, dfn, ha, aa = fit(div, train, cfg)
            hi, ai = div.home[test], div.away[test]
            ok = ~(np.isnan(att[hi]) | np.isnan(att[ai]))
            if not ok.any():
                continue
            test, hi, ai = test[ok], hi[ok], ai[ok]
            lam = ha * att[hi] * dfn[ai]
            mu = aa * att[ai] * dfn[hi]
            p = probs(lam, mu, cfg.rho)
            y = actual[test]
            g = np.bincount(
                np.concatenate([div.home[train], div.away[train]]), minlength=len(div.teams)
            )
            thin = (np.minimum(g[hi], g[ai]) < 6).astype(int)
            pm = div.p_mkt[test]
            ll_m = -np.log(np.maximum(p[np.arange(len(y)), y], 1e-15))
            ll_k = -np.log(np.maximum(pm[np.arange(len(y)), y], 1e-15))
            kl = (pm * np.log(pm / np.maximum(p, 1e-15))).sum(axis=1)
            for j in range(len(y)):
                out.append(
                    (div.seasons[s], float(ll_m[j]), float(ll_k[j]), float(kl[j]), int(thin[j]))
                )
    return out


_DIVS: dict[str, Division] = {}


def _load_cached(db: Path, codes: tuple[str, ...]) -> dict[str, Division]:
    """`load`, cached next to the database (inverting ~100k closing lines takes a while)."""
    import pickle

    # Plain dicts, not Division objects: a pickle made while this runs as __main__ must load
    # from an import of this module too.
    cache = db.with_suffix(f".experiment-{db.stat().st_mtime_ns}.v2.pkl")
    if cache.exists():
        raw: dict[str, dict[str, Any]] = pickle.loads(cache.read_bytes())
        if all(c in raw for c in codes):
            return {c: Division(**raw[c]) for c in codes}
    divs = load(db, codes)
    cache.write_bytes(pickle.dumps({c: asdict(d) for c, d in divs.items()}))
    return divs


def _run(args: tuple[str, Config]) -> tuple[str, str, list[Any]]:
    code, cfg = args
    return code, cfg.label, walk(_DIVS[code], cfg) if code in _DIVS else []


def summarize(recs: dict[str, list[Any]], holdout_from: str) -> dict[str, Any]:
    from soccer.sources.football_data_co_uk import season_sort_key

    def gap(rs: list[Any]) -> tuple[float, int]:
        return (sum(r[1] - r[2] for r in rs) / len(rs), len(rs)) if rs else (float("nan"), 0)

    allr = [r for rs in recs.values() for r in rs]
    hk = season_sort_key(holdout_from)
    tune = [r for r in allr if season_sort_key(r[0]) < hk]
    hold = [r for r in allr if season_sort_key(r[0]) >= hk]
    return {
        "all": gap(allr),
        "tune": gap(tune),
        "holdout": gap(hold),
        "thin": gap([r for r in allr if r[4]]),
        "by_division_holdout": {
            c: gap([r for r in rs if season_sort_key(r[0]) >= hk]) for c, rs in recs.items()
        },
    }


def run_configs(
    db: Path, configs: list[Config], codes: tuple[str, ...], workers: int
) -> dict[str, dict[str, list[Any]]]:
    jobs = [(c, cfg) for cfg in configs for c in codes]
    res: dict[str, dict[str, list[Any]]] = {cfg.label: {} for cfg in configs}
    import multiprocessing as mp

    _DIVS.update(_load_cached(db, codes))  # workers inherit it by fork; one DuckDB reader
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as ex:
        for code, label, recs in ex.map(_run, jobs, chunksize=1):
            res[label][code] = recs
    return res


# ---------------------------------------------------------------- verification


def live_config() -> Config:
    """The dashboard's current forecast settings as a Config."""
    from soccer.dashboard import data as d

    return Config(
        alpha=d.FORECAST_ALPHA,
        shrinkage=d.FORECAST_SHRINKAGE,
        half_life=float(d.FORECAST_TIME_DECAY_DAYS),
        market_weight=d.FORECAST_MARKET_WEIGHT,
        spread=d.FORECAST_SPREAD,
        home_shift=d.FORECAST_HOME_SHIFT,
        rho=d.FORECAST_RHO,
        adjusted=d.FORECAST_OPPONENT_ADJUST,
        label="live",
    )


def verify(db: Path) -> None:
    """The vectorized fit must reproduce the live `_forecast_fit` on a real window."""
    from soccer.dashboard.data import _forecast_fit
    from soccer.storage.analytics_db import AnalyticsDB

    divs = load(db, ("E0", "SP2"))
    worst = 0.0
    for code, div in divs.items():
        s = len(div.seasons) - 2  # last complete season
        with AnalyticsDB(db) as adb:
            window = [
                r for k in range(s - 2, s + 1) for r in adb.outcomes_with_odds(div.seasons[k], code)
            ]
        live = _forecast_fit(window)
        idx = np.flatnonzero((div.season_idx >= s - 2) & (div.season_idx <= s))
        att, dfn, ha, aa = fit(div, idx, live_config())
        for t, st in live.strengths.items():
            k = div.teams.index(t)
            worst = max(worst, abs(st.attack - att[k]), abs(st.defence - dfn[k]))
        worst = max(worst, abs(live.home_avg - ha), abs(live.away_avg - aa))
        print(
            f"{code} {div.seasons[s]}: {len(live.strengths)} teams, max abs diff so far {worst:.2e}"
        )
    if worst > 1e-8:
        sys.exit(f"MISMATCH: vectorized fit differs from _forecast_fit by {worst:.3e}")
    print("OK: vectorized fit reproduces the live _forecast_fit")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/analytics.duckdb")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--variants", default="baseline,adjusted")
    ap.add_argument("--divisions", default=",".join(DIVISIONS))
    ap.add_argument("--holdout-from", default="2021")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="")
    ap.add_argument(
        "--grid",
        default="",
        help='JSON {"field": [values], ...} swept on the first variant; ranked on tune seasons',
    )
    ns = ap.parse_args()
    db = Path(ns.db)
    if ns.verify:
        verify(db)
        return
    presets = {
        "baseline": Config(),
        "live": live_config(),
        "adjusted": Config(adjusted=True, label="adjusted"),
        # Best on the tune seasons (2012/13-2020/21) of the grids in this script's history.
        "retuned": Config(spread=1.12, half_life=90.0, label="retuned"),
        "adjusted_tuned": Config(
            adjusted=True, spread=1.14, half_life=60.0, home_shift=0.01, label="adjusted_tuned"
        ),
        "adjusted_pinned": Config(
            adjusted=True,
            spread=1.14,
            half_life=60.0,
            home_shift=0.01,
            pin_attack_only=True,
            label="adjusted_pinned",
        ),
    }
    from soccer.sources.football_data_co_uk import season_sort_key

    configs = [presets[v] for v in ns.variants.split(",")]
    codes = tuple(ns.divisions.split(","))
    t0 = time.time()
    if ns.grid:
        import itertools

        grid = json.loads(ns.grid)
        keys = list(grid)
        swept = [
            replace(
                configs[0],
                **dict(zip(keys, vals, strict=True)),
                label=json.dumps(dict(zip(keys, vals, strict=True))),
            )
            for vals in itertools.product(*(grid[k] for k in keys))
        ]
        res = run_configs(db, swept, codes, ns.workers)
        rows = sorted(
            ((summarize(r, ns.holdout_from), label) for label, r in res.items()),
            key=lambda t: t[0]["tune"][0],
        )
        for sm, label in rows[:15]:
            print(
                f"tune {sm['tune'][0]:+.5f}  holdout {sm['holdout'][0]:+.5f}  "
                f"thin {sm['thin'][0]:+.5f}  {label}"
            )
        print(f"[{len(swept)} configs, {time.time() - t0:.0f}s]")
        if ns.out:
            Path(ns.out).write_text(json.dumps([(sm, label) for sm, label in rows], indent=1))
        return
    res = run_configs(db, configs, codes, ns.workers)
    summary = {label: summarize(recs, ns.holdout_from) for label, recs in res.items()}
    for label, s in summary.items():
        print(
            f"{label:10s} all {s['all'][0]:+.5f} (n={s['all'][1]})  tune {s['tune'][0]:+.5f}  "
            f"holdout {s['holdout'][0]:+.5f} (n={s['holdout'][1]})  thin {s['thin'][0]:+.5f} "
            f"(n={s['thin'][1]})"
        )
    base = configs[0].label
    for cfg in configs[1:]:
        print(f"\n{cfg.label} vs {base}, paired per match (negative = better):")
        for name, keep in (
            ("all", lambda r: True),
            ("holdout", lambda r: r[0] >= ns.holdout_from[:2]),
        ):
            pairs = [
                (a, b)
                for c in codes
                for a, b in zip(res[base].get(c, []), res[cfg.label].get(c, []), strict=True)
                if keep(a)
            ]
            d = np.array([b[1] - a[1] for a, b in pairs])
            dk = np.array([b[3] - a[3] for a, b in pairs])
            se = math.sqrt(len(d))
            print(
                f"  {name:8s} {d.mean():+.5f} +/- {d.std(ddof=1) / se:.5f} (1 s.e.)  "
                f"KL to market {dk.mean():+.5f} +/- {dk.std(ddof=1) / se:.5f}"
            )
        # Per division on the holdout: the log-loss gap and its paired change, then the paired
        # change in KL to the closing line. One division's log loss is noisy (s.e. ~0.0015-
        # 0.0023 per division); KL is not (~0.0002-0.0004). A division whose log loss worsens
        # while its KL improves by many s.e. moved TOWARD the market and was unlucky with
        # results -- SC0 under opponent adjustment, 2026-11-04 (FORECAST_OPPONENT_ADJUST note).
        better = better_kl = 0
        for c in codes:
            g0 = summary[base]["by_division_holdout"][c][0]
            g1 = summary[cfg.label]["by_division_holdout"][c][0]
            pairs = [
                (a, b)
                for a, b in zip(res[base].get(c, []), res[cfg.label].get(c, []), strict=True)
                if season_sort_key(a[0]) >= season_sort_key(ns.holdout_from)
            ]
            if not pairs:
                continue
            dl = np.array([b[1] - a[1] for a, b in pairs])
            dk = np.array([b[3] - a[3] for a, b in pairs])
            se = math.sqrt(len(pairs))
            better += g1 < g0
            better_kl += dk.mean() < 0
            print(
                f"  {c:4s} holdout gap {g0:+.5f} -> {g1:+.5f}  "
                f"({dl.mean():+.5f} +/- {dl.std(ddof=1) / se:.5f})  "
                f"KL {dk.mean():+.5f} +/- {dk.std(ddof=1) / se:.5f}"
            )
        print(f"  {better}/{len(codes)} divisions improved on holdout log loss, {better_kl} on KL")
        seasons = sorted({r[0] for rs in res[base].values() for r in rs})
        for sn in seasons:
            d = np.array(
                [
                    b[1] - a[1]
                    for c in codes
                    for a, b in zip(res[base].get(c, []), res[cfg.label].get(c, []), strict=True)
                    if a[0] == sn
                ]
            )
            print(
                f"  season {sn}: {d.mean():+.5f} "
                f"+/- {d.std(ddof=1) / math.sqrt(len(d)):.5f} (n={len(d)})"
            )
    print(f"[{time.time() - t0:.0f}s]")
    if ns.out:
        Path(ns.out).write_text(
            json.dumps({"configs": [asdict(c) for c in configs], "summary": summary}, indent=1)
        )


if __name__ == "__main__":
    main()
