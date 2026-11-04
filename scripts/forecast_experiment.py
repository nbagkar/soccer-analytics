"""Walk-forward forecast experiment: the live rating model vs variants, gap to the closing line.

Mirrors the dashboard's live forecast (`soccer.dashboard.data._forecast_fit`): ratings fit on
the last three seasons (`FORECAST_SEASONS`) up to the match, refit every matchday, then the
goal-space recalibration. Each eligible match (both teams in the window, closing 1X2 present)
is scored on log loss against the vig-free closing line; the headline number is the mean gap
``model - market`` (lower is better; 0 would match the market).

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
    # Schedule adjustment (see `rest_buckets`): log-multipliers on expected goals by rest
    # bucket -- `rest_own[b]` on a side's own goals when IT is in bucket b, `rest_opp[b]` on
    # its goals when the OPPONENT is in bucket b. Empty = no adjustment (the live model).
    rest_mode: str = "rest"  # "rest" (days since last match) or "c14" (matches in last 14d)
    rest_own: tuple[float, ...] = ()
    rest_opp: tuple[float, ...] = ()
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


# ---------------------------------------------------------------- schedule

# Days since the side's previous LEAGUE match in the same season (cup and European ties are
# not in this history, so midweek European games are invisible). Buckets:
#   0: <=3d  1: 4-5d  2: 6-7d (the normal week)  3: 8-10d  4: 11-20d (international break)
#   5: first match of the season or >20d
REST_EDGES = np.array([3, 5, 7, 10, 20])
REST_LABELS = ("<=3d", "4-5d", "6-7d", "8-10d", "11-20d", "first/>20d")
# League matches in the 14 days before kick-off: 0, 1, 2, 3+.
C14_LABELS = ("0", "1", "2", "3+")
_REST_CACHE: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}


def rest_days(div: Division) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per row: (home rest days, away rest days, home matches in last 14d, away ...).

    Rest is NaN for a side's first match of the season. Rows are chronological."""
    n = len(div.day)
    rest = np.full((n, 2), np.nan)
    c14 = np.zeros((n, 2))
    history: dict[tuple[int, int], list[int]] = {}
    for i in range(n):
        for side, team in enumerate((div.home[i], div.away[i])):
            past = history.setdefault((int(div.season_idx[i]), int(team)), [])
            d = int(div.day[i])
            if past:
                rest[i, side] = d - past[-1]
                c14[i, side] = sum(1 for p in past[-4:] if d - p <= 14)
            past.append(d)
    return rest[:, 0], rest[:, 1], c14[:, 0], c14[:, 1]


def rest_buckets(div: Division, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Per row bucket index for (home side, away side)."""
    key = (div.code, mode)
    if key not in _REST_CACHE:
        rh, ra, ch, ca = rest_days(div)
        if mode == "rest":

            def b(r: np.ndarray) -> np.ndarray:
                out = np.searchsorted(REST_EDGES, r, side="left")
                return np.where(np.isnan(r), len(REST_EDGES), out)

            _REST_CACHE[key] = (b(rh), b(ra))
        elif mode == "c14":
            _REST_CACHE[key] = (np.minimum(ch, 3).astype(int), np.minimum(ca, 3).astype(int))
        else:
            raise ValueError(mode)
    return _REST_CACHE[key]


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


def walk(div: Division, cfg: Config, first_test_season: int = 2) -> list[tuple[Any, ...]]:
    """Per scored match: (season, model log loss, market log loss, base-rate-free, thin flag,
    row index, model home xG, model away xG) -- xG before any schedule adjustment."""
    out: list[tuple[Any, ...]] = []
    if cfg.rest_own or cfg.rest_opp:
        bh, ba = rest_buckets(div, cfg.rest_mode)
        own = np.asarray(cfg.rest_own) if cfg.rest_own else None
        opp = np.asarray(cfg.rest_opp) if cfg.rest_opp else None
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
            lam0 = ha * att[hi] * dfn[ai]
            mu0 = aa * att[ai] * dfn[hi]
            lam, mu = lam0, mu0
            if cfg.rest_own or cfg.rest_opp:
                adj_h = np.zeros(len(test))
                adj_a = np.zeros(len(test))
                if own is not None:
                    adj_h += own[bh[test]]
                    adj_a += own[ba[test]]
                if opp is not None:
                    adj_h += opp[ba[test]]
                    adj_a += opp[bh[test]]
                lam, mu = lam0 * np.exp(adj_h), mu0 * np.exp(adj_a)
            p = probs(lam, mu, cfg.rho)
            y = actual[test]
            g = np.bincount(
                np.concatenate([div.home[train], div.away[train]]), minlength=len(div.teams)
            )
            thin = (np.minimum(g[hi], g[ai]) < 6).astype(int)
            pm = div.p_mkt[test]
            ll_m = -np.log(np.maximum(p[np.arange(len(y)), y], 1e-15))
            ll_k = -np.log(np.maximum(pm[np.arange(len(y)), y], 1e-15))
            for j in range(len(y)):
                out.append(
                    (
                        div.seasons[s],
                        float(ll_m[j]),
                        float(ll_k[j]),
                        0.0,
                        int(thin[j]),
                        int(test[j]),
                        float(lam0[j]),
                        float(mu0[j]),
                    )
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


# ---------------------------------------------------------------- schedule fit


def poisson_glm(
    y: np.ndarray, x: np.ndarray, offset: np.ndarray, iters: int = 25
) -> tuple[np.ndarray, np.ndarray]:
    """log E[y] = offset + x @ beta, by Newton-Raphson. Returns (beta, quasi-Poisson s.e.).

    `y` need not be integer (market-implied goals): the Poisson score equations are then a
    quasi-likelihood, and the s.e. is scaled by the Pearson dispersion."""
    beta = np.zeros(x.shape[1])
    for _ in range(iters):
        m = np.exp(offset + x @ beta)
        info = x.T @ (x * m[:, None])
        step = np.linalg.solve(info, x.T @ (y - m))
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    m = np.exp(offset + x @ beta)
    info = x.T @ (x * m[:, None])
    phi = float(((y - m) ** 2 / m).sum() / (len(y) - x.shape[1]))
    return beta, np.sqrt(np.diag(np.linalg.inv(info)) * phi)


def fit_rest(
    res: dict[str, list[Any]], mode: str, holdout_from: str, target: str
) -> tuple[tuple[float, ...], tuple[float, ...], float]:
    """Fit (own, opp) rest log-multipliers on the TUNE seasons of a walk-forward run.

    One row per side per match: y = that side's goals (`target="goals"`) or its market-implied
    expected goals (`target="market"`), offset = log of the model's own expected goals, and
    dummies for the side's own rest bucket and the opponent's. An intercept absorbs any
    overall level bias; it is reported but not applied. Reference bucket = 6-7 days (rest) or
    one match in 14 days (c14), fixed at 0."""
    from soccer.sources.football_data_co_uk import season_sort_key

    labels = REST_LABELS if mode == "rest" else C14_LABELS
    ref = 2 if mode == "rest" else 1
    nb = len(labels)
    hk = season_sort_key(holdout_from)
    ys, offs, own_b, opp_b = [], [], [], []
    for code, recs in res.items():
        div = _DIVS[code]
        bh, ba = rest_buckets(div, mode)
        for r in recs:
            if season_sort_key(r[0]) >= hk:
                continue
            i, lam0, mu0 = r[5], r[6], r[7]
            if target == "goals":
                yh, ya = div.hg[i], div.ag[i]
            else:
                yh, ya = div.mh[i], div.ma[i]
                if np.isnan(yh):
                    continue
            ys += [yh, ya]
            offs += [math.log(lam0), math.log(mu0)]
            own_b += [bh[i], ba[i]]
            opp_b += [ba[i], bh[i]]
    y = np.array(ys)
    ob, pb = np.array(own_b), np.array(opp_b)
    keep = [b for b in range(nb) if b != ref]
    x = np.column_stack(
        [np.ones(len(y))] + [(ob == b).astype(float) for b in keep] + [(pb == b) for b in keep]
    ).astype(float)
    beta, se = poisson_glm(y, x, np.array(offs))
    own = np.zeros(nb)
    opp = np.zeros(nb)
    own[keep] = beta[1 : 1 + len(keep)]
    opp[keep] = beta[1 + len(keep) :]
    own_se = np.zeros(nb)
    opp_se = np.zeros(nb)
    own_se[keep] = se[1 : 1 + len(keep)]
    opp_se[keep] = se[1 + len(keep) :]
    print(f"\nrest fit ({mode}, target={target}) on tune seasons, {len(y)} side-matches:")
    print(f"  intercept {beta[0]:+.4f} +/- {se[0]:.4f} (not applied)")
    counts = np.bincount(ob, minlength=nb)
    for b in range(nb):
        print(
            f"  {labels[b]:>10s} n={counts[b]:6d}  own {own[b]:+.4f} +/- {own_se[b]:.4f}"
            f"   opp {opp[b]:+.4f} +/- {opp_se[b]:.4f}"
        )
    return tuple(round(float(v), 4) for v in own), tuple(round(float(v), 4) for v in opp), beta[0]


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
        "--fit-rest",
        default="",
        help="rest|c14: fit schedule multipliers on the tune seasons of the first variant, "
        "then compare it with the fitted variants (goals and market-xG targets)",
    )
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
    configs = [presets[v] for v in ns.variants.split(",")]
    codes = tuple(ns.divisions.split(","))
    t0 = time.time()
    prior: dict[str, dict[str, list[Any]]] = {}
    if ns.fit_rest:
        base_cfg = configs[0]
        first = run_configs(db, [base_cfg], codes, ns.workers)[base_cfg.label]
        fitted = []
        for target in ("goals", "market"):
            own, opp, _ = fit_rest(first, ns.fit_rest, ns.holdout_from, target)
            fitted.append(
                replace(
                    base_cfg,
                    rest_mode=ns.fit_rest,
                    rest_own=own,
                    rest_opp=opp,
                    label=f"{ns.fit_rest}_{target}",
                )
            )
            print(f"  -> rest_own={own} rest_opp={opp}")
        configs = [base_cfg, *fitted]
        prior = {base_cfg.label: first}
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
    todo = [c for c in configs if c.label not in prior]
    res = {**prior, **run_configs(db, todo, codes, ns.workers)}
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
            d = np.array(
                [
                    b[1] - a[1]
                    for c in codes
                    for a, b in zip(res[base].get(c, []), res[cfg.label].get(c, []), strict=True)
                    if keep(a)
                ]
            )
            print(
                f"  {name:8s} {d.mean():+.5f} +/- {d.std(ddof=1) / math.sqrt(len(d)):.5f} (1 s.e.)"
            )
        better = 0
        for c in codes:
            g0 = summary[base]["by_division_holdout"][c][0]
            g1 = summary[cfg.label]["by_division_holdout"][c][0]
            better += g1 < g0
            print(f"  {c:4s} holdout gap {g0:+.5f} -> {g1:+.5f}  ({g1 - g0:+.5f})")
        print(f"  {better}/{len(codes)} divisions improved on holdout")
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
