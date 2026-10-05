"""Cross-league cup model: recovers league strength from results, and feeds CL fixtures."""

from __future__ import annotations

import math
import random
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest

from soccer.models.crossleague import ClubRating, CupMatch, fit_cross_league


def _simulate(n: int, strengths: dict[str, float], seed: int = 7) -> list[CupMatch]:
    """Cup ties between average clubs whose leagues differ only by `strengths`."""
    rng = np.random.default_rng(seed)
    leagues = sorted(strengths)
    out = []
    for i in range(n):
        lh, la = leagues[i % len(leagues)], leagues[(i * 7 + 1) % len(leagues)]
        h, a = ClubRating(lh, 1.0, 1.0), ClubRating(la, 1.0, 1.0)
        s = strengths[lh] - strengths[la]
        out.append(
            CupMatch(
                h, a, int(rng.poisson(1.6 * math.exp(s))), int(rng.poisson(1.2 * math.exp(-s)))
            )
        )
    return out


def test_recovers_league_order_and_rates() -> None:
    truth = {"STRONG": 0.4, "MID": 0.0, "WEAK": -0.4}
    model = fit_cross_league(_simulate(3000, truth), ridge=1.0)
    s = model.league_strength
    assert s["STRONG"] > s["MID"] > s["WEAK"]
    assert s["STRONG"] - s["WEAK"] == pytest.approx(0.8, abs=0.15)
    assert model.home_rate == pytest.approx(1.6, rel=0.15)


def test_without_league_effects_every_league_is_equal() -> None:
    model = fit_cross_league(_simulate(500, {"A": 0.5, "B": -0.5}), league_effects=False)
    assert model.league_strength == {}
    h, a = ClubRating("A", 1.0, 1.0), ClubRating("B", 1.0, 1.0)
    lam, mu = model.expected_goals(h, a)
    assert lam == pytest.approx(model.home_rate) and mu == pytest.approx(model.away_rate)


def test_ridge_shrinks_toward_the_average_league() -> None:
    data = _simulate(200, {"A": 0.5, "B": -0.5})
    loose = fit_cross_league(data, ridge=0.1).league_strength
    tight = fit_cross_league(data, ridge=100.0).league_strength
    assert abs(tight["A"] - tight["B"]) < abs(loose["A"] - loose["B"])


def test_no_matches_is_an_error() -> None:
    with pytest.raises(ValueError):
        fit_cross_league([])


# --- through the data layer: a Champions League fixture gets an experimental forecast ----

LEAGUE_A = [f"Alpha {i}" for i in range(10)]
LEAGUE_B = [f"Beta {i}" for i in range(10)]


def _seed_store(path) -> None:
    from soccer.domain.names import normalize_name
    from soccer.sources.football_data_co_uk import MatchResult
    from soccer.storage.analytics_db import AnalyticsDB

    rng = random.Random(3)

    def row(season: str, division: str, d: date, h: str, a: str, hg: int, ag: int):
        return MatchResult(
            season=season, division=division, match_date=d, home=h, away=a,
            home_norm=normalize_name(h), away_norm=normalize_name(a), fthg=hg, ftag=ag,
            ftr="H" if hg > ag else "A" if ag > hg else "D", hthg=None, htag=None,
            home_shots=None, away_shots=None, home_shots_target=None, away_shots_target=None,
            home_corners=None, away_corners=None, home_yellows=None, away_yellows=None,
            home_reds=None, away_reds=None, referee=None,
        )  # fmt: skip

    rows = []
    for season, year in (("2425", 2024), ("2526", 2025)):
        for division, teams in (("E0", LEAGUE_A), ("SP1", LEAGUE_B)):
            day = date(year, 8, 1)
            for h in teams:
                for a in teams:
                    if h != a:
                        rows.append(
                            row(season, division, day, h, a, rng.randint(0, 3), rng.randint(0, 2))
                        )
                        day += timedelta(days=1)
    # Cup history: league A's clubs are clearly stronger than league B's.
    day = date(2025, 9, 15)
    for i in range(120):
        h, a = (
            (LEAGUE_A[i % 10], LEAGUE_B[(i * 3) % 10])
            if i % 2
            else (LEAGUE_B[i % 10], LEAGUE_A[(i * 3) % 10])
        )
        hg, ag = (3, 0) if h in LEAGUE_A else (0, 2)
        rows.append(row("2526", "UCL", day, h, a, hg, ag))
        day += timedelta(days=1)
    with AnalyticsDB(path) as adb:
        adb.load_results(rows)


def test_champions_league_fixture_gets_an_experimental_forecast(tmp_path) -> None:
    from soccer.dashboard.data import cup_model, fixture_forecasts
    from soccer.domain.match_state import MatchStatus
    from soccer.storage.live_db import LiveDB
    from tests.test_dashboard_data import add_match

    analytics, live = tmp_path / "analytics.duckdb", tmp_path / "live.sqlite"
    _seed_store(analytics)
    fitted = cup_model(analytics)
    assert fitted is not None
    model, _ratings = fitted
    assert model.league_strength["E0"] > model.league_strength["SP1"]

    with LiveDB(live) as db:
        add_match(
            db, match_id="cl", home="Beta 1", away="Alpha 2",
            competition="UEFA Champions League", status=MatchStatus.NOT_STARTED,
            observed_at=datetime.now(UTC) + timedelta(days=2),
        )  # fmt: skip
    (fx,) = fixture_forecasts(live, analytics, limit=10)
    assert fx.experimental and fx.slate is not None
    home_p, _draw, away_p = (m.probability for m in fx.slate.result)
    assert away_p > home_p  # the stronger league's club is favoured, even away


def test_too_little_cup_history_means_no_model(tmp_path) -> None:
    from soccer.dashboard.data import cup_model
    from tests.test_dashboard_data import seed_results

    path = tmp_path / "analytics.duckdb"
    seed_results(path, division="E0", teams=["Arsenal", "Chelsea"])
    assert cup_model(path) is None
