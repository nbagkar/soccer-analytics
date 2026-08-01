"""Dashboard data-layer tests.

The Streamlit render is deliberately thin; the logic worth testing lives in
`dashboard/data.py` -- KPI derivation, filtering, coverage counts, and health mapping.
These run without importing Streamlit.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from soccer.config import Settings
from soccer.dashboard.data import health_snapshot, live_snapshot
from soccer.domain.crosswalk import EntityResolver
from soccer.domain.match_state import MatchState, MatchStateStore, MatchStatus
from soccer.domain.matches import MatchObservation, MatchResolver, SourceRef
from soccer.sources.registry import SourceId
from soccer.storage.live_db import LiveDB


@pytest.fixture
def db(tmp_path) -> LiveDB:
    return LiveDB(tmp_path / "live.sqlite")


def add_match(
    db: LiveDB,
    *,
    match_id: str,
    home: str,
    away: str,
    competition: str,
    status: MatchStatus,
    source: str = SourceId.THESPORTSDB,
    observed_at: datetime = datetime(2026, 8, 8, 18, 0, tzinfo=UTC),
) -> None:
    resolver = MatchResolver(db, EntityResolver(db))
    resolved = resolver.resolve(
        MatchObservation(
            source=source,
            source_match_id=match_id,
            competition=SourceRef(id=f"c-{competition}", name=competition),
            home=SourceRef(id=f"h-{home}", name=home),
            away=SourceRef(id=f"a-{away}", name=away),
            kickoff=datetime(2026, 8, 8, 18, 0, tzinfo=UTC),
        )
    )
    MatchStateStore(db).upsert(
        resolved.internal_id,
        MatchState(
            status=status,
            home_score=1,
            away_score=0,
            minute="60",
            source=source,
            observed_at=observed_at,
            is_stale=False,
        ),
    )


class TestLiveSnapshot:
    def test_kpis_count_correctly(self, db: LiveDB) -> None:
        add_match(
            db, match_id="1", home="A", away="B", competition="EPL", status=MatchStatus.SECOND_HALF
        )
        add_match(
            db, match_id="2", home="C", away="D", competition="EPL", status=MatchStatus.FINISHED
        )
        add_match(
            db,
            match_id="3",
            home="E",
            away="F",
            competition="La Liga",
            status=MatchStatus.NOT_STARTED,
        )

        snap = live_snapshot(db)
        assert snap.kpis.total == 3
        assert snap.kpis.in_play == 1
        assert snap.kpis.finished == 1
        assert snap.kpis.competitions == 2
        assert snap.kpis.sources == 1

    def test_in_play_filter_narrows_rows_but_not_kpis(self, db: LiveDB) -> None:
        add_match(
            db, match_id="1", home="A", away="B", competition="EPL", status=MatchStatus.SECOND_HALF
        )
        add_match(
            db, match_id="2", home="C", away="D", competition="EPL", status=MatchStatus.FINISHED
        )

        snap = live_snapshot(db, in_play_only=True)
        assert len(snap.matches) == 1  # rows filtered
        assert snap.kpis.total == 2  # KPIs stay honest about the whole set

    def test_competition_filter(self, db: LiveDB) -> None:
        add_match(
            db, match_id="1", home="A", away="B", competition="EPL", status=MatchStatus.SECOND_HALF
        )
        add_match(
            db,
            match_id="2",
            home="E",
            away="F",
            competition="La Liga",
            status=MatchStatus.SECOND_HALF,
        )
        snap = live_snapshot(db, competition="La Liga")
        assert {m.competition for m in snap.matches} == {"La Liga"}

    def test_coverage_counts_sorted_desc(self, db: LiveDB) -> None:
        for i in range(3):
            add_match(
                db,
                match_id=f"e{i}",
                home=f"H{i}",
                away=f"A{i}",
                competition="EPL",
                status=MatchStatus.FINISHED,
            )
        add_match(
            db,
            match_id="l1",
            home="X",
            away="Y",
            competition="La Liga",
            status=MatchStatus.FINISHED,
        )
        snap = live_snapshot(db)
        assert snap.competition_counts[0] == ("EPL", 3)
        assert ("La Liga", 1) in snap.competition_counts

    def test_empty_db_yields_zero_kpis(self, db: LiveDB) -> None:
        snap = live_snapshot(db)
        assert snap.kpis.total == 0
        assert snap.kpis.last_updated is None
        assert snap.kpis.freshness_label == "never"


class TestHealthSnapshot:
    def test_disabled_sources_have_a_reason(self, tmp_path, db: LiveDB) -> None:
        settings = Settings(data_dir=tmp_path, football_data_org_token=None)
        snap = health_snapshot(settings, db)
        by_name = {s.name: s for s in snap.sources}
        # No token -> football-data.org disabled with a reason.
        assert not by_name["football-data.org"].enabled
        assert by_name["football-data.org"].reason == "no token"

    def test_live_source_flagged_live(self, tmp_path, db: LiveDB) -> None:
        settings = Settings(data_dir=tmp_path)
        snap = health_snapshot(settings, db)
        tsdb = next(s for s in snap.sources if s.name == "TheSportsDB")
        assert tsdb.is_live
        assert "live" in tsdb.latency_label

    def test_unresolved_licence_flagged(self, tmp_path, db: LiveDB) -> None:
        settings = Settings(data_dir=tmp_path)
        snap = health_snapshot(settings, db)
        # football-data.co.uk has no explicit licence; must be flagged.
        couk = next(s for s in snap.sources if "co.uk" in s.name)
        assert couk.licence_unresolved

    def test_unavailable_capabilities_shown(self, tmp_path, db: LiveDB) -> None:
        settings = Settings(data_dir=tmp_path)
        snap = health_snapshot(settings, db)
        by_cap = {c.capability: c for c in snap.coverage}
        # Nothing free provides live xG -> not available, honestly surfaced.
        assert by_cap["expected_goals"].available is False


class TestAnalyticsSnapshot:
    def _seed_results(self, path) -> None:
        from soccer.domain.names import normalize_name
        from soccer.sources.football_data_co_uk import MatchResult
        from soccer.storage.analytics_db import AnalyticsDB

        def r(home, away, hg, ag, day):
            return MatchResult(
                season="2526",
                division="E0",
                match_date=date(2026, 1, day),
                home=home,
                away=away,
                home_norm=normalize_name(home),
                away_norm=normalize_name(away),
                fthg=hg,
                ftag=ag,
                ftr="H" if hg > ag else "A" if ag > hg else "D",
                hthg=None,
                htag=None,
                home_shots=None,
                away_shots=None,
                home_shots_target=None,
                away_shots_target=None,
                home_corners=None,
                away_corners=None,
                home_yellows=None,
                away_yellows=None,
                home_reds=None,
                away_reds=None,
                referee=None,
            )

        teams = ["Arsenal", "Chelsea", "Fulham", "Brentford"]
        rows, day = [], 1
        for i, h in enumerate(teams):
            for a in teams[i + 1 :]:
                hg, ag = (3, 0) if h == "Arsenal" else (0, 2) if h == "Brentford" else (1, 1)
                rows.append(r(h, a, hg, ag, day))
                rows.append(r(a, h, ag, hg, day + 1))
                day += 2
        with AnalyticsDB(path) as adb:
            adb.load_results(rows)

    def test_snapshot_has_table_ranking_and_odds(self, tmp_path) -> None:
        from soccer.dashboard.data import analytics_snapshot

        path = tmp_path / "analytics.duckdb"
        self._seed_results(path)
        snap = analytics_snapshot(path, "2526", "E0", sims=500, seed=1)
        assert snap is not None
        assert len(snap.table) == 4
        assert len(snap.power) == 4
        assert sum(p.title_pct for p in snap.title_odds) == pytest.approx(1.0, abs=1e-6)
        assert snap.title_odds[0].team == "arsenal"

    def test_missing_slice_returns_none(self, tmp_path) -> None:
        from soccer.dashboard.data import analytics_snapshot

        path = tmp_path / "analytics.duckdb"
        self._seed_results(path)
        assert analytics_snapshot(path, "9999", "ZZ") is None

    def test_available_lists_loaded_slices(self, tmp_path) -> None:
        from soccer.dashboard.data import analytics_available

        path = tmp_path / "analytics.duckdb"
        assert analytics_available(path) == []
        self._seed_results(path)
        assert ("2526", "E0", 12) in analytics_available(path)


class TestShotMap:
    def _seed_shots(self, path) -> None:
        from soccer.sources.statsbomb import Shot
        from soccer.storage.analytics_db import AnalyticsDB

        shots = [
            Shot(
                1, "Argentina", "Messi", 23, 1, 110.0, 40.0, 0.35, "Goal", True, False, "Left Foot"
            ),
            Shot(1, "France", "Mbappé", 80, 2, 108.0, 44.0, 0.76, "Goal", True, True, None),
        ]
        with AnalyticsDB(path) as adb:
            adb.load_shots(shots)

    def test_shot_matches_lists_loaded(self, tmp_path) -> None:
        from soccer.dashboard.data import shot_matches

        path = tmp_path / "analytics.duckdb"
        assert shot_matches(path) == []
        self._seed_shots(path)
        matches = shot_matches(path)
        assert matches[0][0] == 1
        assert "Argentina" in matches[0][1] and "France" in matches[0][1]

    def test_shot_map_returns_shots_and_xg(self, tmp_path) -> None:
        from soccer.dashboard.data import shot_map

        path = tmp_path / "analytics.duckdb"
        self._seed_shots(path)
        data = shot_map(path, 1)
        assert data is not None
        assert len(data.shots) == 2
        assert len(data.team_xg) == 2

    def test_shot_map_missing_returns_none(self, tmp_path) -> None:
        from soccer.dashboard.data import shot_map

        path = tmp_path / "analytics.duckdb"
        self._seed_shots(path)
        assert shot_map(path, 999) is None


class TestAppSmoke:
    """One end-to-end render check so a broken st.* call cannot slip through.

    Streamlit's AppTest actually executes the script (unlike an HTTP fetch of the shell
    HTML), so this catches render-layer regressions the data-layer tests cannot.
    """

    def test_both_pages_render_without_exception(self, tmp_path, monkeypatch) -> None:
        # Streamlit is an optional extra; skip cleanly when only [dev] is installed.
        pytest.importorskip("streamlit")

        from importlib.resources import files

        import soccer.config as config

        # Point the app at a small throwaway database.
        monkeypatch.setenv("SOCCER_DATA_DIR", str(tmp_path))
        config._settings = None
        try:
            build = LiveDB(tmp_path / "live.sqlite")
            add_match(
                build,
                match_id="1",
                home="Arsenal FC",
                away="Chelsea FC",
                competition="Premier League",
                status=MatchStatus.SECOND_HALF,
            )
            add_match(
                build,
                match_id="2",
                home="Ajax",
                away="PSV",
                competition="Eredivisie",
                status=MatchStatus.FINISHED,
            )
            build.close()

            from streamlit.testing.v1 import AppTest

            app_path = str(files("soccer.dashboard") / "app.py")
            # Generous timeout: the first Streamlit run in a process pays cold-start.
            at = AppTest.from_file(app_path, default_timeout=120).run()
            assert not at.exception, f"Live Centre raised: {at.exception}"
            assert any(m.label == "Matches" for m in at.metric)

            at.radio[0].set_value("Data Health").run()
            assert not at.exception, f"Data Health raised: {at.exception}"

            # Analytics page needs an analytics DB; seed one and visit it.
            TestAnalyticsSnapshot()._seed_results(tmp_path / "analytics.duckdb")
            at.radio[0].set_value("Analytics").run()
            assert not at.exception, f"Analytics raised: {at.exception}"

            # Shot Map page needs shots; seed and visit.
            TestShotMap()._seed_shots(tmp_path / "analytics.duckdb")
            at.radio[0].set_value("Shot Map").run()
            assert not at.exception, f"Shot Map raised: {at.exception}"
        finally:
            config._settings = None
