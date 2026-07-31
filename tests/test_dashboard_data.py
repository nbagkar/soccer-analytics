"""Dashboard data-layer tests.

The Streamlit render is deliberately thin; the logic worth testing lives in
`dashboard/data.py` -- KPI derivation, filtering, coverage counts, and health mapping.
These run without importing Streamlit.
"""

from __future__ import annotations

from datetime import UTC, datetime

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
        finally:
            config._settings = None
