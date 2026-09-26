"""Dashboard data-layer tests.

The Streamlit render is deliberately thin; the logic worth testing lives in
`dashboard/data.py` -- KPI derivation, filtering, coverage counts, and health mapping.
These run without importing Streamlit.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

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
    observed_at: datetime | None = None,
) -> None:
    # Relative to "now" so the fixture doesn't age out of the 7-day "recent" window this
    # module tests against -- a fixed past date silently fell outside it once enough real
    # time had passed (was 2026-08-08, tests started failing by 2026-09-02). NOT_STARTED
    # defaults to a future kickoff instead: `upcoming()` filters out anything already kicked
    # off regardless of cached status, so a "yesterday" default would make every NOT_STARTED
    # fixture invisible to it unless a caller overrides `observed_at` explicitly. In-play
    # statuses default to a recent kickoff too: `list_current(in_play_only=True)` now drops
    # anything older than MAX_PLAUSIBLE_MATCH_AGE, so a "yesterday" default would make every
    # in-play fixture invisible to it as well.
    if observed_at is None:
        if status is MatchStatus.NOT_STARTED:
            observed_at = datetime.now(UTC) + timedelta(days=1)
        elif status.is_in_play:
            observed_at = datetime.now(UTC) - timedelta(minutes=45)
        else:
            observed_at = datetime.now(UTC) - timedelta(days=1)
    resolver = MatchResolver(db, EntityResolver(db))
    resolved = resolver.resolve(
        MatchObservation(
            source=source,
            source_match_id=match_id,
            competition=SourceRef(id=f"c-{competition}", name=competition),
            home=SourceRef(id=f"h-{home}", name=home),
            away=SourceRef(id=f"a-{away}", name=away),
            kickoff=observed_at,
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


def seed_results(path, *, division: str, teams: list[str], season: str = "2526") -> None:
    """Seed a division's double round-robin so models have a connected, fittable graph."""
    from soccer.domain.names import normalize_name
    from soccer.sources.football_data_co_uk import MatchResult
    from soccer.storage.analytics_db import AnalyticsDB

    rows, day = [], 1
    for i, h in enumerate(teams):
        for a in teams[i + 1 :]:
            for home, away, hg, ag in ((h, a, 2, 0), (a, h, 1, 1)):
                rows.append(
                    MatchResult(
                        season=season,
                        division=division,
                        match_date=date(2026, 1, (day % 27) + 1),
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
                )
            day += 1
    with AnalyticsDB(path) as adb:
        adb.load_results(rows)


def seed_player_events(path) -> None:
    """Seed full-event player stats + matching shots so the rich Players view has data."""
    from soccer.sources.statsbomb import PlayerMatchStats, Shot
    from soccer.storage.analytics_db import AnalyticsDB

    def pms(player: str, team: str, **kw) -> PlayerMatchStats:
        base = dict(
            match_id=1,
            player=player,
            team=team,
            position=kw.pop("position", "Center Midfield"),
            minutes=kw.pop("minutes", 300),
            passes=0,
            passes_completed=0,
            key_passes=0,
            assists=0,
            xa=0.0,
            progressive_passes=0,
            carries=0,
            progressive_carries=0,
            dribbles=0,
            dribbles_completed=0,
            tackles=0,
            tackles_won=0,
            interceptions=0,
            blocks=0,
            clearances=0,
            ball_recoveries=0,
            pressures=0,
            fouls=0,
            fouled=0,
            yellow_cards=0,
            red_cards=0,
            touches=0,
        )
        base.update(kw)
        return PlayerMatchStats(**base)

    stats = [
        pms(
            "Messi",
            "Argentina",
            passes=200,
            passes_completed=170,
            key_passes=10,
            assists=3,
            xa=2.0,
            progressive_passes=40,
            progressive_carries=30,
            dribbles_completed=20,
        ),
        pms(
            "Otamendi",
            "Argentina",
            position="Center Back",
            passes=150,
            passes_completed=135,
            tackles=8,
            interceptions=6,
            blocks=4,
            clearances=20,
            ball_recoveries=15,
        ),
    ]
    shots = [
        Shot(1, "Argentina", "Messi", 23, 1, 110.0, 40.0, 0.35, "Goal", True, False, "Left Foot"),
    ]
    with AnalyticsDB(path) as adb:
        adb.load_player_stats(stats)
        adb.load_shots(shots)


class TestLiveSnapshot:
    def test_kpis_reflect_the_live_view(self, db: LiveDB) -> None:
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

        # A live match exists, so the live centre shows the live view -- KPIs describe that,
        # not the whole tracked set (which is dominated by fixtures and old results).
        snap = live_snapshot(db)
        assert snap.mode == "live"
        assert snap.kpis.in_play == 1
        assert snap.kpis.total == 1
        assert snap.kpis.competitions == 1
        assert snap.kpis.sources == 1

    def test_live_view_excludes_finished_while_something_is_on(self, db: LiveDB) -> None:
        add_match(
            db, match_id="1", home="A", away="B", competition="EPL", status=MatchStatus.SECOND_HALF
        )
        add_match(
            db, match_id="2", home="C", away="D", competition="EPL", status=MatchStatus.FINISHED
        )

        snap = live_snapshot(db)
        assert snap.mode == "live"
        assert len(snap.matches) == 1
        assert snap.matches[0].status.is_in_play

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


class TestLiveKpisAging:
    """`is_aging` flags a refresh old enough that its "in play" matches are more likely
    finished than actually live -- nothing re-polls them without `soccer serve` running.
    Tested directly on the dataclass: it depends purely on wall-clock age vs `last_updated`,
    which a DB round-trip can't control precisely (the store always stamps `updated_at` with
    the real current time, not a test's simulated one)."""

    def _kpis(self, *, in_play: int, last_updated):
        from soccer.dashboard.data import LiveKpis

        return LiveKpis(
            total=in_play,
            in_play=in_play,
            finished=0,
            competitions=1,
            sources=1,
            last_updated=last_updated,
            any_stale=False,
        )

    def test_aging_when_old_and_something_in_play(self) -> None:
        from soccer.dashboard.data import STALE_LIVE_AGE_MINUTES

        old = datetime.now(UTC) - timedelta(minutes=STALE_LIVE_AGE_MINUTES + 1)
        assert self._kpis(in_play=3, last_updated=old).is_aging

    def test_not_aging_when_recent(self) -> None:
        from soccer.dashboard.data import STALE_LIVE_AGE_MINUTES

        recent = datetime.now(UTC) - timedelta(minutes=STALE_LIVE_AGE_MINUTES - 1)
        assert not self._kpis(in_play=3, last_updated=recent).is_aging

    def test_not_aging_when_nothing_in_play(self) -> None:
        from soccer.dashboard.data import STALE_LIVE_AGE_MINUTES

        # A purely historical listing doesn't go stale by going untouched -- only a claimed
        # "in play" status can be undermined by time passing.
        old = datetime.now(UTC) - timedelta(minutes=STALE_LIVE_AGE_MINUTES + 1)
        assert not self._kpis(in_play=0, last_updated=old).is_aging

    def test_not_aging_when_never_updated(self) -> None:
        assert not self._kpis(in_play=0, last_updated=None).is_aging


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
        # FPL carries Opta xG but is grey-area and off by default; with it off, no *free*
        # source provides live xG, so the capability is honestly surfaced as unavailable.
        # Pin the flag rather than inherit the ambient .env so the assertion is deterministic.
        settings = Settings(data_dir=tmp_path, enable_fpl=False)
        snap = health_snapshot(settings, db)
        by_cap = {c.capability: c for c in snap.coverage}
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


class TestTrends:
    def test_team_form_computed_and_sorted(self, tmp_path) -> None:
        from soccer.dashboard.data import team_form

        path = tmp_path / "analytics.duckdb"
        TestAnalyticsSnapshot()._seed_results(path)  # Arsenal strong, Brentford weak in E0 2526
        forms = team_form(path, "2526", "E0", last_n=3)
        assert forms
        by = {f.team: f for f in forms}
        assert by["Arsenal"].ppg > by["Brentford"].ppg  # strong team, higher PPG
        assert all(len(f.recent_form) <= 3 for f in forms)  # window respected
        assert forms[0].recent_ppg >= forms[-1].recent_ppg  # hottest first
        # over25/btts rates are valid fractions
        assert all(0.0 <= f.over25_rate <= 1.0 and 0.0 <= f.btts_rate <= 1.0 for f in forms)

    def test_team_form_empty_for_unknown_slice(self, tmp_path) -> None:
        from soccer.dashboard.data import team_form

        path = tmp_path / "analytics.duckdb"
        TestAnalyticsSnapshot()._seed_results(path)
        assert team_form(path, "9999", "ZZ") == []


class TestSeasonBriefing:
    def test_projects_full_table(self, tmp_path) -> None:
        from soccer.dashboard.data import season_briefing

        path = tmp_path / "analytics.duckdb"
        TestAnalyticsSnapshot()._seed_results(path)  # Arsenal strong, Brentford weak
        briefing = season_briefing(path, "2526", "E0", n_sims=300, seed=1)
        assert briefing is not None
        assert len(briefing.projections) == 4  # the four seeded teams
        # Someone wins every simulated season -> title probabilities sum to 1.
        assert sum(p.title_pct for p in briefing.projections) == pytest.approx(1.0, abs=1e-6)
        # The strongest seeded team is the title favourite.
        favourite = max(briefing.projections, key=lambda p: p.title_pct)
        assert briefing.names[favourite.team] == "Arsenal"

    def test_none_for_unknown_slice(self, tmp_path) -> None:
        from soccer.dashboard.data import season_briefing

        path = tmp_path / "analytics.duckdb"
        TestAnalyticsSnapshot()._seed_results(path)
        assert season_briefing(path, "9999", "ZZ") is None


class TestSeasonRecords:
    def test_streaks_and_notable_matches(self, tmp_path) -> None:
        from soccer.dashboard.data import season_records

        path = tmp_path / "analytics.duckdb"
        TestAnalyticsSnapshot()._seed_results(path)  # Arsenal win 3-0s, Brentford lose 0-2s
        records = season_records(path, "2526", "E0")
        assert records is not None
        assert records.streaks  # non-empty, sorted by active unbeaten
        assert records.streaks == sorted(records.streaks, key=lambda s: (-s.unbeaten, -s.winning))
        # The biggest win is by the largest margin.
        top = records.biggest_wins[0]
        assert top.margin == max(m.margin for m in records.biggest_wins)
        # Highest scoring is ordered by total goals.
        assert records.highest_scoring[0].total >= records.highest_scoring[-1].total

    def test_none_for_unknown_slice(self, tmp_path) -> None:
        from soccer.dashboard.data import season_records

        path = tmp_path / "analytics.duckdb"
        TestAnalyticsSnapshot()._seed_results(path)
        assert season_records(path, "9999", "ZZ") is None


class TestDataActions:
    def test_status_counts_loaded_leagues(self, tmp_path) -> None:
        from soccer.config import Settings
        from soccer.dashboard.actions import data_status

        settings = Settings(data_dir=tmp_path)
        seed_results(settings.analytics_db, division="E0", teams=["Arsenal", "Chelsea"])
        status = data_status(settings)
        assert status["leagues"] == 1
        assert status["history_matches"] == 2  # two teams, home and away
        assert status["player_competitions"] == 0

    def test_status_empty_when_nothing_loaded(self, tmp_path) -> None:
        from soccer.config import Settings
        from soccer.dashboard.actions import data_status

        status = data_status(Settings(data_dir=tmp_path / "empty"))
        assert status == {
            "leagues": 0,
            "history_matches": 0,
            "player_competitions": 0,
            "upcoming": 0,
            "squad_players": 0,
            "injuries": 0,
        }

    def test_load_full_history_honours_depth_and_is_idempotent(self, tmp_path, monkeypatch) -> None:
        # A fake source with history for one league only: every requested season yields one
        # match, other leagues 404 (return []). Proves the depth flows through and that a
        # second run replaces units rather than duplicating them.
        from datetime import date

        from soccer.config import Settings
        from soccer.dashboard import actions
        from soccer.domain.names import normalize_name
        from soccer.sources.football_data_co_uk import MatchResult
        from soccer.storage.analytics_db import AnalyticsDB

        def mr(season: str) -> MatchResult:
            return MatchResult(
                season=season,
                division="E0",
                match_date=date(2020, 1, 1),
                home="Arsenal",
                away="Chelsea",
                home_norm=normalize_name("Arsenal"),
                away_norm=normalize_name("Chelsea"),
                fthg=2,
                ftag=0,
                ftr="H",
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

        class _FakeSource:
            def __init__(self, *a, **k) -> None: ...
            def __enter__(self):
                return self

            def __exit__(self, *a) -> bool:
                return False

            def fetch_division(self, season, division):
                return [mr(season)] if division == "E0" else []

            def fetch_new_league(self, code, *, recent_seasons):
                return []

        monkeypatch.setattr(actions, "FootballDataCoUk", _FakeSource)
        settings = Settings(data_dir=tmp_path)

        actions.load_full_history(settings, seasons=6)
        with AnalyticsDB(settings.analytics_db) as adb:
            loaded = adb.seasons_loaded()
        assert len({s for s, d, _n in loaded if d == "E0"}) == 6  # depth honoured
        assert sum(n for _s, _d, n in loaded) == 6  # one match per season, no others

        actions.load_full_history(settings, seasons=6)  # re-run
        with AnalyticsDB(settings.analytics_db) as adb:
            assert sum(n for _s, _d, n in adb.seasons_loaded()) == 6  # idempotent, not 12


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

    def test_xg_timeline_accumulates_per_team(self, tmp_path) -> None:
        from soccer.dashboard.data import shot_map

        path = tmp_path / "analytics.duckdb"
        self._seed_shots(path)  # Messi/Argentina 0.35, Mbappé/France 0.76
        data = shot_map(path, 1)
        assert data.timeline[0]["cum_xg"] == 0.0  # each team starts at zero
        finals = {}
        for point in data.timeline:
            finals[point["team"]] = point["cum_xg"]
        assert finals["Argentina"] == pytest.approx(0.35)
        assert finals["France"] == pytest.approx(0.76)

    def test_shot_matches_carries_competition(self, tmp_path) -> None:
        from soccer.dashboard.data import shot_matches

        path = tmp_path / "analytics.duckdb"
        self._seed_shots(path)
        matches = shot_matches(path)
        assert len(matches[0]) == 4  # (match_id, label, competition, season)
        assert matches[0][2] == "Other"  # no metadata seeded -> falls back honestly
        assert matches[0][3] == ""  # season blank without metadata


class TestForecastData:
    def test_forecast_teams_lists_names(self, tmp_path) -> None:
        from soccer.dashboard.data import forecast_teams

        path = tmp_path / "analytics.duckdb"
        TestAnalyticsSnapshot()._seed_results(path)
        teams = forecast_teams(path, "2526", "E0")
        assert "Arsenal" in teams and "Brentford" in teams

    def test_forecast_slate_returns_markets(self, tmp_path) -> None:
        from soccer.dashboard.data import forecast_slate

        path = tmp_path / "analytics.duckdb"
        TestAnalyticsSnapshot()._seed_results(path)
        slate = forecast_slate(path, "2526", "E0", "Arsenal", "Brentford")
        assert slate is not None
        assert sum(m.probability for m in slate.result) == pytest.approx(1.0, abs=1e-6)
        # Arsenal (strong) favoured over Brentford (weak).
        result = {m.name: m.probability for m in slate.result}
        assert result["Arsenal"] > result["Brentford"]

    def test_forecast_slate_unknown_team_is_none(self, tmp_path) -> None:
        from soccer.dashboard.data import forecast_slate

        path = tmp_path / "analytics.duckdb"
        TestAnalyticsSnapshot()._seed_results(path)
        assert forecast_slate(path, "2526", "E0", "Arsenal", "Nobody") is None

    def test_forecast_explanation_attributes_the_forecast(self, tmp_path) -> None:
        from soccer.dashboard.data import forecast_explanation

        path = tmp_path / "analytics.duckdb"
        TestAnalyticsSnapshot()._seed_results(path)
        exp = forecast_explanation(path, "2526", "E0", "Arsenal", "Brentford")
        assert exp is not None
        # Strong home side out-projects the weak away side, and its attack rates higher.
        assert exp.home_xg > exp.away_xg
        assert exp.home_factor.attack > exp.away_factor.attack
        assert exp.confidence in ("High", "Moderate", "Low")
        assert exp.summary  # a non-empty plain-English "why"
        assert forecast_explanation(path, "2526", "E0", "Arsenal", "Nobody") is None


class TestAvailabilityAdjustedSlate:
    """The PL-only forecast nudge: raw and adjusted side by side, moving the number toward the
    side of the ball a club's absences belong to -- and a strict no-op (None) otherwise."""

    def _seed_availability(self, live_path, team, players) -> None:
        """players: (name, status, element_type, price, chance) tuples for one club."""
        from soccer.domain.availability import (
            AvailabilityStore,
            PlayerAvailability,
            status_label,
        )
        from soccer.domain.names import normalize_name

        recs = [
            PlayerAvailability(
                source="fpl",
                team=team,
                team_norm=normalize_name(team),
                player=name,
                full_name=None,
                status=status,
                availability=status_label(status),
                chance=chance,
                news=None,
                news_added=None,
                fetched_at="2026-08-04T00:00:00+00:00",
                element_type=etype,
                price=price,
            )
            for (name, status, etype, price, chance) in players
        ]
        with LiveDB(live_path) as db:
            AvailabilityStore(db).replace_source("fpl", recs)

    # A small but plausible Brentford squad; the two forwards are the ones we injure.
    _BRENTFORD_FIT = (
        ("Flekken", "a", 1, 45, None),  # GK
        ("Collins", "a", 2, 45, None),  # DEF
        ("Pinnock", "a", 2, 45, None),  # DEF
        ("Janelt", "a", 3, 50, None),  # MID
        ("Norgaard", "a", 3, 50, None),  # MID
        ("Mbeumo", "a", 4, 70, None),  # FWD
        ("Wissa", "a", 4, 65, None),  # FWD
    )

    def test_injured_forwards_shift_the_forecast_toward_the_opponent(self, tmp_path) -> None:
        from soccer.dashboard.data import availability_adjusted_slate

        analytics = tmp_path / "analytics.duckdb"
        live = tmp_path / "live.sqlite"
        TestAnalyticsSnapshot()._seed_results(analytics)  # Arsenal strong, Brentford weak
        # Brentford lose both strikers; Arsenal have no news at all.
        squad = [
            (n, "i" if n in ("Mbeumo", "Wissa") else s, e, p, c)
            for (n, s, e, p, c) in self._BRENTFORD_FIT
        ]
        self._seed_availability(live, "Brentford", squad)

        adj = availability_adjusted_slate(analytics, live, "2526", "E0", "Arsenal", "Brentford")
        assert adj is not None
        assert adj.is_material
        # Arsenal (home) carry no availability rows -> a strict neutral adjustment.
        assert not adj.home_adj.is_material
        # Brentford's loss is on the attack, so their expected goals fall...
        assert adj.away_adj.attack_factor < 1.0
        assert adj.away_adj.attack_factor >= 0.75  # bounded
        assert "Mbeumo" in adj.away_adj.missing
        assert adj.adjusted.away_expected < adj.raw.away_expected
        # ...and Arsenal's win probability rises versus the base model.
        raw = {m.name: m.probability for m in adj.raw.result}
        moved = {m.name: m.probability for m in adj.adjusted.result}
        assert moved["Arsenal"] > raw["Arsenal"]
        assert moved["Brentford"] < raw["Brentford"]

    def test_none_when_nobody_is_flagged(self, tmp_path) -> None:
        from soccer.dashboard.data import availability_adjusted_slate

        analytics = tmp_path / "analytics.duckdb"
        live = tmp_path / "live.sqlite"
        TestAnalyticsSnapshot()._seed_results(analytics)
        self._seed_availability(live, "Brentford", self._BRENTFORD_FIT)  # all available
        # A fully fit pair is identical to the plain forecast, so there is nothing to compare.
        assert availability_adjusted_slate(
            analytics, live, "2526", "E0", "Arsenal", "Brentford"
        ) is None

    def test_none_outside_the_premier_league(self, tmp_path) -> None:
        from soccer.dashboard.data import availability_adjusted_slate

        analytics = tmp_path / "analytics.duckdb"
        live = tmp_path / "live.sqlite"
        TestAnalyticsSnapshot()._seed_results(analytics)
        self._seed_availability(live, "Brentford", self._BRENTFORD_FIT)
        # The availability feed is PL-only, so a non-E0 division never gets the nudge.
        assert availability_adjusted_slate(
            analytics, live, "2526", "SP1", "Arsenal", "Brentford"
        ) is None

    def test_matches_the_plain_slate_on_the_base_numbers(self, tmp_path) -> None:
        from soccer.dashboard.data import availability_adjusted_slate, forecast_slate

        analytics = tmp_path / "analytics.duckdb"
        live = tmp_path / "live.sqlite"
        TestAnalyticsSnapshot()._seed_results(analytics)
        squad = [
            (n, "i" if n == "Mbeumo" else s, e, p, c) for (n, s, e, p, c) in self._BRENTFORD_FIT
        ]
        self._seed_availability(live, "Brentford", squad)
        adj = availability_adjusted_slate(analytics, live, "2526", "E0", "Arsenal", "Brentford")
        plain = forecast_slate(analytics, "2526", "E0", "Arsenal", "Brentford")
        # The "raw" half must be exactly the plain forecast -- same seam, no drift.
        assert adj is not None and plain is not None
        assert adj.raw.home_expected == pytest.approx(plain.home_expected)
        assert adj.raw.away_expected == pytest.approx(plain.away_expected)

    def test_confirmed_lineup_catches_an_absence_fpl_missed(self, tmp_path) -> None:
        from soccer.dashboard.data import availability_adjusted_slate
        from soccer.domain.availability import (
            AvailabilityStore,
            ConfirmedLineupStore,
            PlayerAvailability,
            status_label,
        )
        from soccer.domain.names import normalize_name

        analytics = tmp_path / "analytics.duckdb"
        live = tmp_path / "live.sqlite"
        TestAnalyticsSnapshot()._seed_results(analytics)

        # FPL's season-long snapshot shows the whole Brentford squad "available" (it hasn't
        # caught a late change), but season-to-date minutes mark Mbeumo a clear regular.
        minutes_by_player = {
            "Flekken": 2700, "Collins": 2600, "Pinnock": 2500,
            "Janelt": 2400, "Norgaard": 2300, "Mbeumo": 2700, "Wissa": 2600,
        }
        squad = [
            PlayerAvailability(
                source="fpl",
                team="Brentford",
                team_norm=normalize_name("Brentford"),
                player=name,
                full_name=None,
                status="a",
                availability=status_label("a"),
                chance=None,
                news=None,
                news_added=None,
                fetched_at="2026-08-04T00:00:00+00:00",
                element_type=4 if name in ("Mbeumo", "Wissa") else 2,
                price=70,
                minutes=minutes,
            )
            for name, minutes in minutes_by_player.items()
        ]
        with LiveDB(live) as db:
            AvailabilityStore(db).replace_source("fpl", squad)
            # Today's confirmed matchday squad -- announced close to kickoff -- doesn't
            # include Mbeumo at all (a late, undisclosed absence FPL hasn't reflected yet).
            confirmed = [p.player for p in squad if p.player != "Mbeumo"]
            ConfirmedLineupStore(db).replace_team(
                normalize_name("Brentford"), confirmed, "2026-08-04T12:00:00+00:00"
            )

        adj = availability_adjusted_slate(analytics, live, "2526", "E0", "Arsenal", "Brentford")
        assert adj is not None
        assert adj.confirmed_lineup_used
        assert adj.away_adj.is_material  # FPL alone would have said "nobody flagged"
        assert "Mbeumo" in adj.away_adj.missing

    def test_no_confirmed_lineup_cached_falls_back_to_fpl_only(self, tmp_path) -> None:
        from soccer.dashboard.data import availability_adjusted_slate

        analytics = tmp_path / "analytics.duckdb"
        live = tmp_path / "live.sqlite"
        TestAnalyticsSnapshot()._seed_results(analytics)
        squad = [
            (n, "i" if n == "Mbeumo" else s, e, p, c) for (n, s, e, p, c) in self._BRENTFORD_FIT
        ]
        self._seed_availability(live, "Brentford", squad)  # no ConfirmedLineupStore data at all
        adj = availability_adjusted_slate(analytics, live, "2526", "E0", "Arsenal", "Brentford")
        assert adj is not None
        assert not adj.confirmed_lineup_used


class TestUnderlyingTable:
    def _seed_with_shots(self, path):
        from datetime import date

        from soccer.domain.names import normalize_name
        from soccer.sources.football_data_co_uk import MatchResult
        from soccer.storage.analytics_db import AnalyticsDB

        def mr(h, a, hg, ag, hst, ast, day):
            return MatchResult(
                season="2526", division="E0", match_date=date(2026, 1, day),
                home=h, away=a, home_norm=normalize_name(h), away_norm=normalize_name(a),
                fthg=hg, ftag=ag, ftr="H" if hg > ag else "A" if ag > hg else "D",
                hthg=None, htag=None, home_shots=None, away_shots=None,
                home_shots_target=hst, away_shots_target=ast, home_corners=None,
                away_corners=None, home_yellows=None, away_yellows=None, home_reds=None,
                away_reds=None, referee=None,
            )

        # A dominates chances (SoT 10/9/8) but converts poorly; B barely creates.
        rows = [
            mr("A", "B", 1, 0, 10, 2, 1),
            mr("B", "A", 0, 1, 3, 9, 2),
            mr("A", "B", 1, 1, 8, 4, 3),
        ]
        with AnalyticsDB(path) as adb:
            adb.load_results(rows)

    def test_none_without_shot_data(self, tmp_path) -> None:
        from soccer.dashboard.data import underlying_table

        path = tmp_path / "analytics.duckdb"
        TestAnalyticsSnapshot()._seed_results(path)  # seeded results carry no shots
        assert underlying_table(path, "2526", "E0") is None

    def test_expected_points_reflect_chance_quality(self, tmp_path) -> None:
        from soccer.dashboard.data import underlying_table

        path = tmp_path / "analytics.duckdb"
        self._seed_with_shots(path)
        table = underlying_table(path, "2526", "E0")
        assert table is not None and len(table) == 2
        teams = {r.team: r for r in table}
        assert teams["A"].played == 3 and teams["B"].played == 3
        assert teams["A"].xpoints > 0 and teams["B"].xpoints > 0
        # A creates far more, so it deserves more points than B regardless of finishing.
        assert teams["A"].xpoints > teams["B"].xpoints


class TestTeamDossier:
    def test_gathers_a_team(self, tmp_path) -> None:
        from soccer.dashboard.data import team_dossier

        path = tmp_path / "analytics.duckdb"
        TestAnalyticsSnapshot()._seed_results(path)
        d = team_dossier(path, "E0", "2526", "Arsenal")
        assert d is not None
        assert d.team == "Arsenal" and d.position == 1  # strongest seeded team leads
        assert d.played > 0 and d.points > 0
        assert d.recent  # its recent results
        # cumulative trajectory ends at the season points total
        assert d.trajectory and d.trajectory[-1]["points"] == d.points
        assert d.xpoints is None  # seed carries no shots, so no expected points
        assert team_dossier(path, "E0", "2526", "Nobody") is None


class TestLeagueHistory:
    def test_all_time_leaderboards(self, tmp_path) -> None:
        from soccer.dashboard.data import league_history

        path = tmp_path / "analytics.duckdb"
        teams = ["Arsenal", "Chelsea", "Fulham", "Brentford"]
        seed_results(path, division="E0", teams=teams, season="2425")
        seed_results(path, division="E0", teams=teams, season="2526")
        hist = league_history(path, "E0")
        assert hist is not None
        assert hist.seasons == 2
        # Same seeded shape both seasons, so one team tops both -> two titles.
        assert hist.title_counts[0][1] == 2
        assert hist.record_points is not None
        assert hist.biggest_wins and hist.highest_scoring
        assert league_history(path, "ZZ9") is None


class TestCanonicalNames:
    def test_resolve_bridges_aliases_and_overrides_bad_subset(self) -> None:
        from soccer.dashboard.data import resolve_canonical_name

        registry = {
            "bayern munich": "Bayern Munich",
            "paris sg": "Paris SG",
            "paris": "Paris FC",  # the new Ligue 1 club a subset would wrongly grab
            "dortmund": "Dortmund",
        }
        # curated alias bridges the verbose football-data.org name
        assert resolve_canonical_name("FC Bayern München", registry) == (
            "Bayern Munich",
            "bayern munich",
        )
        # the alias wins over a token-subset, so PSG never becomes Paris FC
        assert resolve_canonical_name("Paris Saint-Germain FC", registry)[0] == "Paris SG"
        # a clean token-subset still bridges
        assert resolve_canonical_name("Borussia Dortmund", registry)[0] == "Dortmund"
        # a club from an unloaded league keeps its own name
        assert resolve_canonical_name("Shakhtar Donetsk", registry)[0] == "Shakhtar Donetsk"

    def test_racing_santander_alias_beats_ambiguous_subset(self) -> None:
        from soccer.dashboard.data import resolve_canonical_name

        # with Argentina loaded, "racing club" makes the token-subset ambiguous
        registry = {"santander": "Santander", "racing club": "Racing Club"}
        assert resolve_canonical_name("Real Racing Club de Santander", registry)[0] == "Santander"


class TestLeagueProfile:
    def test_style_fingerprint(self, tmp_path) -> None:
        from soccer.dashboard.data import league_profile

        path = tmp_path / "analytics.duckdb"
        seed_results(path, division="E0", teams=["Arsenal", "Chelsea", "Fulham", "Brentford"])
        prof = league_profile(path, "2526", "E0")
        assert prof is not None
        assert prof.played > 0
        # The result split is a partition of 100%, and rates stay in range.
        assert abs(prof.home_win_pct + prof.draw_pct + prof.away_win_pct - 100.0) < 1e-6
        assert 0.0 <= prof.over25_pct <= 100.0
        assert prof.goals_per_game >= 0.0
        assert league_profile(path, "2526", "ZZ9") is None


class TestFixtureForecasts:
    def test_upcoming_orders_and_filters(self, db: LiveDB) -> None:
        add_match(
            db, match_id="1", home="A", away="B", competition="EPL", status=MatchStatus.FINISHED
        )
        add_match(
            db, match_id="2", home="C", away="D", competition="EPL", status=MatchStatus.NOT_STARTED
        )
        ups = MatchStateStore(db).upcoming()
        assert [v.status for v in ups] == [MatchStatus.NOT_STARTED]
        assert ups[0].home == "C"

    def test_upcoming_drops_a_stale_not_started_fixture_whose_kickoff_has_passed(
        self, db: LiveDB
    ) -> None:
        """A fixture's cached status only updates when something re-polls it. Without a
        continuously running scheduler, a match that has actually kicked off (or finished)
        can sit at NOT_STARTED indefinitely -- it must not still show up as "upcoming"."""
        add_match(
            db,
            match_id="1",
            home="A",
            away="B",
            competition="EPL",
            status=MatchStatus.NOT_STARTED,
            observed_at=datetime.now(UTC) - timedelta(days=2),  # kickoff already passed
        )
        add_match(
            db,
            match_id="2",
            home="C",
            away="D",
            competition="EPL",
            status=MatchStatus.NOT_STARTED,
            observed_at=datetime.now(UTC) + timedelta(days=1),
        )
        ups = MatchStateStore(db).upcoming()
        assert [v.home for v in ups] == ["C"]

    def test_forecastable_fixture_gets_a_slate(self, tmp_path) -> None:
        from soccer.dashboard.data import fixture_forecasts

        live = tmp_path / "live.sqlite"
        with LiveDB(live) as build:
            add_match(
                build,
                match_id="1",
                home="Arsenal",
                away="Brentford",
                competition="Premier League",  # -> E0, the seeded division
                status=MatchStatus.NOT_STARTED,
            )
        analytics = tmp_path / "analytics.duckdb"
        TestAnalyticsSnapshot()._seed_results(analytics)

        fixtures = fixture_forecasts(live, analytics)
        assert len(fixtures) == 1
        f = fixtures[0]
        assert (f.home, f.away) == ("Arsenal", "Brentford")
        assert f.slate is not None  # both teams in the E0 model
        result = {m.name: m.probability for m in f.slate.result}
        assert result["Arsenal"] > result["Brentford"]  # strong over weak

    def test_uncovered_competition_listed_without_forecast(self, tmp_path) -> None:
        from soccer.dashboard.data import fixture_forecasts

        live = tmp_path / "live.sqlite"
        with LiveDB(live) as build:
            add_match(
                build,
                match_id="1",
                home="Someone",
                away="Nobody",
                competition="Kazakhstan Cup",  # maps to no division
                status=MatchStatus.NOT_STARTED,
            )
        analytics = tmp_path / "analytics.duckdb"
        TestAnalyticsSnapshot()._seed_results(analytics)

        fixtures = fixture_forecasts(live, analytics)
        assert len(fixtures) == 1
        assert fixtures[0].slate is None  # honest: no model, no forecast

    def test_latest_season_picked_per_division(self, tmp_path) -> None:
        from soccer.storage.analytics_db import AnalyticsDB

        path = tmp_path / "analytics.duckdb"
        seed_results(path, division="SP1", teams=["Ath Madrid", "Barcelona"], season="2425")
        seed_results(path, division="SP1", teams=["Ath Madrid", "Barcelona"], season="2526")
        with AnalyticsDB(path) as adb:
            assert adb.latest_season("SP1") == "2526"
            assert adb.latest_season("ZZ") is None

    def test_recent_outcomes_window_spans_seasons(self, tmp_path) -> None:
        from soccer.storage.analytics_db import AnalyticsDB

        path = tmp_path / "analytics.duckdb"
        seed_results(path, division="E0", teams=["Arsenal", "Chelsea"], season="2425")
        seed_results(path, division="E0", teams=["Arsenal", "Chelsea"], season="2526")
        with AnalyticsDB(path) as adb:
            window3 = adb.recent_outcomes_through("E0", "2526", n_seasons=3)
            window1 = adb.recent_outcomes_through("E0", "2526", n_seasons=1)
        assert len(window1) == 2  # only 2526 (2 teams, home-and-away)
        assert len(window3) == 4  # 2526 + 2425 both included

    def test_curated_alias_resolves_verbose_name(self, tmp_path) -> None:
        # "Club Atlético de Madrid" (football-data.org) must reach "Ath Madrid" (co.uk).
        from soccer.dashboard.data import fixture_forecasts

        analytics = tmp_path / "analytics.duckdb"
        seed_results(
            analytics, division="SP1", teams=["Ath Madrid", "Barcelona", "Getafe", "Sevilla"]
        )
        live = tmp_path / "live.sqlite"
        with LiveDB(live) as build:
            add_match(
                build,
                match_id="1",
                home="Club Atlético de Madrid",
                away="Barcelona",
                competition="La Liga",  # -> SP1
                status=MatchStatus.NOT_STARTED,
            )
        fixtures = fixture_forecasts(live, analytics)
        assert len(fixtures) == 1
        assert fixtures[0].slate is not None  # alias bridged the verbose name


class TestPlayerProfileData:
    def test_profiles_and_percentiles(self, tmp_path) -> None:
        from soccer.dashboard.data import (
            has_player_events,
            player_percentiles,
            player_profiles,
        )

        path = tmp_path / "analytics.duckdb"
        assert has_player_events(path) is False
        seed_player_events(path)
        assert has_player_events(path) is True

        profiles = player_profiles(path, min_minutes=1, order="contributions")
        by = {p.player for p in profiles}
        assert {"Messi", "Otamendi"} <= by

        # Messi (goals + assists) ranks above the defender by contributions.
        assert profiles[0].player == "Messi"

        # Percentile fingerprint: the defender tops defending, Messi tops attacking.
        otam = {
            mp.label: mp.percentile for mp in player_percentiles(path, "Otamendi", min_minutes=1)
        }
        messi = {mp.label: mp.percentile for mp in player_percentiles(path, "Messi", min_minutes=1)}
        assert otam["Tackles"] >= messi["Tackles"]
        assert messi["xA"] >= otam["xA"]

    def test_percentiles_empty_for_unknown_player(self, tmp_path) -> None:
        from soccer.dashboard.data import player_percentiles

        path = tmp_path / "analytics.duckdb"
        seed_player_events(path)
        assert player_percentiles(path, "Nobody", min_minutes=1) == []

    def test_match_log(self, tmp_path) -> None:
        from soccer.dashboard.data import player_match_log

        path = tmp_path / "analytics.duckdb"
        seed_player_events(path)
        log = player_match_log(path, "Messi")
        assert len(log) == 1
        assert log[0].match_id == 1

    def test_match_log_empty_when_db_missing(self, tmp_path) -> None:
        from soccer.dashboard.data import player_match_log

        assert player_match_log(tmp_path / "nope.duckdb", "Messi") == []


def _seed_similarity_players(path) -> None:
    """Two near-identical attacking wingers plus one very different defender, so a
    similarity ranking has an unambiguous right answer to assert against."""
    from soccer.sources.statsbomb import PlayerMatchStats
    from soccer.storage.analytics_db import AnalyticsDB

    def pms(player: str, **kw) -> PlayerMatchStats:
        base = dict(
            match_id=1,
            player=player,
            team="Club",
            position="Right Wing",
            minutes=900,
            passes=0,
            passes_completed=0,
            key_passes=0,
            assists=0,
            xa=0.0,
            progressive_passes=0,
            carries=0,
            progressive_carries=0,
            dribbles=0,
            dribbles_completed=0,
            tackles=0,
            tackles_won=0,
            interceptions=0,
            blocks=0,
            clearances=0,
            ball_recoveries=0,
            pressures=0,
            fouls=0,
            fouled=0,
            yellow_cards=0,
            red_cards=0,
            touches=0,
        )
        base.update(kw)
        return PlayerMatchStats(**base)

    winger_profile = dict(
        passes=300,
        passes_completed=250,
        key_passes=20,
        assists=8,
        xa=6.0,
        progressive_passes=30,
        progressive_carries=60,
        dribbles=100,
        dribbles_completed=70,
    )
    stats = [
        pms("WingerA", **winger_profile),
        pms("WingerB", **{**winger_profile, "assists": 9, "xa": 6.5}),  # near-identical
        pms(
            "DefenderC",
            position="Center Back",
            passes=400,
            passes_completed=370,
            tackles=60,
            tackles_won=45,
            interceptions=50,
            blocks=30,
            clearances=100,
            ball_recoveries=80,
        ),
    ]
    with AnalyticsDB(path) as adb:
        adb.load_player_stats(stats)


class TestPlayerSimilarity:
    def test_closest_match_is_the_near_identical_profile(self, tmp_path) -> None:
        from soccer.dashboard.data import player_similarity

        path = tmp_path / "analytics.duckdb"
        _seed_similarity_players(path)
        results = player_similarity(path, "WingerA", min_minutes=1)
        assert results[0].player == "WingerB"
        assert results[-1].player == "DefenderC"
        # A near-identical profile scores much higher similarity than a very different one.
        assert results[0].similarity > results[-1].similarity

    def test_limit_caps_the_result_count(self, tmp_path) -> None:
        from soccer.dashboard.data import player_similarity

        path = tmp_path / "analytics.duckdb"
        _seed_similarity_players(path)
        assert len(player_similarity(path, "WingerA", min_minutes=1, limit=1)) == 1

    def test_target_never_appears_in_its_own_results(self, tmp_path) -> None:
        from soccer.dashboard.data import player_similarity

        path = tmp_path / "analytics.duckdb"
        _seed_similarity_players(path)
        results = player_similarity(path, "WingerA", min_minutes=1)
        assert "WingerA" not in {r.player for r in results}

    def test_unknown_player_is_empty(self, tmp_path) -> None:
        from soccer.dashboard.data import player_similarity

        path = tmp_path / "analytics.duckdb"
        _seed_similarity_players(path)
        assert player_similarity(path, "Nobody", min_minutes=1) == []

    def test_target_filtered_out_by_min_minutes_is_empty(self, tmp_path) -> None:
        from soccer.dashboard.data import player_similarity

        path = tmp_path / "analytics.duckdb"
        seed_player_events(path)  # Messi + Otamendi, but min_minutes=1000 admits neither
        assert player_similarity(path, "Messi", min_minutes=1000) == []

    def test_a_single_qualifying_player_has_nothing_to_compare_against(self, tmp_path) -> None:
        from soccer.dashboard.data import player_similarity
        from soccer.sources.statsbomb import PlayerMatchStats
        from soccer.storage.analytics_db import AnalyticsDB

        path = tmp_path / "analytics.duckdb"
        with AnalyticsDB(path) as adb:
            adb.load_player_stats(
                [
                    PlayerMatchStats(
                        match_id=1, player="Solo", team="Club", position="Striker",
                        minutes=900, passes=0, passes_completed=0, key_passes=0,
                        assists=0, xa=0.0, progressive_passes=0, carries=0,
                        progressive_carries=0, dribbles=0, dribbles_completed=0,
                        tackles=0, tackles_won=0, interceptions=0, blocks=0,
                        clearances=0, ball_recoveries=0, pressures=0, fouls=0,
                        fouled=0, yellow_cards=0, red_cards=0, touches=0,
                    )
                ]
            )
        assert player_similarity(path, "Solo", min_minutes=1) == []


class TestLiveCentreModes:
    def test_shows_live_matches_first(self, tmp_path) -> None:
        from soccer.dashboard.data import live_snapshot

        db = LiveDB(tmp_path / "live.sqlite")
        add_match(
            db,
            match_id="1",
            home="Arsenal",
            away="Chelsea",
            competition="Premier League",
            status=MatchStatus.SECOND_HALF,
        )
        add_match(
            db,
            match_id="2",
            home="Barca",
            away="Madrid",
            competition="La Liga",
            status=MatchStatus.FINISHED,
        )
        snap = live_snapshot(db)
        assert snap.mode == "live"
        assert snap.matches and all(v.status.is_in_play for v in snap.matches)
        db.close()

    def test_falls_back_to_recent_results_when_nothing_live(self, tmp_path) -> None:
        from soccer.dashboard.data import live_snapshot

        db = LiveDB(tmp_path / "live.sqlite")
        add_match(
            db,
            match_id="2",
            home="Barca",
            away="Madrid",
            competition="La Liga",
            status=MatchStatus.FINISHED,
        )
        snap = live_snapshot(db)
        assert snap.mode == "recent"

    def test_a_match_stuck_in_play_for_weeks_is_treated_as_a_zombie_not_live(
        self, tmp_path
    ) -> None:
        """Reproduces the real bug: TheSportsDB's livescore feed only lists matches
        currently in play, so once a match finishes it simply disappears from that feed --
        nothing ever tells the store the match is over. A row can then sit frozen at e.g.
        SECOND_HALF indefinitely. Confirmed live: 275 such rows in the real store, one from
        33 days earlier. Without an age cutoff, the Live Centre shows these as "live" forever
        and never falls back to real recent results."""
        from soccer.dashboard.data import live_snapshot

        db = LiveDB(tmp_path / "live.sqlite")
        add_match(
            db,
            match_id="1",
            home="Fafe",
            away="Maria da Fonte",
            competition="Portugal Amateur",
            status=MatchStatus.SECOND_HALF,
            observed_at=datetime.now(UTC) - timedelta(days=33),
        )
        add_match(
            db,
            match_id="2",
            home="Barca",
            away="Madrid",
            competition="La Liga",
            status=MatchStatus.FINISHED,
        )
        snap = live_snapshot(db)
        assert snap.mode == "recent"
        assert snap.kpis.in_play == 0
        assert all(v.home != "Fafe" for v in snap.matches)
        assert len(snap.matches) == 1 and snap.matches[0].status.is_concluded
        db.close()


class TestUpcomingSeasonBriefing:
    def test_projects_the_fixture_lineup_with_promoted_prior(self, tmp_path) -> None:
        from soccer.dashboard.data import upcoming_season_briefing

        analytics = tmp_path / "analytics.duckdb"
        seed_results(analytics, division="E0", teams=["Arsenal", "Chelsea", "Fulham", "Brentford"])

        # Upcoming PL fixtures: three established clubs plus one with no top-flight history.
        live = tmp_path / "live.sqlite"
        db = LiveDB(live)
        teams = ["Arsenal", "Chelsea", "Fulham", "Newcomers FC"]
        pairs = [(h, a) for h in teams for a in teams if h != a]
        for i, (home, away) in enumerate(pairs):
            add_match(
                db,
                match_id=str(i),
                home=home,
                away=away,
                competition="Premier League",
                status=MatchStatus.NOT_STARTED,
            )
        db.close()

        result = upcoming_season_briefing(live, analytics, "E0", n_sims=500, relegation=1)
        assert result is not None
        briefing, promoted = result
        projected = {briefing.names.get(p.team, p.team) for p in briefing.projections}
        assert projected == set(teams)  # exactly the fixtures' line-up, not last season's
        assert "Newcomers FC" in promoted  # the club with no history got the promoted prior
        assert "Brentford" not in projected  # in results but not in the new fixtures -> dropped

    def test_thin_sample_team_is_shrunk_not_left_to_an_unregularised_mle(self, tmp_path) -> None:
        """Reproduces the real bug: once a new season's own results start loading (which
        happens every season, as soon as a few matches are played), a team the fit DID see
        -- but on only a couple of matches -- used to get an unregularised Dixon-Coles MLE
        fit instead of the safe 'promoted' prior. One upset win replayed thousands of times
        by the Monte Carlo turned a newly promoted side into a ~94% title favourite. This
        also pins the season label: it must stay the season actually in progress, not
        silently advance to the one after it just because that season now has results.
        """
        from soccer.dashboard.data import upcoming_season_briefing
        from soccer.domain.names import normalize_name
        from soccer.sources.football_data_co_uk import MatchResult
        from soccer.storage.analytics_db import AnalyticsDB

        def row(season: str, home: str, away: str, hg: int, ag: int, d: date) -> MatchResult:
            return MatchResult(
                season=season,
                division="E0",
                match_date=d,
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

        # A prior season's full round robin among four established sides, with real (not
        # uniform) variance, so the fit has genuine spread to work with.
        scores = {
            ("Arsenal", "Chelsea"): (2, 1),
            ("Chelsea", "Arsenal"): (0, 2),
            ("Arsenal", "Fulham"): (3, 0),
            ("Fulham", "Arsenal"): (0, 1),
            ("Arsenal", "Brentford"): (2, 0),
            ("Brentford", "Arsenal"): (1, 2),
            ("Chelsea", "Fulham"): (1, 1),
            ("Fulham", "Chelsea"): (0, 1),
            ("Chelsea", "Brentford"): (2, 1),
            ("Brentford", "Chelsea"): (1, 2),
            ("Fulham", "Brentford"): (1, 0),
            ("Brentford", "Fulham"): (0, 0),
        }
        rows = [
            row("2526", home, away, hg, ag, date(2025, 9, 1 + i))
            for i, ((home, away), (hg, ag)) in enumerate(scores.items())
        ]
        # The new season, a few days old. One promoted side has played twice and won both --
        # including a lopsided win over last season's strongest side. Another has been shut
        # out in both its games -- the exact shape (0 goals, twice) that pins a small-sample
        # Dixon-Coles fit's attack rating at the optimiser's own boundary rather than a real
        # estimate, which a naive weighted blend does not fully protect against.
        today = date.today()
        rows += [
            row("2627", "Newcomers FC", "Arsenal", 3, 0, today - timedelta(days=10)),
            row("2627", "Chelsea", "Newcomers FC", 0, 1, today - timedelta(days=3)),
            row("2627", "Arsenal", "Strugglers FC", 3, 0, today - timedelta(days=9)),
            row("2627", "Strugglers FC", "Chelsea", 0, 1, today - timedelta(days=2)),
        ]
        analytics = tmp_path / "analytics.duckdb"
        with AnalyticsDB(analytics) as adb:
            adb.load_results(rows)

        live = tmp_path / "live.sqlite"
        db = LiveDB(live)
        teams = ["Arsenal", "Chelsea", "Fulham", "Newcomers FC", "Strugglers FC"]
        pairs = [(h, a) for h in teams for a in teams if h != a]
        for i, (home, away) in enumerate(pairs):
            add_match(
                db,
                match_id=str(i),
                home=home,
                away=away,
                competition="Premier League",
                status=MatchStatus.NOT_STARTED,
            )
        db.close()

        result = upcoming_season_briefing(live, analytics, "E0", n_sims=2000, relegation=1)
        assert result is not None
        briefing, promoted = result

        # Labelled the season actually in progress (2026/27), not a year ahead of it.
        assert briefing.season == "2627"

        newcomers_norm = normalize_name("Newcomers FC")
        strugglers_norm = normalize_name("Strugglers FC")
        assert newcomers_norm not in promoted  # it DID play -- this is the thin-sample path
        assert strugglers_norm not in promoted
        title_pct = {p.team: p.title_pct for p in briefing.projections}
        assert title_pct[newcomers_norm] < 0.5  # two flattering games must not make it a lock

        relegation_pct = {p.team: p.relegation_pct for p in briefing.projections}
        # Shut out twice must not be treated as a near-certainty -- that symptom (relegation
        # pinned close to 1.0) is what an unclipped small-sample fit produced. This end-to-end
        # check is a smoke test; TestStabilizeThinSamples below pins the exact mechanism.
        assert relegation_pct[strugglers_norm] < 0.8


class TestStabilizeThinSamples:
    """Direct tests of the blending `_stabilize_thin_samples` does, without the Monte Carlo
    noise of a full season simulation on top -- that noise is too coarse, at a test-sized
    team count, to reliably distinguish a clipped fix from the unclipped bug it replaced."""

    def _model(self):
        from soccer.models.dixon_coles import DixonColesModel

        # Three established teams spanning a normal, realistic range.
        return DixonColesModel(
            attack={"a": 0.5, "b": -0.2, "c": 0.0},
            defence={"a": 0.4, "b": -0.1, "c": 0.0},
            home_advantage=0.2,
            rho=-0.05,
        )

    def test_a_boundary_pinned_thin_sample_is_clipped_before_blending(self) -> None:
        """A team the fit only saw twice, both shutouts, gets pinned at the optimiser's own
        bound (+-3) -- a degenerate artifact of the fit, not a real estimate of its strength.
        Blending that raw, unclipped value in (even down-weighted) can rate it worse than
        every genuinely weak, well-supported team in the league; this is the exact defect
        that turned a real newly promoted side into a ~100%-certain relegation candidate."""
        from soccer.dashboard.data import SEASON_SIM_MIN_MATCHES, _stabilize_thin_samples

        model = self._model()
        model.add_team("thin", -3.0, -0.5)  # boundary-pinned attack, from 2 shutout losses
        match_counts = {"a": 10, "b": 10, "c": 10, "thin": 2}
        assert match_counts["thin"] < SEASON_SIM_MIN_MATCHES

        promoted = _stabilize_thin_samples(model, ["a", "b", "c", "thin"], match_counts)

        assert promoted == []  # the fit DID see it -- this is the thin-sample path, not it
        established_min_attack = min(model.strengths[t] for t in ("a", "b", "c"))
        # Never worse than the worst REAL, well-supported team -- the boundary artifact must
        # not leak through the blend.
        assert model.strengths["thin"] >= established_min_attack

    def test_a_team_with_enough_matches_is_left_alone(self) -> None:
        from soccer.dashboard.data import _stabilize_thin_samples

        model = self._model()
        original = dict(model.strengths)
        match_counts = {"a": 10, "b": 10, "c": 10}

        promoted = _stabilize_thin_samples(model, ["a", "b", "c"], match_counts)

        assert promoted == []
        assert model.strengths == original  # nothing to stabilise -- untouched


class TestStabilizeThinSamplesPoisson:
    """`_stabilize_thin_samples` also runs on `PoissonModel` (the shots-on-target blend
    season_briefing/upcoming_season_briefing now fit). PoissonModel's defence is the
    OPPOSITE sign convention from DixonColesModel's: higher = concedes MORE (leakier),
    not fewer -- so naively reusing `attack + defence` to rank "weakest" would rank a
    leaky-but-potent attacking side as weak and a tight, low-scoring side as strong. These
    pin the corrected `attack - defence` goodness score against that exact failure mode."""

    def _model(self):
        from soccer.models.poisson import PoissonModel, TeamStrength

        return PoissonModel(
            strengths={
                # Genuinely weak: poor attack, leaky defence.
                "weak1": TeamStrength(attack=0.5, defence=1.4),
                "weak2": TeamStrength(attack=0.4, defence=1.3),
                # High-scoring but also leaky -- naive attack+defence ranks this "strong"
                # (3.4, the highest of any team here); correct attack-defence (0.2) ranks
                # it middling, well above the two genuinely weak sides.
                "leaky_scorer": TeamStrength(attack=1.8, defence=1.6),
                # The actual strongest side: strong attack AND tight defence. Naive
                # attack+defence (2.3) sits BELOW leaky_scorer's 3.4, so a naive ranking
                # would wrongly call this one weaker than the leaky scorer.
                "strong": TeamStrength(attack=1.8, defence=0.5),
            },
            home_avg=1.5,
            away_avg=1.1,
            rho=-0.05,
        )

    def test_weakest_three_uses_attack_minus_defence_not_the_sum(self) -> None:
        from soccer.dashboard.data import _stabilize_thin_samples

        model = self._model()
        match_counts = {"weak1": 10, "weak2": 10, "leaky_scorer": 10, "strong": 10}

        _stabilize_thin_samples(
            model, ["weak1", "weak2", "leaky_scorer", "strong", "new"], match_counts
        )

        # The promoted team's prior is the mean of the correctly-identified weakest three
        # (weak1, weak2, leaky_scorer) -- NOT "strong", which a naive attack+defence sum
        # would have mistakenly included instead of leaky_scorer.
        expected_attack = (0.5 + 0.4 + 1.8) / 3
        expected_defence = (1.4 + 1.3 + 1.6) / 3
        new = model.strengths["new"]
        assert new.attack == pytest.approx(expected_attack)
        assert new.defence == pytest.approx(expected_defence)

    def test_thin_sample_boundary_artifact_is_clipped(self) -> None:
        """A team the fit only saw twice with a wild scoreline can land far outside any
        established team's range; clip to that range before blending, as for DixonColesModel."""
        from soccer.dashboard.data import SEASON_SIM_MIN_MATCHES, _stabilize_thin_samples
        from soccer.models.poisson import TeamStrength

        model = self._model()
        model.strengths["thin"] = TeamStrength(attack=5.0, defence=0.05)  # degenerate outlier
        match_counts = {
            "weak1": 10,
            "weak2": 10,
            "leaky_scorer": 10,
            "strong": 10,
            "thin": 2,
        }
        assert match_counts["thin"] < SEASON_SIM_MIN_MATCHES

        promoted = _stabilize_thin_samples(
            model, ["weak1", "weak2", "leaky_scorer", "strong", "thin"], match_counts
        )

        assert promoted == []  # the fit DID see it -- thin-sample path, not the no-data one
        established_max_attack = max(
            model.strengths[t].attack for t in ("weak1", "weak2", "leaky_scorer", "strong")
        )
        # The outlier's raw 5.0 must be clipped to the established range before blending in,
        # not let through to swamp the weight-average.
        assert model.strengths["thin"].attack <= established_max_attack


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
            # Seed some league data up front so the Home page shows content instead of
            # firing its first-launch auto-download (which would hit the network).
            TestAnalyticsSnapshot()._seed_results(tmp_path / "analytics.duckdb")

            from streamlit.testing.v1 import AppTest

            app_path = str(files("soccer.dashboard") / "app.py")
            # Generous timeout: the first Streamlit run in a process pays cold-start.
            at = AppTest.from_file(app_path, default_timeout=120).run()
            assert not at.exception, f"Home (default) raised: {at.exception}"

            # A Home call-to-action button (_go) navigates without a session_state error.
            cta = [b for b in at.button if "assistant" in b.label.lower()]
            if cta:
                cta[0].click().run()
                assert not at.exception, f"Home navigation raised: {at.exception}"

            at.radio[0].set_value("Live scores").run()
            assert not at.exception, f"Live Centre raised: {at.exception}"
            assert any(m.label == "Matches" for m in at.metric)

            at.radio[0].set_value("About & sources").run()
            assert not at.exception, f"Data Health raised: {at.exception}"

            # Predictions merges upcoming fixtures + matchup + season into one page. Visit it
            # before seeding league data: the Upcoming tab lists fixtures (none forecastable
            # here), and the Matchup/Season tabs fall back to their "add data" prompts.
            at.radio[0].set_value("Predictions").run()
            assert not at.exception, f"Predictions (empty) raised: {at.exception}"

            # League tables now folds the form guide in as a second tab, and adds a Form
            # column to the table itself. AppTest executes both tab bodies on one visit, so
            # this single navigation covers the standings render and the trends render.
            TestAnalyticsSnapshot()._seed_results(tmp_path / "analytics.duckdb")
            at.radio[0].set_value("League tables").run()
            assert not at.exception, f"League tables raised: {at.exception}"
            # The form guide is folded in as a Form column on the standings table itself.
            assert any(">Form</th>" in (m.value or "") for m in at.markdown), (
                "League table is missing its Form column"
            )

            # Teams hub: pick a club and render its dossier (standing, form, trajectory).
            at.radio[0].set_value("Teams").run()
            assert not at.exception, f"Teams raised: {at.exception}"

            # Assistant answers a real question now that data is present.
            at.radio[0].set_value("Ask a question").run()
            assert not at.exception, f"Assistant raised: {at.exception}"
            at.chat_input[0].set_value("who is top of the premier league?").run()
            assert not at.exception, f"Assistant query raised: {at.exception}"

            # A dossier answer renders its points-trajectory chart inline in the chat,
            # exercising _render_chat_chart + st.altair_chart on the chat page itself.
            at.chat_input[0].set_value("tell me about Arsenal").run()
            assert not at.exception, f"Assistant chat chart raised: {at.exception}"

            # With league data seeded, revisit Predictions: AppTest executes all three tab
            # bodies, so this covers the upcoming-fixtures list, the matchup forecast slate
            # and the season simulation in one visit.
            at.radio[0].set_value("Predictions").run()
            assert not at.exception, f"Predictions raised: {at.exception}"

            at.radio[0].set_value("Records").run()
            assert not at.exception, f"Records raised: {at.exception}"

            # Analysis merges match xG + players into one page (two tabs). Seed both shots
            # and full-event player stats so each tab has data, then a single visit renders
            # both tab bodies.
            seed_player_events(tmp_path / "analytics.duckdb")
            TestShotMap()._seed_shots(tmp_path / "analytics.duckdb")
            at.radio[0].set_value("Analysis").run()
            assert not at.exception, f"Analysis raised: {at.exception}"
            # Switch the Players tab to the per-player profile view (percentile fingerprint).
            at.segmented_control[0].set_value("Player profile").run()
            assert not at.exception, f"Analysis players profile raised: {at.exception}"
        finally:
            config._settings = None

    def test_upcoming_competition_filter(self, tmp_path, monkeypatch) -> None:
        # With upcoming fixtures across two competitions, the Upcoming tab offers a
        # Competition filter, and picking one narrows the list without error.
        pytest.importorskip("streamlit")
        from importlib.resources import files

        import soccer.config as config

        analytics = tmp_path / "analytics.duckdb"
        seed_results(analytics, division="E0", teams=["Arsenal", "Chelsea", "Fulham", "Brentford"])
        seed_results(analytics, division="N1", teams=["Ajax", "PSV", "Feyenoord", "AZ"])
        build = LiveDB(tmp_path / "live.sqlite")
        add_match(
            build,
            match_id="1",
            home="Arsenal",
            away="Chelsea",
            competition="Premier League",
            status=MatchStatus.NOT_STARTED,
        )
        add_match(
            build,
            match_id="2",
            home="Ajax",
            away="PSV",
            competition="Eredivisie",
            status=MatchStatus.NOT_STARTED,
        )
        build.close()

        monkeypatch.setenv("SOCCER_DATA_DIR", str(tmp_path))
        config._settings = None
        try:
            from streamlit.testing.v1 import AppTest

            app_path = str(files("soccer.dashboard") / "app.py")
            at = AppTest.from_file(app_path, default_timeout=120).run()
            at.radio[0].set_value("Predictions").run()
            assert not at.exception, f"Predictions raised: {at.exception}"
            sb = [s for s in at.selectbox if s.label == "Competition"]
            assert sb, "expected a Competition filter on the Upcoming tab"
            assert {"Premier League", "Eredivisie"} <= set(sb[0].options)
            sb[0].set_value("Eredivisie").run()
            assert not at.exception, f"filtering raised: {at.exception}"
        finally:
            config._settings = None

    def test_predictions_locks_to_latest_season(self, tmp_path, monkeypatch) -> None:
        # Predictions must not offer past seasons -- projecting a finished season is not a
        # forecast. With two seasons loaded, the Matchup/Season pickers show only the latest.
        pytest.importorskip("streamlit")
        from importlib.resources import files

        import soccer.config as config

        teams = ["Arsenal", "Chelsea", "Fulham", "Brentford"]
        seed_results(tmp_path / "analytics.duckdb", division="E0", teams=teams, season="2425")
        seed_results(tmp_path / "analytics.duckdb", division="E0", teams=teams, season="2526")
        monkeypatch.setenv("SOCCER_DATA_DIR", str(tmp_path))
        config._settings = None
        try:
            from streamlit.testing.v1 import AppTest

            app_path = str(files("soccer.dashboard") / "app.py")
            at = AppTest.from_file(app_path, default_timeout=120).run()
            at.radio[0].set_value("Predictions").run()
            assert not at.exception, f"Predictions raised: {at.exception}"
            seasons = [s for s in at.selectbox if s.label == "Season"]
            assert seasons, "expected a Season selector on Predictions"
            for s in seasons:
                assert s.options == ["2025/26"], f"expected only the latest season: {s.options}"
                assert s.disabled, "the Predictions season must be locked"
        finally:
            config._settings = None

    def test_password_gate_blocks_until_entered(self, tmp_path, monkeypatch) -> None:
        # With SOCCER_DASHBOARD_PASSWORD set (a public tunnel), the app must stop at a
        # password prompt before rendering the nav or reaching any data action.
        pytest.importorskip("streamlit")
        from importlib.resources import files

        import soccer.config as config

        monkeypatch.setenv("SOCCER_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("SOCCER_DASHBOARD_PASSWORD", "secret")
        config._settings = None
        try:
            from streamlit.testing.v1 import AppTest

            app_path = str(files("soccer.dashboard") / "app.py")
            at = AppTest.from_file(app_path, default_timeout=60).run()
            assert not at.exception
            assert at.text_input, "expected a password field"
            assert not at.radio, "nav must not render before the password is entered"
        finally:
            config._settings = None


class TestPreviousSeason:
    """The pure season-lookup half of the trajectory-overlay feature -- which season, if
    any, counts as "last season" for a given (division, season), for the Team page's
    "compare with previous season" toggle."""

    def test_finds_the_immediately_prior_season_for_that_division(self) -> None:
        from soccer.dashboard.app import _previous_season

        available = [("2526", "E0", 380), ("2425", "E0", 380), ("2324", "E0", 380)]
        assert _previous_season(available, "E0", "2526") == "2425"

    def test_chronological_not_lexical_ordering(self) -> None:
        # "9900" (1999/2000) must not lexically outrank "2526" -- season_sort_key, not str.
        from soccer.dashboard.app import _previous_season

        available = [("2526", "E0", 380), ("9900", "E0", 380)]
        assert _previous_season(available, "E0", "2526") == "9900"

    def test_earliest_loaded_season_has_no_predecessor(self) -> None:
        from soccer.dashboard.app import _previous_season

        available = [("2526", "E0", 380)]
        assert _previous_season(available, "E0", "2526") is None

    def test_scoped_to_the_same_division(self) -> None:
        # An earlier season loaded for a DIFFERENT division must not leak in.
        from soccer.dashboard.app import _previous_season

        available = [("2526", "E0", 380), ("2425", "SP1", 380)]
        assert _previous_season(available, "E0", "2526") is None


class TestParseEventKickoff:
    """The pure timestamp-parsing half of `update_confirmed_lineups`'s pre-kickoff window
    check -- the network-fetch half isn't separately tested at the actions layer, matching
    how `update_availability`'s live fetch also isn't (only its disabled-gate is, in
    test_fpl.py); the adapter methods it calls are unit-tested in test_thesportsdb.py."""

    def test_parses_a_real_timestamp_as_utc(self) -> None:
        from soccer.dashboard.actions import _parse_event_kickoff

        kickoff = _parse_event_kickoff({"strTimestamp": "2026-09-18T19:00:00"})
        assert kickoff == datetime(2026, 9, 18, 19, 0, tzinfo=UTC)

    def test_missing_timestamp_is_none(self) -> None:
        from soccer.dashboard.actions import _parse_event_kickoff

        assert _parse_event_kickoff({}) is None
        assert _parse_event_kickoff({"strTimestamp": None}) is None

    def test_malformed_timestamp_is_none_not_a_raise(self) -> None:
        from soccer.dashboard.actions import _parse_event_kickoff

        assert _parse_event_kickoff({"strTimestamp": "not-a-date"}) is None
