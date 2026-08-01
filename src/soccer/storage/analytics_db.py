"""DuckDB analytics store -- historical results and the queries over them.

Separate engine from the live SQLite by design (the plan's split): SQLite holds small,
mutable current state; DuckDB holds the large, append-mostly history that analytics and
forecasting read. Columnar and bulk-loaded, so a full results archive (tens of seasons,
~250k matches) queries fast.

Results are keyed by normalized team name, produced by the same `normalize_name` the
live crosswalk uses -- so a later join to canonical live entities is a plain
normalized-name join, not a second identity system.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import duckdb
import polars as pl

from soccer.sources.football_data_co_uk import MatchResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS results (
    season             VARCHAR NOT NULL,
    division           VARCHAR NOT NULL,
    match_date         DATE    NOT NULL,
    home               VARCHAR NOT NULL,
    away               VARCHAR NOT NULL,
    home_norm          VARCHAR NOT NULL,
    away_norm          VARCHAR NOT NULL,
    fthg               INTEGER NOT NULL,
    ftag               INTEGER NOT NULL,
    ftr                VARCHAR NOT NULL,
    hthg               INTEGER,
    htag               INTEGER,
    home_shots         INTEGER,
    away_shots         INTEGER,
    home_shots_target  INTEGER,
    away_shots_target  INTEGER,
    home_corners       INTEGER,
    away_corners       INTEGER,
    home_yellows       INTEGER,
    away_yellows       INTEGER,
    home_reds          INTEGER,
    away_reds          INTEGER,
    referee            VARCHAR,
    close_home_odds    DOUBLE,
    close_draw_odds    DOUBLE,
    close_away_odds    DOUBLE
);

CREATE TABLE IF NOT EXISTS shots (
    match_id    INTEGER NOT NULL,
    team        VARCHAR NOT NULL,
    player      VARCHAR NOT NULL,
    minute      INTEGER NOT NULL,
    period      INTEGER NOT NULL,
    x           DOUBLE,
    y           DOUBLE,
    xg          DOUBLE NOT NULL,
    outcome     VARCHAR NOT NULL,
    is_goal     BOOLEAN NOT NULL,
    is_penalty  BOOLEAN NOT NULL,
    body_part   VARCHAR
);

CREATE TABLE IF NOT EXISTS player_match_stats (
    match_id            INTEGER NOT NULL,
    player              VARCHAR NOT NULL,
    team                VARCHAR NOT NULL,
    position            VARCHAR,
    minutes             INTEGER NOT NULL,
    passes              INTEGER NOT NULL,
    passes_completed    INTEGER NOT NULL,
    key_passes          INTEGER NOT NULL,
    assists             INTEGER NOT NULL,
    xa                  DOUBLE  NOT NULL,
    progressive_passes  INTEGER NOT NULL,
    carries             INTEGER NOT NULL,
    progressive_carries INTEGER NOT NULL,
    dribbles            INTEGER NOT NULL,
    dribbles_completed  INTEGER NOT NULL,
    tackles             INTEGER NOT NULL,
    tackles_won         INTEGER NOT NULL,
    interceptions       INTEGER NOT NULL,
    blocks              INTEGER NOT NULL,
    clearances          INTEGER NOT NULL,
    ball_recoveries     INTEGER NOT NULL,
    pressures           INTEGER NOT NULL,
    fouls               INTEGER NOT NULL,
    fouled              INTEGER NOT NULL,
    yellow_cards        INTEGER NOT NULL,
    red_cards           INTEGER NOT NULL,
    touches             INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS match_meta (
    match_id       INTEGER PRIMARY KEY,
    competition    VARCHAR,
    season         VARCHAR,
    competition_id INTEGER,
    season_id      INTEGER,
    match_date     VARCHAR,
    home_team      VARCHAR,
    away_team      VARCHAR
);
"""

# Sortable expressions for the player-profile leaderboard, allowlisted so `order` can
# never reach the SQL string directly. Keys are what the CLI/dashboard pass.
_PROFILE_ORDER = {
    "goals": "COALESCE(sh.goals,0)",
    "xg": "COALESCE(sh.xg,0)",
    "npxg": "COALESCE(sh.npxg,0)",
    "assists": "p.assists",
    "xa": "p.xa",
    "contributions": "COALESCE(sh.goals,0)+p.assists",
    "xgxa": "COALESCE(sh.xg,0)+p.xa",
    "key_passes": "p.key_passes",
    "passes": "p.passes",
    "progressive": "p.progressive_passes+p.progressive_carries",
    "tackles": "p.tackles",
    "interceptions": "p.interceptions",
    "pressures": "p.pressures",
    "defensive": "p.tackles+p.interceptions+p.blocks+p.clearances+p.ball_recoveries",
    "minutes": "p.minutes",
    "touches": "p.touches",
}


# Full profile per player: event stats (pstat) LEFT JOINed onto shot aggregates (sh), so
# a non-shooting defender still gets a row. Column order matches PlayerProfile's fields.
# An optional competition filter restricts BOTH sides to one competition's matches, so
# percentiles compare like with like (a league season, not a mix of tournaments).
def _profile_select(competition: str | None) -> tuple[str, list]:
    filt, params = "", []
    if competition:
        filt = "WHERE match_id IN (SELECT match_id FROM match_meta WHERE competition = ?)"
        params = [competition, competition]  # one per CTE below
    sql = f"""
        WITH pstat AS (
            SELECT player, mode(team) AS team, mode(position) AS position,
                   COUNT(DISTINCT match_id) AS matches, SUM(minutes) AS minutes,
                   SUM(assists) AS assists, SUM(xa) AS xa, SUM(key_passes) AS key_passes,
                   SUM(passes) AS passes, SUM(passes_completed) AS passes_completed,
                   SUM(progressive_passes) AS progressive_passes, SUM(carries) AS carries,
                   SUM(progressive_carries) AS progressive_carries, SUM(dribbles) AS dribbles,
                   SUM(dribbles_completed) AS dribbles_completed, SUM(tackles) AS tackles,
                   SUM(tackles_won) AS tackles_won, SUM(interceptions) AS interceptions,
                   SUM(blocks) AS blocks, SUM(clearances) AS clearances,
                   SUM(ball_recoveries) AS ball_recoveries, SUM(pressures) AS pressures,
                   SUM(fouls) AS fouls, SUM(fouled) AS fouled, SUM(yellow_cards) AS yellow_cards,
                   SUM(red_cards) AS red_cards, SUM(touches) AS touches
            FROM player_match_stats {filt} GROUP BY player
        ),
        sh AS (
            SELECT player, SUM(xg) AS xg,
                   COALESCE(SUM(xg) FILTER (WHERE NOT is_penalty), 0) AS npxg,
                   SUM(is_goal::INT) AS goals, COUNT(*) AS shots
            FROM shots {filt} GROUP BY player
        )
        SELECT p.player, p.team, p.position, p.matches, p.minutes,
               COALESCE(sh.goals, 0), COALESCE(sh.xg, 0.0), COALESCE(sh.npxg, 0.0),
               COALESCE(sh.shots, 0), p.assists, p.xa, p.key_passes, p.passes,
               p.passes_completed, p.progressive_passes, p.carries, p.progressive_carries,
               p.dribbles, p.dribbles_completed, p.tackles, p.tackles_won, p.interceptions,
               p.blocks, p.clearances, p.ball_recoveries, p.pressures, p.fouls, p.fouled,
               p.yellow_cards, p.red_cards, p.touches
        FROM pstat p LEFT JOIN sh ON sh.player = p.player
    """
    return sql, params


@dataclass(frozen=True)
class TableRow:
    position: int
    team: str
    played: int
    won: int
    drawn: int
    lost: int
    goals_for: int
    goals_against: int
    goal_difference: int
    points: int


@dataclass(frozen=True)
class XgRow:
    name: str  # team or player
    team: str | None
    xg: float
    goals: int
    shots: int


@dataclass(frozen=True)
class PlayerRow:
    player: str
    team: str
    xg: float
    npxg: float  # non-penalty xG
    goals: int
    shots: int

    @property
    def xg_diff(self) -> float:
        """Goals minus xG -- finishing over/under-performance."""
        return self.goals - self.xg


@dataclass(frozen=True)
class PlayerProfile:
    """A player's full contribution across all loaded matches, shooting + everything else.

    Shooting (goals/xg/npxg/shots) comes from the shots table; the rest from aggregated
    per-match event stats. Rate stats are exposed via `per90` so a substitute and an
    ever-present are comparable.
    """

    player: str
    team: str
    position: str | None
    matches: int
    minutes: int
    goals: int
    xg: float
    npxg: float
    shots: int
    assists: int
    xa: float
    key_passes: int
    passes: int
    passes_completed: int
    progressive_passes: int
    carries: int
    progressive_carries: int
    dribbles: int
    dribbles_completed: int
    tackles: int
    tackles_won: int
    interceptions: int
    blocks: int
    clearances: int
    ball_recoveries: int
    pressures: int
    fouls: int
    fouled: int
    yellow_cards: int
    red_cards: int
    touches: int

    @property
    def goal_contributions(self) -> int:
        return self.goals + self.assists

    @property
    def xg_plus_xa(self) -> float:
        return self.xg + self.xa

    @property
    def xg_diff(self) -> float:
        return self.goals - self.xg

    @property
    def pass_pct(self) -> float:
        return 100.0 * self.passes_completed / self.passes if self.passes else 0.0

    @property
    def defensive_actions(self) -> int:
        """Tackles + interceptions + blocks + clearances + recoveries -- a defending tally."""
        return (
            self.tackles + self.interceptions + self.blocks + self.clearances + self.ball_recoveries
        )

    def per90(self, value: float) -> float:
        return value / self.minutes * 90.0 if self.minutes else 0.0


@dataclass(frozen=True)
class ResultRow:
    """The slim result shape the models consume (satisfies their Outcome protocols)."""

    match_date: date
    home: str
    away: str
    home_norm: str
    away_norm: str
    fthg: int
    ftag: int


@dataclass(frozen=True)
class OddsRow:
    """A dated result plus its closing 1X2 odds, for market-edge analysis.

    Satisfies the same DatedOutcome shape the models fit on (match_date, home_norm,
    away_norm, fthg, ftag), with the closing decimal odds alongside. Odds may be None.
    """

    match_date: date
    home: str
    away: str
    home_norm: str
    away_norm: str
    fthg: int
    ftag: int
    close_home_odds: float | None
    close_draw_odds: float | None
    close_away_odds: float | None

    @property
    def has_odds(self) -> bool:
        return None not in (self.close_home_odds, self.close_draw_odds, self.close_away_odds)


@dataclass(frozen=True)
class MatchMeta:
    """StatsBomb match identity: what competition/season a match_id belongs to."""

    match_id: int
    competition: str
    season: str
    competition_id: int | None
    season_id: int | None
    match_date: str | None
    home_team: str
    away_team: str

    @property
    def label(self) -> str:
        return f"{self.home_team} v {self.away_team}"


class AnalyticsDB:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if self.path.parent != Path():
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(str(self.path))
        self._con.execute("SET enable_progress_bar = false")  # keep CLI output clean
        self._con.execute(_SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """Idempotent column additions for databases created before a column existed."""
        for column in ("close_home_odds", "close_draw_odds", "close_away_odds"):
            self._con.execute(f"ALTER TABLE results ADD COLUMN IF NOT EXISTS {column} DOUBLE")

    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> AnalyticsDB:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def load_results(self, results: list[MatchResult]) -> int:
        """Load results, replacing each (season, division) unit so re-ingest is idempotent.

        A division-season is the natural reload unit: football-data.co.uk revises a
        season's file as matches are played, so replacing the whole unit keeps it current
        without duplicating rows.
        """
        if not results:
            return 0

        frame = pl.DataFrame([asdict(r) for r in results])
        units = {(r.season, r.division) for r in results}

        # DuckDB reads the Polars frame directly by name via a replacement scan.
        self._con.register("incoming", frame)
        try:
            self._con.execute("BEGIN")
            for season, division in units:
                self._con.execute(
                    "DELETE FROM results WHERE season = ? AND division = ?",
                    [season, division],
                )
            self._con.execute("INSERT INTO results BY NAME SELECT * FROM incoming")
            self._con.execute("COMMIT")
        except Exception:
            self._con.execute("ROLLBACK")
            raise
        finally:
            self._con.unregister("incoming")
        return len(results)

    def league_table(self, season: str, division: str) -> list[TableRow]:
        """Compute a standings table from results -- validates the data end to end."""
        rows = self._con.execute(
            """
            WITH sides AS (
                SELECT home AS team, fthg AS gf, ftag AS ga,
                       CASE WHEN fthg>ftag THEN 3 WHEN fthg=ftag THEN 1 ELSE 0 END AS pts,
                       (fthg>ftag)::INT AS w, (fthg=ftag)::INT AS d, (fthg<ftag)::INT AS l
                FROM results WHERE season=? AND division=?
                UNION ALL
                SELECT away AS team, ftag AS gf, fthg AS ga,
                       CASE WHEN ftag>fthg THEN 3 WHEN ftag=fthg THEN 1 ELSE 0 END AS pts,
                       (ftag>fthg)::INT AS w, (ftag=fthg)::INT AS d, (ftag<fthg)::INT AS l
                FROM results WHERE season=? AND division=?
            )
            SELECT team, COUNT(*) AS played, SUM(w) AS won, SUM(d) AS drawn,
                   SUM(l) AS lost, SUM(gf) AS gf, SUM(ga) AS ga,
                   SUM(gf)-SUM(ga) AS gd, SUM(pts) AS pts
            FROM sides GROUP BY team
            ORDER BY pts DESC, gd DESC, gf DESC, team
            """,
            [season, division, season, division],
        ).fetchall()

        return [
            TableRow(
                position=i + 1,
                team=r[0],
                played=r[1],
                won=r[2],
                drawn=r[3],
                lost=r[4],
                goals_for=r[5],
                goals_against=r[6],
                goal_difference=r[7],
                points=r[8],
            )
            for i, r in enumerate(rows)
        ]

    def result_count(self, season: str | None = None, division: str | None = None) -> int:
        clauses, params = [], []
        if season is not None:
            clauses.append("season = ?")
            params.append(season)
        if division is not None:
            clauses.append("division = ?")
            params.append(division)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        return self._con.execute(f"SELECT COUNT(*) FROM results{where}", params).fetchone()[0]

    def seasons_loaded(self) -> list[tuple[str, str, int]]:
        """(season, division, match count) for everything loaded, newest season first."""
        return self._con.execute(
            "SELECT season, division, COUNT(*) FROM results "
            "GROUP BY season, division ORDER BY season DESC, division"
        ).fetchall()

    def latest_season(self, division: str) -> str | None:
        """The most-recent loaded season for a division, or None if it has no results.

        Lets forecasting fit on whatever a division's newest data is without hardcoding a
        season string -- European files use "2526", the extra-country files use calendar
        years like "2026", and both sort most-recent-last lexically.
        """
        row = self._con.execute(
            "SELECT MAX(season) FROM results WHERE division = ?", [division]
        ).fetchone()
        return row[0] if row and row[0] else None

    def outcomes_for(self, season: str, division: str) -> list[ResultRow]:
        """Results for one (season, division) in date order, for the models to fit on."""
        rows = self._con.execute(
            "SELECT match_date, home, away, home_norm, away_norm, fthg, ftag "
            "FROM results WHERE season=? AND division=? ORDER BY match_date, home",
            [season, division],
        ).fetchall()
        return [ResultRow(*r) for r in rows]

    def outcomes_with_odds(self, season: str, division: str) -> list[OddsRow]:
        """Results for one (season, division) carrying closing 1X2 odds, date order.

        For the market-edge analysis. Rows without a full odds triple keep None there;
        the value backtest skips them, so pre-odds seasons degrade rather than break.
        """
        rows = self._con.execute(
            "SELECT match_date, home, away, home_norm, away_norm, fthg, ftag, "
            "       close_home_odds, close_draw_odds, close_away_odds "
            "FROM results WHERE season=? AND division=? ORDER BY match_date, home",
            [season, division],
        ).fetchall()
        return [OddsRow(*r) for r in rows]

    def odds_coverage(self, season: str, division: str) -> tuple[int, int]:
        """(matches with a full closing-odds triple, total matches) for a slice."""
        row = self._con.execute(
            "SELECT COUNT(*) FILTER (WHERE close_home_odds IS NOT NULL "
            "  AND close_draw_odds IS NOT NULL AND close_away_odds IS NOT NULL), COUNT(*) "
            "FROM results WHERE season=? AND division=?",
            [season, division],
        ).fetchone()
        return (row[0], row[1]) if row else (0, 0)

    # --- StatsBomb shots / xG -------------------------------------------------

    def load_shots(self, shots: list) -> int:
        """Load shots, replacing each match's set so re-ingest is idempotent."""
        if not shots:
            return 0
        frame = pl.DataFrame([asdict(s) for s in shots])
        match_ids = {s.match_id for s in shots}
        self._con.register("incoming_shots", frame)
        try:
            self._con.execute("BEGIN")
            for match_id in match_ids:
                self._con.execute("DELETE FROM shots WHERE match_id = ?", [match_id])
            self._con.execute("INSERT INTO shots BY NAME SELECT * FROM incoming_shots")
            self._con.execute("COMMIT")
        except Exception:
            self._con.execute("ROLLBACK")
            raise
        finally:
            self._con.unregister("incoming_shots")
        return len(shots)

    def team_xg(self, match_id: int) -> list[XgRow]:
        """Per-team xG, goals and shot count for a match, highest xG first."""
        rows = self._con.execute(
            "SELECT team, SUM(xg), SUM(is_goal::INT), COUNT(*) FROM shots "
            "WHERE match_id=? GROUP BY team ORDER BY SUM(xg) DESC",
            [match_id],
        ).fetchall()
        return [XgRow(name=r[0], team=None, xg=r[1], goals=r[2], shots=r[3]) for r in rows]

    def top_shooters(self, match_id: int, limit: int = 10) -> list[XgRow]:
        """Players by total xG in a match."""
        rows = self._con.execute(
            "SELECT player, team, SUM(xg), SUM(is_goal::INT), COUNT(*) FROM shots "
            "WHERE match_id=? GROUP BY player, team ORDER BY SUM(xg) DESC LIMIT ?",
            [match_id, limit],
        ).fetchall()
        return [XgRow(name=r[0], team=r[1], xg=r[2], goals=r[3], shots=r[4]) for r in rows]

    def shots_for(self, match_id: int) -> list[dict]:
        """All shots in a match with location, for shot-map rendering."""
        rows = self._con.execute(
            "SELECT team, player, minute, x, y, xg, outcome, is_goal FROM shots "
            "WHERE match_id=? ORDER BY minute",
            [match_id],
        ).fetchall()
        return [
            {
                "team": r[0],
                "player": r[1],
                "minute": r[2],
                "x": r[3],
                "y": r[4],
                "xg": r[5],
                "outcome": r[6],
                "is_goal": r[7],
            }
            for r in rows
        ]

    def matches_with_shots(self) -> list[int]:
        return [
            r[0]
            for r in self._con.execute(
                "SELECT DISTINCT match_id FROM shots ORDER BY match_id"
            ).fetchall()
        ]

    def shot_match_labels(self) -> list[tuple[int, str]]:
        """(match_id, 'Team A vs Team B') for every match with shots, for a selector."""
        rows = self._con.execute(
            "SELECT match_id, string_agg(DISTINCT team, ' vs ') "
            "FROM shots GROUP BY match_id ORDER BY match_id"
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def shot_matches_indexed(self) -> list[tuple[int, str, str, str | None]]:
        """(match_id, label, competition, date) for matches with shots, metadata-aware.

        Uses the real home/away and competition from match_meta when present, falling back
        to the aggregated team names and 'Other' so matches ingested before metadata still
        appear. Ordered by competition then date for a grouped selector.
        """
        rows = self._con.execute(
            "SELECT s.match_id, "
            "  COALESCE(m.home_team || ' v ' || m.away_team, "
            "           string_agg(DISTINCT s.team, ' v ')) AS label, "
            "  COALESCE(m.competition, 'Other') AS competition, m.match_date "
            "FROM shots s LEFT JOIN match_meta m ON m.match_id = s.match_id "
            "GROUP BY s.match_id, m.home_team, m.away_team, m.competition, m.match_date "
            "ORDER BY competition, m.match_date NULLS FIRST, s.match_id"
        ).fetchall()
        return [(r[0], r[1], r[2], r[3]) for r in rows]

    def player_leaderboard(
        self, *, limit: int = 20, min_shots: int = 3, order: str = "xg"
    ) -> list[PlayerRow]:
        """Players ranked across all loaded shots -- xG, non-penalty xG, goals, shots.

        `order` is 'xg', 'goals' or 'npxg'. `min_shots` filters out cameo appearances.
        """
        order_col = {"xg": "xg", "goals": "goals", "npxg": "npxg"}.get(order, "xg")
        rows = self._con.execute(
            f"""
            SELECT player, mode(team) AS team, SUM(xg) AS xg,
                   COALESCE(SUM(xg) FILTER (WHERE NOT is_penalty), 0) AS npxg,
                   SUM(is_goal::INT) AS goals, COUNT(*) AS shots
            FROM shots GROUP BY player
            HAVING COUNT(*) >= ?
            ORDER BY {order_col} DESC
            LIMIT ?
            """,  # order_col is from a fixed allowlist above
            [min_shots, limit],
        ).fetchall()
        return [
            PlayerRow(player=r[0], team=r[1], xg=r[2], npxg=r[3], goals=r[4], shots=r[5])
            for r in rows
        ]

    def player_shot_log(self, player: str) -> list[dict]:
        """One player's shots across all loaded matches, for a profile view."""
        rows = self._con.execute(
            "SELECT match_id, minute, xg, outcome, is_goal, is_penalty, body_part "
            "FROM shots WHERE player=? ORDER BY match_id, minute",
            [player],
        ).fetchall()
        return [
            {
                "match_id": r[0],
                "minute": r[1],
                "xg": r[2],
                "outcome": r[3],
                "is_goal": r[4],
                "is_penalty": r[5],
                "body_part": r[6],
            }
            for r in rows
        ]

    def player_count(self) -> int:
        return self._con.execute("SELECT COUNT(DISTINCT player) FROM shots").fetchone()[0]

    # --- StatsBomb full-event player stats ------------------------------------

    def load_player_stats(self, stats: list) -> int:
        """Load per-match player stats, replacing each match's set so re-ingest is idempotent."""
        if not stats:
            return 0
        frame = pl.DataFrame([asdict(s) for s in stats])
        match_ids = {s.match_id for s in stats}
        self._con.register("incoming_pstats", frame)
        try:
            self._con.execute("BEGIN")
            for match_id in match_ids:
                self._con.execute("DELETE FROM player_match_stats WHERE match_id = ?", [match_id])
            self._con.execute(
                "INSERT INTO player_match_stats BY NAME SELECT * FROM incoming_pstats"
            )
            self._con.execute("COMMIT")
        except Exception:
            self._con.execute("ROLLBACK")
            raise
        finally:
            self._con.unregister("incoming_pstats")
        return len(stats)

    def player_stats_count(self) -> int:
        return self._con.execute(
            "SELECT COUNT(DISTINCT player) FROM player_match_stats"
        ).fetchone()[0]

    def player_profiles(
        self,
        *,
        limit: int = 30,
        min_minutes: int = 90,
        order: str = "contributions",
        competition: str | None = None,
    ) -> list[PlayerProfile]:
        """Full player profiles across loaded matches, ranked by `order`.

        `order` is one of the keys in `_PROFILE_ORDER` (goals, xg, assists, xa,
        contributions, progressive, defensive, ...). `min_minutes` drops cameos;
        `competition` restricts to one competition's matches (comparable percentiles).
        """
        order_expr = _PROFILE_ORDER.get(order, _PROFILE_ORDER["contributions"])
        select, params = _profile_select(competition)
        rows = self._con.execute(
            f"{select} WHERE p.minutes >= ? "
            f"ORDER BY {order_expr} DESC, p.minutes DESC LIMIT ?",  # order_expr is allowlisted
            [*params, min_minutes, limit],
        ).fetchall()
        return [PlayerProfile(*r) for r in rows]

    def player_profile(
        self, player: str, *, competition: str | None = None
    ) -> PlayerProfile | None:
        """One player's full profile, or None if they have no event stats loaded."""
        select, params = _profile_select(competition)
        rows = self._con.execute(f"{select} WHERE p.player = ?", [*params, player]).fetchall()
        return PlayerProfile(*rows[0]) if rows else None

    # --- StatsBomb match metadata ---------------------------------------------

    def load_match_meta(self, metas: list[MatchMeta]) -> int:
        """Upsert match identities so player stats can be filtered by competition."""
        if not metas:
            return 0
        frame = pl.DataFrame([asdict(m) for m in metas])
        self._con.register("incoming_meta", frame)
        try:
            self._con.execute("BEGIN")
            for meta in metas:
                self._con.execute("DELETE FROM match_meta WHERE match_id = ?", [meta.match_id])
            self._con.execute("INSERT INTO match_meta BY NAME SELECT * FROM incoming_meta")
            self._con.execute("COMMIT")
        except Exception:
            self._con.execute("ROLLBACK")
            raise
        finally:
            self._con.unregister("incoming_meta")
        return len(metas)

    def competitions_loaded(self) -> list[tuple[str, int]]:
        """(competition, matches with player stats) for the loaded-competition filter."""
        return self._con.execute(
            "SELECT m.competition, COUNT(DISTINCT p.match_id) "
            "FROM player_match_stats p JOIN match_meta m ON m.match_id = p.match_id "
            "GROUP BY m.competition ORDER BY COUNT(DISTINCT p.match_id) DESC"
        ).fetchall()

    def match_label(self, match_id: int) -> str | None:
        """'Home v Away (competition season)' for a match, if its metadata is loaded."""
        row = self._con.execute(
            "SELECT home_team, away_team, competition, season FROM match_meta WHERE match_id = ?",
            [match_id],
        ).fetchone()
        if not row:
            return None
        return f"{row[0]} v {row[1]} ({row[2]} {row[3]})"
