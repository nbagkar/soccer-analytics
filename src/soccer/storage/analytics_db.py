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
    referee            VARCHAR
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
"""


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
class ResultRow:
    """The slim result shape the models consume (satisfies their Outcome protocols)."""

    match_date: date
    home: str
    away: str
    home_norm: str
    away_norm: str
    fthg: int
    ftag: int


class AnalyticsDB:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if self.path.parent != Path():
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(str(self.path))
        self._con.execute("SET enable_progress_bar = false")  # keep CLI output clean
        self._con.execute(_SCHEMA)

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
