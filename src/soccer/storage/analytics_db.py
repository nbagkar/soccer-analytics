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

    def outcomes_for(self, season: str, division: str) -> list[ResultRow]:
        """Results for one (season, division) in date order, for the models to fit on."""
        rows = self._con.execute(
            "SELECT match_date, home, away, home_norm, away_norm, fthg, ftag "
            "FROM results WHERE season=? AND division=? ORDER BY match_date, home",
            [season, division],
        ).fetchall()
        return [ResultRow(*r) for r in rows]
