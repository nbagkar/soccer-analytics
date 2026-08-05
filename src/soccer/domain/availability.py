"""Player availability: who is injured, suspended, or a doubt right now.

A current-state cache, not history. The whole point of holding it in the live SQLite store
(and replacing it wholesale on each refresh) is that the only provider for it -- Fantasy
Premier League -- is licensed for private, ephemeral use and bars 'creating a database'.
So there is exactly one row per (source, club, player), always the latest snapshot.

`team_norm` is the reconciled, normalized club name, so availability joins straight onto
the squads and results the rest of the app already knows -- "who's injured at Arsenal"
resolves the club the same way every other intent does.
"""

from __future__ import annotations

from dataclasses import dataclass

from soccer.storage.live_db import LiveDB

# FPL status codes, worst-first. Drives both display order and what counts as "team news":
# an injury, suspension, doubt or other unavailability is news; "available" and the
# loan/not-registered "not in squad" are not. Ranks are stable so callers can sort by them.
_STATUS_RANK: dict[str, int] = {"i": 0, "s": 1, "u": 2, "d": 3, "n": 4, "a": 5}
_STATUS_LABEL: dict[str, str] = {
    "i": "Injured",
    "s": "Suspended",
    "u": "Unavailable",
    "d": "Doubtful",
    "n": "Not in squad",
    "a": "Available",
}
# The statuses that constitute team news -- what "who's injured/out" should surface.
FLAGGED_STATUSES: tuple[str, ...] = ("i", "s", "d", "u")


def status_label(status: str) -> str:
    return _STATUS_LABEL.get(status, status or "Unknown")


def status_rank(status: str) -> int:
    return _STATUS_RANK.get(status, 9)


@dataclass(frozen=True)
class PlayerAvailability:
    """One player's current availability, as produced by a source mapper and stored verbatim."""

    source: str
    team: str
    team_norm: str
    player: str
    full_name: str | None
    status: str
    availability: str
    chance: int | None
    news: str | None
    news_added: str | None
    fetched_at: str


@dataclass(frozen=True)
class AvailabilityRow:
    """A row read back for display."""

    team: str
    team_norm: str
    player: str
    full_name: str | None
    status: str
    availability: str
    chance: int | None
    news: str | None
    news_added: str | None

    @property
    def is_flagged(self) -> bool:
        return self.status in FLAGGED_STATUSES

    @property
    def label(self) -> str:
        """Availability with the chance-of-playing appended when it adds information."""
        if self.chance is not None and 0 <= self.chance < 100:
            return f"{self.availability} ({self.chance}%)"
        return self.availability


# Read order: worst status first, then club, then player -- a stable, human-sensible sort.
_ORDER_BY = (
    "CASE status WHEN 'i' THEN 0 WHEN 's' THEN 1 WHEN 'u' THEN 2 "
    "WHEN 'd' THEN 3 WHEN 'n' THEN 4 ELSE 5 END, team, player"
)
_COLUMNS = "team, team_norm, player, full_name, status, availability, chance, news, news_added"


class AvailabilityStore:
    def __init__(self, db: LiveDB) -> None:
        self._conn = db.connection

    def replace_source(self, source: str, records: list[PlayerAvailability]) -> int:
        """Replace all of one source's availability wholesale. Returns rows written.

        Atomic replace, not upsert: availability is a snapshot of *now*, so a player who
        recovered must vanish, not linger. Scoped to the source so a second provider added
        later refreshes independently.
        """
        with self._conn:  # BEGIN/COMMIT; rolls back on error
            self._conn.execute("DELETE FROM player_availability WHERE source=?", (source,))
            self._conn.executemany(
                "INSERT INTO player_availability "
                "(source, team, team_norm, player, full_name, status, availability, "
                " chance, news, news_added, fetched_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        r.source,
                        r.team,
                        r.team_norm,
                        r.player,
                        r.full_name,
                        r.status,
                        r.availability,
                        r.chance,
                        r.news,
                        r.news_added,
                        r.fetched_at,
                    )
                    for r in records
                ],
            )
        return len(records)

    def _rows(
        self, where: str, params: tuple, *, limit: int | None = None
    ) -> list[AvailabilityRow]:
        sql = f"SELECT {_COLUMNS} FROM player_availability WHERE {where} ORDER BY {_ORDER_BY}"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        return [
            AvailabilityRow(
                team=row["team"],
                team_norm=row["team_norm"],
                player=row["player"],
                full_name=row["full_name"],
                status=row["status"],
                availability=row["availability"],
                chance=row["chance"],
                news=row["news"],
                news_added=row["news_added"],
            )
            for row in self._conn.execute(sql, params)
        ]

    def for_team(self, team_norm: str) -> list[AvailabilityRow]:
        """Every stored player for a club, worst status first."""
        return self._rows("team_norm=?", (team_norm,))

    def flagged_for_team(self, team_norm: str) -> list[AvailabilityRow]:
        """A club's injured / suspended / doubtful players only."""
        marks = ",".join("?" * len(FLAGGED_STATUSES))
        return self._rows(f"team_norm=? AND status IN ({marks})", (team_norm, *FLAGGED_STATUSES))

    def flagged(self, *, limit: int = 40) -> list[AvailabilityRow]:
        """League-wide team news: everyone injured / suspended / doubtful, worst first."""
        marks = ",".join("?" * len(FLAGGED_STATUSES))
        return self._rows(f"status IN ({marks})", FLAGGED_STATUSES, limit=limit)

    def for_source(self, source: str) -> list[AvailabilityRow]:
        """Every row for a source -- used to match a named player across all clubs."""
        return self._rows("source=?", (source,))

    def count(self) -> int:
        """Total players held (any status)."""
        return self._conn.execute("SELECT COUNT(*) FROM player_availability").fetchone()[0]

    def flagged_count(self) -> int:
        """Players currently flagged as team news (injured / suspended / doubtful)."""
        marks = ",".join("?" * len(FLAGGED_STATUSES))
        return self._conn.execute(
            f"SELECT COUNT(*) FROM player_availability WHERE status IN ({marks})",
            FLAGGED_STATUSES,
        ).fetchone()[0]
