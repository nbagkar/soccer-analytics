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
    # Role and price drive the forecast adjustment: FPL element_type (1 GK, 2 DEF, 3 MID,
    # 4 FWD) routes a loss to attack vs defence; now_cost (price x10) is a quality proxy that,
    # unlike minutes/points, is meaningful pre-season. Optional so a source without them still
    # stores availability (it just can't weight the adjustment).
    element_type: int | None = None
    price: int | None = None


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
    element_type: int | None = None
    price: int | None = None

    @property
    def is_flagged(self) -> bool:
        return self.status in FLAGGED_STATUSES

    @property
    def label(self) -> str:
        """Availability with the chance-of-playing appended when it adds information."""
        if self.chance is not None and 0 <= self.chance < 100:
            return f"{self.availability} ({self.chance}%)"
        return self.availability


# --- forecast adjustment -----------------------------------------------------
#
# Turning current team news into a nudge on expected goals. This is a transparent PRIOR, not a
# fitted or backtested effect: there is no history of who was injured before past matches to
# score it against (FPL is a snapshot, and its terms bar accumulating one), so the weights and
# caps below are deliberately conservative. The base rating already absorbs a club's *typical*
# injuries, so this only bends the forecast for the notable, current absences.

# FPL element_type -> (attack_weight, defence_weight). A forward's absence hurts scoring most,
# a keeper's/defender's hurts conceding most, a midfielder splits. These route each missing
# player's price (its quality weight) to the correct side of the ball.
_ROLE_WEIGHTS: dict[int, tuple[float, float]] = {
    1: (0.0, 1.0),  # goalkeeper
    2: (0.1, 1.0),  # defender
    3: (0.6, 0.4),  # midfielder
    4: (1.0, 0.1),  # forward
}
# How hard a fully-missing contingent bends expected goals, and the caps that stop any single
# snapshot swinging a forecast implausibly (never more than a quarter either way).
_ATTACK_SENSITIVITY = 0.6
_DEFENCE_SENSITIVITY = 0.6
_ATTACK_FLOOR = 0.75
_LEAK_CEIL = 1.25


def _unavailability(status: str, chance: int | None) -> float:
    """How unavailable a player is: 1.0 when out (i/s/u), scaled by chance for a doubt (d)."""
    if status in ("i", "s", "u"):
        return 1.0
    if status == "d":
        return 1.0 - (chance / 100.0) if chance is not None else 0.5
    return 0.0  # available (or not-in-squad, excluded upstream) -> no loss


@dataclass(frozen=True)
class AvailabilityAdjustment:
    """A club's forecast nudge from current team news, split by side of the ball."""

    attack_factor: float  # multiply this club's expected goals by this (<= 1.0)
    leak_factor: float  # multiply the OPPONENT's expected goals by this (>= 1.0)
    lost_attack: float  # price-weighted fraction of attacking strength missing (0-1)
    lost_defence: float  # price-weighted fraction of defensive strength missing (0-1)
    missing: tuple[str, ...]  # the flagged players driving it, most important first

    @property
    def is_material(self) -> bool:
        """Whether it actually moves a number -- worth applying and showing."""
        return abs(self.attack_factor - 1.0) > 1e-9 or abs(self.leak_factor - 1.0) > 1e-9


NEUTRAL_ADJUSTMENT = AvailabilityAdjustment(1.0, 1.0, 0.0, 0.0, ())


def team_adjustment(rows: list[AvailabilityRow]) -> AvailabilityAdjustment:
    """Attack/leak factors for a club from its stored availability, price- and role-weighted.

    The denominator is the club's whole priced squad (bar loaned-out 'n' players), so depth is
    resilience -- losing one star from a deep squad costs proportionally less. Price stands in
    for quality: the one importance signal that is meaningful before the season's own minutes
    and points exist. Returns a neutral (no-op) adjustment when nothing is priced or nobody is
    flagged, so a fully fit team leaves the forecast exactly as the base model produced it.
    """
    att_total = att_lost = def_total = def_lost = 0.0
    missing: list[tuple[float, str]] = []
    for r in rows:
        if r.price is None or r.element_type not in _ROLE_WEIGHTS or r.status == "n":
            continue  # unpriced, unknown role, or not in the squad -> not part of the XI
        aw, dw = _ROLE_WEIGHTS[r.element_type]
        att_w, def_w = aw * r.price, dw * r.price
        att_total += att_w
        def_total += def_w
        u = _unavailability(r.status, r.chance)
        if u > 0.0:
            att_lost += att_w * u
            def_lost += def_w * u
            missing.append((att_w + def_w, r.player))
    if att_total == 0.0 and def_total == 0.0:
        return NEUTRAL_ADJUSTMENT
    lost_attack = att_lost / att_total if att_total else 0.0
    lost_defence = def_lost / def_total if def_total else 0.0
    attack_factor = max(_ATTACK_FLOOR, 1.0 - _ATTACK_SENSITIVITY * lost_attack)
    leak_factor = min(_LEAK_CEIL, 1.0 + _DEFENCE_SENSITIVITY * lost_defence)
    ordered = tuple(name for _, name in sorted(missing, reverse=True))
    return AvailabilityAdjustment(attack_factor, leak_factor, lost_attack, lost_defence, ordered)


# Read order: worst status first, then club, then player -- a stable, human-sensible sort.
_ORDER_BY = (
    "CASE status WHEN 'i' THEN 0 WHEN 's' THEN 1 WHEN 'u' THEN 2 "
    "WHEN 'd' THEN 3 WHEN 'n' THEN 4 ELSE 5 END, team, player"
)
_COLUMNS = (
    "team, team_norm, player, full_name, status, availability, chance, news, news_added, "
    "element_type, price"
)


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
                " chance, news, news_added, fetched_at, element_type, price) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                        r.element_type,
                        r.price,
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
                element_type=row["element_type"],
                price=row["price"],
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
