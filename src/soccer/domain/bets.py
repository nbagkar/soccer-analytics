"""Personal bet ledger: what was staked, at what odds, against what the model said --
and, once known, what actually happened.

Pure record-keeping. Nothing here places a bet, fetches odds, or moves money; it exists
because everything else in this app is judged by a walk-forward backtest, and a backtest
cannot score a decision that has not happened yet. The model already loses to the closing
line overall (see the forecast-model-quality memory / Predictions -> Scorecard) and a
weekly-accumulator strategy backtests worse than betting the same picks straight (see
models/parlay.py) -- so the honest next question is whether real decisions, tracked over
time, do any better. A backtest cannot answer that; only tracking real results can.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from soccer.storage.live_db import LiveDB


class BetStatus(StrEnum):
    PENDING = "pending"
    WON = "won"
    LOST = "lost"
    VOID = "void"
    """Push or cancelled: the stake is returned, no profit or loss."""


_DEFAULT_PAYOUT: dict[BetStatus, Callable[[float, float], float]] = {
    BetStatus.WON: lambda odds, stake: odds * stake,
    BetStatus.LOST: lambda odds, stake: 0.0,
    BetStatus.VOID: lambda odds, stake: stake,
}


@dataclass(frozen=True)
class Bet:
    id: int
    placed_at: str
    competition: str
    home: str
    away: str
    match_date: str
    selection: str
    """Free text: "Arsenal win", "Home win + Over 2.5", "2-leg: Arsenal win, Chelsea win"."""
    model_probability: float | None
    """The model's estimate for this exact selection at the time of placing, if known --
    lets a later calibration check compare what the model said to what actually happened."""
    odds: float
    stake: float
    status: BetStatus
    payout: float | None
    """None while pending; set on settle()."""
    notes: str | None
    settled_at: str | None

    @property
    def profit(self) -> float | None:
        return None if self.payout is None else self.payout - self.stake


@dataclass(frozen=True)
class LedgerSummary:
    n_settled: int
    n_pending: int
    wins: int
    losses: int
    voids: int
    staked: float
    returned: float

    @property
    def profit(self) -> float:
        return self.returned - self.staked

    @property
    def yield_pct(self) -> float:
        return 100.0 * self.profit / self.staked if self.staked else 0.0

    @property
    def win_rate(self) -> float:
        """Wins as a fraction of decided (won/lost) bets -- voids and pending excluded."""
        decided = self.wins + self.losses
        return self.wins / decided if decided else 0.0


def _row_to_bet(row: sqlite3.Row) -> Bet:
    return Bet(
        id=row["id"],
        placed_at=row["placed_at"],
        competition=row["competition"],
        home=row["home"],
        away=row["away"],
        match_date=row["match_date"],
        selection=row["selection"],
        model_probability=row["model_probability"],
        odds=row["odds"],
        stake=row["stake"],
        status=BetStatus(row["status"]),
        payout=row["payout"],
        notes=row["notes"],
        settled_at=row["settled_at"],
    )


class BetLedger:
    def __init__(self, db: LiveDB) -> None:
        self._conn = db.connection

    def add(
        self,
        *,
        competition: str,
        home: str,
        away: str,
        match_date: str,
        selection: str,
        odds: float,
        stake: float,
        model_probability: float | None = None,
        notes: str | None = None,
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO bet (placed_at, competition, home, away, match_date, selection, "
            " model_probability, odds, stake, status, payout, notes, settled_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, NULL)",
            (
                datetime.now(UTC).isoformat(),
                competition,
                home,
                away,
                match_date,
                selection,
                model_probability,
                odds,
                stake,
                BetStatus.PENDING.value,
                notes,
            ),
        )
        assert cur.lastrowid is not None
        return cur.lastrowid

    def settle(self, bet_id: int, status: BetStatus, *, payout: float | None = None) -> None:
        """Record a bet's outcome.

        `payout` defaults by status: stake*odds if won, 0 if lost, the stake back if void --
        pass it explicitly only when a book pays out a promo boost or a partial cash-out.
        """
        if status is BetStatus.PENDING:
            raise ValueError("settle() needs a final status, not pending")
        if payout is None:
            row = self._conn.execute(
                "SELECT odds, stake FROM bet WHERE id=?", (bet_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"no bet with id {bet_id}")
            payout = _DEFAULT_PAYOUT[status](row["odds"], row["stake"])
        self._conn.execute(
            "UPDATE bet SET status=?, payout=?, settled_at=? WHERE id=?",
            (status.value, payout, datetime.now(UTC).isoformat(), bet_id),
        )

    def delete(self, bet_id: int) -> None:
        self._conn.execute("DELETE FROM bet WHERE id=?", (bet_id,))

    def list(self, *, status: BetStatus | None = None) -> list[Bet]:
        """Every bet, most recent match first. Filter to one status, or leave it None for all."""
        if status is None:
            rows = self._conn.execute(
                "SELECT * FROM bet ORDER BY match_date DESC, id DESC"
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM bet WHERE status=? ORDER BY match_date DESC, id DESC",
                (status.value,),
            ).fetchall()
        return [_row_to_bet(row) for row in rows]

    def summary(self) -> LedgerSummary:
        """Real, forward-looking yield -- over settled bets only; pending bets haven't
        resolved yet so they contribute a count but no stake/return to the yield."""
        all_bets = self.list()
        settled = [b for b in all_bets if b.status is not BetStatus.PENDING]
        return LedgerSummary(
            n_settled=len(settled),
            n_pending=sum(1 for b in all_bets if b.status is BetStatus.PENDING),
            wins=sum(1 for b in settled if b.status is BetStatus.WON),
            losses=sum(1 for b in settled if b.status is BetStatus.LOST),
            voids=sum(1 for b in settled if b.status is BetStatus.VOID),
            staked=sum(b.stake for b in settled),
            returned=sum(b.payout or 0.0 for b in settled),
        )
