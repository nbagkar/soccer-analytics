"""BetLedger behaviour: add/settle/list, default payouts by outcome, and the summary yield.

The ledger is the only place in this app that measures a REAL, forward-looking result
rather than a walk-forward backtest -- so its arithmetic (yield, win rate, what a "won"
bet pays without an explicit payout) has to be exactly right.
"""

from __future__ import annotations

import pytest

from soccer.domain.bets import Bet, BetLedger, BetStatus
from soccer.storage.live_db import LiveDB


@pytest.fixture
def ledger(tmp_path) -> BetLedger:
    db = LiveDB(tmp_path / "live.sqlite")
    return BetLedger(db)


def _add(ledger: BetLedger, **overrides) -> int:
    defaults = dict(
        competition="Premier League",
        home="Arsenal",
        away="Chelsea",
        match_date="2026-09-20",
        selection="Arsenal win",
        odds=2.0,
        stake=10.0,
    )
    return ledger.add(**{**defaults, **overrides})


class TestAddAndList:
    def test_a_new_bet_is_pending_with_no_payout(self, ledger: BetLedger) -> None:
        bet_id = _add(ledger)
        bet = ledger.list()[0]
        assert bet.id == bet_id
        assert bet.status is BetStatus.PENDING
        assert bet.payout is None
        assert bet.profit is None

    def test_list_orders_most_recent_match_first(self, ledger: BetLedger) -> None:
        _add(ledger, match_date="2026-09-01", home="Team A")
        _add(ledger, match_date="2026-09-30", home="Team B")
        bets = ledger.list()
        assert [b.home for b in bets] == ["Team B", "Team A"]

    def test_list_filters_by_status(self, ledger: BetLedger) -> None:
        won_id = _add(ledger, home="Winner")
        _add(ledger, home="StillOpen")
        ledger.settle(won_id, BetStatus.WON)
        pending = ledger.list(status=BetStatus.PENDING)
        assert len(pending) == 1
        assert pending[0].home == "StillOpen"

    def test_optional_fields_round_trip(self, ledger: BetLedger) -> None:
        _add(ledger, model_probability=0.62, notes="closing line was 1.85")
        bet = ledger.list()[0]
        assert bet.model_probability == pytest.approx(0.62)
        assert bet.notes == "closing line was 1.85"


class TestSettle:
    def test_won_defaults_to_stake_times_odds(self, ledger: BetLedger) -> None:
        bet_id = _add(ledger, odds=2.5, stake=10.0)
        ledger.settle(bet_id, BetStatus.WON)
        bet = ledger.list()[0]
        assert bet.payout == pytest.approx(25.0)
        assert bet.profit == pytest.approx(15.0)
        assert bet.settled_at is not None

    def test_lost_defaults_to_zero(self, ledger: BetLedger) -> None:
        bet_id = _add(ledger, stake=10.0)
        ledger.settle(bet_id, BetStatus.LOST)
        bet = ledger.list()[0]
        assert bet.payout == pytest.approx(0.0)
        assert bet.profit == pytest.approx(-10.0)

    def test_void_returns_the_stake_with_no_profit_or_loss(self, ledger: BetLedger) -> None:
        bet_id = _add(ledger, stake=10.0)
        ledger.settle(bet_id, BetStatus.VOID)
        bet = ledger.list()[0]
        assert bet.payout == pytest.approx(10.0)
        assert bet.profit == pytest.approx(0.0)

    def test_explicit_payout_overrides_the_default(self, ledger: BetLedger) -> None:
        bet_id = _add(ledger, odds=2.0, stake=10.0)
        ledger.settle(bet_id, BetStatus.WON, payout=30.0)  # e.g. a promo-boosted price
        assert ledger.list()[0].payout == pytest.approx(30.0)

    def test_settling_as_pending_is_rejected(self, ledger: BetLedger) -> None:
        bet_id = _add(ledger)
        with pytest.raises(ValueError, match="not pending"):
            ledger.settle(bet_id, BetStatus.PENDING)

    def test_settling_an_unknown_id_raises(self, ledger: BetLedger) -> None:
        with pytest.raises(KeyError):
            ledger.settle(999, BetStatus.WON)


class TestDelete:
    def test_delete_removes_the_bet(self, ledger: BetLedger) -> None:
        bet_id = _add(ledger)
        ledger.delete(bet_id)
        assert ledger.list() == []


class TestSummary:
    def test_empty_ledger_is_all_zeros(self, ledger: BetLedger) -> None:
        s = ledger.summary()
        assert s.n_settled == 0 and s.n_pending == 0
        assert s.staked == 0.0 and s.returned == 0.0
        assert s.yield_pct == 0.0
        assert s.win_rate == 0.0

    def test_pending_bets_count_but_do_not_affect_yield(self, ledger: BetLedger) -> None:
        _add(ledger, stake=10.0)
        s = ledger.summary()
        assert s.n_pending == 1
        assert s.n_settled == 0
        assert s.staked == 0.0

    def test_yield_and_win_rate_over_a_mixed_ledger(self, ledger: BetLedger) -> None:
        won = _add(ledger, odds=3.0, stake=10.0)
        lost = _add(ledger, odds=2.0, stake=10.0, home="Other")
        void = _add(ledger, odds=1.5, stake=10.0, home="Postponed")
        _add(ledger, home="StillPending")  # excluded from settled totals
        ledger.settle(won, BetStatus.WON)
        ledger.settle(lost, BetStatus.LOST)
        ledger.settle(void, BetStatus.VOID)

        s = ledger.summary()
        assert s.n_settled == 3
        assert s.n_pending == 1
        assert (s.wins, s.losses, s.voids) == (1, 1, 1)
        assert s.staked == pytest.approx(30.0)
        assert s.returned == pytest.approx(30.0 + 0.0 + 10.0)  # 30 won, 0 lost, 10 void back
        assert s.profit == pytest.approx(10.0)
        assert s.yield_pct == pytest.approx(100.0 * 10.0 / 30.0)
        assert s.win_rate == pytest.approx(0.5)  # 1 win / (1 win + 1 loss), void excluded


class TestBetProperties:
    def test_profit_is_none_while_pending(self) -> None:
        bet = Bet(
            id=1,
            placed_at="2026-01-01T00:00:00+00:00",
            competition="Premier League",
            home="A",
            away="B",
            match_date="2026-01-05",
            selection="A win",
            model_probability=None,
            odds=2.0,
            stake=10.0,
            status=BetStatus.PENDING,
            payout=None,
            notes=None,
            settled_at=None,
        )
        assert bet.profit is None
