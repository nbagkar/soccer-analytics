"""AvailabilityStore behaviour: wholesale replace, severity ordering, and the flagged views.

Availability is a snapshot of *now*, so the store's defining property is that a refresh
replaces a source wholesale -- a recovered player must vanish, not linger. The read paths
sort worst-status-first and split "team news" (injury/suspension/doubt) from the merely
available, which is what every availability question downstream relies on.
"""

from __future__ import annotations

import pytest

from soccer.domain.availability import (
    _ATTACK_FLOOR,
    _LEAK_CEIL,
    NEUTRAL_ADJUSTMENT,
    AvailabilityRow,
    AvailabilityStore,
    PlayerAvailability,
    status_label,
    team_adjustment,
)
from soccer.domain.names import normalize_name
from soccer.storage.live_db import LiveDB


def _rec(
    team: str,
    player: str,
    status: str,
    *,
    chance: int | None = None,
    news: str | None = None,
    full_name: str | None = None,
    element_type: int | None = None,
    price: int | None = None,
) -> PlayerAvailability:
    return PlayerAvailability(
        source="fpl",
        team=team,
        team_norm=normalize_name(team),
        player=player,
        full_name=full_name,
        status=status,
        availability=status_label(status),
        chance=chance,
        news=news,
        news_added=None,
        fetched_at="2026-08-04T00:00:00+00:00",
        element_type=element_type,
        price=price,
    )


def _store(tmp_path) -> AvailabilityStore:
    return AvailabilityStore(LiveDB(tmp_path / "live.sqlite"))


class TestReplaceSource:
    def test_replace_is_wholesale_not_upsert(self, tmp_path) -> None:
        store = _store(tmp_path)
        store.replace_source("fpl", [_rec("Arsenal", "Saka", "i")])
        # A second refresh in which Saka has recovered and Rice is now hurt: Saka must be gone.
        store.replace_source("fpl", [_rec("Arsenal", "Rice", "d")])
        assert [r.player for r in store.for_team("arsenal")] == ["Rice"]

    def test_replace_is_scoped_to_the_source(self, tmp_path) -> None:
        store = _store(tmp_path)
        store.replace_source("fpl", [_rec("Arsenal", "Saka", "i")])
        # A different source refreshing must not wipe FPL's rows.
        store.replace_source("other", [_rec("Chelsea", "James", "d")])
        assert store.count() == 2
        assert [r.player for r in store.for_team("arsenal")] == ["Saka"]


class TestOrdering:
    def test_for_team_orders_worst_status_first(self, tmp_path) -> None:
        store = _store(tmp_path)
        store.replace_source(
            "fpl",
            [
                _rec("Arsenal", "Raya", "a"),
                _rec("Arsenal", "Rice", "d"),
                _rec("Arsenal", "Saka", "i"),
                _rec("Arsenal", "Timber", "s"),
            ],
        )
        # injured < suspended < unavailable < doubtful < not-in-squad < available
        assert [r.player for r in store.for_team("arsenal")] == ["Saka", "Timber", "Rice", "Raya"]


class TestFlaggedViews:
    def test_flagged_excludes_available_and_not_in_squad(self, tmp_path) -> None:
        store = _store(tmp_path)
        store.replace_source(
            "fpl",
            [
                _rec("Arsenal", "Raya", "a"),
                _rec("Arsenal", "Nwaneri", "n"),
                _rec("Arsenal", "Saka", "i"),
                _rec("Chelsea", "James", "d"),
            ],
        )
        assert {r.player for r in store.flagged()} == {"Saka", "James"}
        assert store.flagged_count() == 2
        assert store.count() == 4

    def test_flagged_for_team_scopes_to_one_club(self, tmp_path) -> None:
        store = _store(tmp_path)
        store.replace_source(
            "fpl",
            [_rec("Arsenal", "Saka", "i"), _rec("Chelsea", "James", "d")],
        )
        assert [r.player for r in store.flagged_for_team("arsenal")] == ["Saka"]

    def test_flagged_respects_limit(self, tmp_path) -> None:
        store = _store(tmp_path)
        store.replace_source(
            "fpl", [_rec("Arsenal", f"P{i}", "i") for i in range(10)]
        )
        assert len(store.flagged(limit=3)) == 3
        assert store.flagged_count() == 10  # the count is unaffected by a read limit


class TestLabel:
    def test_doubtful_label_appends_chance(self, tmp_path) -> None:
        store = _store(tmp_path)
        store.replace_source("fpl", [_rec("Arsenal", "Rice", "d", chance=75)])
        (row,) = store.for_team("arsenal")
        assert row.label == "Doubtful (75%)"

    def test_available_label_has_no_chance_noise(self, tmp_path) -> None:
        store = _store(tmp_path)
        # 100% == fully available; appending "(100%)" would be noise, so it's suppressed.
        store.replace_source("fpl", [_rec("Arsenal", "Raya", "a", chance=100)])
        (row,) = store.for_team("arsenal")
        assert row.label == "Available"
        assert not row.is_flagged

    def test_role_and_price_survive_the_round_trip(self, tmp_path) -> None:
        # The forecast adjustment reads element_type/price straight off the stored row, so
        # they must persist through replace_source -> for_team unchanged.
        store = _store(tmp_path)
        store.replace_source("fpl", [_rec("Arsenal", "Saka", "i", element_type=3, price=100)])
        (row,) = store.for_team("arsenal")
        assert row.element_type == 3
        assert row.price == 100


# A plausible priced Arsenal XI: (player, FPL element_type, now_cost). Enough depth that one
# absence is a fraction of the whole, and a spread of prices so quality actually weights.
_SQUAD: tuple[tuple[str, int, int], ...] = (
    ("Raya", 1, 55),  # GK
    ("Saliba", 2, 60),  # DEF
    ("Gabriel", 2, 55),  # DEF
    ("White", 2, 50),  # DEF
    ("Timber", 2, 45),  # DEF
    ("Rice", 3, 65),  # MID
    ("Odegaard", 3, 85),  # MID
    ("Saka", 3, 100),  # MID (winger; FPL classes wingers as MID)
    ("Merino", 3, 55),  # MID
    ("Havertz", 4, 80),  # FWD
    ("Jesus", 4, 70),  # FWD
)


def _arow(
    player: str,
    status: str,
    element_type: int | None,
    price: int | None,
    *,
    chance: int | None = None,
) -> AvailabilityRow:
    return AvailabilityRow(
        team="Arsenal",
        team_norm="arsenal",
        player=player,
        full_name=None,
        status=status,
        availability=status_label(status),
        chance=chance,
        news=None,
        news_added=None,
        element_type=element_type,
        price=price,
    )


def _squad(flags: dict[str, tuple[str, int | None]] | None = None) -> list[AvailabilityRow]:
    """The squad as availability rows: every player available unless `flags` marks one
    with a (status, chance) pair."""
    flags = flags or {}
    rows: list[AvailabilityRow] = []
    for player, etype, price in _SQUAD:
        status, chance = flags.get(player, ("a", None))
        rows.append(_arow(player, status, etype, price, chance=chance))
    return rows


class TestTeamAdjustment:
    """The forecast nudge: a fully fit squad is a strict no-op, and any absence bends the
    number toward the side of the ball it belongs to, bounded and quality-weighted."""

    def test_full_strength_is_a_strict_no_op(self) -> None:
        # The keystone: nobody flagged must leave the base forecast *exactly* untouched,
        # so the model's calibration is preserved whenever there is no news.
        adj = team_adjustment(_squad())
        assert adj == NEUTRAL_ADJUSTMENT
        assert not adj.is_material
        assert adj.attack_factor == 1.0
        assert adj.leak_factor == 1.0
        assert adj.missing == ()

    def test_missing_forward_cuts_attack_far_more_than_leak(self) -> None:
        adj = team_adjustment(_squad({"Havertz": ("i", None)}))
        assert adj.is_material
        assert adj.attack_factor < 1.0
        assert adj.attack_factor >= _ATTACK_FLOOR  # bounded, never absurd
        # A forward barely defends, so his loss moves attack far more than the opponent's leak.
        assert (1.0 - adj.attack_factor) > (adj.leak_factor - 1.0)
        assert "Havertz" in adj.missing

    def test_missing_keeper_raises_leak_and_leaves_attack_untouched(self) -> None:
        adj = team_adjustment(_squad({"Raya": ("i", None)}))
        assert adj.is_material
        assert adj.leak_factor > 1.0
        assert adj.leak_factor <= _LEAK_CEIL  # bounded
        assert adj.attack_factor == 1.0  # a keeper carries zero attacking weight

    def test_doubtful_is_scaled_by_chance_of_playing(self) -> None:
        injured = team_adjustment(_squad({"Havertz": ("i", None)}))
        doubtful = team_adjustment(_squad({"Havertz": ("d", 75)}))
        assert doubtful.is_material
        # 75% chance of playing => only a quarter of the loss counts.
        assert doubtful.attack_factor > injured.attack_factor
        assert (1.0 - doubtful.attack_factor) == pytest.approx(0.25 * (1.0 - injured.attack_factor))

    def test_doubtful_without_a_chance_is_a_coin_flip(self) -> None:
        injured = team_adjustment(_squad({"Havertz": ("i", None)}))
        unknown = team_adjustment(_squad({"Havertz": ("d", None)}))
        # No percentage published => treat a doubt as half-out.
        assert (1.0 - unknown.attack_factor) == pytest.approx(0.5 * (1.0 - injured.attack_factor))

    def test_not_in_squad_and_unpriced_and_unknown_role_are_ignored(self) -> None:
        base = team_adjustment(_squad())
        noisy = _squad()
        noisy.append(_arow("Loanee", "n", 4, 120))  # loaned out -> not in the XI
        noisy.append(_arow("Ghost", "i", 4, None))  # injured but unpriced -> cannot weight
        noisy.append(_arow("Coach", "i", None, 90))  # no role -> cannot route the loss
        assert team_adjustment(noisy) == base == NEUTRAL_ADJUSTMENT

    def test_missing_is_ordered_most_important_first(self) -> None:
        adj = team_adjustment(
            _squad(
                {
                    "Timber": ("i", None),  # cheap defender
                    "Havertz": ("i", None),  # expensive forward
                    "Merino": ("s", None),  # mid-priced midfielder
                }
            )
        )
        assert adj.missing[0] == "Havertz"  # priciest, most attacking -> first
        assert set(adj.missing) == {"Havertz", "Timber", "Merino"}

    def test_factors_are_bounded_by_their_caps(self) -> None:
        # Everyone who attacks is out: the attack cut floors rather than running away.
        attackers_out = {p: ("i", None) for p, etype, _ in _SQUAD if etype in (3, 4)}
        adj = team_adjustment(_squad(attackers_out))
        assert adj.attack_factor == pytest.approx(_ATTACK_FLOOR)
        # Every keeper and defender out: the leak ceils.
        defenders_out = {p: ("i", None) for p, etype, _ in _SQUAD if etype in (1, 2)}
        adj2 = team_adjustment(_squad(defenders_out))
        assert adj2.leak_factor == pytest.approx(_LEAK_CEIL)

    def test_empty_input_is_neutral(self) -> None:
        assert team_adjustment([]) == NEUTRAL_ADJUSTMENT
