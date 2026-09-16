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
    ConfirmedLineupStore,
    PlayerAvailability,
    apply_confirmed_gap,
    confirmed_squad_gap,
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
    minutes: int | None = None,
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
        minutes=minutes,
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

    def test_minutes_survive_the_round_trip(self, tmp_path) -> None:
        # confirmed_squad_gap ranks by minutes straight off the stored row.
        store = _store(tmp_path)
        store.replace_source("fpl", [_rec("Arsenal", "Saka", "a", minutes=2500)])
        (row,) = store.for_team("arsenal")
        assert row.minutes == 2500


# A plausible priced Arsenal XI: (player, FPL element_type, now_cost, season minutes).
# Enough depth that one absence is a fraction of the whole, a spread of prices so quality
# actually weights, and a spread of minutes so regulars (Saka, Odegaard, ...) are clearly
# separated from fringe squad players (Merino, Jesus) for the confirmed-squad-gap tests.
_SQUAD: tuple[tuple[str, int, int, int], ...] = (
    ("Raya", 1, 55, 2700),  # GK
    ("Saliba", 2, 60, 2600),  # DEF
    ("Gabriel", 2, 55, 2500),  # DEF
    ("White", 2, 50, 2400),  # DEF
    ("Timber", 2, 45, 2000),  # DEF
    ("Rice", 3, 65, 2600),  # MID
    ("Odegaard", 3, 85, 2500),  # MID
    ("Saka", 3, 100, 2700),  # MID (winger; FPL classes wingers as MID)
    ("Merino", 3, 55, 900),  # MID -- fringe, low minutes
    ("Havertz", 4, 80, 2200),  # FWD
    ("Jesus", 4, 70, 600),  # FWD -- fringe, low minutes
    # Deeper bench, so the squad exceeds `_REGULARS_SQUAD_SIZE` (14) and the "who's a
    # regular" cutoff actually excludes someone -- the youngest fringe player, below.
    ("Nwaneri", 3, 45, 500),
    ("Norton-Cuffy", 2, 40, 400),
    ("Kiwior", 2, 45, 300),
    ("YouthPlayer", 2, 40, 200),  # the least-used squad player -- outside the top 14
)


def _arow(
    player: str,
    status: str,
    element_type: int | None,
    price: int | None,
    *,
    chance: int | None = None,
    minutes: int | None = None,
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
        minutes=minutes,
    )


def _squad(flags: dict[str, tuple[str, int | None]] | None = None) -> list[AvailabilityRow]:
    """The squad as availability rows: every player available unless `flags` marks one
    with a (status, chance) pair."""
    flags = flags or {}
    rows: list[AvailabilityRow] = []
    for player, etype, price, minutes in _SQUAD:
        status, chance = flags.get(player, ("a", None))
        rows.append(_arow(player, status, etype, price, chance=chance, minutes=minutes))
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
        attackers_out = {p: ("i", None) for p, etype, _, _ in _SQUAD if etype in (3, 4)}
        adj = team_adjustment(_squad(attackers_out))
        assert adj.attack_factor == pytest.approx(_ATTACK_FLOOR)
        # Every keeper and defender out: the leak ceils.
        defenders_out = {p: ("i", None) for p, etype, _, _ in _SQUAD if etype in (1, 2)}
        adj2 = team_adjustment(_squad(defenders_out))
        assert adj2.leak_factor == pytest.approx(_LEAK_CEIL)

    def test_empty_input_is_neutral(self) -> None:
        assert team_adjustment([]) == NEUTRAL_ADJUSTMENT


class TestTeamAdjustmentSanity:
    """Ablation guards on the prior's *routing and aggregation* -- the parts that have a right
    answer. (The magnitude knobs -- sensitivity, caps -- deliberately aren't asserted here: with
    no injury history to score against they're an honest prior, not a fitted quantity.) What must
    always hold: more absences never soften the nudge, a dearer loss of the same role hurts at
    least as much, and a deeper squad absorbs the same loss more."""

    def test_more_absences_never_soften_the_nudge(self) -> None:
        one = team_adjustment(_squad({"Havertz": ("i", None)}))
        two = team_adjustment(_squad({"Havertz": ("i", None), "Jesus": ("i", None)}))
        assert two.is_material
        # Monotone: adding an absence can only cut attack further and leak no less.
        assert two.attack_factor <= one.attack_factor
        assert two.leak_factor >= one.leak_factor
        assert two.lost_attack > one.lost_attack

    def test_a_costlier_loss_of_the_same_role_hurts_at_least_as_much(self) -> None:
        # Price is the only difference (both forwards): the dearer absence bends attack more.
        dear = team_adjustment(_squad({"Havertz": ("i", None)}))  # priced 80
        cheap = team_adjustment(_squad({"Jesus": ("i", None)}))  # priced 70
        assert dear.attack_factor < cheap.attack_factor

    def test_depth_is_resilience(self) -> None:
        # The same forward's absence costs proportionally less once the squad is deeper, because
        # the whole priced squad is the denominator -- losing one of many dilutes the fraction.
        shallow = team_adjustment(_squad({"Havertz": ("i", None)}))
        deeper = _squad({"Havertz": ("i", None)})
        deeper.append(_arow("Nketiah", "a", 4, 55))  # extra priced cover -> bigger denominator
        deeper.append(_arow("Trossard", "a", 3, 65))
        deep = team_adjustment(deeper)
        assert deep.is_material
        assert deep.attack_factor > shallow.attack_factor


class TestConfirmedLineupStore:
    def test_no_data_is_none_not_an_empty_set(self, tmp_path) -> None:
        store = ConfirmedLineupStore(LiveDB(tmp_path / "live.sqlite"))
        assert store.for_team("arsenal") is None

    def test_replace_and_read_back(self, tmp_path) -> None:
        store = ConfirmedLineupStore(LiveDB(tmp_path / "live.sqlite"))
        store.replace_team("arsenal", ["Raya", "Saka"], "2026-09-16T12:00:00+00:00")
        assert store.for_team("arsenal") == {"Raya", "Saka"}

    def test_replace_is_wholesale_not_upsert(self, tmp_path) -> None:
        store = ConfirmedLineupStore(LiveDB(tmp_path / "live.sqlite"))
        store.replace_team("arsenal", ["Raya", "Saka"], "t1")
        store.replace_team("arsenal", ["Raya"], "t2")  # Saka dropped from the new squad
        assert store.for_team("arsenal") == {"Raya"}

    def test_replace_is_scoped_to_one_team(self, tmp_path) -> None:
        store = ConfirmedLineupStore(LiveDB(tmp_path / "live.sqlite"))
        store.replace_team("arsenal", ["Raya"], "t1")
        store.replace_team("chelsea", ["Sanchez"], "t1")
        store.replace_team("arsenal", ["Raya", "Saka"], "t2")
        assert store.for_team("chelsea") == {"Sanchez"}  # untouched by arsenal's refresh


class TestConfirmedSquadGap:
    """The real-time supplement to FPL's season-long snapshot: a regular missing from
    today's confirmed matchday squad is team news; a fringe player's absence never was."""

    def test_a_regular_missing_from_the_confirmed_squad_is_flagged(self) -> None:
        rows = _squad()  # everyone "available" per FPL -- nothing flagged there
        confirmed = {p for p, *_ in _SQUAD if p != "Saka"}  # Saka didn't make the squad at all
        gap = confirmed_squad_gap(rows, confirmed)
        assert "Saka" in gap

    def test_a_fringe_players_absence_is_not_flagged(self) -> None:
        rows = _squad()  # 15 players; YouthPlayer (200 mins) falls outside the top 14
        confirmed = {p for p, *_ in _SQUAD if p != "YouthPlayer"}
        gap = confirmed_squad_gap(rows, confirmed)
        assert "YouthPlayer" not in gap

    def test_everyone_confirmed_present_is_a_clean_no_op(self) -> None:
        rows = _squad()
        confirmed = {p for p, *_ in _SQUAD}
        assert confirmed_squad_gap(rows, confirmed) == ()

    def test_rows_without_minutes_data_cannot_be_ranked(self) -> None:
        rows = [_arow(p, "a", et, pr) for p, et, pr, _ in _SQUAD]  # minutes=None throughout
        assert confirmed_squad_gap(rows, set()) == ()

    def test_already_fpl_flagged_players_are_excluded_from_the_ranking(self) -> None:
        # A regular FPL already reports injured ('n' == not in squad, excluded like the
        # existing team_adjustment filter) shouldn't count toward "who should be here".
        rows = _squad({"Saka": ("n", None)})
        confirmed = {p for p, *_ in _SQUAD if p not in ("Saka", "Havertz")}
        gap = confirmed_squad_gap(rows, confirmed)
        assert "Saka" not in gap  # already excluded from the regulars ranking (status "n")
        assert "Havertz" in gap  # a genuinely regular player missing from today's squad


class TestApplyConfirmedGap:
    def test_empty_gap_is_a_strict_no_op(self) -> None:
        rows = _squad()
        assert apply_confirmed_gap(rows, ()) is rows

    def test_gap_player_becomes_fully_unavailable(self) -> None:
        rows = _squad()
        updated = apply_confirmed_gap(rows, ("Saka",))
        saka = next(r for r in updated if r.player == "Saka")
        assert saka.status == "u"
        assert saka.availability == status_label("u")
        # team_adjustment then treats it exactly like an FPL-flagged absence.
        assert team_adjustment(updated).is_material

    def test_already_flagged_player_keeps_its_own_fpl_status(self) -> None:
        # FPL already says Havertz is doubtful at 40% -- the confirmed-squad-gap signal
        # (which only means "fully out today") must not override that finer-grained read.
        rows = _squad({"Havertz": ("d", 40)})
        updated = apply_confirmed_gap(rows, ("Havertz",))
        havertz = next(r for r in updated if r.player == "Havertz")
        assert havertz.status == "d"
        assert havertz.chance == 40

    def test_untouched_players_are_unaffected(self) -> None:
        rows = _squad()
        updated = apply_confirmed_gap(rows, ("Saka",))
        others = [r for r in updated if r.player != "Saka"]
        assert others == [r for r in rows if r.player != "Saka"]
