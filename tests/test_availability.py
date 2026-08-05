"""AvailabilityStore behaviour: wholesale replace, severity ordering, and the flagged views.

Availability is a snapshot of *now*, so the store's defining property is that a refresh
replaces a source wholesale -- a recovered player must vanish, not linger. The read paths
sort worst-status-first and split "team news" (injury/suspension/doubt) from the merely
available, which is what every availability question downstream relies on.
"""

from __future__ import annotations

from soccer.domain.availability import AvailabilityStore, PlayerAvailability, status_label
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
