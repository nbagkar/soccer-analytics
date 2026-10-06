"""Expected threat: action parsing, the grid fit, and the stored per-player values."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from soccer.models.xthreat import GRID_Y, KIND_CARRY, KIND_PASS, KIND_SHOT, cell_index, fit_xt
from soccer.sources.statsbomb import Action, parse_actions


def _event(kind: str, loc: list[float], **detail: object) -> dict[str, object]:
    event: dict[str, object] = {
        "type": {"name": kind},
        "location": loc,
        "team": {"name": "Home"},
        "player": {"name": "Ana"},
    }
    if detail:
        event[kind.lower()] = detail
    return event


class TestParseActions:
    def test_kinds_outcomes_and_set_pieces(self) -> None:
        events = [
            _event("Pass", [60, 40], end_location=[80, 40]),
            _event("Pass", [60, 40], end_location=[90, 10], outcome={"name": "Incomplete"}),
            _event("Pass", [119, 1], end_location=[110, 40], type={"name": "Corner"}),
            _event("Pass", [60, 40], end_location=[50, 40], type={"name": "Kick Off"}),
            _event("Carry", [80, 40], end_location=[95, 35]),
            _event("Shot", [108, 40], outcome={"name": "Goal"}),
            _event("Shot", [108, 40], outcome={"name": "Saved"}, type={"name": "Penalty"}),
            {"type": {"name": "Pressure"}, "location": [50, 50], "player": {"name": "Ana"}},
        ]
        actions = parse_actions(events, match_id=7)
        assert [a.kind for a in actions] == ["pass", "pass", "pass", "carry", "shot", "shot"]
        assert [a.ok for a in actions] == [True, False, True, True, True, False]
        assert [a.set_piece for a in actions] == [False, False, True, False, False, True]
        assert actions[0].x1 == 80 and actions[4].x1 == 108  # a shot ends where it starts
        assert all(a.match_id == 7 and a.player == "Ana" for a in actions)


def _arrays(rows: list[tuple[int, float, float, float, float, bool, bool]]) -> list[np.ndarray]:
    cols = list(zip(*rows, strict=True))
    return [
        np.array(cols[0], np.int64),
        *(np.array(c, np.float64) for c in cols[1:5]),
        np.array(cols[5], bool),
        np.array(cols[6], bool),
    ]


class TestFitXt:
    def _league(self) -> list[tuple[int, float, float, float, float, bool, bool]]:
        # Every move from a cell succeeds half the time; in the box half the actions are
        # shots that score half the time. So xT is exactly 0.25 (box), 0.125 (midfield,
        # half its moves reach the box) and 0.0625 (own half).
        rows = []
        for _ in range(50):
            rows += [(KIND_PASS, 30, 40, 70, 40, ok, False) for ok in (True, False)]
            rows += [(KIND_CARRY, 70, 40, 105, 40, ok, False) for ok in (True, False)]
            rows += [(KIND_SHOT, 105, 40, 105, 40, ok, False) for ok in (True, False)]
            rows += [(KIND_PASS, 105, 40, 110, 20, False, False)] * 2
        return rows

    def test_threat_rises_toward_goal_and_values_follow(self) -> None:
        xt = fit_xt(*_arrays(self._league()))
        start, mid, box = (
            int(cell_index(np.array([x]), np.array([40.0]))[0]) for x in (30, 70, 105)
        )
        assert (xt[start], xt[mid], xt[box]) == (
            pytest.approx(0.0625, abs=1e-6),
            pytest.approx(0.125, abs=1e-6),
            pytest.approx(0.25, abs=1e-6),
        )

    def test_penalties_do_not_make_the_spot_valuable(self) -> None:
        rows = self._league() + [(KIND_SHOT, 108, 40, 108, 40, True, True)] * 500
        spot = int(cell_index(np.array([108.0]), np.array([40.0]))[0])
        assert fit_xt(*_arrays(rows))[spot] == pytest.approx(fit_xt(*_arrays(self._league()))[spot])

    def test_failed_moves_lower_a_cells_threat(self) -> None:
        rows = self._league()
        worse = rows + [(KIND_CARRY, 70, 40, 105, 40, False, False)] * 50
        mid = int(cell_index(np.array([70.0]), np.array([40.0]))[0])
        assert fit_xt(*_arrays(worse))[mid] < fit_xt(*_arrays(rows))[mid]

    def test_cells_clamp_to_the_pitch(self) -> None:
        assert cell_index(np.array([-5.0]), np.array([-5.0]))[0] == 0
        assert cell_index(np.array([200.0]), np.array([200.0]))[0] == 16 * GRID_Y - 1


def test_store_refits_and_credits_players(tmp_path) -> None:
    from soccer.sources.statsbomb import PlayerMatchStats
    from soccer.storage.analytics_db import AnalyticsDB, MatchMeta

    def action(player: str, kind: str, x0: float, x1: float, ok: bool = True) -> Action:
        return Action(1, "Home", player, kind, x0, 40, x1, 40, ok, False)

    actions = (
        [action("Ana", "carry", 70, 105)] * 20  # Ana takes it into the box
        + [action("Bo", "pass", 70, 30)] * 20  # Bo plays it backwards
        + [action("Ana", "shot", 105, 105, ok=i % 2 == 0) for i in range(20)]
    )
    stats = [
        PlayerMatchStats(**{**_zero_stats(), "match_id": 1, "player": p, "team": "Home"})
        for p in ("Ana", "Bo")
    ]
    with AnalyticsDB(tmp_path / "a.duckdb") as adb:
        adb.load_match_meta(
            [MatchMeta(1, "L", "2015/2016", 1, 1, str(date(2016, 1, 1)), "Home", "Away")]
        )
        adb.load_player_stats(stats)
        assert adb.load_actions(actions) == 60
        assert adb.load_actions(actions) == 60  # reloading a match replaces it
        assert adb.refresh_xt() == 60
        assert len(adb.xt_grid()) == 16 * GRID_Y
        profiles = {p.player: p for p in adb.player_profiles(min_minutes=0)}
    assert profiles["Ana"].xt > 0 > profiles["Bo"].xt


def _zero_stats() -> dict[str, object]:
    from dataclasses import fields

    from soccer.sources.statsbomb import PlayerMatchStats

    zero: dict[str, object] = {f.name: 0 for f in fields(PlayerMatchStats)}
    zero.update(position=None, xa=0.0, minutes=90)
    return zero
