"""StatsBomb open-data adapter -- historical event data, the analytics lab.

Fetches raw event JSON, snapshots it, and parses shots (location, StatsBomb's own xG,
outcome, player). No kloppy or modelling needed: StatsBomb ships xG per shot, and their
JSON is well-structured, so this fits the same fetch-snapshot-parse shape as the other
adapters.

**Licence -- read before shipping anything derived.** StatsBomb open data is governed by
a proprietary EULA (see docs/source-verification.md and sources/datasets.py): no
redistribution of the data, no commercial use of the data OR analysis derived from it,
and attribution must use the StatsBomb logo. So: the parsed shots live only in the local
gitignored `data/`, are never exposed verbatim, and every surface that shows them credits
StatsBomb. Off by default; a deliberate opt-in.

The canonical repo moved statsbomb/open-data -> hudl/open-data; the old raw URLs redirect,
so requests must follow redirects (kloppy's loader does not, which is why this fetches
directly).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

from soccer.sources.registry import SourceId
from soccer.storage.raw import RawStore

logger = logging.getLogger(__name__)

BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
ATTRIBUTION = "Event data from StatsBomb open data (attribution requires the StatsBomb logo)"


@dataclass(frozen=True)
class Shot:
    match_id: int
    team: str
    player: str
    minute: int
    period: int
    x: float | None
    y: float | None
    xg: float
    outcome: str
    is_goal: bool
    is_penalty: bool
    body_part: str | None


@dataclass(frozen=True)
class PlayerMatchStats:
    """One player's aggregated contribution in one match, from the full event stream.

    Everything a StatsBomb events file can attribute to a player beyond shooting:
    passing (volume, completion, progression, chance creation), ball carrying and
    dribbling, defensive actions, and discipline -- plus an appearance-minutes estimate
    so the dashboard can normalize to per-90. xG/goals stay in the `shots` table; xa here
    is expected assists (the xG of shots a player's passes set up, via key_pass_id).
    """

    match_id: int
    player: str
    team: str
    position: str | None
    minutes: int
    passes: int
    passes_completed: int
    key_passes: int
    assists: int
    xa: float
    progressive_passes: int
    carries: int
    progressive_carries: int
    dribbles: int
    dribbles_completed: int
    tackles: int
    tackles_won: int
    interceptions: int
    blocks: int
    clearances: int
    ball_recoveries: int
    pressures: int
    fouls: int
    fouled: int
    yellow_cards: int
    red_cards: int
    touches: int


# A pass moving the ball this far up-pitch (StatsBomb x, 0..120 toward goal) into at
# least the middle third counts as progressive; a carry gaining this much likewise.
_PROG_PASS_GAIN = 15.0
_PROG_CARRY_GAIN = 10.0
_TACKLE_WON = {"Won", "Success", "Success In Play", "Success Out"}
_RED_CARDS = {"Red Card", "Second Yellow"}


def parse_shots(events: list[dict], match_id: int) -> list[Shot]:
    """Extract shots from a StatsBomb events array. Tolerant of missing sub-fields."""
    shots: list[Shot] = []
    for event in events:
        if event.get("type", {}).get("name") != "Shot":
            continue
        shot = event.get("shot", {})
        location = event.get("location") or [None, None]
        outcome = shot.get("outcome", {}).get("name", "Unknown")
        shots.append(
            Shot(
                match_id=match_id,
                team=event.get("team", {}).get("name", "Unknown"),
                player=event.get("player", {}).get("name", "Unknown"),
                minute=event.get("minute", 0),
                period=event.get("period", 0),
                x=location[0],
                y=location[1],
                xg=float(shot.get("statsbomb_xg", 0.0)),
                outcome=outcome,
                is_goal=outcome == "Goal",
                is_penalty=shot.get("type", {}).get("name") == "Penalty",
                body_part=shot.get("body_part", {}).get("name"),
            )
        )
    return shots


def parse_player_stats(events: list[dict], match_id: int) -> list[PlayerMatchStats]:
    """Aggregate per-player match contributions from a StatsBomb events array.

    One pass over the events accumulates counters per player; a second short pass links
    shots back to the passes that created them (via `shot.key_pass_id`) for key passes and
    expected assists. Appearance minutes come from the Starting XI and Substitution events.
    Defensive throughout -- StatsBomb omits sub-objects (a completed pass has no `outcome`),
    so every access tolerates absence.
    """
    from collections import defaultdict

    match_end = max((e.get("minute", 0) for e in events), default=90)
    starters: dict[str, str | None] = {}
    on_min: dict[str, int] = {}
    off_min: dict[str, int] = {}
    teams: dict[str, str] = {}
    positions: dict[str, str | None] = {}
    pass_player: dict[str, str] = {}
    stat: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for e in events:
        etype = e.get("type", {}).get("name")

        if etype == "Starting XI":
            team = e.get("team", {}).get("name", "Unknown")
            for slot in e.get("tactics", {}).get("lineup", []):
                name = slot.get("player", {}).get("name")
                if name:
                    starters[name] = slot.get("position", {}).get("name")
                    on_min[name] = 0
                    teams.setdefault(name, team)
            continue

        if etype == "Substitution":
            off = e.get("player", {}).get("name")
            repl = e.get("substitution", {}).get("replacement", {}).get("name")
            minute = e.get("minute", match_end)
            if off:
                off_min[off] = minute
            if repl:
                on_min[repl] = minute
                teams.setdefault(repl, e.get("team", {}).get("name", "Unknown"))
            continue

        player = e.get("player", {}).get("name")
        if not player:
            continue
        teams.setdefault(player, e.get("team", {}).get("name", "Unknown"))
        if e.get("position"):
            positions.setdefault(player, e.get("position", {}).get("name"))
        loc = e.get("location") or [None, None]
        s = stat[player]

        if etype == "Pass":
            if e.get("id"):
                pass_player[e["id"]] = player
            pass_obj = e.get("pass", {})
            s["passes"] += 1
            s["touches"] += 1
            if "outcome" not in pass_obj:  # StatsBomb marks only NON-completions
                s["passes_completed"] += 1
                end = pass_obj.get("end_location") or [None, None]
                if (
                    loc[0] is not None
                    and end[0] is not None
                    and end[0] - loc[0] >= _PROG_PASS_GAIN
                    and end[0] >= 60
                ):
                    s["progressive_passes"] += 1
            if pass_obj.get("goal_assist"):
                s["assists"] += 1
        elif etype == "Carry":
            s["carries"] += 1
            s["touches"] += 1
            end = e.get("carry", {}).get("end_location") or [None, None]
            if loc[0] is not None and end[0] is not None and end[0] - loc[0] >= _PROG_CARRY_GAIN:
                s["progressive_carries"] += 1
        elif etype == "Dribble":
            s["dribbles"] += 1
            s["touches"] += 1
            if e.get("dribble", {}).get("outcome", {}).get("name") == "Complete":
                s["dribbles_completed"] += 1
        elif etype == "Shot":
            s["touches"] += 1
        elif etype == "Duel":
            if e.get("duel", {}).get("type", {}).get("name") == "Tackle":
                s["tackles"] += 1
                if e.get("duel", {}).get("outcome", {}).get("name") in _TACKLE_WON:
                    s["tackles_won"] += 1
        elif etype == "Interception":
            s["interceptions"] += 1
        elif etype == "Block":
            s["blocks"] += 1
        elif etype == "Clearance":
            s["clearances"] += 1
        elif etype == "Ball Recovery":
            if not e.get("ball_recovery", {}).get("recovery_failure"):
                s["ball_recoveries"] += 1
        elif etype == "Pressure":
            s["pressures"] += 1
        elif etype == "Foul Committed":
            s["fouls"] += 1
            card = e.get("foul_committed", {}).get("card", {}).get("name", "")
            if card == "Yellow Card":
                s["yellow_cards"] += 1
            elif card in _RED_CARDS:
                s["red_cards"] += 1
        elif etype == "Foul Won":
            s["fouled"] += 1
        elif etype == "Bad Behaviour":
            card = e.get("bad_behaviour", {}).get("card", {}).get("name", "")
            if card == "Yellow Card":
                s["yellow_cards"] += 1
            elif card in _RED_CARDS:
                s["red_cards"] += 1

    # Second pass: credit the passer of each shot's key pass with a key pass and its xG (xA).
    for e in events:
        if e.get("type", {}).get("name") != "Shot":
            continue
        key_pass_id = e.get("shot", {}).get("key_pass_id")
        passer = pass_player.get(key_pass_id) if key_pass_id else None
        if passer:
            stat[passer]["key_passes"] += 1
            stat[passer]["xa"] += float(e.get("shot", {}).get("statsbomb_xg", 0.0))

    rows: list[PlayerMatchStats] = []
    for player in set(stat) | set(on_min) | set(starters):
        s = stat[player]
        minutes = max(0, int(off_min.get(player, match_end) - on_min.get(player, 0)))
        rows.append(
            PlayerMatchStats(
                match_id=match_id,
                player=player,
                team=teams.get(player, "Unknown"),
                position=starters.get(player) or positions.get(player),
                minutes=minutes,
                passes=int(s["passes"]),
                passes_completed=int(s["passes_completed"]),
                key_passes=int(s["key_passes"]),
                assists=int(s["assists"]),
                xa=round(s["xa"], 4),
                progressive_passes=int(s["progressive_passes"]),
                carries=int(s["carries"]),
                progressive_carries=int(s["progressive_carries"]),
                dribbles=int(s["dribbles"]),
                dribbles_completed=int(s["dribbles_completed"]),
                tackles=int(s["tackles"]),
                tackles_won=int(s["tackles_won"]),
                interceptions=int(s["interceptions"]),
                blocks=int(s["blocks"]),
                clearances=int(s["clearances"]),
                ball_recoveries=int(s["ball_recoveries"]),
                pressures=int(s["pressures"]),
                fouls=int(s["fouls"]),
                fouled=int(s["fouled"]),
                yellow_cards=int(s["yellow_cards"]),
                red_cards=int(s["red_cards"]),
                touches=int(s["touches"]),
            )
        )
    return rows


class StatsBomb:
    def __init__(
        self,
        raw_store: RawStore,
        *,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._raw = raw_store
        self._client = client or httpx.Client(timeout=timeout, follow_redirects=True)

    def __enter__(self) -> StatsBomb:
        return self

    def __exit__(self, *_: object) -> None:
        self._client.close()

    def competitions(self) -> list[dict]:
        response = self._client.get(f"{BASE_URL}/competitions.json")
        response.raise_for_status()
        return response.json()

    def matches(self, competition_id: int, season_id: int) -> list[dict]:
        response = self._client.get(f"{BASE_URL}/matches/{competition_id}/{season_id}.json")
        if response.status_code == 404:
            return []
        response.raise_for_status()
        return response.json()

    def fetch_events(self, match_id: int) -> list[dict]:
        """Fetch a match's raw events and snapshot them. [] if not in the open data (404).

        The single network read behind both shots and player stats -- parse the returned
        list with `parse_shots` and `parse_player_stats` so one fetch feeds both.
        """
        response = self._client.get(f"{BASE_URL}/events/{match_id}.json")
        if response.status_code == 404:
            logger.info("No events for match %s (404)", match_id)
            return []
        response.raise_for_status()
        events = response.json()

        # Snapshot the raw events so anything derived can be re-parsed after a parser change.
        self._raw.write(
            SourceId.STATSBOMB,
            f"events_{match_id}",
            events,
            request_meta={"match_id": match_id},
        )
        return events

    def fetch_shots(self, match_id: int) -> list[Shot]:
        """Fetch a match's events, snapshot them, and parse the shots."""
        return parse_shots(self.fetch_events(match_id), match_id)
