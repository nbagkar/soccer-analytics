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

    def fetch_shots(self, match_id: int) -> list[Shot]:
        """Fetch a match's events, snapshot them, and parse the shots.

        Returns [] for a match whose events are not in the open data (some listed matches
        have none) rather than raising, so a competition sweep skips them cleanly.
        """
        response = self._client.get(f"{BASE_URL}/events/{match_id}.json")
        if response.status_code == 404:
            logger.info("No events for match %s (404)", match_id)
            return []
        response.raise_for_status()
        events = response.json()

        # Snapshot the raw events so shots can be re-parsed after a parser change.
        self._raw.write(
            SourceId.STATSBOMB,
            f"events_{match_id}",
            events,
            request_meta={"match_id": match_id},
        )
        return parse_shots(events, match_id)
