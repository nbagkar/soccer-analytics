"""Historical dataset registry.

Separate from `registry.py` because datasets differ from live sources in the way that
matters most here: their licences vary from fully permissive to outright proprietary,
and getting that wrong has consequences that no amount of correct code fixes.

The `commercial_use` and `may_redistribute` flags exist to be checked, not merely read.
StatsBomb's public data carries a proprietary EULA barring both -- and the bar extends
to "any analysis derived from" the data, not just the raw files. Any export, public
API, or monetization path must consult these flags first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class DatasetId(StrEnum):
    STATSBOMB = "statsbomb"
    WYSCOUT = "wyscout"
    SKILLCORNER = "skillcorner"


class DataKind(StrEnum):
    EVENTS = "events"
    THREE_SIXTY = "three_sixty"
    TRACKING = "tracking"
    LINEUPS = "lineups"


@dataclass(frozen=True)
class Dataset:
    id: DatasetId
    name: str
    url: str
    kinds: frozenset[DataKind]

    licence: str
    licence_source: str
    """Where the binding licence actually lives. Not always the repo README."""

    may_redistribute: bool
    commercial_use: bool
    attribution: str

    matches: int
    size_gb: float
    kloppy_support: bool

    logo_attribution_required: bool = False
    caveats: tuple[str, ...] = field(default_factory=tuple)


DATASETS: dict[DatasetId, Dataset] = {
    DatasetId.STATSBOMB: Dataset(
        id=DatasetId.STATSBOMB,
        name="StatsBomb Open Data",
        url="https://github.com/hudl/open-data",
        kinds=frozenset({DataKind.EVENTS, DataKind.THREE_SIXTY, DataKind.LINEUPS}),
        licence="StatsBomb Public Data User Agreement (proprietary EULA)",
        licence_source="LICENSE.pdf in the repo -- NOT the README's softer wording",
        may_redistribute=False,
        commercial_use=False,
        attribution="StatsBomb",
        logo_attribution_required=True,
        matches=4_235,
        size_gb=16.13,
        kloppy_support=True,
        caveats=(
            "NOT open data despite the name. EULA s1.2.1 bars distributing or "
            "reproducing the data to any third party; s1.2.2 bars commercially "
            "exploiting the data OR any analysis derived from it.",
            "Analysis and research are permitted (s1.1). Building tools is fine; "
            "redistributing what they ingest is not.",
            "Attribution requires the StatsBomb LOGO (s1.4), not a text credit.",
            "Never clone -- 16.13 GB working tree, ~23.5 GB with history. Fetch "
            "per-match via kloppy load_open_data() and cache only what is used.",
            "360 availability: key on match_available_360, NOT match_updated_360. "
            "Rows with a non-null match_updated_360 may have no 360 data at all.",
            "Governed by England and Wales law.",
        ),
    ),
    DatasetId.WYSCOUT: Dataset(
        id=DatasetId.WYSCOUT,
        name="Wyscout public event dataset",
        url="https://github.com/koenvo/wyscout-soccer-match-event-dataset",
        kinds=frozenset({DataKind.EVENTS}),
        licence="CC BY 4.0",
        licence_source="figshare articles 7770599 / 7770422 / 7765196",
        may_redistribute=True,
        commercial_use=True,
        attribution=(
            "Pappalardo, L., Cintia, P., Rossi, A. et al. A public data set of "
            "spatio-temporal match events in soccer competitions. Sci Data 6, 236 "
            "(2019). https://doi.org/10.1038/s41597-019-0247-7"
        ),
        matches=1_941,
        size_gb=0.285,
        kloppy_support=True,
        caveats=(
            "The most permissive event dataset available -- commercial use and "
            "redistribution both allowed.",
            "The GitHub repo has NO LICENSE file and the API reports license: null. "
            "Its CC BY 4.0 claim is inherited. Attribute via figshare, which is the "
            "verified source of record.",
            "Static since 2023-12-04. That is fine -- it reformats a frozen dataset.",
            "Use processed-v2/ for kloppy >= 3.14.",
        ),
    ),
    DatasetId.SKILLCORNER: Dataset(
        id=DatasetId.SKILLCORNER,
        name="SkillCorner Open Data",
        url="https://github.com/SkillCorner/opendata",
        kinds=frozenset({DataKind.TRACKING}),
        licence="MIT",
        licence_source="LICENSE in the repo",
        may_redistribute=True,
        commercial_use=True,
        attribution="SkillCorner (requested, not required)",
        matches=10,
        size_gb=0.164,
        kloppy_support=True,
        caveats=(
            "10 matches, Australian A-League 2024/25. Older write-ups citing 9 "
            "matches or European fixtures are out of date.",
            "BROADCAST tracking, not full tracking. Off-camera players are "
            "EXTRAPOLATED, not omitted -- every frame looks complete, so naive code "
            "silently treats guesses as observations.",
            "Filter on is_detected before any analysis needing complete simultaneous "
            "positions: pitch control, packing, off-ball shape, team centroid, "
            "spatial dominance.",
            "Valid without filtering: on-ball work, physical metrics over detected "
            "spans, anything restricted to detected players.",
            "socceraction cannot consume this -- it is event-only. Tracking work is "
            "kloppy plus custom code.",
            "~97% player-identity accuracy; smooth speed/acceleration before use.",
        ),
    ),
}


def redistributable() -> list[Dataset]:
    """Datasets whose data may be exposed to third parties. Check before any export."""
    return [d for d in DATASETS.values() if d.may_redistribute]


def commercially_usable() -> list[Dataset]:
    """Datasets usable if this platform is ever monetized."""
    return [d for d in DATASETS.values() if d.commercial_use]


def total_size_gb(*, include_360: bool = True) -> float:
    total = sum(d.size_gb for d in DATASETS.values())
    if not include_360:
        total -= 3.21  # StatsBomb three-sixty
    return round(total, 2)
