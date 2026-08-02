"""football-data.co.uk adapter -- historical results, the analytics backbone.

Static season CSVs (one per division), so this is a batch source, not part of the live
async pipeline: no rate limit, no live state, just download -> snapshot -> parse. It is
a name-only source (no match or team ids), so identity is by normalized name -- the same
`normalize_name` the live crosswalk uses, so football-data.co.uk "Arsenal" and
football-data.org "Arsenal FC" share a join key without a second identity system.

Terms note: no explicit licence exists (see docs/source-verification.md). Downloading and
analysing is plainly the intended use; the CSVs must never be redistributed verbatim.
"""

from __future__ import annotations

import csv
import io
import logging
from dataclasses import dataclass
from datetime import date, datetime

import httpx

from soccer.domain.names import normalize_name
from soccer.sources.registry import SourceId
from soccer.storage.raw import RawStore

logger = logging.getLogger(__name__)

BASE_URL = "https://www.football-data.co.uk/mmz4281"
# The "extra" leagues (Brazil, Argentina, USA, ...) live under a different path with a
# different schema: one file per country holding every season, no shot/card columns.
NEW_LEAGUE_BASE = "https://www.football-data.co.uk/new"

# football-data.co.uk division codes -> the leagues they actually are, for display. The
# codes are the storage key everywhere; this maps them to human names at the edges (the
# dashboard selectors, CLI titles). Unknown codes fall back to the code itself.
DIVISION_NAMES = {
    "E0": "Premier League",
    "E1": "Championship",
    "E2": "League One",
    "E3": "League Two",
    "EC": "National League",
    "SC0": "Scottish Premiership",
    "SC1": "Scottish Championship",
    "SC2": "Scottish League One",
    "SC3": "Scottish League Two",
    "D1": "Bundesliga",
    "D2": "2. Bundesliga",
    "I1": "Serie A",
    "I2": "Serie B",
    "SP1": "La Liga",
    "SP2": "La Liga 2",
    "F1": "Ligue 1",
    "F2": "Ligue 2",
    "N1": "Eredivisie",
    "B1": "Belgian Pro League",
    "P1": "Primeira Liga",
    "T1": "Super Lig",
    "G1": "Super League Greece",
    # "new leagues" country codes
    "BRA": "Brazil Serie A",
    "ARG": "Argentina Primera",
    "USA": "MLS",
    "MEX": "Liga MX",
}


def division_name(code: str) -> str:
    """Human league name for a division code, or the code itself if unmapped."""
    return DIVISION_NAMES.get(code, code)


def season_label(season: str) -> str:
    """Readable season: European "2526" -> "2025/26"; a calendar year "2026" stays as is.

    football-data.co.uk's European files use a two-year code (consecutive halves), while
    the extra-country files use a single calendar year. The consecutive-halves test tells
    them apart without a per-division flag.
    """
    if len(season) == 4 and season.isdigit():
        first, second = int(season[:2]), int(season[2:])
        if (first + 1) % 100 == second:
            # Two-digit year -> century. football-data.co.uk history starts in 1993, so a
            # high prefix (>=90) is a 1990s season; everything else is 20xx. Without this,
            # "9394" rendered as "2093/94".
            century = "19" if first >= 90 else "20"
            return f"{century}{season[:2]}/{season[2:]}"
    return season


# football-data.co.uk "new leagues" country codes (the /new/ path), distinct from the
# European division codes (E0, SP1, ...). Used to route a loaded division to the right
# fetcher when refreshing the current season.
NEW_LEAGUE_CODES = frozenset(
    {"ARG", "AUT", "BRA", "CHN", "DNK", "FIN", "IRL", "JPN", "MEX", "NOR", "POL", "ROU", "USA"}
)


def current_season_code(today: date) -> str:
    """The football-data.co.uk season code for the European season current on `today`.

    Seasons span August->May, so from July onward the *new* season's code applies (its
    file may 404 until the season actually starts -- callers treat that as a no-op). E.g.
    2026-08-01 -> "2627"; 2026-03-01 -> "2526".
    """
    start = today.year if today.month >= 7 else today.year - 1
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


@dataclass(frozen=True)
class MatchResult:
    season: str
    division: str
    match_date: date
    home: str
    away: str
    home_norm: str
    away_norm: str
    fthg: int
    ftag: int
    ftr: str  # 'H' | 'D' | 'A'
    hthg: int | None
    htag: int | None
    home_shots: int | None
    away_shots: int | None
    home_shots_target: int | None
    away_shots_target: int | None
    home_corners: int | None
    away_corners: int | None
    home_yellows: int | None
    away_yellows: int | None
    home_reds: int | None
    away_reds: int | None
    referee: str | None
    # Closing 1X2 decimal odds (best available), for market-edge analysis. Defaulted so
    # existing constructions that predate odds still build.
    close_home_odds: float | None = None
    close_draw_odds: float | None = None
    close_away_odds: float | None = None


# Preference order for closing 1X2 odds: Pinnacle closing (sharpest) -> market-average
# closing -> Bet365 closing -> opening lines as a last resort. The first fully-present
# triple wins. Column names are shared between the mmz4281 and new-league schemas.
_ODDS_KEYS = (
    ("PSCH", "PSCD", "PSCA"),
    ("AvgCH", "AvgCD", "AvgCA"),
    ("B365CH", "B365CD", "B365CA"),
    ("PSH", "PSD", "PSA"),
    ("AvgH", "AvgD", "AvgA"),
    ("B365H", "B365D", "B365A"),
)


def _float(row: dict[str, str], key: str) -> float | None:
    raw = (row.get(key) or "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _best_close_odds(row: dict[str, str]) -> tuple[float | None, float | None, float | None]:
    """The best available closing 1X2 odds triple, or (None, None, None) if none parse."""
    for h_key, d_key, a_key in _ODDS_KEYS:
        h, d, a = _float(row, h_key), _float(row, d_key), _float(row, a_key)
        if h and d and a and h > 1 and d > 1 and a > 1:
            return h, d, a
    return None, None, None


def _parse_date(value: str) -> date | None:
    """football-data.co.uk uses DD/MM/YYYY, or DD/MM/YY in older seasons."""
    value = value.strip()
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def _int(row: dict[str, str], key: str) -> int | None:
    raw = (row.get(key) or "").strip()
    if not raw:
        return None
    try:
        return int(float(raw))
    except ValueError:
        return None


def parse_results_csv(text: str, *, season: str, division: str) -> list[MatchResult]:
    """Parse one division CSV into results, tolerating the format's rough edges.

    Real files carry trailing blank lines, occasional rows with a date but no result
    (abandoned matches), and fewer stat columns in older seasons. Anything without a
    usable date and full-time score is skipped rather than allowed to corrupt the set.
    """
    reader = csv.DictReader(io.StringIO(text))
    results: list[MatchResult] = []
    skipped = 0

    for row in reader:
        home = (row.get("HomeTeam") or "").strip()
        away = (row.get("AwayTeam") or "").strip()
        match_date = _parse_date(row.get("Date") or "")
        fthg, ftag = _int(row, "FTHG"), _int(row, "FTAG")

        if not home or not away or match_date is None or fthg is None or ftag is None:
            skipped += 1
            continue

        close_h, close_d, close_a = _best_close_odds(row)
        results.append(
            MatchResult(
                season=season,
                division=division,
                match_date=match_date,
                home=home,
                away=away,
                home_norm=normalize_name(home),
                away_norm=normalize_name(away),
                fthg=fthg,
                ftag=ftag,
                ftr=(row.get("FTR") or "").strip() or _result_from_score(fthg, ftag),
                hthg=_int(row, "HTHG"),
                htag=_int(row, "HTAG"),
                home_shots=_int(row, "HS"),
                away_shots=_int(row, "AS"),
                home_shots_target=_int(row, "HST"),
                away_shots_target=_int(row, "AST"),
                home_corners=_int(row, "HC"),
                away_corners=_int(row, "AC"),
                home_yellows=_int(row, "HY"),
                away_yellows=_int(row, "AY"),
                home_reds=_int(row, "HR"),
                away_reds=_int(row, "AR"),
                referee=(row.get("Referee") or "").strip() or None,
                close_home_odds=close_h,
                close_draw_odds=close_d,
                close_away_odds=close_a,
            )
        )

    if skipped:
        logger.info("Skipped %d unusable row(s) in %s/%s", skipped, season, division)
    return results


def _result_from_score(home: int, away: int) -> str:
    return "H" if home > away else "A" if away > home else "D"


def parse_new_league_csv(
    text: str, *, division: str, recent_seasons: int | None = None
) -> list[MatchResult]:
    """Parse a football-data.co.uk "new leagues" CSV (Brazil, Argentina, USA, ...).

    A different schema from the European division files: a single file holds every
    season, with columns Country/League/Season/Date/Home/Away/HG/AG/Res and betting odds
    -- but no shot, corner or card stats. `recent_seasons` keeps only the N most-recent
    seasons, so a decade-long file becomes a current, relevant model instead of loading
    years of stale rows (and the model is fit on the newest season regardless).
    """
    rows = list(csv.DictReader(io.StringIO(text)))
    seasons = sorted(
        {(r.get("Season") or "").strip() for r in rows if (r.get("Season") or "").strip()}
    )
    keep = set(seasons[-recent_seasons:]) if recent_seasons else set(seasons)

    results: list[MatchResult] = []
    skipped = 0
    for row in rows:
        season = (row.get("Season") or "").strip()
        home = (row.get("Home") or "").strip()
        away = (row.get("Away") or "").strip()
        match_date = _parse_date(row.get("Date") or "")
        hg, ag = _int(row, "HG"), _int(row, "AG")

        if season not in keep or not home or not away or match_date is None or hg is None:
            skipped += 1
            continue
        if ag is None:
            skipped += 1
            continue

        close_h, close_d, close_a = _best_close_odds(row)
        results.append(
            MatchResult(
                season=season,
                division=division,
                match_date=match_date,
                home=home,
                away=away,
                home_norm=normalize_name(home),
                away_norm=normalize_name(away),
                fthg=hg,
                ftag=ag,
                ftr=(row.get("Res") or "").strip() or _result_from_score(hg, ag),
                hthg=None,
                htag=None,
                home_shots=None,
                away_shots=None,
                home_shots_target=None,
                away_shots_target=None,
                home_corners=None,
                away_corners=None,
                home_yellows=None,
                away_yellows=None,
                home_reds=None,
                away_reds=None,
                referee=None,
                close_home_odds=close_h,
                close_draw_odds=close_d,
                close_away_odds=close_a,
            )
        )

    if skipped:
        logger.info("Skipped %d row(s) outside kept seasons / unusable in %s", skipped, division)
    return results


class FootballDataCoUk:
    def __init__(
        self,
        raw_store: RawStore,
        *,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._raw = raw_store
        self._client = client or httpx.Client(timeout=timeout, follow_redirects=True)

    def __enter__(self) -> FootballDataCoUk:
        return self

    def __exit__(self, *_: object) -> None:
        self._client.close()

    def fetch_division(self, season: str, division: str) -> list[MatchResult]:
        """Download and parse one (season, division). Stores the raw CSV as a snapshot.

        `season` is football-data.co.uk's 4-digit form, e.g. "2526" for 2025/26.
        Returns [] for a 404 (a division/season that does not exist) rather than raising,
        since callers sweep many combinations of which some are absent.
        """
        url = f"{BASE_URL}/{season}/{division}.csv"
        response = self._client.get(url)
        if response.status_code == 404:
            logger.info("No data for %s/%s (404)", season, division)
            return []
        response.raise_for_status()

        # Snapshot the CSV text verbatim so history can be re-parsed after a parser fix.
        self._raw.write(
            SourceId.FOOTBALL_DATA_CO_UK,
            f"{season}_{division}",
            response.text,
            request_meta={"url": url, "season": season, "division": division},
        )
        return parse_results_csv(response.text, season=season, division=division)

    def fetch_new_league(
        self, code: str, *, division: str | None = None, recent_seasons: int | None = 3
    ) -> list[MatchResult]:
        """Download and parse a "new leagues" country file (e.g. "BRA" for Brazil).

        `code` is football-data.co.uk's country code; `division` is the code stored in the
        analytics DB (defaults to `code`). Keeps the `recent_seasons` most-recent seasons.
        Returns [] on a 404 so a sweep of several countries skips absent ones cleanly.
        """
        division = division or code
        url = f"{NEW_LEAGUE_BASE}/{code}.csv"
        response = self._client.get(url)
        if response.status_code == 404:
            logger.info("No new-league file for %s (404)", code)
            return []
        response.raise_for_status()

        self._raw.write(
            SourceId.FOOTBALL_DATA_CO_UK,
            f"new_{code}",
            response.text,
            request_meta={"url": url, "code": code, "division": division},
        )
        return parse_new_league_csv(response.text, division=division, recent_seasons=recent_seasons)
