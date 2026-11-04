"""Forecast ledger: the forecasts actually published before kickoff, and how they did.

Every forecast-quality number elsewhere is a walk-forward backtest -- a re-enactment, scored
by code that knows how the story ends. This keeps the live track record instead: each time
the upcoming fixtures are forecast (the Predictions page, its background warm-up, `soccer
serve`), one row per fixture is written with what was shown, and it is frozen at kickoff.
Once the result lands it is scored the same way the backtest is (log loss, RPS, Brier),
against the bookmaker's closing line where the result carries one, and against base rates.

Per fixture the ledger keeps two forecasts: the FIRST one ever published, and the LAST one
published before kickoff (the "closing" forecast, comparable to the closing line). A
recomputation that produces exactly the last forecast again writes nothing, so recording on
every page view is idempotent. Nothing is written once kickoff has passed -- not even a late
correction -- because a forecast amended after the whistle is not a forecast.

Its own SQLite file (`Settings.ledger_db`), like the usage log: writing it must never move
the live or analytics stores' mtimes, which key the dashboard's data caches.

Grading here is pure: callers hand in the candidate results (from DuckDB and the live store)
and base rates, so the matching rules are unit-testable without any database.
"""

from __future__ import annotations

import logging
import math
import sqlite3
from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from soccer.domain.names import normalize_name
from soccer.models.evaluation import Probs, Score, _brier, _log_loss, rps

logger = logging.getLogger(__name__)

# A result may be dated the day before or after the UTC kickoff (a 01:00 UTC kickoff in
# Brazil is the previous local evening; football-data.co.uk dates are local).
MATCH_WINDOW_DAYS = 1
# Kicked off this long ago with no result anywhere -> "unmatched" rather than "awaiting":
# football-data.co.uk usually posts results within a few days.
UNMATCHED_AFTER = timedelta(days=10)
# Below this many graded forecasts, the live record is called out as too small to read.
SMALL_SAMPLE = 100


@dataclass(frozen=True)
class PublishedForecast:
    """One fixture's forecast as shown, the input to `ForecastLedger.record`."""

    match_id: str | None
    """The canonical match id from the live store, when the fixture came from there."""
    competition: str
    division: str | None
    """The results division it is graded against ("E0", "UCL", ...), None if unknown."""
    kickoff_utc: datetime
    home: str
    away: str
    home_norm: str
    away_norm: str
    p_home: float
    p_draw: float
    p_away: float
    home_xg: float
    away_xg: float
    model_version: str
    team_news: bool = False
    """Whether the forecast had the team-news (availability) adjustment applied."""

    @property
    def fixture_key(self) -> str:
        return fixture_key(
            self.match_id,
            self.division or self.competition,
            self.kickoff_utc,
            self.home_norm,
            self.away_norm,
        )


def fixture_key(
    match_id: str | None, division: str, kickoff_utc: datetime, home_norm: str, away_norm: str
) -> str:
    """A fixture's stable identity: the canonical match id where one exists, else division +
    kickoff date + normalized teams (so the same fixture from a re-run maps to one row)."""
    if match_id:
        return f"match:{match_id}"
    day = kickoff_utc.astimezone(UTC).date().isoformat()
    return f"{division}|{day}|{normalize_name(home_norm)}|{normalize_name(away_norm)}"


@dataclass(frozen=True)
class ForecastSnapshot:
    recorded_at: datetime
    probs: Probs
    home_xg: float
    away_xg: float
    model_version: str
    team_news: bool


@dataclass(frozen=True)
class LedgerEntry:
    fixture_key: str
    match_id: str | None
    competition: str
    division: str | None
    kickoff_utc: datetime
    home: str
    away: str
    home_norm: str
    away_norm: str
    first: ForecastSnapshot
    last: ForecastSnapshot
    revisions: int
    """How many distinct forecasts were published for it (1 = never changed)."""


_SCHEMA = """
CREATE TABLE IF NOT EXISTS forecast (
    fixture_key        TEXT PRIMARY KEY,
    match_id           TEXT,
    competition        TEXT NOT NULL,
    division           TEXT,
    kickoff_utc        TEXT NOT NULL,
    home               TEXT NOT NULL,
    away               TEXT NOT NULL,
    home_norm          TEXT NOT NULL,
    away_norm          TEXT NOT NULL,
    first_recorded_at  TEXT NOT NULL,
    first_p_home       REAL NOT NULL,
    first_p_draw       REAL NOT NULL,
    first_p_away       REAL NOT NULL,
    first_home_xg      REAL NOT NULL,
    first_away_xg      REAL NOT NULL,
    first_model        TEXT NOT NULL,
    first_team_news    INTEGER NOT NULL,
    last_recorded_at   TEXT NOT NULL,
    last_p_home        REAL NOT NULL,
    last_p_draw        REAL NOT NULL,
    last_p_away        REAL NOT NULL,
    last_home_xg       REAL NOT NULL,
    last_away_xg       REAL NOT NULL,
    last_model         TEXT NOT NULL,
    last_team_news     INTEGER NOT NULL,
    revisions          INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_forecast_kickoff ON forecast (kickoff_utc);
"""

# Probabilities and expected goals are stored at this precision, and a recomputation that
# matches the stored last forecast at it is "the same forecast" (no write).
_DP = 6


class LedgerDB:
    """The ledger's own SQLite file (`Settings.ledger_db`)."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)

    @property
    def connection(self) -> sqlite3.Connection:
        return self._conn

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> LedgerDB:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _values(f: PublishedForecast) -> tuple[float, float, float, float, float, str, int]:
    return (
        round(f.p_home, _DP),
        round(f.p_draw, _DP),
        round(f.p_away, _DP),
        round(f.home_xg, _DP),
        round(f.away_xg, _DP),
        f.model_version,
        int(f.team_news),
    )


def _row_last(r: sqlite3.Row) -> tuple[float, float, float, float, float, str, int]:
    return (
        r["last_p_home"],
        r["last_p_draw"],
        r["last_p_away"],
        r["last_home_xg"],
        r["last_away_xg"],
        r["last_model"],
        r["last_team_news"],
    )


@dataclass(frozen=True)
class RecordOutcome:
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    skipped_kicked_off: int = 0


class ForecastLedger:
    def __init__(self, db: LedgerDB) -> None:
        self._conn = db.connection

    def record(
        self, forecasts: Iterable[PublishedForecast], *, now: datetime | None = None
    ) -> RecordOutcome:
        """Store each fixture's forecast: insert the first, update the last, until kickoff.

        A forecast whose fixture has already kicked off -- by its own kickoff time or the one
        the ledger already holds -- is dropped. A changed kickoff (rescheduled before it was
        reached) is followed, so the freeze happens at the real kickoff.
        """
        now = now or datetime.now(UTC)
        inserted = updated = unchanged = skipped = 0
        stamp = now.isoformat()
        with self._conn:
            for f in forecasts:
                kickoff = f.kickoff_utc.astimezone(UTC)
                if kickoff <= now:
                    skipped += 1
                    continue
                key = f.fixture_key
                values = _values(f)
                existing = self._conn.execute(
                    "SELECT * FROM forecast WHERE fixture_key=?", (key,)
                ).fetchone()
                if existing is None:
                    self._conn.execute(
                        "INSERT INTO forecast VALUES "
                        "(?,?,?,?,?,?,?,?,?, ?,?,?,?,?,?,?,?, ?,?,?,?,?,?,?,?, 1)",
                        (
                            key,
                            f.match_id,
                            f.competition,
                            f.division,
                            kickoff.isoformat(),
                            f.home,
                            f.away,
                            f.home_norm,
                            f.away_norm,
                            stamp,
                            *values,
                            stamp,
                            *values,
                        ),
                    )
                    inserted += 1
                    continue
                if datetime.fromisoformat(existing["kickoff_utc"]) <= now:
                    skipped += 1  # frozen: the stored kickoff has passed
                    continue
                kickoff_moved = existing["kickoff_utc"] != kickoff.isoformat()
                if _row_last(existing) == values and not kickoff_moved:
                    unchanged += 1
                    continue
                changed = _row_last(existing) != values
                self._conn.execute(
                    "UPDATE forecast SET kickoff_utc=?, last_recorded_at=?, last_p_home=?, "
                    "last_p_draw=?, last_p_away=?, last_home_xg=?, last_away_xg=?, "
                    "last_model=?, last_team_news=?, revisions=revisions+? "
                    "WHERE fixture_key=? AND kickoff_utc > ?",
                    (
                        kickoff.isoformat(),
                        stamp if changed else existing["last_recorded_at"],
                        *values,
                        int(changed),
                        key,
                        stamp,
                    ),
                )
                updated += 1
        return RecordOutcome(inserted, updated, unchanged, skipped)

    def entries(self) -> list[LedgerEntry]:
        rows = self._conn.execute("SELECT * FROM forecast ORDER BY kickoff_utc, fixture_key")
        return [_entry(r) for r in rows.fetchall()]

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM forecast").fetchone()[0])


def _snapshot(r: sqlite3.Row, prefix: str) -> ForecastSnapshot:
    return ForecastSnapshot(
        recorded_at=datetime.fromisoformat(r[f"{prefix}_recorded_at"]),
        probs=(r[f"{prefix}_p_home"], r[f"{prefix}_p_draw"], r[f"{prefix}_p_away"]),
        home_xg=r[f"{prefix}_home_xg"],
        away_xg=r[f"{prefix}_away_xg"],
        model_version=r[f"{prefix}_model"],
        team_news=bool(r[f"{prefix}_team_news"]),
    )


def _entry(r: sqlite3.Row) -> LedgerEntry:
    return LedgerEntry(
        fixture_key=r["fixture_key"],
        match_id=r["match_id"],
        competition=r["competition"],
        division=r["division"],
        kickoff_utc=datetime.fromisoformat(r["kickoff_utc"]),
        home=r["home"],
        away=r["away"],
        home_norm=r["home_norm"],
        away_norm=r["away_norm"],
        first=_snapshot(r, "first"),
        last=_snapshot(r, "last"),
        revisions=r["revisions"],
    )


# --- Grading -------------------------------------------------------------------------------


@dataclass(frozen=True)
class FinalResult:
    """A played match a ledger row can be graded against."""

    match_date: date
    home: str
    away: str
    home_norm: str
    away_norm: str
    home_goals: int
    away_goals: int
    closing_odds: tuple[float, float, float] | None = None
    source: str = "results"  # "results" (football-data.co.uk / DuckDB) or "live"


@dataclass(frozen=True)
class LiveOutcome:
    """The live store's word on a match id: concluded with a score, or called off."""

    status: str  # MatchStatus value
    home_goals: int | None
    away_goals: int | None


@dataclass(frozen=True)
class GradedForecast:
    entry: LedgerEntry
    result: FinalResult
    actual: int  # 0 home / 1 draw / 2 away
    base: Probs
    market: Probs | None

    @property
    def model(self) -> Probs:
        return self.entry.last.probs


@dataclass(frozen=True)
class RecordSummary:
    n: int
    model: Score
    first: Score
    """The same matches scored on the FIRST forecast published, not the last."""
    baseline: Score
    n_with_odds: int = 0
    model_on_odds: Score | None = None
    market: Score | None = None
    gap_se: float | None = None
    """Standard error of the per-match log-loss gap (model minus market), when n_with_odds > 1."""

    @property
    def gap(self) -> float | None:
        if self.model_on_odds is None or self.market is None:
            return None
        return self.model_on_odds.log_loss - self.market.log_loss

    @property
    def skill_vs_base(self) -> float:
        """Log-loss skill over base rates: 1 - model/base (positive = better)."""
        if self.baseline.log_loss <= 0:
            return 0.0
        return 1.0 - self.model.log_loss / self.baseline.log_loss


@dataclass(frozen=True)
class LiveRecord:
    recorded: int
    upcoming: int
    awaiting: int
    unmatched: int
    void: int
    graded: list[GradedForecast] = field(default_factory=list)
    """Most recent kickoff first."""
    since: datetime | None = None
    """When the first forecast in the ledger was recorded."""

    @property
    def n(self) -> int:
        return len(self.graded)

    @property
    def is_small(self) -> bool:
        return self.n < SMALL_SAMPLE

    @property
    def summary(self) -> RecordSummary | None:
        return summarize(self.graded)

    def by_version(self) -> list[tuple[str, RecordSummary]]:
        return _grouped(self.graded, lambda g: g.entry.last.model_version)

    def by_competition(self) -> list[tuple[str, RecordSummary]]:
        return _grouped(self.graded, lambda g: g.entry.competition)


def _grouped(
    graded: list[GradedForecast], key: Callable[[GradedForecast], str]
) -> list[tuple[str, RecordSummary]]:
    groups: dict[str, list[GradedForecast]] = {}
    for g in graded:
        groups.setdefault(key(g), []).append(g)
    out = [(k, s) for k, gs in groups.items() if (s := summarize(gs)) is not None]
    return sorted(out, key=lambda kv: -kv[1].n)


def _score(preds: list[tuple[Probs, int]]) -> Score:
    n = len(preds)
    return Score(
        log_loss=sum(_log_loss(p, a) for p, a in preds) / n,
        rps=sum(rps(p, a) for p, a in preds) / n,
        brier=sum(_brier(p, a) for p, a in preds) / n,
    )


def summarize(graded: list[GradedForecast]) -> RecordSummary | None:
    if not graded:
        return None
    with_odds = [g for g in graded if g.market is not None]
    model_on_odds = market = None
    gap_se = None
    if with_odds:
        model_on_odds = _score([(g.model, g.actual) for g in with_odds])
        market = _score([(g.market, g.actual) for g in with_odds if g.market is not None])
        gaps = [
            _log_loss(g.model, g.actual) - _log_loss(g.market, g.actual)
            for g in with_odds
            if g.market is not None
        ]
        if len(gaps) > 1:
            mean = sum(gaps) / len(gaps)
            var = sum((x - mean) ** 2 for x in gaps) / (len(gaps) - 1)
            gap_se = math.sqrt(var / len(gaps))
    return RecordSummary(
        n=len(graded),
        model=_score([(g.model, g.actual) for g in graded]),
        first=_score([(g.entry.first.probs, g.actual) for g in graded]),
        baseline=_score([(g.base, g.actual) for g in graded]),
        n_with_odds=len(with_odds),
        model_on_odds=model_on_odds,
        market=market,
        gap_se=gap_se,
    )


def _outcome(home_goals: int, away_goals: int) -> int:
    return 0 if home_goals > away_goals else 2 if away_goals > home_goals else 1


def _implied(odds: tuple[float, float, float] | None) -> Probs | None:
    if odds is None or any(o is None or o <= 1.0 for o in odds):
        return None
    raw = [1.0 / o for o in odds]
    total = sum(raw)
    return (raw[0] / total, raw[1] / total, raw[2] / total)


Matcher = Callable[[str, Collection[str]], str | None]


def _exact(name: str, candidates: Collection[str]) -> str | None:
    n = normalize_name(name)
    return n if n in candidates else None


def find_result(
    entry: LedgerEntry, candidates: Iterable[FinalResult], matcher: Matcher = _exact
) -> FinalResult | None:
    """The result for a ledger row: same home and away side (in that orientation), dated
    within MATCH_WINDOW_DAYS of kickoff, nearest date first. A reversed fixture (the venue
    swapped) is not a match -- the forecast was for the other ground. Ambiguity (two results
    equally near) is no match, never a guess.
    """
    kickoff_day = entry.kickoff_utc.astimezone(UTC).date()
    near = [r for r in candidates if abs((r.match_date - kickoff_day).days) <= MATCH_WINDOW_DAYS]
    if not near:
        return None
    homes = {r.home_norm for r in near}
    aways = {r.away_norm for r in near}

    def side(norm: str, display: str, pool: Collection[str]) -> str | None:
        if norm in pool:
            return norm
        return matcher(display, pool) or matcher(norm, pool)

    hn = side(entry.home_norm, entry.home, homes)
    an = side(entry.away_norm, entry.away, aways)
    if hn is None or an is None:
        return None
    hits = [r for r in near if r.home_norm == hn and r.away_norm == an]
    if not hits:
        return None

    def distance(r: FinalResult) -> int:
        return abs((r.match_date - kickoff_day).days)

    hits.sort(key=distance)
    if len(hits) > 1 and distance(hits[0]) == distance(hits[1]):
        return None  # two equally near candidates: refuse to guess
    return hits[0]


_VOID_STATUSES = frozenset({"postponed", "cancelled", "awarded", "suspended"})


def grade(
    entries: Iterable[LedgerEntry],
    results: Mapping[str, list[FinalResult]],
    *,
    live: Mapping[str, LiveOutcome] | None = None,
    base_rates: Mapping[str, Probs] | None = None,
    matcher: Matcher = _exact,
    now: datetime | None = None,
) -> LiveRecord:
    """Match each kicked-off ledger row to its result and score it.

    `results` is the candidate results per division; `live` the live store's state per match
    id (the fallback for a result football-data.co.uk hasn't posted yet -- graded without
    odds); `base_rates` each division's historical home/draw/away shares (uniform where
    missing). A postponed, cancelled or administratively awarded match is void, not graded.
    """
    now = now or datetime.now(UTC)
    live = live or {}
    base_rates = base_rates or {}
    all_entries = list(entries)
    upcoming = awaiting = unmatched = void = 0
    graded: list[GradedForecast] = []
    for e in all_entries:
        if e.kickoff_utc > now:
            upcoming += 1
            continue
        state = live.get(e.match_id) if e.match_id else None
        result = find_result(e, results.get(e.division or "", []), matcher)
        if result is None and state is not None:
            if state.status in _VOID_STATUSES:
                void += 1
                continue
            if (
                state.status == "finished"
                and state.home_goals is not None
                and state.away_goals is not None
            ):
                result = FinalResult(
                    match_date=e.kickoff_utc.date(),
                    home=e.home,
                    away=e.away,
                    home_norm=e.home_norm,
                    away_norm=e.away_norm,
                    home_goals=state.home_goals,
                    away_goals=state.away_goals,
                    source="live",
                )
        if result is None:
            if now - e.kickoff_utc > UNMATCHED_AFTER:
                unmatched += 1
            else:
                awaiting += 1
            continue
        graded.append(
            GradedForecast(
                entry=e,
                result=result,
                actual=_outcome(result.home_goals, result.away_goals),
                base=base_rates.get(e.division or "", (1 / 3, 1 / 3, 1 / 3)),
                market=_implied(result.closing_odds),
            )
        )
    graded.sort(key=lambda g: g.entry.kickoff_utc, reverse=True)
    since = min((e.first.recorded_at for e in all_entries), default=None)
    return LiveRecord(
        recorded=len(all_entries),
        upcoming=upcoming,
        awaiting=awaiting,
        unmatched=unmatched,
        void=void,
        graded=graded,
        since=since,
    )


def safe_record(path: Path, forecasts: Iterable[PublishedForecast]) -> RecordOutcome | None:
    """Record, never letting a ledger failure break the page or job that called it."""
    try:
        with LedgerDB(path) as db:
            return ForecastLedger(db).record(forecasts)
    except (sqlite3.Error, OSError) as exc:
        logger.warning("forecast ledger write failed: %s", exc)
        return None


def verdict(record: LiveRecord, *, sample_note: bool = True) -> str:
    """One honest sentence on the live record -- shared by the Scorecard, CLI and assistant.
    `sample_note=False` leaves out the small-sample caveat, for a page that shows its own."""
    s = record.summary
    if s is None:
        if record.recorded == 0:
            return (
                "No forecasts recorded yet: the ledger fills as upcoming fixtures are forecast "
                "(the Predictions page, or `soccer serve`), and grades them as results land."
            )
        return (
            f"{record.recorded} forecasts recorded, none graded yet -- results are still to "
            "come in."
        )
    parts = [
        f"Over {s.n} graded forecasts the model's log loss is {s.model.log_loss:.3f} vs "
        f"{s.baseline.log_loss:.3f} for base rates ({s.skill_vs_base:+.1%} skill)"
    ]
    gap, se = s.gap, s.gap_se
    if gap is not None and s.market is not None:
        where = (
            f"on the {s.n_with_odds} with closing odds the market scored {s.market.log_loss:.3f}"
        )
        if se is not None and abs(gap) < 2 * se:
            parts.append(f"{where} -- level with the closing line within noise")
        elif gap > 0:
            parts.append(f"{where} -- the model trails the closing line by {gap:.3f}")
        else:
            parts.append(f"{where} -- the model is ahead of the closing line by {-gap:.3f}")
    sentence = "; ".join(parts) + "."
    if record.is_small and sample_note:
        sentence += (
            f" Only {s.n} matches so far: far too few to tell skill from luck (a few hundred "
            "are needed), so read the backtest, not this, for now."
        )
    return sentence
