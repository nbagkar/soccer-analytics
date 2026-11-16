"""Pre-match odds for upcoming fixtures: stored, and matched to the fixtures they price.

football-data.co.uk's fixtures.csv carries the coming round's fixtures with pre-match 1X2 and
over/under 2.5 odds. Measured against the closing line (`scripts/prematch_odds_backtest.py`,
2021/22-2026/27, 34,891 matches, 18 divisions) that line is far closer than the forecast
model -- +0.00408 vs +0.01088 log loss -- and blending the model in adds nothing measurable.
So where a fixture has a pre-match price, that price is the strongest forecast available,
and the dashboard leads with it; the model's own number stays beside it.

The store lives in the live database (`upcoming_odds`): new odds change what the Upcoming
tab shows, so they should move that store's mtime and invalidate the forecast caches.

Matching is deliberately strict -- same division, both sides resolved to the odds file's own
team names, dated within a day of kickoff, and unique. Anything less is no match: a fixture
then simply shows the model's forecast, which is the honest fallback.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from soccer.sources.football_data_co_uk import UpcomingOdds
from soccer.storage.live_db import LiveDB

# A stored date may be the day before or after the UTC kickoff (the file's dates are local).
MATCH_WINDOW_DAYS = 1
# Rows are dropped this long after their match date: the file only covers the coming round,
# and a played fixture's price has no further use here (the result file carries the close).
KEEP_DAYS = 3

Matcher = Callable[[str, Collection[str]], str | None]


@dataclass(frozen=True)
class PrematchLine:
    """The vig-free pre-match 1X2 for one fixture, and where it came from."""

    p_home: float
    p_draw: float
    p_away: float
    p_over25: float | None
    book: str  # "avg" | "b365"
    fetched_at: datetime

    @property
    def probs(self) -> tuple[float, float, float]:
        return (self.p_home, self.p_draw, self.p_away)


def vig_free(home_odds: float, draw_odds: float, away_odds: float) -> tuple[float, float, float]:
    """1X2 probabilities with the bookmaker's margin divided out proportionally -- the same
    de-vig the closing-line benchmark uses (`models.value.implied_probabilities`)."""
    raw = (1.0 / home_odds, 1.0 / draw_odds, 1.0 / away_odds)
    total = sum(raw)
    return (raw[0] / total, raw[1] / total, raw[2] / total)


def line_from(odds: UpcomingOdds, fetched_at: datetime) -> PrematchLine:
    p_home, p_draw, p_away = vig_free(odds.home_odds, odds.draw_odds, odds.away_odds)
    p_over = None
    if odds.over25_odds and odds.under25_odds:
        over, under = 1.0 / odds.over25_odds, 1.0 / odds.under25_odds
        p_over = over / (over + under)
    return PrematchLine(p_home, p_draw, p_away, p_over, odds.book, fetched_at)


class UpcomingOddsStore:
    def __init__(self, db: LiveDB) -> None:
        self._conn = db.connection

    def replace(self, rows: Iterable[UpcomingOdds], *, now: datetime | None = None) -> int:
        """Upsert the fetched rows and drop those whose match is more than KEEP_DAYS past.

        An upsert, not a wipe: the file rolls on to the next round before every match of the
        last one has kicked off (a Monday game), and that fixture's price must survive.
        Returns the number of rows written.
        """
        now = now or datetime.now(UTC)
        stamp = now.isoformat()
        written = 0
        with self._conn:
            for r in rows:
                self._conn.execute(
                    "INSERT INTO upcoming_odds (division, match_date, home_norm, away_norm, "
                    "home, away, book, home_odds, draw_odds, away_odds, over25_odds, "
                    "under25_odds, fetched_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?) "
                    "ON CONFLICT(division, match_date, home_norm, away_norm) DO UPDATE SET "
                    "home=excluded.home, away=excluded.away, book=excluded.book, "
                    "home_odds=excluded.home_odds, draw_odds=excluded.draw_odds, "
                    "away_odds=excluded.away_odds, over25_odds=excluded.over25_odds, "
                    "under25_odds=excluded.under25_odds, fetched_at=excluded.fetched_at",
                    (
                        r.division,
                        r.match_date.isoformat(),
                        r.home_norm,
                        r.away_norm,
                        r.home,
                        r.away,
                        r.book,
                        r.home_odds,
                        r.draw_odds,
                        r.away_odds,
                        r.over25_odds,
                        r.under25_odds,
                        stamp,
                    ),
                )
                written += 1
            cutoff = (now.date() - timedelta(days=KEEP_DAYS)).isoformat()
            self._conn.execute("DELETE FROM upcoming_odds WHERE match_date < ?", (cutoff,))
        return written

    def all(self) -> list[tuple[UpcomingOdds, datetime]]:
        rows = self._conn.execute(
            "SELECT * FROM upcoming_odds ORDER BY division, match_date, home_norm"
        ).fetchall()
        return [
            (
                UpcomingOdds(
                    division=r["division"],
                    match_date=date.fromisoformat(r["match_date"]),
                    home=r["home"],
                    away=r["away"],
                    home_norm=r["home_norm"],
                    away_norm=r["away_norm"],
                    book=r["book"],
                    home_odds=r["home_odds"],
                    draw_odds=r["draw_odds"],
                    away_odds=r["away_odds"],
                    over25_odds=r["over25_odds"],
                    under25_odds=r["under25_odds"],
                ),
                datetime.fromisoformat(r["fetched_at"]),
            )
            for r in rows
        ]

    def count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM upcoming_odds").fetchone()[0])


def _exact(name: str, candidates: Collection[str]) -> str | None:
    from soccer.domain.names import normalize_name

    n = normalize_name(name)
    return n if n in candidates else None


def find_line(
    stored: Iterable[tuple[UpcomingOdds, datetime]],
    *,
    division: str,
    kickoff_utc: datetime,
    home: str,
    away: str,
    matcher: Matcher = _exact,
) -> PrematchLine | None:
    """The pre-match line for a fixture, or None.

    Same division, dated within MATCH_WINDOW_DAYS of kickoff, home and away each resolved
    (by `matcher`, the app's one cross-source name matcher) against the home / away names
    the odds file lists that day -- in that orientation, so a reversed fixture never matches.
    Two candidates equally near is no match, never a guess.
    """
    day = kickoff_utc.astimezone(UTC).date()
    near = [
        (o, at)
        for o, at in stored
        if o.division == division and abs((o.match_date - day).days) <= MATCH_WINDOW_DAYS
    ]
    if not near:
        return None
    hn = matcher(home, {o.home_norm for o, _ in near})
    an = matcher(away, {o.away_norm for o, _ in near})
    if hn is None or an is None:
        return None
    hits = [(o, at) for o, at in near if o.home_norm == hn and o.away_norm == an]
    if not hits:
        return None
    hits.sort(key=lambda t: abs((t[0].match_date - day).days))
    if len(hits) > 1 and abs((hits[0][0].match_date - day).days) == abs(
        (hits[1][0].match_date - day).days
    ):
        return None
    return line_from(*hits[0])
