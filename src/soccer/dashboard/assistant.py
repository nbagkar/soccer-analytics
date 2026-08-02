"""A built-in, offline football assistant -- rule-based, no LLM, no network.

Maps a plain-language question to one of a fixed set of intents (standings, top
scorers, a player's stats, a match forecast, current form, records, title odds, fixtures)
and answers it from the local store. Deterministic and private: nothing leaves the
machine, and it says plainly what it can and cannot answer rather than guessing.

Kept free of Streamlit so the intent logic is unit-testable; the chat page is a thin
render over `answer()`.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from soccer.sources.football_data_co_uk import division_name, season_label
from soccer.storage.analytics_db import AnalyticsDB


@dataclass
class Reply:
    text: str
    """Markdown answer."""
    table: list[dict] | None = None
    suggestions: list[str] = field(default_factory=list)


# Plain-language league names/nicknames -> football-data.co.uk division codes (results
# data). Only divisions actually loaded are offered; the rest degrade to a friendly note.
_LEAGUE_ALIASES = {
    "premier league": "E0",
    "prem": "E0",
    "epl": "E0",
    "english": "E0",
    "championship": "E1",
    "la liga": "SP1",
    "laliga": "SP1",
    "spain": "SP1",
    "spanish": "SP1",
    "eredivisie": "N1",
    "dutch": "N1",
    "netherlands": "N1",
    "holland": "N1",
    "primeira": "P1",
    "portugal": "P1",
    "portuguese": "P1",
    "bundesliga": "D1",
    "germany": "D1",
    "german": "D1",
    "serie a": "I1",
    "italy": "I1",
    "italian": "I1",
    "ligue 1": "F1",
    "france": "F1",
    "french": "F1",
    "brazil": "BRA",
    "brazilian": "BRA",
    "brasileiro": "BRA",
}

# Common team nicknames -> the football-data.co.uk spelling, for name resolution.
_TEAM_ALIASES = {
    "spurs": "tottenham",
    "man city": "man city",
    "city": "man city",
    "man utd": "man united",
    "man u": "man united",
    "united": "man united",
    "wolves": "wolves",
    "gunners": "arsenal",
    "reds": "liverpool",
    "toffees": "everton",
    "hammers": "west ham",
}

_STOP = {"fc", "afc", "cf", "real", "the", "de", "city", "united", "town", "club"}


def _norm(text: str) -> str:
    """Light normalization for both questions and names: lowercase, fold accents, drop
    punctuation -- but KEEP every word.

    The team-name normalizer is too aggressive here: it strips stopwords ("of", "the") and
    splits "who's" into "who s", which breaks keyword matching on a full sentence. This
    folds "Mbappé"->"mbappe" and "who's"->"whos" while leaving the sentence structure
    intact, so intent keywords survive and entity names still match.
    """
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))  # fold accents
    text = text.lower().replace("'", "").replace("\u2019", "")  # straight and smart apostrophes
    text = re.sub(r"[^a-z0-9 ]+", " ", text)  # other punctuation -> space
    return re.sub(r"\s+", " ", text).strip()


def answer(question: str, analytics_db: Path, live_db: Path | None = None) -> Reply:
    """Route a question to an intent handler and answer it from the local store."""
    q = _norm(question)
    if not q:
        return _help()
    if not Path(analytics_db).exists():
        return Reply(
            "I don't have any data loaded yet. Head to the **Home** page to download "
            "some — no terminal needed."
        )

    for handler in (
        _intent_help,
        _intent_forecast,
        _intent_top_scorers,
        _intent_player,
        _intent_title_odds,
        _intent_records,
        _intent_form,
        _intent_standings,
        _intent_fixtures,
    ):
        reply = handler(q, analytics_db, live_db)
        if reply is not None:
            return reply
    return _fallback(q, analytics_db)


# --- entity resolution -------------------------------------------------------


def _loaded_divisions(adb: AnalyticsDB) -> dict[str, str]:
    """{division: latest_season} for divisions with results, newest season each."""
    out: dict[str, str] = {}
    for season, division, _n in adb.seasons_loaded():
        if division not in out or season > out[division]:
            out[division] = season
    return out


def _resolve_league(q: str, loaded: dict[str, str]) -> str | None:
    """A division mentioned in the question (by name or nickname), if it's loaded."""
    for alias, division in _LEAGUE_ALIASES.items():
        if alias in q and division in loaded:
            return division
    return None


def _default_league(loaded: dict[str, str]) -> str | None:
    for preferred in ("E0", "SP1", "N1", "P1", "E1", "BRA"):
        if preferred in loaded:
            return preferred
    return next(iter(loaded), None)


def _team_index(adb: AnalyticsDB, loaded: dict[str, str]) -> dict[str, tuple[str, str, str]]:
    """{normalized team name: (display, division, season)} across each league's latest season."""
    index: dict[str, tuple[str, str, str]] = {}
    for division, season in loaded.items():
        for outcome in adb.outcomes_for(season, division):
            for display in (outcome.home, outcome.away):
                index.setdefault(_norm(display), (display, division, season))
    return index


def _resolve_teams(q: str, index: dict[str, tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    """Teams named in the question, in the order they appear (so 'A vs B' keeps A, B)."""
    aliased = q
    for alias, canonical in _TEAM_ALIASES.items():
        if alias in q:
            aliased = f"{aliased} {canonical}"
    hits: dict[str, tuple[int, tuple[str, str, str]]] = {}
    for norm_name in sorted(index, key=len, reverse=True):  # longest first, so 'man city' wins
        if not norm_name:
            continue
        pos = aliased.find(norm_name)
        display = index[norm_name][0]
        if pos != -1 and display not in hits:
            hits[display] = (pos, index[norm_name])
    return [entry for _pos, entry in sorted(hits.values(), key=lambda pe: pe[0])]


def _resolve_player(q: str, adb: AnalyticsDB) -> str | None:
    """The most prominent player (by minutes) whose name appears in the question.

    StatsBomb stores full legal names ("Lionel Andrés Messi Cuccittini"); users type
    "Messi". So match on the full name OR any distinctive whole-word token (a surname),
    and disambiguate common tokens by picking the player with the most minutes.
    """
    if adb.player_stats_count() == 0:
        return None
    q_words = set(q.split())
    best, best_mins = None, -1
    for player, mins in adb.player_minutes():
        name = _norm(player)
        tokens = [t for t in name.split() if len(t) >= 5 and t not in _STOP]
        if (name in q or any(t in q_words for t in tokens)) and mins > best_mins:
            best, best_mins = player, mins
    return best


# --- intents -----------------------------------------------------------------

_EXAMPLES = [
    "Who's top of the Premier League?",
    "Arsenal vs Chelsea — who wins?",
    "Top scorers in La Liga",
    "How is Man City's form?",
    "Longest unbeaten run in the Premier League",
    "Who will win the league?",
    "Show upcoming fixtures",
]


def _help() -> Reply:
    return Reply(
        "I'm your football assistant — I answer questions from the data you've loaded, "
        "no typing commands. Try one of these:\n\n" + "\n".join(f"- {e}" for e in _EXAMPLES),
        suggestions=_EXAMPLES[:4],
    )


def _intent_help(q: str, analytics_db: Path, live_db: Path | None) -> Reply | None:
    if re.search(r"\b(help|what can you|what do you|who are you|hello|hey|examples?)\b", q):
        return _help()
    return None


def _intent_forecast(q: str, analytics_db: Path, live_db: Path | None) -> Reply | None:
    triggers = re.search(r"\b(vs|versus|beat|predict|forecast|wins?|winner|odds|score)\b", q)
    if not triggers and " v " not in f" {q} ":
        return None
    with AnalyticsDB(analytics_db) as adb:
        loaded = _loaded_divisions(adb)
        if not loaded:
            return None
        index = _team_index(adb, loaded)
    teams = _resolve_teams(q, index)
    by_div: dict[str, list] = {}
    for display, division, season in teams:
        by_div.setdefault(division, []).append((display, season))
    pair = next(((d, t) for d, t in by_div.items() if len(t) >= 2), None)
    if pair is None:
        return None

    from soccer.dashboard.data import forecast_slate

    division, (home_e, away_e) = pair[0], pair[1][:2]
    home, away = home_e[0], away_e[0]
    slate = forecast_slate(analytics_db, home_e[1], division, home, away)
    if slate is None:
        return None
    res = {m.name: m.probability for m in slate.result}
    x, y, _ = slate.most_likely_score
    over = next(o for o in slate.over_under if o.line == 2.5).over
    btts = next(m.probability for m in slate.btts if m.name == "Yes")
    lead = max(res, key=res.get)
    text = (
        f"**{home} vs {away}** ({division_name(division)})\n\n"
        f"- Most likely score: **{x}-{y}**\n"
        f"- {home} win **{res[home]:.0%}** · draw **{res['Draw']:.0%}** · "
        f"{away} win **{res[away]:.0%}**\n"
        f"- Over 2.5 goals **{over:.0%}** · both teams score **{btts:.0%}**\n\n"
        f"Model leans **{lead if lead != 'Draw' else 'a draw'}**. Directional, not advice."
    )
    return Reply(
        text, suggestions=[f"Top scorers in {division_name(division)}", "Who will win the league?"]
    )


def _intent_top_scorers(q: str, analytics_db: Path, live_db: Path | None) -> Reply | None:
    if not re.search(
        r"\b(top scorer|scorers?|most goals|golden boot|leading scorer|best striker)\b", q
    ):
        return None
    with AnalyticsDB(analytics_db) as adb:
        if adb.player_stats_count() == 0:
            return Reply(
                "I don't have per-player data loaded yet. Go to "
                "**Home → Add player data** to add some."
            )
        competition = _resolve_statsbomb_competition(q, adb)
        profiles = adb.player_profiles(
            limit=8, min_minutes=270, order="goals", competition=competition
        )
    if not profiles:
        return None
    scope = f" in {competition}" if competition else ""
    rows = [
        {
            "Player": p.player,
            "Team": p.team,
            "Goals": p.goals,
            "xG": round(p.xg, 1),
            "Assists": p.assists,
        }
        for p in profiles
    ]
    lead = profiles[0]
    return Reply(
        f"**Top scorers{scope}** (across all loaded seasons) — {lead.player} leads with "
        f"**{lead.goals} goals** (xG {lead.xg:.1f}).",
        table=rows,
        suggestions=[f"Tell me about {lead.player}", "Who is the best playmaker?"],
    )


def _resolve_statsbomb_competition(q: str, adb: AnalyticsDB) -> str | None:
    comps = {c.lower(): c for c, _n in adb.competitions_loaded()}
    for low, display in comps.items():
        if low in q:
            return display
    # league nickname -> StatsBomb competition name
    for alias, comp in {
        "premier league": "Premier League",
        "prem": "Premier League",
        "la liga": "La Liga",
        "serie a": "Serie A",
        "bundesliga": "1. Bundesliga",
        "ligue 1": "Ligue 1",
        "world cup": "FIFA World Cup",
    }.items():
        if alias in q and comp.lower() in comps:
            return comps[comp.lower()]
    return None


def _intent_player(q: str, analytics_db: Path, live_db: Path | None) -> Reply | None:
    with AnalyticsDB(analytics_db) as adb:
        player = _resolve_player(q, adb)
        if player is None:
            return None
        profile = adb.player_profile(player)
    if profile is None:
        return None
    text = (
        f"**{profile.player}** — {profile.team}"
        f"{' · ' + profile.position if profile.position else ''}\n\n"
        f"- {profile.matches} matches, {profile.minutes} minutes\n"
        f"- **{profile.goals} goals** (xG {profile.xg:.1f}) · **{profile.assists} assists** "
        f"(xA {profile.xa:.1f})\n"
        f"- {profile.key_passes} key passes · {profile.pass_pct:.0f}% passing · "
        f"{profile.progressive_passes + profile.progressive_carries} progressive actions\n"
        f"- Defending: {profile.tackles} tackles, {profile.interceptions} interceptions"
    )
    return Reply(text, suggestions=["Top scorers", "Who is the best playmaker?"])


def _intent_title_odds(q: str, analytics_db: Path, live_db: Path | None) -> Reply | None:
    if not re.search(
        r"\b(win the league|who will win|who wins|title|champions?|relegat|top four|top 4"
        r"|finish|odds to win)\b",
        q,
    ):
        return None
    with AnalyticsDB(analytics_db) as adb:
        loaded = _loaded_divisions(adb)
        division = _resolve_league(q, loaded) or _default_league(loaded)
        if division is None:
            return None
        season = loaded[division]

    from soccer.dashboard.data import season_briefing

    briefing = season_briefing(analytics_db, season, division, n_sims=4000)
    if briefing is None:
        return None
    projs = sorted(briefing.projections, key=lambda p: -p.title_pct)[:6]
    names = briefing.names
    rows = [
        {
            "Team": names.get(p.team, p.team),
            "Title %": round(100 * p.title_pct, 1),
            "Top 4 %": round(100 * p.top_pct),
            "Relegation %": round(100 * p.relegation_pct),
        }
        for p in projs
    ]
    fav = projs[0]
    return Reply(
        f"**{division_name(division)} {season_label(season)} projection** — "
        f"{names.get(fav.team, fav.team)} are favourites at **{fav.title_pct:.0%}** to win it. "
        "A pre-season model off last season's strengths, blind to transfers.",
        table=rows,
        suggestions=["Who is in form?", "Show upcoming fixtures"],
    )


def _intent_records(q: str, analytics_db: Path, live_db: Path | None) -> Reply | None:
    if not re.search(r"\b(unbeaten|streak|biggest win|highest scoring|record|thrash)\b", q):
        return None
    with AnalyticsDB(analytics_db) as adb:
        loaded = _loaded_divisions(adb)
        division = _resolve_league(q, loaded) or _default_league(loaded)
        if division is None:
            return None
        season = loaded[division]
        streaks = adb.team_streaks(season, division)
        wins = adb.biggest_wins(season, division, limit=3)
        scoring = adb.highest_scoring(season, division, limit=3)
    if not streaks:
        return None
    unbeaten = streaks[0]
    text = (
        f"**{division_name(division)} {season_label(season)} records**\n\n"
        f"- Longest active unbeaten run: **{unbeaten.team}, {unbeaten.unbeaten} games**\n"
        f"- Biggest win: {wins[0].home} {wins[0].score} {wins[0].away}\n"
        f"- Highest scoring: {scoring[0].home} {scoring[0].score} {scoring[0].away} "
        f"({scoring[0].total} goals)"
    )
    return Reply(text, suggestions=["Who is in form?", "Who will win the league?"])


def _intent_form(q: str, analytics_db: Path, live_db: Path | None) -> Reply | None:
    if not re.search(r"\b(form|in form|hot|cold|struggling|slump|rising|momentum)\b", q):
        return None
    with AnalyticsDB(analytics_db) as adb:
        loaded = _loaded_divisions(adb)
        division = _resolve_league(q, loaded) or _default_league(loaded)
        if division is None:
            return None
        season = loaded[division]
        forms = adb.team_form(season, division, last_n=5)
    if not forms:
        return None
    index = None
    # If a specific team is named, answer about it.
    with AnalyticsDB(analytics_db) as adb:
        index = _team_index(adb, loaded)
    named = _resolve_teams(q, index)
    if named:
        target = next((f for f in forms if _norm(f.team) == _norm(named[0][0])), None)
        if target:
            arrow = (
                "rising" if target.trend > 0.15 else "sliding" if target.trend < -0.15 else "steady"
            )
            return Reply(
                f"**{target.team}** — last 5: **{target.recent_form}** "
                f"({target.recent_ppg:.2f} ppg vs {target.ppg:.2f} season, {arrow}). "
                f"Over 2.5 goals in {target.over25_rate:.0%} of games."
            )
    hot, cold = forms[0], forms[-1]
    rows = [
        {"Team": f.team, "Last 5": f.recent_form, "Recent ppg": round(f.recent_ppg, 2)}
        for f in forms[:6]
    ]
    return Reply(
        f"**{division_name(division)} form** — hottest is **{hot.team}** ({hot.recent_form}), "
        f"coldest **{cold.team}** ({cold.recent_form}).",
        table=rows,
        suggestions=["Longest unbeaten run", "Who will win the league?"],
    )


def _intent_standings(q: str, analytics_db: Path, live_db: Path | None) -> Reply | None:
    if not re.search(
        r"\b(table|standings?|top of|leading|who is top|whos top|position|rank|where are)\b", q
    ):
        return None
    with AnalyticsDB(analytics_db) as adb:
        loaded = _loaded_divisions(adb)
        division = _resolve_league(q, loaded) or _default_league(loaded)
        if division is None:
            return None
        season = loaded[division]
        table = adb.league_table(season, division)
        index = _team_index(adb, loaded)
    if not table:
        return None
    named = _resolve_teams(q, index)
    if named and re.search(r"\b(where|position|rank)\b", q):
        display = named[0][0]
        row = next((r for r in table if _norm(r.team) == _norm(display)), None)
        if row:
            return Reply(
                f"**{row.team}** are **{_ordinal(row.position)}** in "
                f"{division_name(division)} {season_label(season)} — {row.points} points, "
                f"{row.won}W {row.drawn}D {row.lost}L, GD {row.goal_difference:+d}."
            )
    rows = [
        {
            "#": r.position,
            "Team": r.team,
            "P": r.played,
            "W": r.won,
            "D": r.drawn,
            "L": r.lost,
            "GD": r.goal_difference,
            "Pts": r.points,
        }
        for r in table[:8]
    ]
    leader = table[0]
    return Reply(
        f"**{division_name(division)} {season_label(season)}** — "
        f"**{leader.team}** lead on {leader.points} points.",
        table=rows,
        suggestions=[f"Who is in form in {division_name(division)}?", "Who will win the league?"],
    )


def _intent_fixtures(q: str, analytics_db: Path, live_db: Path | None) -> Reply | None:
    if not re.search(
        r"\b(fixtures?|upcoming|next (game|match)|who ?s playing|this weekend|schedule)\b",
        q,
    ):
        return None
    if live_db is None or not Path(live_db).exists():
        return Reply("No fixtures loaded yet. Go to **Home → Update fixtures**.")

    from soccer.dashboard.data import fixture_forecasts

    fixtures = [
        f for f in fixture_forecasts(live_db, analytics_db, limit=200) if f.slate is not None
    ]
    if not fixtures:
        return Reply("No upcoming fixtures with a forecast yet — try **Home → Update fixtures**.")
    rows = []
    for f in fixtures[:8]:
        x, y, _ = f.slate.most_likely_score
        res = [m.probability for m in f.slate.result]
        rows.append(
            {
                "Date": f.kickoff_utc.strftime("%m-%d"),
                "Match": f"{f.home} v {f.away}",
                "Pred": f"{x}-{y}",
                "Home %": round(100 * res[0]),
                "Away %": round(100 * res[2]),
            }
        )
    return Reply(
        f"**{len(fixtures)} upcoming matches** with forecasts. The next few:",
        table=rows,
        suggestions=["Who will win the league?", "Who is in form?"],
    )


def _ordinal(n: int) -> str:
    suffix = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _fallback(q: str, analytics_db: Path) -> Reply:
    return Reply(
        "I'm not sure how to answer that yet. I can help with league tables, top scorers, "
        "a player's stats, match forecasts, current form, records, title odds and fixtures.",
        suggestions=_EXAMPLES[:4],
    )
