"""A built-in, offline football assistant -- rule-based, no LLM, no network.

Maps a plain-language question to one of a fixed set of intents (match forecasts with
predicted scorelines, team dossiers, standings for any loaded league or past season, top
scorers, a player's stats and player comparisons, head-to-head records, current form,
over/under-performance vs xG, in-season and all-time records, title odds, model accuracy,
fixtures) and answers it from the local store. Deterministic and private: nothing leaves the
machine, and it says plainly what it can and cannot answer rather than guessing.

Kept free of Streamlit so the intent logic is unit-testable; the chat page is a thin
render over `answer()`.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from soccer.sources.football_data_co_uk import division_name, season_label, season_sort_key
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
    "league one": "E2",
    "league 1": "E2",
    "league two": "E3",
    "league 2": "E3",
    "la liga": "SP1",
    "laliga": "SP1",
    "spain": "SP1",
    "spanish": "SP1",
    "la liga 2": "SP2",
    "segunda": "SP2",
    "bundesliga": "D1",
    "germany": "D1",
    "german": "D1",
    "2 bundesliga": "D2",
    "bundesliga 2": "D2",
    "serie a": "I1",
    "italy": "I1",
    "italian": "I1",
    "serie b": "I2",
    "ligue 1": "F1",
    "france": "F1",
    "french": "F1",
    "ligue 2": "F2",
    "mls": "USA",
    "usa": "USA",
    "america": "USA",
    "major league soccer": "USA",
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
        _intent_compare,
        _intent_h2h,
        _intent_match_centre,
        _intent_forecast,
        _intent_model,
        _intent_honours,
        _intent_scout,
        _intent_top_scorers,
        _intent_player,
        _intent_title_odds,
        _intent_underlying,
        _intent_records,
        _intent_form,
        _intent_standings,
        _intent_fixtures,
        _intent_team,
    ):
        reply = handler(q, analytics_db, live_db)
        if reply is not None:
            return reply
    return _fallback(q, analytics_db)


# --- entity resolution -------------------------------------------------------


def _loaded_divisions(adb: AnalyticsDB) -> dict[str, str]:
    """{division: latest_season} for divisions with results, newest season each.

    Newest by chronology, not string order -- a 1990s code like "9900" is lexically larger
    than "2526" but chronologically older.
    """
    out: dict[str, str] = {}
    for season, division, _n in adb.seasons_loaded():
        if division not in out or season_sort_key(season) > season_sort_key(out[division]):
            out[division] = season
    return out


def _resolve_league(q: str, loaded: dict[str, str]) -> str | None:
    """A division mentioned in the question (by name or nickname), if it's loaded.

    Longest alias first, so 'bundesliga 2' resolves to D2 rather than matching 'bundesliga'.
    """
    for alias in sorted(_LEAGUE_ALIASES, key=len, reverse=True):
        division = _LEAGUE_ALIASES[alias]
        if alias in q and division in loaded:
            return division
    return None


def _default_league(loaded: dict[str, str]) -> str | None:
    for preferred in ("E0", "SP1", "D1", "I1", "F1", "E1", "USA"):
        if preferred in loaded:
            return preferred
    return next(iter(loaded), None)


def _season_years(code: str) -> tuple[int, int]:
    """(start_year, end_year) for a season code: European halves span two years, calendar one."""
    if len(code) == 4 and code.isdigit():
        first, second = int(code[:2]), int(code[2:])
        if (first + 1) % 100 == second:
            start = (1900 if first >= 90 else 2000) + first
            return start, start + 1
    try:
        return int(code), int(code)
    except ValueError:
        return 0, 0


def _resolve_season(q: str, seasons: list[str]) -> str | None:
    """A loaded season the question refers to (year, 'last season', ...), else None for default."""
    if not seasons:
        return None
    ordered = sorted(seasons, key=season_sort_key)  # oldest -> newest
    if re.search(r"\blast season\b", q):
        return ordered[-2] if len(ordered) >= 2 else ordered[-1]
    if re.search(r"\b(this season|current|currently|right now)\b", q):
        return ordered[-1]
    # explicit range e.g. 2003/04, 03/04 (normalisation turns the slash into a space)
    m = re.search(r"\b(\d{2}(?:\d{2})?)[\s/-]+(\d{2}(?:\d{2})?)\b", q)
    if m:
        a2, b2 = int(m.group(1)[-2:]), int(m.group(2)[-2:])
        if (a2 + 1) % 100 == b2:  # consecutive halves -> a European season range
            end = (1900 if b2 >= 90 else 2000) + b2
            for s in seasons:
                if _season_years(s)[1] == end:
                    return s
    # a bare year: prefer the season ENDING that year (when a title is awarded), then starting it
    for ym in re.findall(r"\b(?:19|20)\d{2}\b", q):
        year = int(ym)
        for s in seasons:
            if _season_years(s)[1] == year:
                return s
        for s in seasons:
            if _season_years(s)[0] == year:
                return s
    return None


def _league_and_season(q: str, adb: AnalyticsDB, loaded: dict[str, str]) -> tuple[str | None, str]:
    """Resolve both the division and the season the question asks about (default: latest)."""
    division = _resolve_league(q, loaded) or _default_league(loaded)
    if division is None:
        return None, ""
    seasons = [s for s, d, _n in adb.seasons_loaded() if d == division]
    return division, (_resolve_season(q, seasons) or loaded[division])


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


def _team_tokens(adb: AnalyticsDB) -> frozenset[str]:
    """Distinctive words that name a loaded team (len>=5), to stop a club name matching a
    player. A club literally named after a player exists (e.g. a keeper 'Chelsea Ashurst'),
    so a lone 'chelsea' must resolve to the club, not her -- the full player name still wins.
    """
    loaded = _loaded_divisions(adb)
    tokens: set[str] = set()
    for norm in _team_index(adb, loaded):
        tokens.update(w for w in norm.split() if len(w) >= 5)
    return frozenset(tokens)


def _resolve_player(
    q: str, adb: AnalyticsDB, team_tokens: frozenset[str] = frozenset()
) -> str | None:
    """The most prominent player (by minutes) whose name appears in the question.

    StatsBomb stores full legal names ("Lionel Andrés Messi Cuccittini"); users type
    "Messi". So match on the full name OR any distinctive whole-word token (a surname),
    and disambiguate common tokens by picking the player with the most minutes. Tokens that
    are also a loaded club name are skipped unless the full player name is present.
    """
    if adb.player_stats_count() == 0:
        return None
    q_words = set(q.split())
    best, best_mins = None, -1
    for player, mins in adb.player_minutes():
        name = _norm(player)
        tokens = [
            t for t in name.split() if len(t) >= 5 and t not in _STOP and t not in team_tokens
        ]
        if (name in q or any(t in q_words for t in tokens)) and mins > best_mins:
            best, best_mins = player, mins
    return best


def _resolve_players(
    q: str, adb: AnalyticsDB, team_tokens: frozenset[str], limit: int = 2
) -> list[str]:
    """Distinct players named in the question, in the order they appear -- for comparisons.

    Like `_resolve_player` but keeps every match, ordered by first mention (ties broken by
    minutes), so 'compare Messi and Ronaldo' yields both, Messi first.
    """
    if adb.player_stats_count() == 0:
        return []
    q_words = set(q.split())
    found: dict[str, tuple[int, int]] = {}  # player -> (position, minutes)
    for player, mins in adb.player_minutes():
        name = _norm(player)
        pos = q.find(name) if name in q else -1
        if pos == -1:
            for t in name.split():
                if len(t) >= 5 and t not in _STOP and t not in team_tokens and t in q_words:
                    p = q.find(t)
                    if p != -1 and (pos == -1 or p < pos):
                        pos = p
        if pos != -1:
            found[player] = (pos, mins)
    ordered = sorted(found.items(), key=lambda kv: (kv[1][0], -kv[1][1]))
    return [player for player, _ in ordered[:limit]]


# --- intents -----------------------------------------------------------------

_EXAMPLES = [
    "What's the predicted score for Arsenal vs Chelsea?",
    "Tell me about Liverpool",
    "Who's top of the Premier League?",
    "Who has won the most titles?",
    "Compare Messi and Ronaldo",
    "How accurate is your model?",
    "Top scorers in La Liga",
    "How is Man City's form?",
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
    triggers = re.search(
        r"\b(vs|versus|beat|predict|forecasts?|prediction|wins?|winner|odds"
        r"|scoreline|score|result|chances?)\b",
        q,
    )
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
        # A clear match-forecast ask ("match forecasts", "predict Arsenal") but not two
        # teams -> guide, rather than falling through to a blank "I'm not sure".
        if re.search(r"\b(predict|forecasts?|prediction|scoreline)\b", q) or " vs " in f" {q} ":
            return Reply(
                "To forecast a match, name two teams — e.g. **Arsenal vs Chelsea**. "
                "For upcoming real fixtures, ask for **fixtures**.",
                suggestions=["Arsenal vs Chelsea, who wins?", "Show upcoming fixtures"],
            )
        return None

    from soccer.dashboard.data import forecast_explanation, forecast_slate

    division, (home_e, away_e) = pair[0], pair[1][:2]
    home, away = home_e[0], away_e[0]
    season = home_e[1]
    slate = forecast_slate(analytics_db, season, division, home, away)
    if slate is None:
        return None
    res = {m.name: m.probability for m in slate.result}
    x, y, p0 = slate.most_likely_score
    others = ", ".join(f"{a}-{b} ({p:.0%})" for a, b, p in slate.correct_scores[1:3])
    over = next(o for o in slate.over_under if o.line == 2.5).over
    btts = next(m.probability for m in slate.btts if m.name == "Yes")
    lead = max(res, key=res.get)
    text = (
        f"**{home} vs {away}** ({division_name(division)})\n\n"
        f"- Predicted score: **{x}-{y}** ({p0:.0%}) — then {others}\n"
        f"- {home} win **{res[home]:.0%}** · draw **{res['Draw']:.0%}** · "
        f"{away} win **{res[away]:.0%}**\n"
        f"- Over 2.5 goals **{over:.0%}** · both teams score **{btts:.0%}**"
    )
    expl = forecast_explanation(analytics_db, season, division, home, away)
    if expl is not None:
        text += f"\n\n{expl.summary} (confidence: {expl.confidence.lower()})"
    else:
        text += (
            f"\n\nModel leans **{lead if lead != 'Draw' else 'a draw'}**. Directional, not advice."
        )
    return Reply(
        text,
        suggestions=[
            f"{home} vs {away} head to head",
            f"How is {home}'s form?",
        ],
    )


def _intent_top_scorers(q: str, analytics_db: Path, live_db: Path | None) -> Reply | None:
    if not re.search(
        r"top scorer|scorers?|most goals|golden boot|leading scorer|goalscorer"
        r"|best (striker|forward|player|playmaker|passer|creator|midfielder)"
        r"|most assists|player of the (season|year)|goal contributions|most creative",
        q,
    ):
        return None
    if re.search(r"playmaker|assist|passer|creator|creative|key pass", q):
        order = "assists"
    elif re.search(r"best player|player of the|contribution|goals and assists", q):
        order = "contributions"
    else:
        order = "goals"
    with AnalyticsDB(analytics_db) as adb:
        if adb.player_stats_count() == 0:
            return Reply(
                "I don't have per-player data loaded yet. Go to "
                "**Home → Add player data** to add some."
            )
        competition = _resolve_statsbomb_competition(q, adb)
        profiles = adb.player_profiles(
            limit=8, min_minutes=270, order=order, competition=competition
        )
    if not profiles:
        return None
    scope = f" in {competition}" if competition else ""
    rows = [
        {
            "Player": p.player,
            "Team": p.team,
            "Goals": p.goals,
            "Assists": p.assists,
            "xG": round(p.xg, 1),
        }
        for p in profiles
    ]
    lead = profiles[0]
    if order == "assists":
        headline = (
            f"**Top playmakers{scope}** — {lead.player} leads with "
            f"**{lead.assists} assists** (xA {lead.xa:.1f})."
        )
    elif order == "contributions":
        headline = (
            f"**Best players{scope}** by goal involvement — {lead.player} leads with "
            f"**{lead.goal_contributions}** ({lead.goals}G, {lead.assists}A)."
        )
    else:
        headline = (
            f"**Top scorers{scope}** — {lead.player} leads with "
            f"**{lead.goals} goals** (xG {lead.xg:.1f})."
        )
    return Reply(
        headline + " Across all loaded seasons.",
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
        player = _resolve_player(q, adb, _team_tokens(adb))
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


def _vs_avg(multiplier: float) -> str:
    """A strength multiplier as a readable deviation: 1.2 -> '20% above average'."""
    d = multiplier - 1.0
    return f"{abs(d):.0%} {'above' if d >= 0 else 'below'} average"


def _intent_compare(q: str, analytics_db: Path, live_db: Path | None) -> Reply | None:
    if not re.search(r"\bcompare\b|\bvs\b|\bversus\b|\bbetter\b|compared to", q):
        return None
    with AnalyticsDB(analytics_db) as adb:
        if adb.player_stats_count() == 0:
            return None
        names = _resolve_players(q, adb, _team_tokens(adb), limit=2)
        profiles = [adb.player_profile(n) for n in names]
    profiles = [p for p in profiles if p is not None]
    if len(profiles) < 2:
        return None
    a, b = profiles[0], profiles[1]
    rows = [
        {"Metric": m, a.player: av, b.player: bv}
        for m, av, bv in (
            ("Matches", a.matches, b.matches),
            ("Goals", a.goals, b.goals),
            ("Assists", a.assists, b.assists),
            ("xG", round(a.xg, 1), round(b.xg, 1)),
            ("xA", round(a.xa, 1), round(b.xa, 1)),
            ("Key passes", a.key_passes, b.key_passes),
            ("Pass %", round(a.pass_pct), round(b.pass_pct)),
        )
    ]
    return Reply(
        f"**{a.player}** vs **{b.player}** — {a.goals}G/{a.assists}A vs "
        f"{b.goals}G/{b.assists}A over {a.matches} and {b.matches} matches loaded.",
        table=rows,
        suggestions=[f"Tell me about {a.player}", f"Tell me about {b.player}"],
    )


def _intent_model(q: str, analytics_db: Path, live_db: Path | None) -> Reply | None:
    if not re.search(
        r"how (accurate|reliable)\b|how good (is|are) (your|the) (model|forecast|prediction)"
        r"|model'?s? (accuracy|accurate)|forecast accuracy|accuracy of (your|the)"
        r"|beat the (market|bookies?|book)|can you beat|do you beat|edge over"
        r"|how (do|does) (your|the) (model|forecast)|calibrat|log ?loss|brier|\brps\b|backtest",
        q,
    ):
        return None
    with AnalyticsDB(analytics_db) as adb:
        loaded = _loaded_divisions(adb)
        if not loaded:
            return None
        division, _season = _league_and_season(q, adb, loaded)
        if division is None:
            return None

    from soccer.dashboard.data import forecast_report

    report = forecast_report(analytics_db, division)
    if report is None:
        return Reply(
            "I can only score forecast accuracy where bookmaker odds are loaded, and I don't "
            f"have any for {division_name(division)} yet."
        )
    m, mk = report.model, report.market
    if report.best_weight >= 0.10 and report.blend_beats_market:
        blendline = (
            f"a **{report.best_weight:.0%}** model blend beats the closing line by a hair, "
            "but not enough to clear the bookmaker's margin"
        )
    else:
        blendline = "mixing the model into the market barely moves the needle"
    return Reply(
        f"**Forecast accuracy — {division_name(division)}** "
        f"(walk-forward over {report.n} matches):\n\n"
        f"- Model: RPS **{m.rps:.3f}**, log-loss **{m.log_loss:.3f}**\n"
        f"- Market close: RPS **{mk.rps:.3f}**, log-loss **{mk.log_loss:.3f}** (lower is better)\n"
        f"- Best blend weight on the model: **{report.best_weight:.0%}**\n\n"
        f"Honest read: the market is sharper — {blendline}. There's no exploitable edge on free "
        "public data, so these forecasts are for insight, not betting.",
        suggestions=["Arsenal vs Chelsea, who wins?", "Who will win the league?"],
    )


def _intent_honours(q: str, analytics_db: Path, live_db: Path | None) -> Reply | None:
    if not re.search(
        r"most (league )?titles?|won the most|most successful|most champions|most trophies"
        r"|record (points|for points)|all[ -]?time|hall of fame|how many titles"
        r"|biggest (ever|ever win|win in history)|record (win|victory)",
        q,
    ):
        return None
    with AnalyticsDB(analytics_db) as adb:
        loaded = _loaded_divisions(adb)
        division, _season = _league_and_season(q, adb, loaded)
        if division is None:
            return None

    from soccer.dashboard.data import league_history

    hist = league_history(analytics_db, division)
    if hist is None or not hist.title_counts:
        return None
    rows = [{"Team": t, "Titles": n} for t, n in hist.title_counts[:8]]
    leader, count = hist.title_counts[0]
    text = (
        f"**{division_name(division)} — all-time** (loaded {hist.oldest} to {hist.newest}, "
        f"{hist.seasons} seasons)\n\n"
        f"- Most titles: **{leader}** with **{count}**\n"
    )
    if hist.record_points:
        rt, rs, rp = hist.record_points
        text += f"- Record points haul: **{rt}, {rp} pts** ({rs})\n"
    if hist.biggest_wins:
        w = hist.biggest_wins[0]
        text += f"- Biggest win on record: {w['Result']} ({w['Season']})"
    return Reply(
        text,
        table=rows,
        suggestions=["Who will win the league?", "Who is in form?"],
    )


def _intent_team(q: str, analytics_db: Path, live_db: Path | None) -> Reply | None:
    kw = re.search(
        r"tell me about|how (are|is|good|s)\b|hows\b|what about|whats up with|profile of"
        r"|overview|report on|scout|any good|doing|season so far|rate\b|breakdown|dossier",
        q,
    )
    if not kw and len(q.split()) > 3:  # a bare club name is a fine "everything about X" ask
        return None
    with AnalyticsDB(analytics_db) as adb:
        loaded = _loaded_divisions(adb)
        if not loaded:
            return None
        index = _team_index(adb, loaded)
        named = _resolve_teams(q, index)
        if not named:
            return None
        display, division, season = named[0]

    from soccer.dashboard.data import team_dossier

    d = team_dossier(analytics_db, division, season, display)
    if d is None:
        return None
    trend = "rising" if d.trend > 0.15 else "sliding" if d.trend < -0.15 else "steady"
    lines = [
        f"**{d.team}** — {division_name(division)} {season_label(season)}",
        "",
        f"- **{_ordinal(d.position)}** on {d.points} pts "
        f"({d.won}W {d.drawn}D {d.lost}L), GD {d.goal_difference:+d}",
        f"- Last 5: **{d.recent_form or 'n/a'}** ({d.recent_ppg:.2f} ppg vs "
        f"{d.season_ppg:.2f} season, {trend})",
    ]
    if d.xpoints is not None:
        diff = d.points - d.xpoints
        read = "overperforming" if diff > 1.5 else "underperforming" if diff < -1.5 else "about par"
        lines.append(
            f"- Underlying: **{d.xpoints:.1f} xP** ({diff:+.1f}, {read}); "
            f"xGF {d.xgf:.1f} / xGA {d.xga:.1f}"
        )
    lines.append(f"- Attack {_vs_avg(d.attack)}; defence {_vs_avg(d.solidity)}")
    if d.winning >= 3:
        lines.append(f"- On a **{d.winning}-game winning run**")
    elif d.unbeaten >= 4:
        lines.append(f"- **Unbeaten in {d.unbeaten}**")
    rows = [
        {"Opponent": r["opponent"], "H/A": r["venue"], "Score": r["score"], "Res": r["result"]}
        for r in d.recent
    ]
    return Reply(
        "\n".join(lines),
        table=rows,
        suggestions=[f"Is {d.team} overperforming their xG?", f"How is {d.team}'s form?"],
    )


def _intent_title_odds(q: str, analytics_db: Path, live_db: Path | None) -> Reply | None:
    if not re.search(
        r"win the (league|title|season)|winning the (league|title|season)"
        r"|chances? (of|to) win|chance to win|odds (of|to) win|likely to win"
        r"|who will win|who wins|\btitle\b|champions?|relegat|top four|top 4"
        r"|finish|title race|win it\b",
        q,
    ):
        return None
    with AnalyticsDB(analytics_db) as adb:
        loaded = _loaded_divisions(adb)
        if not loaded:
            return None
        index = _team_index(adb, loaded)
        named = _resolve_teams(q, index)
        if named:  # a club is named -> use its own league
            _display, division, season = named[0]
        else:
            division, season = _league_and_season(q, adb, loaded)
        if division is None:
            return None

    from soccer.dashboard.data import season_briefing

    briefing = season_briefing(analytics_db, season, division, n_sims=4000)
    if briefing is None:
        return None
    names = briefing.names

    if named:
        target = _norm(named[0][0])
        proj = next(
            (p for p in briefing.projections if _norm(names.get(p.team, p.team)) == target), None
        )
        if proj is not None:
            name = names.get(proj.team, proj.team)
            return Reply(
                f"**{name}** — {division_name(division)} "
                f"{season_label(season)}: **{proj.title_pct:.0%}** to win the title, "
                f"**{proj.top_pct:.0%}** top four, **{proj.relegation_pct:.0%}** relegation. "
                "A Monte-Carlo sim off recent strengths, blind to transfers.",
                suggestions=["Who are the favourites?", f"How is {name}'s form?"],
            )

    projs = sorted(briefing.projections, key=lambda p: -p.title_pct)[:6]
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
        division, season = _league_and_season(q, adb, loaded)
        if division is None:
            return None
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
        division, season = _league_and_season(q, adb, loaded)
        if division is None:
            return None
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
        r"\b(table|standings?|top of|leading|who is top|whos top|position|rank|where are"
        r"|who won|who lifted|who topped|winners?)\b",
        q,
    ):
        return None
    with AnalyticsDB(analytics_db) as adb:
        loaded = _loaded_divisions(adb)
        division, season = _league_and_season(q, adb, loaded)
        if division is None:
            return None
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
        r"\b(fixtures?|upcoming|next (game|match|fixture|up)|who ?s playing|this weekend"
        r"|schedule|play next|playing next)\b|when (do|does|will|are|is) \w+",
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

    with AnalyticsDB(analytics_db) as adb:
        index = _team_index(adb, _loaded_divisions(adb))
    named = _resolve_teams(q, index)
    if named:  # a club is named -> that club's own schedule, team-first
        tnorm = _norm(named[0][0])
        mine = [f for f in fixtures if tnorm in _norm(f.home) or tnorm in _norm(f.away)]
        if not mine:
            return Reply(
                f"I don't see an upcoming fixture for **{named[0][0]}** in the loaded schedule."
            )
        rows = []
        for f in mine[:6]:
            hx, ax, _ = f.slate.most_likely_score
            res = [m.probability for m in f.slate.result]
            home_is = tnorm in _norm(f.home)
            rows.append(
                {
                    "Date": f.kickoff_utc.strftime("%b %d"),
                    "Opponent": f.away if home_is else f.home,
                    "H/A": "H" if home_is else "A",
                    "Pred": f"{hx}-{ax}" if home_is else f"{ax}-{hx}",
                    "Win %": round(100 * (res[0] if home_is else res[2])),
                }
            )
        nxt = mine[0]
        home_is = tnorm in _norm(nxt.home)
        opp = nxt.away if home_is else nxt.home
        res = [m.probability for m in nxt.slate.result]
        hx, ax, _ = nxt.slate.most_likely_score
        pred = f"{hx}-{ax}" if home_is else f"{ax}-{hx}"
        return Reply(
            f"**{named[0][0]}'s next match** — {'home to' if home_is else 'away to'} "
            f"**{opp}** on {nxt.kickoff_utc.strftime('%a %b %d')}: predicted **{pred}**, "
            f"{named[0][0]} win **{(res[0] if home_is else res[2]):.0%}**.",
            table=rows,
            suggestions=[f"Tell me about {named[0][0]}", "Show all upcoming fixtures"],
        )

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


def _intent_h2h(q: str, analytics_db: Path, live_db: Path | None) -> Reply | None:
    if not re.search(
        r"head ?to ?head|\bh2h\b|record against|\bmeetings?\b|\bagainst\b|"
        r"history (with|against|between)",
        q,
    ):
        return None
    with AnalyticsDB(analytics_db) as adb:
        loaded = _loaded_divisions(adb)
        if not loaded:
            return None
        index = _team_index(adb, loaded)
        teams = _resolve_teams(q, index)
        if len(teams) < 2:
            return None
        a, b = teams[0], teams[1]
        rows = adb.head_to_head(_norm(a[0]), _norm(b[0]), limit=200)
    if not rows:
        return Reply(f"No meetings between **{a[0]}** and **{b[0]}** in the loaded data.")
    a_wins = b_wins = draws = 0
    for _s, _d, _date, home, away, hg, ag in rows:
        if hg == ag:
            draws += 1
        elif _norm(home if hg > ag else away) == _norm(a[0]):
            a_wins += 1
        else:
            b_wins += 1
    recent = [
        {
            "Date": md.strftime("%Y-%m-%d") if hasattr(md, "strftime") else str(md)[:10],
            "Competition": division_name(div),
            "Result": f"{home} {hg}-{ag} {away}",
        }
        for _s, div, md, home, away, hg, ag in rows[:6]
    ]
    return Reply(
        f"**{a[0]} vs {b[0]} — head to head** ({len(rows)} meetings in the loaded data)\n\n"
        f"- **{a[0]} {a_wins}** · **{draws} draws** · **{b[0]} {b_wins}**",
        table=recent,
        suggestions=[f"{a[0]} vs {b[0]}, who wins?", f"How is {a[0]}'s form?"],
    )


def _intent_underlying(q: str, analytics_db: Path, live_db: Path | None) -> Reply | None:
    if not re.search(
        r"overperform\w*|underperform\w*|over ?achiev\w*|under ?achiev\w*|\blucky\b|unlucky|"
        r"deserv\w*|expected points|\bxp\b|xg table|regress\w*|flatter\w*|riding .*luck|"
        r"punching above",
        q,
    ):
        return None
    with AnalyticsDB(analytics_db) as adb:
        loaded = _loaded_divisions(adb)
        division, season = _league_and_season(q, adb, loaded)
        if division is None:
            return None
        index = _team_index(adb, loaded)

    from soccer.dashboard.data import underlying_table

    table = underlying_table(analytics_db, season, division)
    if not table:
        return Reply(
            f"I don't have shot data for {division_name(division)} {season_label(season)}, so I "
            "can't work out expected points there — try a recent season of a major league."
        )
    named = _resolve_teams(q, index)
    if named:
        row = next((r for r in table if _norm(r.team) == _norm(named[0][0])), None)
        if row:
            if row.points_diff > 1.5:
                verdict = "**overperforming** their underlying play (running hot, prone to cool)"
            elif row.points_diff < -1.5:
                verdict = "**underperforming** — creating more than the table shows (unlucky)"
            else:
                verdict = "roughly in line with their underlying numbers"
            return Reply(
                f"**{row.team}** ({division_name(division)} {season_label(season)}) — "
                f"{row.points} pts vs **{row.xpoints:.1f} expected** ({row.points_diff:+.1f}). "
                f"They're {verdict}. Goals {row.goals_for} vs {row.xgf:.1f} xG."
            )
    over = max(table, key=lambda r: r.points_diff)
    under = min(table, key=lambda r: r.points_diff)
    rows = [
        {
            "Team": r.team,
            "Pts": r.points,
            "xP": round(r.xpoints, 1),
            "Diff": round(r.points_diff, 1),
        }
        for r in table[:8]
    ]
    return Reply(
        f"**{division_name(division)} {season_label(season)} — over/under-performance.** "
        f"Running hottest: **{over.team}** ({over.points_diff:+.1f} vs expected). "
        f"Unluckiest: **{under.team}** ({under.points_diff:+.1f}).",
        table=rows,
        suggestions=["Who is in form?", "Who will win the league?"],
    )


_SB_STOP = {"real", "club", "borussia", "olympique", "deportivo", "sporting", "racing"}


def _covered_tokens(name: str, q_words: set[str]) -> set[str]:
    """Distinctive club-name tokens (len>=4) from `name` that appear in the question."""
    return {w for w in _norm(name).split() if len(w) >= 4 and w not in _SB_STOP} & q_words


def _find_shot_match(q: str, matches: list[tuple]) -> tuple[int, str, str] | None:
    """The loaded StatsBomb match whose two clubs are both named in the question.

    Scores every "Home v Away" label by how many distinctive name tokens the question covers
    on each side; a valid match needs both sides covered by different clubs. Ties go to the
    newest StatsBomb id (~ most recent meeting). None if no match is clearly identified. This
    matches whole games rather than picking two club names, so "Bayer Leverkusen" and the
    alias "TSV Bayer 04 Leverkusen" can't be mistaken for two different sides.
    """
    q_words = set(q.split())
    best: tuple[int, int, str, str] | None = None  # (score, mid, home, away)
    for mid, label, _comp, _season in matches:
        if " v " not in label:
            continue
        home, away = (x.strip() for x in label.split(" v ", 1))
        ch, ca = _covered_tokens(home, q_words), _covered_tokens(away, q_words)
        if ch and ca and _norm(home) != _norm(away):
            score = len(ch) + len(ca)
            if best is None or (score, mid) > (best[0], best[1]):
                best = (score, mid, home, away)
    return (best[1], best[2], best[3]) if best else None


def _intent_match_centre(q: str, analytics_db: Path, live_db: Path | None) -> Reply | None:
    if not re.search(
        r"shot map|shot log|shots? (in|from|for|by|chart)|match (centre|center|report)"
        r"|xg (timeline|race|map|breakdown)|chances (created|in the (game|match))"
        r"|how many shots|xg (in|for) (the )?(game|match)",
        q,
    ):
        return None

    from soccer.dashboard.data import shot_map, shot_matches

    matches = shot_matches(analytics_db)
    found = _find_shot_match(q, matches) if matches else None
    if found is None:
        return None
    mid, home, away = found
    data = shot_map(analytics_db, mid)
    if data is None:
        return None
    xg = sorted(data.team_xg, key=lambda r: -r.xg)
    xg_line = " · ".join(f"{r.name} **{r.xg:.1f} xG** ({r.goals}g, {r.shots} sh)" for r in xg)
    top = sorted(data.shots, key=lambda s: -s["xg"])[:8]
    rows = [
        {
            "Min": s["minute"],
            "Player": s["player"],
            "Team": s["team"],
            "xG": round(s["xg"], 2),
            "Outcome": s.get("outcome", ""),
        }
        for s in top
    ]
    return Reply(
        f"**{data.label}**\n\n- xG race: {xg_line}\n- Biggest chances below.",
        table=rows,
        suggestions=[f"Tell me about {home}", f"How is {away}'s form?"],
    )


def _intent_scout(q: str, analytics_db: Path, live_db: Path | None) -> Reply | None:
    if not re.search(
        r"scout|scouting|percentiles?|\bradar\b|fingerprint|strengths and weakness"
        r"|how does .* (rank|compare)|rank among|percentile rank",
        q,
    ):
        return None
    with AnalyticsDB(analytics_db) as adb:
        if adb.player_stats_count() == 0:
            return None
        player = _resolve_player(q, adb, _team_tokens(adb))
    if player is None:
        return None

    from soccer.dashboard.data import player_percentiles

    pcts = player_percentiles(analytics_db, player)
    if not pcts:
        return None
    best = max(pcts, key=lambda m: m.percentile)
    worst = min(pcts, key=lambda m: m.percentile)
    rows = [{"Metric": m.label, "Per 90": m.value, "Pctl": round(m.percentile)} for m in pcts]
    return Reply(
        f"**{player} — scouting profile** (percentile vs all loaded players, 200+ mins)\n\n"
        f"- Elite: **{best.label}** — {round(best.percentile)}th pct ({best.value} per 90)\n"
        f"- Weakest: {worst.label} — {round(worst.percentile)}th pct\n\n"
        "Percentiles pool every loaded competition and era together, so read them as a rough "
        "shape rather than a single-league rank.",
        table=rows,
        suggestions=["Top scorers", "Who is the best playmaker?"],
    )


def _ordinal(n: int) -> str:
    suffix = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _fallback(q: str, analytics_db: Path) -> Reply:
    return Reply(
        "I'm not sure how to answer that yet. I can help with match forecasts and predicted "
        "scores, team reports, league tables (any loaded league or past season), top scorers "
        "and player comparisons, head-to-head records, current form, over/under-performance "
        "(xG), in-season and all-time records, title odds, model accuracy and fixtures.",
        suggestions=_EXAMPLES[:4],
    )
