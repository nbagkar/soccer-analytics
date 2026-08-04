"""Streamlit dashboard -- a thin render over `dashboard/data.py`.

Read-only. Every page is a consistent header (Material icon + one-line blurb) over data
the store can actually fill; a surface with no data says so rather than showing an empty
frame. Theme lives in `.streamlit/config.toml` (loaded because `soccer dashboard` runs
streamlit from this directory).

Status is shown as a reserved palette that always carries a text label, never colour
alone: live is green, delayed/postponed amber, cancelled red, concluded a muted grey.
Run with `soccer dashboard` (or `streamlit run src/soccer/dashboard/app.py`).
"""

from __future__ import annotations

import html

import altair as alt
import polars as pl
import streamlit as st

from soccer.config import get_settings
from soccer.dashboard.data import (
    AnalyticsSnapshot,
    HealthSnapshot,
    LiveSnapshot,
    analytics_available,
    analytics_snapshot,
    fixture_forecasts,
    forecast_explanation,
    forecast_report,
    forecast_slate,
    forecast_teams,
    has_player_events,
    health_snapshot,
    league_history,
    live_snapshot,
    market_edge,
    player_board,
    player_competitions,
    player_percentiles,
    player_profile,
    player_profiles,
    player_seasons,
    season_briefing,
    season_records,
    shot_map,
    shot_matches,
    team_dossier,
    team_form,
    underlying_table,
)
from soccer.domain.match_state import MatchStatus, MatchView
from soccer.sources.football_data_co_uk import division_name, season_label, season_sort_key
from soccer.sources.statsbomb import ATTRIBUTION as ATTRIBUTION_STATSBOMB
from soccer.storage.live_db import LiveDB

# Reserved status palette: (text colour, short label). Every status renders with its
# label, so meaning never rests on colour alone.
_STATUS = {
    MatchStatus.IN_PLAY: ("#16a34a", "LIVE"),
    MatchStatus.FIRST_HALF: ("#16a34a", "1H"),
    MatchStatus.HALF_TIME: ("#16a34a", "HT"),
    MatchStatus.SECOND_HALF: ("#16a34a", "2H"),
    MatchStatus.EXTRA_TIME: ("#16a34a", "ET"),
    MatchStatus.PENALTIES: ("#16a34a", "PENS"),
    MatchStatus.FINISHED: ("#8b8b8b", "FT"),
    MatchStatus.AWARDED: ("#8b8b8b", "AWD"),
    MatchStatus.NOT_STARTED: ("#8b8b8b", ""),
    MatchStatus.POSTPONED: ("#d97706", "PSTP"),
    MatchStatus.SUSPENDED: ("#d97706", "SUSP"),
    MatchStatus.CANCELLED: ("#dc2626", "CANC"),
    MatchStatus.UNKNOWN: ("#8b8b8b", "?"),
}


def _marker(view: MatchView) -> tuple[str, str]:
    """(colour, text) for a match's status cell -- live minute, FT, or kickoff time."""
    colour, label = _STATUS.get(view.status, ("#8b8b8b", "?"))
    if view.status.is_in_play:
        return colour, view.minute or label or "LIVE"
    if view.status is MatchStatus.NOT_STARTED:
        return colour, view.kickoff_utc.strftime("%H:%M")
    return colour, label


def _go(page: str) -> None:
    """Request a navigation change from a call-to-action button (takes a routing key).

    Stashes the target label in a separate key rather than writing the radio's own ``nav``
    key, which Streamlit forbids once the widget has been instantiated this run; main()
    applies the request before the radio is created.
    """
    st.session_state._nav_to = _LABEL_BY_KEY.get(page, page)
    st.rerun()


def _render_home(settings) -> None:
    from soccer.dashboard import actions

    status = actions.data_status(settings)

    # First launch: set the user up automatically -- no clicks, no terminal. The data
    # can't ship bundled (source licences forbid redistribution), so we fetch a starter
    # set live, once.
    if status["leagues"] == 0 and not st.session_state.get("_setup_tried"):
        st.session_state._setup_tried = True
        st.info("Welcome! Setting up a starter set of leagues for you — a one-time download.")
        bar = st.progress(0.0, "Downloading…")
        try:
            message = actions.starter_setup(
                settings, on_progress=lambda d, t: bar.progress(d / t, f"{d}/{t} leagues")
            )
            bar.empty()
            st.toast(message, icon="✅")
            st.rerun()
        except Exception as exc:  # offline/first-run; fall back to the manual buttons
            bar.empty()
            st.warning(f"Couldn't auto-download (are you online?). Use the buttons below. — {exc}")

    st.markdown(
        "Welcome! This is your local football intelligence centre — everything runs on "
        "your machine. Ask questions in plain English, browse forecasts, tables, form and "
        "player stats. It sets itself up on first launch; top up any time below. "
        "**No terminal needed.**"
    )
    c = st.columns(4)
    c[0].metric("Leagues", status["leagues"], border=True)
    c[1].metric("History matches", f"{status['history_matches']:,}", border=True)
    c[2].metric("Player datasets", status["player_competitions"], border=True)
    c[3].metric("Upcoming fixtures", status["upcoming"], border=True)

    left, right = st.columns(2)
    with left, st.container(border=True):
        st.markdown("#### :material/rocket_launch: Start here")
        if st.button("Ask the assistant", icon=":material/chat:", width="stretch", type="primary"):
            _go("Assistant")
        if st.button("See predictions & fixtures", icon=":material/insights:", width="stretch"):
            _go("Predictor")
    with right, st.container(border=True):
        st.markdown("#### :material/refresh: Keep it current")
        if st.button("Refresh live scores", icon=":material/bolt:", width="stretch"):
            with st.spinner("Fetching the latest scores…"):
                st.toast(actions.refresh_scores(settings), icon="✅")
            _go("Home")
        if st.button("Update fixtures", icon=":material/event:", width="stretch"):
            with st.spinner("Fetching upcoming fixtures…"):
                st.toast(actions.update_fixtures(settings), icon="✅")
            _cached_fixture_forecasts.clear()  # show the freshly pulled schedule at once
            _go("Home")

    st.caption(
        "Everything's loaded and ready — nothing to add. To pull in an extra league or "
        "more player data later, see **About & sources**."
    )


def _render_data_manager(settings) -> None:
    """Optional data top-ups, tucked away from the main flow -- most users never need it."""
    from soccer.dashboard import actions

    st.markdown("#### :material/tune: Manage data")
    st.caption("Your data is already loaded; this is only for adding extras.")

    with st.expander("Add a league (results & forecasts)", icon=":material/add_circle:"):
        choice = st.selectbox("Which league?", list(actions.LEAGUE_CHOICES))
        if st.button("Download recent seasons", key="add_league"):
            with st.spinner(f"Downloading {choice}…"):
                st.success(actions.add_league_history(settings, actions.LEAGUE_CHOICES[choice]))
        st.caption(
            f"Or go deep: every league's results back to the 1990s "
            f"(~{actions.FULL_HISTORY_SEASONS} seasons). Richer tables, forecasts and records."
        )
        if st.button("Load full history (all leagues)", key="add_full_history"):
            bar = st.progress(0.0, "Starting…")
            msg = actions.load_full_history(
                settings, on_progress=lambda d, t: bar.progress(d / t, f"league {d}/{t}")
            )
            bar.empty()
            st.success(msg)
        st.caption(
            "Champions League results (recent seasons, via football-data.org) — powers "
            "head-to-head and records. Needs a free football-data.org token."
        )
        if st.button("Load Champions League", key="add_ucl"):
            with st.spinner("Fetching Champions League results…"):
                st.success(actions.load_champions_league(settings))

    with st.expander("Add player data (StatsBomb events)", icon=":material/groups:"):
        st.caption(
            "StatsBomb open data — free for personal/research use, logo attribution "
            "required. Loading everything takes a few minutes."
        )
        if st.button("Load all player data", key="add_all_events"):
            bar = st.progress(0.0, "Starting…")
            msg = actions.load_all_events(
                settings, on_progress=lambda d, t: bar.progress(d / t, f"dataset {d}/{t}")
            )
            bar.empty()
            st.success(msg)
        pack = st.selectbox("Or a single dataset", list(actions.EVENT_PACKS))
        if st.button("Load this one", key="add_events"):
            comp_id, season_id, _n = actions.EVENT_PACKS[pack]
            bar = st.progress(0.0, "Starting…")
            msg = actions.load_event_pack(
                settings,
                comp_id,
                season_id,
                on_progress=lambda d, t: bar.progress(d / t, f"{d}/{t} matches"),
            )
            bar.empty()
            st.success(msg)


def _render_chat_chart(chart) -> None:
    """Render an assistant Reply.chart spec with the same builders the pages use."""
    if not chart or not chart.get("data"):
        return
    kind = chart["kind"]
    if kind == "xg_race":
        st.altair_chart(_xg_race_chart(chart["data"]), width="stretch")
    elif kind == "trajectory":
        st.altair_chart(_team_trajectory_chart(chart["data"]), width="stretch")
    elif kind == "percentiles":
        st.altair_chart(_percentile_bars_chart(chart["data"]), width="stretch")
    elif kind == "result_bar":
        st.altair_chart(_outcome_bar_chart(chart["data"]), width="stretch")


def _outcome_bar_chart(rows: list[dict]):
    """Win/draw/away probability bars from [{outcome, pct}, ...] (home, draw, away order)."""
    frame = pl.DataFrame(rows).to_pandas()
    order = [r["outcome"] for r in rows]
    colours = ["#16c784", "#8b95a1", "#ea3943"]  # home green / draw grey / away red
    bars = (
        alt.Chart(frame)
        .mark_bar(cornerRadiusEnd=3, height=26)
        .encode(
            x=alt.X("pct:Q", scale=alt.Scale(domain=[0, 100]), title="Win probability (%)"),
            y=alt.Y("outcome:N", sort=order, title=None),
            color=alt.Color("outcome:N", scale=alt.Scale(domain=order, range=colours), legend=None),
            tooltip=["outcome", alt.Tooltip("pct", title="%")],
        )
    )
    labels = bars.mark_text(align="left", dx=3).encode(text=alt.Text("pct:Q", format=".0f"))
    return (bars + labels).properties(height=110)


def _render_assistant(settings) -> None:
    from soccer.dashboard.assistant import answer as assistant_answer

    st.caption(
        "Ask in plain English about tables, top scorers, a player, a match forecast, form, "
        "records or fixtures. Runs entirely on your machine — nothing is sent anywhere."
    )
    greeting = {
        "role": "assistant",
        "text": (
            "Hi! I'm your football assistant. Ask me things like "
            "*“who's top of the Premier League?”* or *“Arsenal vs Chelsea, who wins?”*"
        ),
    }
    if "chat" not in st.session_state:
        st.session_state.chat = [greeting]

    prompt = st.chat_input("Ask a question…") or st.session_state.pop("_suggestion", None)
    if prompt:
        st.session_state.chat.append({"role": "user", "text": prompt})
        with st.spinner("Thinking…"):
            reply = assistant_answer(prompt, settings.analytics_db, settings.live_db)
        st.session_state.chat.append(
            {
                "role": "assistant",
                "text": reply.text,
                "table": reply.table,
                "suggestions": reply.suggestions,
                "chart": reply.chart,
            }
        )

    for message in st.session_state.chat:
        avatar = "⚽" if message["role"] == "assistant" else None
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["text"])
            if message.get("table"):
                st.dataframe(
                    pl.DataFrame(message["table"]).to_pandas(), hide_index=True, width="stretch"
                )
            _render_chat_chart(message.get("chart"))

    last = st.session_state.chat[-1]
    if last["role"] == "assistant" and last.get("suggestions"):
        st.caption("Try:")
        cols = st.columns(len(last["suggestions"]))
        for col, suggestion in zip(cols, last["suggestions"], strict=False):
            if col.button(suggestion, key=f"sug_{suggestion}", width="stretch"):
                st.session_state._suggestion = suggestion
                st.rerun()


def _render_live(snap: LiveSnapshot) -> None:
    k = snap.kpis
    cols = st.columns(5)
    cols[0].metric("Matches", k.total, border=True)
    cols[1].metric("In play", k.in_play, border=True)
    cols[2].metric("Competitions", k.competitions, border=True)
    cols[3].metric("Sources", k.sources, border=True)
    cols[4].metric("Updated", k.freshness_label, border=True)

    if k.any_stale:
        st.warning(
            "⚠ Some rows are **stale** — served from cache after a source failed. "
            "They are labelled in the Source column.",
            icon="⚠️",
        )
    if k.total == 0:
        st.info(
            "No scores yet. Go to **Home** → **Refresh live scores** to fetch them.",
            icon=":material/bolt:",
        )
        return

    rows = []
    for v in snap.matches:
        colour, text = _marker(v)
        source = v.source.replace("_", "-") + (" ⚠" if v.is_stale else "")
        rows.append(
            {
                "": f'<span style="color:{colour};font-weight:600">{text}</span>',
                "Home": _esc(v.home),
                "Score": f"<b>{_esc(v.score)}</b>",
                "Away": _esc(v.away),
                "Competition": f'<span style="color:#8b8b8b">{_esc(v.competition)}</span>',
                "Source": f'<span style="color:#8b8b8b">{source}</span>',
            }
        )
    st.markdown(_html_table(rows), unsafe_allow_html=True)

    if snap.competition_counts:
        with st.expander("Coverage — matches by competition"):
            _render_coverage_chart(snap.competition_counts)


def _render_coverage_chart(counts: list[tuple[str, int]]) -> None:
    # Single-series magnitude: one hue, sorted, direct value labels, recessive axes.
    frame = pl.DataFrame({"competition": [c for c, _ in counts], "matches": [n for _, n in counts]})
    base = alt.Chart(frame.to_pandas()).encode(
        x=alt.X("matches:Q", axis=alt.Axis(title=None, grid=False, tickMinStep=1)),
        y=alt.Y("competition:N", sort="-x", axis=alt.Axis(title=None)),
    )
    bars = base.mark_bar(color="#2563eb", cornerRadiusEnd=4, size=14)
    labels = base.mark_text(align="left", dx=4, color="#8b8b8b").encode(text="matches:Q")
    st.altair_chart((bars + labels).properties(height=max(120, 22 * len(counts))))


def _render_health(snap: HealthSnapshot) -> None:
    st.caption(
        "Free coverage is patchy by nature — this panel makes the gaps explicit so "
        "missing data never looks like a bug."
    )

    src_rows = []
    for s in snap.sources:
        status = (
            '<span style="color:#16a34a">● enabled</span>'
            if s.enabled
            else f'<span style="color:#8b8b8b">○ disabled ({s.reason})</span>'
        )
        latency = (
            f'<span style="color:#16a34a">{s.latency_label}</span>'
            if s.is_live
            else f'<span style="color:#8b8b8b">{s.latency_label}</span>'
        )
        licence = (
            f'<span style="color:#d97706">⚠ {s.licence}</span>'
            if s.licence_unresolved
            else f'<span style="color:#8b8b8b">{s.licence}</span>'
        )
        src_rows.append(
            {
                "Source": s.name,
                "Status": status,
                "Trust": s.trust,
                "Latency": latency,
                "Licence": licence,
            }
        )
    st.markdown(_html_table(src_rows), unsafe_allow_html=True)

    st.markdown("**Capability coverage**")
    cap_rows = [
        {
            "Capability": c.capability,
            "Served by": (
                ", ".join(c.providers)
                if c.available
                else '<span style="color:#dc2626">not available</span>'
            ),
        }
        for c in snap.coverage
    ]
    st.markdown(_html_table(cap_rows), unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Aliases**")
        if snap.aliases:
            for a in snap.aliases:
                st.write(f"`{a.alias_name}` → **{a.canonical_name}**")
        else:
            st.caption("None defined.")
    with col_b:
        st.markdown("**Probable duplicates to review**")
        if snap.duplicate_suggestions:
            for d in snap.duplicate_suggestions:
                st.write(f"{d.name_a} ~ {d.name_b}")
                st.code(d.alias_command(), language="bash")
        else:
            st.caption("None found.")

    st.divider()
    for line in snap.attributions:
        st.caption(line)


def _esc(value: object) -> str:
    """Escape a data value (team/player/competition name, etc.) for safe interpolation
    into the HTML we render with unsafe_allow_html. Content-only, so quotes are left as-is
    -- keeps apostrophes readable ("Nott'm Forest") while neutralising & < >."""
    return html.escape(str(value), quote=False)


def _html_table(rows: list[dict]) -> str:
    if not rows:
        return ""
    headers = rows[0].keys()
    head = "".join(f'<th style="text-align:left;padding:4px 12px 4px 0">{h}</th>' for h in headers)
    body = ""
    for row in rows:
        cells = "".join(f'<td style="padding:3px 12px 3px 0">{v}</td>' for v in row.values())
        body += f'<tr style="border-top:1px solid #33333322">{cells}</tr>'
    return (
        f'<table style="width:100%;border-collapse:collapse;font-size:0.9rem">{head}{body}</table>'
    )


@st.cache_data(show_spinner="Simulating season…")
def _cached_analytics(db_path: str, season: str, division: str) -> AnalyticsSnapshot | None:
    # Cached on plain args so switching pages does not re-run the Monte Carlo each time.
    from pathlib import Path

    return analytics_snapshot(Path(db_path), season, division)


@st.cache_data(show_spinner="Testing the model against the closing line…")
def _cached_market_edge(db_path: str, season: str, division: str):
    # The walk-forward value backtest is heavy; cache it per (path, season, division).
    from pathlib import Path

    return market_edge(Path(db_path), season, division)


@st.cache_data(show_spinner="Scoring the model against the market over recent seasons…")
def _cached_forecast_report(db_path: str, division: str, n_seasons: int):
    from pathlib import Path

    return forecast_report(Path(db_path), division, n_seasons=n_seasons)


@st.cache_data(show_spinner="Building the team dossier…")
def _cached_team_dossier(db_path: str, division: str, season: str, team: str):
    from pathlib import Path

    return team_dossier(Path(db_path), division, season, team)


@st.cache_data(show_spinner="Gathering the all-time records…")
def _cached_league_history(db_path: str, division: str):
    from pathlib import Path

    return league_history(Path(db_path), division)


@st.cache_data(show_spinner="Simulating the season…")
def _cached_season_briefing(db_path: str, season: str, division: str):
    from pathlib import Path

    return season_briefing(Path(db_path), season, division)


@st.cache_data(ttl=600, show_spinner="Forecasting the season's fixtures…")
def _cached_fixture_forecasts(live_db: str, analytics_db: str, limit: int):
    """Cached full-season fixtures + forecasts. Forecasting a whole season's ~3k fixtures
    is a few seconds, so it's cached (TTL, and cleared when fixtures are refreshed) rather
    than recomputed on every rerun and filter change."""
    from pathlib import Path

    return fixture_forecasts(Path(live_db), Path(analytics_db), limit=limit)


def _render_season(briefing) -> None:
    st.subheader(
        f"{division_name(briefing.division)} {season_label(briefing.season)}", anchor=False
    )
    st.caption(
        f"{briefing.n_sims:,} Monte Carlo seasons from the recency-weighted model, every team "
        f"playing a full round-robin. A pre-season projection off {season_label(briefing.season)} "
        "strengths — blind to summer transfers and injuries, so treat the extremes with care."
    )
    names = briefing.names
    projs = sorted(briefing.projections, key=lambda p: -p.expected_points)

    fav = projs[0]
    drop = max(projs, key=lambda p: p.relegation_pct)
    tightest = min(projs, key=lambda p: abs(p.title_pct - 0.5))
    k = st.columns(3)
    k[0].metric(
        "Title favourite", names.get(fav.team, fav.team), f"{fav.title_pct:.0%}", border=True
    )
    k[1].metric(
        "Most at risk",
        names.get(drop.team, drop.team),
        f"{drop.relegation_pct:.0%} down",
        border=True,
    )
    k[2].metric(
        "On the bubble",
        names.get(tightest.team, tightest.team),
        f"{tightest.title_pct:.0%} title",
        border=True,
    )

    frame = pl.DataFrame(
        {
            "#": list(range(1, len(projs) + 1)),
            "Team": [names.get(p.team, p.team) for p in projs],
            "xPts": [round(p.expected_points) for p in projs],
            "Title %": [round(100 * p.title_pct, 1) for p in projs],
            f"Top {briefing.top_n} %": [round(100 * p.top_pct, 1) for p in projs],
            "Relegation %": [round(100 * p.relegation_pct, 1) for p in projs],
        }
    ).to_pandas()

    st.dataframe(
        frame,
        width="stretch",
        hide_index=True,
        height=min(760, 60 + 35 * len(projs)),
        column_config={
            "xPts": st.column_config.NumberColumn("xPts", format="%d", help="Expected points"),
            "Title %": st.column_config.ProgressColumn(
                "Title %", format="%.1f%%", min_value=0, max_value=100
            ),
            f"Top {briefing.top_n} %": st.column_config.ProgressColumn(
                f"Top {briefing.top_n} %", format="%.0f%%", min_value=0, max_value=100
            ),
            "Relegation %": st.column_config.ProgressColumn(
                "Relegation %", format="%.0f%%", min_value=0, max_value=100
            ),
        },
    )
    st.caption("xPts = expected final points. Probabilities are Monte Carlo frequencies.")


def _league_season_pickers(
    available: list[tuple[str, str, int]],
    *,
    extra: int = 0,
    key: str = "ls",
    latest_only: bool = False,
) -> tuple:
    """Dependent League + Season selectors laid out as a row on the page.

    Returns (season, division). Pass `extra` to reserve that many trailing columns for a
    page's own controls (the returned tuple then also yields those columns). The season
    list follows the chosen league, newest first with its latest entry tagged. Widget keys
    derive from `key`, so pages that show two pickers at once pass distinct keys.

    With `latest_only`, the season is locked (read-only) to each league's most recent loaded
    season -- for the Predictions tabs, where projecting a finished past season is not a
    forecast.
    """
    by_league: dict[str, list] = {}
    for season, division, _n in available:
        entry = by_league.setdefault(division_name(division), [division, []])
        entry[1].append(season)

    cols = st.columns(2 + extra)
    league = cols[0].selectbox("League", sorted(by_league), key=f"{key}_league")
    division, seasons = by_league[league]
    seasons = sorted(set(seasons), key=season_sort_key, reverse=True)
    if latest_only:
        season = seasons[0]
        cols[1].selectbox("Season", [season_label(season)], disabled=True, key=f"{key}_season")
    else:
        tagged = {
            f"{season_label(s)}{' · latest' if i == 0 else ''}": s for i, s in enumerate(seasons)
        }
        season = tagged[cols[1].selectbox("Season", list(tagged), key=f"{key}_season")]
    if extra:
        return season, division, cols[2:]
    return season, division


def _render_records(records) -> None:
    st.caption(
        "Active runs counted back from each team's most recent match, plus the season's "
        "standout results. Streaks are the records to watch."
    )
    streaks = records.streaks
    on_fire = max(streaks, key=lambda s: s.winning)
    unbeaten = streaks[0]  # already sorted by active unbeaten
    struggling = max(streaks, key=lambda s: s.winless)
    k = st.columns(3)
    k[0].metric("Longest unbeaten", unbeaten.team, f"{unbeaten.unbeaten} games", border=True)
    k[1].metric("Hot streak", on_fire.team, f"{on_fire.winning} wins", border=True)
    k[2].metric("Winless run", struggling.team, f"{struggling.winless} games", border=True)

    st.markdown("**Active streaks**")
    rows = [
        {
            "Team": _esc(s.team),
            "Unbeaten": _streak_span(s.unbeaten, good=True),
            "Winning": _streak_span(s.winning, good=True),
            "Winless": _streak_span(s.winless, good=False),
            "Scoring": _streak_span(s.scoring, good=True),
            "Best unbeaten": f'<span style="color:#8b8b8b">{s.longest_unbeaten}</span>',
        }
        for s in streaks
    ]
    st.markdown(_html_table(rows), unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Biggest wins**")
        st.markdown(_match_record_list(records.biggest_wins, tag="margin"), unsafe_allow_html=True)
    with col2:
        st.markdown("**Highest scoring**")
        st.markdown(
            _match_record_list(records.highest_scoring, tag="goals"), unsafe_allow_html=True
        )


def _title_bar_chart(counts):
    top = counts[:10]
    frame = pl.DataFrame({"team": [t for t, _n in top], "titles": [n for _t, n in top]}).to_pandas()
    base = alt.Chart(frame).encode(
        x=alt.X("titles:Q", title=None, axis=alt.Axis(grid=False, tickMinStep=1)),
        y=alt.Y("team:N", sort="-x", title=None),
    )
    bars = base.mark_bar(color="#16c784", cornerRadiusEnd=4, size=16)
    labels = base.mark_text(align="left", dx=4, color="#8b95a1").encode(text="titles:Q")
    return (bars + labels).properties(height=max(120, 26 * len(top)))


def _render_league_history(hist) -> None:
    st.caption(
        f"All-time across {hist.seasons} loaded seasons ({hist.oldest} to {hist.newest}). "
        "The latest season's leader is provisional — it may not be finished."
    )
    if hist.record_points:
        team, season, pts = hist.record_points
        st.metric("Record points campaign", f"{team} — {pts} pts", season, border=True)
    st.markdown("**Most table-toppers**")
    st.altair_chart(_title_bar_chart(hist.title_counts), width="stretch")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Biggest wins ever**")
        rows = [{"Season": r["Season"], "Result": _esc(r["Result"])} for r in hist.biggest_wins]
        st.markdown(_html_table(rows), unsafe_allow_html=True)
    with col2:
        st.markdown("**Highest scoring ever**")
        rows = [{"Season": r["Season"], "Result": _esc(r["Result"])} for r in hist.highest_scoring]
        st.markdown(_html_table(rows), unsafe_allow_html=True)


def _streak_span(n: int, *, good: bool) -> str:
    if n == 0:
        return '<span style="color:#8b8b8b">0</span>'
    colour = "#16a34a" if good else "#dc2626"
    weight = "700" if n >= 5 else "400"
    return f'<span style="color:{colour};font-weight:{weight}">{n}</span>'


def _match_record_list(matches, *, tag: str) -> str:
    items = ""
    for m in matches:
        extra = f"{m.margin}" if tag == "margin" else f"{m.total} goals"
        items += (
            f'<div style="padding:2px 0;font-size:0.9rem">'
            f"{_esc(m.home)} <b>{_esc(m.score)}</b> {_esc(m.away)} "
            f'<span style="color:#8b8b8b">· {extra}</span></div>'
        )
    return items


def _form_html(form: str) -> str:
    """Colour a W/D/L form string: win green, draw grey, loss red."""
    colour = {"W": "#16a34a", "D": "#8b8b8b", "L": "#dc2626"}
    letters = "".join(
        f'<span style="color:{colour.get(c, "#8b8b8b")};font-weight:700">{c}</span>' for c in form
    )
    return f'<span style="letter-spacing:1px">{letters}</span>'


def _trend_span(trend: float) -> str:
    if trend > 0.15:
        return f'<span style="color:#16a34a">▲ {trend:+.2f}</span>'
    if trend < -0.15:
        return f'<span style="color:#dc2626">▼ {trend:+.2f}</span>'
    return f'<span style="color:#8b8b8b">{trend:+.2f}</span>'


def _render_trends(forms, *, last_n: int) -> None:
    st.caption(
        f"Last {last_n} matches vs the season baseline. ▲ rising, ▼ sliding. "
        "O2.5 = share of a team's games with over 2.5 goals; BTTS = both teams scored."
    )
    hottest, coldest = forms[0], forms[-1]
    goalfest = max(forms, key=lambda f: f.over25_rate)
    k = st.columns(3)
    k[0].metric(
        "Hottest",
        hottest.team,
        f"{hottest.recent_form} · {hottest.recent_ppg:.2f} ppg",
        border=True,
    )
    k[1].metric(
        "Coldest",
        coldest.team,
        f"{coldest.recent_form} · {coldest.recent_ppg:.2f} ppg",
        border=True,
    )
    k[2].metric("Most goals", goalfest.team, f"{goalfest.over25_rate:.0%} over 2.5", border=True)

    rows = [
        {
            "Team": _esc(f.team),
            "Form": _form_html(f.recent_form),
            "Recent ppg": f"{f.recent_ppg:.2f}",
            "Season ppg": f'<span style="color:#8b8b8b">{f.ppg:.2f}</span>',
            "Trend": _trend_span(f.trend),
            "GF-GA": f'<span style="color:#8b8b8b">{f.goals_for}-{f.goals_against}</span>',
            "O2.5": f"{f.over25_rate:.0%}",
            "BTTS": f"{f.btts_rate:.0%}",
        }
        for f in forms
    ]
    st.markdown(_html_table(rows), unsafe_allow_html=True)


def _render_analytics(snap: AnalyticsSnapshot, forms=None, *, last_n: int = 5) -> None:
    st.subheader(f"{division_name(snap.division)} {season_label(snap.season)}", anchor=False)

    form_by_team = {f.team: f.recent_form for f in (forms or [])}
    left, right = st.columns([3, 2])
    with left:
        label = f"**League table** (form = last {last_n})" if form_by_team else "**League table**"
        st.markdown(label)
        rows = []
        for r in snap.table:
            row = {
                "#": r.position,
                "Team": _esc(r.team),
                "P": r.played,
                "GD": f"{r.goal_difference:+d}",
                "Pts": f"<b>{r.points}</b>",
            }
            if form_by_team:
                row["Form"] = _form_html(form_by_team.get(r.team, ""))
            rows.append(row)
        st.markdown(_html_table(rows), unsafe_allow_html=True)

    with right:
        st.markdown("**Title odds** (Monte Carlo, full-season replay)")
        contenders = [p for p in snap.title_odds if p.title_pct >= 0.005][:8]
        if contenders:
            _render_title_odds(contenders, snap.names)
        else:
            st.caption("No clear favourites — an open race.")

        st.markdown("**Elo power ranking**")
        elo_rows = [
            {
                "#": r.position,
                "Team": _esc(snap.names.get(r.team, r.team)),
                "Elo": f"{r.rating:.0f}",
            }
            for r in snap.power[:8]
        ]
        st.markdown(_html_table(elo_rows), unsafe_allow_html=True)


def _render_title_odds(contenders: list, names: dict[str, str]) -> None:
    frame = pl.DataFrame(
        {
            "team": [names.get(p.team, p.team) for p in contenders],
            "pct": [round(p.title_pct * 100, 1) for p in contenders],
        }
    )
    base = alt.Chart(frame.to_pandas()).encode(
        x=alt.X("pct:Q", axis=alt.Axis(title="title %", grid=False)),
        y=alt.Y("team:N", sort="-x", axis=alt.Axis(title=None)),
    )
    bars = base.mark_bar(color="#16a34a", cornerRadiusEnd=4, size=16)
    labels = base.mark_text(align="left", dx=4, color="#8b8b8b").encode(text="pct:Q")
    st.altair_chart((bars + labels).properties(height=max(120, 26 * len(contenders))))


def _xpoints_scatter(rows):
    frame = pl.DataFrame(
        {
            "team": [r.team for r in rows],
            "xP": [r.xpoints for r in rows],
            "pts": [float(r.points) for r in rows],
        }
    ).to_pandas()
    lo = min(frame["xP"].min(), frame["pts"].min()) - 3
    hi = max(frame["xP"].max(), frame["pts"].max()) + 3
    diag = (
        alt.Chart(pl.DataFrame({"x": [lo, hi], "y": [lo, hi]}).to_pandas())
        .mark_line(color="#8b95a1", strokeDash=[4, 4])
        .encode(x="x:Q", y="y:Q")
    )
    pts = (
        alt.Chart(frame)
        .mark_circle(size=90, color="#16c784", opacity=0.8)
        .encode(
            x=alt.X("xP:Q", title="expected points (xP)", scale=alt.Scale(domain=[lo, hi])),
            y=alt.Y("pts:Q", title="actual points", scale=alt.Scale(domain=[lo, hi])),
            tooltip=["team", "pts", "xP"],
        )
    )
    return (diag + pts).properties(height=320)


def _render_underlying(rows) -> None:
    st.caption(
        "Expected points (xP) from shots-on-target chance quality, vs the real table. Above xP "
        "= results are flattering the underlying play (running hot, prone to cool); below = "
        "creating more than the table shows (unlucky, prone to rise). Insight, not a bet."
    )
    over = max(rows, key=lambda r: r.points_diff)
    under = min(rows, key=lambda r: r.points_diff)
    k = st.columns(2)
    k[0].metric("Running hottest", over.team, f"{over.points_diff:+.1f} pts vs xP", border=True)
    k[1].metric("Unluckiest", under.team, f"{under.points_diff:+.1f} pts vs xP", border=True)

    body = []
    for i, r in enumerate(rows, 1):
        diff = r.points_diff
        colour = "#f0a020" if diff > 1.5 else "#3b82f6" if diff < -1.5 else "#8b95a1"
        body.append(
            {
                "#": i,
                "Team": _esc(r.team),
                "Pts": f"<b>{r.points}</b>",
                "xP": f"{r.xpoints:.1f}",
                "Diff": f"<b style='color:{colour}'>{diff:+.1f}</b>",
                "GF": r.goals_for,
                "xGF": f"{r.xgf:.1f}",
                "GA": r.goals_against,
                "xGA": f"{r.xga:.1f}",
            }
        )
    st.markdown(_html_table(body), unsafe_allow_html=True)
    st.altair_chart(_xpoints_scatter(rows), width="stretch")


def _team_strength_bars(d):
    frame = pl.DataFrame(
        {"metric": ["Attack", "Defence"], "value": [d.attack, d.solidity]}
    ).to_pandas()
    bars = (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X("value:Q", title=None, scale=alt.Scale(domainMin=0)),
            y=alt.Y("metric:N", sort=None, title=None),
            color=alt.condition("datum.value >= 1", alt.value("#16c784"), alt.value("#ea3943")),
        )
    )
    rule = (
        alt.Chart(pl.DataFrame({"x": [1.0]}).to_pandas())
        .mark_rule(color="#8b95a1", strokeDash=[4, 4])
        .encode(x="x:Q")
    )
    return (bars + rule).properties(height=110)


def _team_trajectory_chart(trajectory):
    records = []
    for p in trajectory:
        records.append({"matchday": p["matchday"], "value": p["points"], "series": "Points"})
        if "xpoints" in p:
            records.append({"matchday": p["matchday"], "value": p["xpoints"], "series": "Expected"})
    frame = pl.DataFrame(records).to_pandas()
    return (
        alt.Chart(frame)
        .mark_line()
        .encode(
            x=alt.X("matchday:Q", title="matchday"),
            y=alt.Y("value:Q", title="cumulative points"),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(domain=["Points", "Expected"], range=["#16c784", "#8b95a1"]),
                legend=alt.Legend(title=None, orient="top-left"),
            ),
            strokeDash=alt.condition(
                "datum.series == 'Expected'", alt.value([4, 4]), alt.value([1, 0])
            ),
        )
        .properties(height=260)
    )


def _render_team(d) -> None:
    st.subheader(d.team, anchor=False)
    st.caption(f"{division_name(d.division)} {season_label(d.season)} · {d.played} played")
    c = st.columns(5)
    c[0].metric("Position", f"#{d.position}", border=True)
    c[1].metric("Points", d.points, border=True)
    c[2].metric(
        "Goal diff", f"{d.goal_difference:+d}", f"{d.goals_for}-{d.goals_against}", border=True
    )
    c[3].metric("Points/game", f"{d.season_ppg:.2f}", f"{d.trend:+.2f} recent", border=True)
    if d.xpoints is not None:
        c[4].metric(
            "Expected pts",
            f"{d.xpoints:.0f}",
            f"{d.points - d.xpoints:+.1f} vs actual",
            border=True,
        )
    else:
        c[4].metric("Recent form", d.recent_form or "—", border=True)

    left, right = st.columns([3, 2])
    with left:
        st.markdown(f"**Recent form** &nbsp; {_form_html(d.recent_form)}", unsafe_allow_html=True)
        if d.xpoints is not None:
            diff = d.points - d.xpoints
            verdict = (
                "overperforming their chances (running hot)"
                if diff > 1.5
                else "underperforming — creating more than the table shows (unlucky)"
                if diff < -1.5
                else "roughly in line with their underlying numbers"
            )
            colour = "#f0a020" if diff > 1.5 else "#3b82f6" if diff < -1.5 else "#8b95a1"
            st.markdown(
                f"<div style='margin:6px 0'>Underlying: <b>{d.points}</b> pts vs "
                f"<b style='color:{colour}'>{d.xpoints:.1f} expected</b> ({diff:+.1f}) "
                f"— {verdict}. <span style='color:{_MUTE}'>"
                f"xG for {d.xgf:.1f}, against {d.xga:.1f}.</span></div>",
                unsafe_allow_html=True,
            )
        st.caption(
            f"Active streaks — unbeaten {d.unbeaten} · winning {d.winning} · scoring {d.scoring}"
        )
        st.markdown("**Recent results**")
        rows = [
            {
                "": r["venue"],
                "Opponent": _esc(r["opponent"]),
                "Score": r["score"],
                "Result": _form_html(r["result"]),
            }
            for r in d.recent
        ]
        st.markdown(_html_table(rows), unsafe_allow_html=True)
    with right:
        st.markdown("**Strength vs league** (1.0 = average)")
        st.altair_chart(_team_strength_bars(d), width="stretch")

    st.markdown("**Season trajectory** — points earned vs deserved (the gap is luck)")
    st.altair_chart(_team_trajectory_chart(d.trajectory), width="stretch")


def _render_shot_map(data) -> None:
    st.subheader(data.label, anchor=False)
    st.caption("StatsBomb event data. Circle size ∝ xG; filled = goal. Both teams attack →")

    cols = st.columns(len(data.team_xg) or 1)
    for col, row in zip(cols, data.team_xg, strict=False):
        col.metric(f"{row.name} xG", f"{row.xg:.2f}", f"{row.goals} goals", border=True)

    tab_race, tab_map, tab_log = st.tabs(["xG timeline", "Shot map", "Shot log"])

    with tab_race:
        st.altair_chart(_xg_race_chart(data.timeline), width="stretch")
        st.caption(
            "Cumulative xG over the match — the 'xG race'. A team above on xG but behind on "
            "goals was wasteful or unlucky; steps are shots, dots are goals."
        )
    with tab_map:
        frame = pl.DataFrame(
            {
                "x": [s["x"] for s in data.shots if s["x"] is not None],
                "y": [s["y"] for s in data.shots if s["x"] is not None],
                "xg": [s["xg"] for s in data.shots if s["x"] is not None],
                "team": [s["team"] for s in data.shots if s["x"] is not None],
                "player": [s["player"] for s in data.shots if s["x"] is not None],
                "outcome": [s["outcome"] for s in data.shots if s["x"] is not None],
                "goal": [bool(s["is_goal"]) for s in data.shots if s["x"] is not None],
            }
        ).to_pandas()
        st.altair_chart(_shot_chart(frame), width="stretch")
    with tab_log:
        log = (
            pl.DataFrame(
                {
                    "Min": [s["minute"] for s in data.shots],
                    "Team": [s["team"] for s in data.shots],
                    "Player": [s["player"] for s in data.shots],
                    "xG": [round(s["xg"], 2) for s in data.shots],
                    "Outcome": [s["outcome"] for s in data.shots],
                }
            )
            .sort("Min")
            .to_pandas()
        )
        st.dataframe(
            log,
            width="stretch",
            hide_index=True,
            column_config={"xG": st.column_config.NumberColumn("xG", format="%.2f")},
            height=min(520, 60 + 32 * len(data.shots)),
        )
    st.caption(ATTRIBUTION_STATSBOMB)


def _xg_race_chart(timeline: list[dict]):
    frame = pl.DataFrame(timeline).to_pandas()
    line = (
        alt.Chart(frame)
        .mark_line(interpolate="step-after", strokeWidth=2)
        .encode(
            x=alt.X(
                "minute:Q",
                title="Minute",
                scale=alt.Scale(domain=[0, max(95, frame["minute"].max())]),
            ),
            y=alt.Y("cum_xg:Q", title="Cumulative xG"),
            color=alt.Color("team:N", legend=alt.Legend(title=None, orient="top")),
        )
    )
    goals = (
        alt.Chart(frame[frame["is_goal"]])
        .mark_point(size=90, filled=True, opacity=0.9)
        .encode(
            x="minute:Q",
            y="cum_xg:Q",
            color=alt.Color("team:N", legend=None),
            tooltip=["player", "team", alt.Tooltip("minute", title="min")],
        )
    )
    return (line + goals).properties(height=300).configure_view(strokeWidth=0)


def _shot_chart(frame):
    # StatsBomb frame is 120x80; show the attacking half (x 60-120) where shots live.
    # Recessive pitch lines, then shots as circles sized by xG, coloured by team, with
    # goals drawn solid and everything else hollow so identity never rests on colour.
    lines = pl.DataFrame(
        {
            "x": [60, 120, 120, 60, 60, 102, 102, 120, 114, 114, 120],
            "y": [0, 0, 80, 80, 0, 18, 62, 62, 30, 50, 50],
            "seg": [0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2],
        }
    ).to_pandas()
    pitch = (
        alt.Chart(lines)
        .mark_line(color="#9aa0a6", strokeWidth=1)
        .encode(
            x=alt.X("x", scale=alt.Scale(domain=[58, 122]), axis=None),
            y=alt.Y("y", scale=alt.Scale(domain=[-2, 82]), axis=None),
            detail="seg",
            order="seg",
        )
    )
    base = alt.Chart(frame).encode(
        x=alt.X("x", scale=alt.Scale(domain=[58, 122]), axis=None),
        y=alt.Y("y", scale=alt.Scale(domain=[-2, 82]), axis=None),
        size=alt.Size("xg", scale=alt.Scale(range=[30, 700]), legend=alt.Legend(title="xG")),
        color=alt.Color("team", legend=alt.Legend(title=None)),
        tooltip=["player", "team", "outcome", alt.Tooltip("xg", format=".2f")],
    )
    goals = base.transform_filter(alt.datum.goal).mark_circle(opacity=0.9)
    misses = base.transform_filter(~alt.datum.goal).mark_point(filled=False, strokeWidth=1.5)
    return (pitch + misses + goals).properties(height=380).configure_view(strokeWidth=0)


# --- Kalshi-style market rendering: a probability is shown as a cent "price" (100c =
# certain), the leading outcome in Yes-green, fill bars proportional to the price, and
# tabular numerals so columns of prices line up like an order book. ---
_YES = "#16c784"
_MUTE = "#8b95a1"


def _cents(p: float) -> str:
    return f"{round(p * 100)}¢"


def _mkt_title(title: str) -> str:
    return (
        f"<div style='color:{_MUTE};font-size:0.72rem;text-transform:uppercase;"
        f"letter-spacing:0.6px;margin:2px 0 4px'>{title}</div>"
    )


def _market_table(title: str, markets, *, odds: bool = True) -> str:
    """A market as Kalshi-style price rows: name, fill bar, cent price (leader green)."""
    if not markets:
        return ""
    mx = max(m.probability for m in markets)
    rows = []
    for m in markets:
        lead = m.probability >= mx
        tint = "rgba(22,199,132,0.16)" if lead else "rgba(139,149,161,0.13)"
        fill = round(m.probability * 100)
        bar = f"linear-gradient(90deg,{tint} {fill}%,rgba(255,255,255,0.03) {fill}%)"
        tail = (
            f"<span style='color:{_MUTE};font-size:0.78rem;margin-left:8px'>"
            f"{m.fair_odds:.2f}</span>"
            if odds
            else ""
        )
        rows.append(
            f"<div style='display:flex;align-items:center;justify-content:space-between;"
            f"background:{bar};border-radius:6px;padding:5px 10px;margin:3px 0'>"
            f"<span>{_esc(m.name)}</span>"
            f"<span style='font-variant-numeric:tabular-nums'>"
            f"<b style='color:{_YES if lead else _MUTE}'>"
            f"{_cents(m.probability)}</b>{tail}</span></div>"
        )
    return f"<div style='margin-bottom:12px'>{_mkt_title(title)}{''.join(rows)}</div>"


def _price_tiles(markets) -> str:
    """Headline outcomes as big Kalshi price tiles (the 1X2 result)."""
    mx = max(m.probability for m in markets)
    tiles = []
    for m in markets:
        lead = m.probability >= mx
        border = _YES if lead else "#262b36"
        bg = "rgba(22,199,132,0.08)" if lead else "rgba(255,255,255,0.02)"
        color = _YES if lead else "#e6e9ef"
        tiles.append(
            f"<div style='flex:1;border:1px solid {border};border-radius:10px;"
            f"padding:12px 14px;background:{bg}'>"
            f"<div style='color:{_MUTE};font-size:0.82rem;white-space:nowrap;overflow:hidden;"
            f"text-overflow:ellipsis'>{_esc(m.name)}</div>"
            f"<div style='font-size:1.9rem;font-weight:700;line-height:1.15;"
            f"font-variant-numeric:tabular-nums;color:{color}'>{_cents(m.probability)}</div>"
            f"<div style='color:{_MUTE};font-size:0.72rem'>{m.fair_odds:.2f} fair</div></div>"
        )
    return f"<div style='display:flex;gap:10px;margin:4px 0 10px'>{''.join(tiles)}</div>"


def _ou_block(over_unders) -> str:
    """Over/Under lines as Over/Under (Yes/No) price rows."""
    rows = []
    for ou in over_unders:
        over_lead = ou.over >= ou.under
        rows.append(
            f"<div style='display:flex;justify-content:space-between;padding:5px 10px;margin:3px 0;"
            f"background:rgba(255,255,255,0.03);border-radius:6px'>"
            f"<span>{ou.line} goals</span>"
            f"<span style='font-variant-numeric:tabular-nums'>"
            f"<b style='color:{_YES if over_lead else _MUTE}'>O&nbsp;{_cents(ou.over)}</b>"
            f"&nbsp;&nbsp;<b style='color:{_YES if not over_lead else _MUTE}'>"
            f"U&nbsp;{_cents(ou.under)}</b>"
            f"</span></div>"
        )
    return f"<div style='margin-bottom:12px'>{_mkt_title('Total goals lines')}{''.join(rows)}</div>"


def _render_forecast(slate) -> None:
    st.markdown(f"#### {slate.home}  ·  {slate.away}")
    st.markdown(_price_tiles(slate.result), unsafe_allow_html=True)
    x, y, p = slate.most_likely_score
    c = st.columns(3)
    c[0].metric(f"{slate.home} xG", f"{slate.home_expected:.2f}", border=True)
    c[1].metric(f"{slate.away} xG", f"{slate.away_expected:.2f}", border=True)
    c[2].metric("Likely score", f"{x}-{y}", _cents(p), border=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(_market_table("Double chance", slate.double_chance), unsafe_allow_html=True)
        st.markdown(_market_table("Both teams to score", slate.btts), unsafe_allow_html=True)
    with col2:
        st.markdown(_ou_block(slate.over_under), unsafe_allow_html=True)
        st.markdown(_market_table("Clean sheet", slate.clean_sheet), unsafe_allow_html=True)
        st.markdown(_market_table("Win to nil", slate.win_to_nil), unsafe_allow_html=True)
    with col3:
        st.markdown(
            _market_table("Total goals", slate.total_goals, odds=False), unsafe_allow_html=True
        )
        cs_rows = "".join(
            f"<div style='display:flex;justify-content:space-between;padding:3px 10px;margin:2px 0;"
            f"background:rgba(255,255,255,0.03);border-radius:6px'><span>{x}-{y}</span>"
            f"<b style='font-variant-numeric:tabular-nums'>{_cents(p)}</b></div>"
            for x, y, p in slate.correct_scores
        )
        st.markdown(
            f"<div style='margin-bottom:12px'>{_mkt_title('Correct score')}{cs_rows}</div>",
            unsafe_allow_html=True,
        )
    st.caption(
        "Prices are the model's fair probability in cents (100¢ = certain), no bookmaker "
        "margin. Shots-on-target model over ~3 recent seasons, re-fit as results land. "
        "Directional, not advice."
    )


def _strength_bars(exp):
    rows = [
        (f"{exp.home} attack", exp.home_factor.attack),
        (f"{exp.home} defence", exp.home_factor.solidity),
        (f"{exp.away} attack", exp.away_factor.attack),
        (f"{exp.away} defence", exp.away_factor.solidity),
    ]
    frame = pl.DataFrame({"label": [r[0] for r in rows], "value": [r[1] for r in rows]}).to_pandas()
    bars = (
        alt.Chart(frame)
        .mark_bar()
        .encode(
            x=alt.X(
                "value:Q",
                title="rating vs league average (dashed = 1.0)",
                scale=alt.Scale(domainMin=0),
            ),
            y=alt.Y("label:N", sort=None, title=None),
            color=alt.condition("datum.value >= 1", alt.value("#16c784"), alt.value("#ea3943")),
        )
    )
    rule = (
        alt.Chart(pl.DataFrame({"x": [1.0]}).to_pandas())
        .mark_rule(color="#8b95a1", strokeDash=[4, 4])
        .encode(x="x:Q")
    )
    return (bars + rule).properties(height=150)


def _render_forecast_explanation(exp) -> None:
    """The honest 'why': attribute the forecast to team ratings and flag how much data backs it."""
    st.markdown("**Why this forecast**")
    conf_colour = {"High": _YES, "Moderate": "#f0a020", "Low": "#ea3943"}[exp.confidence]
    st.markdown(
        f"<span style='color:{_MUTE}'>Confidence </span>"
        f"<b style='color:{conf_colour}'>{exp.confidence}</b>"
        f"<span style='color:{_MUTE}'> — ratings from {exp.home_factor.games} "
        f"({_esc(exp.home)}) and {exp.away_factor.games} ({_esc(exp.away)}) recent matches.</span>",
        unsafe_allow_html=True,
    )
    if exp.confidence == "Low":
        st.caption(
            "⚠ Limited data on one side (newly promoted or early season), so the forecast "
            "leans on the league average — treat it with extra caution."
        )
    st.markdown(f"<span style='color:#e6e9ef'>{_esc(exp.summary)}</span>", unsafe_allow_html=True)
    home_leak = 1.0 / max(exp.away_factor.solidity, 0.05)
    away_leak = 1.0 / max(exp.home_factor.solidity, 0.05)
    st.markdown(
        f"<div style='font-size:0.9rem;color:{_MUTE};margin:6px 0'>"
        f"{_esc(exp.home)} xG = {exp.league_home_avg:.2f} <i>home baseline</i> &times; "
        f"{exp.home_factor.attack:.2f} <i>attack</i> &times; {home_leak:.2f} "
        f"<i>opp. leakiness</i> = <b style='color:#e6e9ef'>{exp.home_xg:.2f}</b><br>"
        f"{_esc(exp.away)} xG = {exp.league_away_avg:.2f} <i>away baseline</i> &times; "
        f"{exp.away_factor.attack:.2f} <i>attack</i> &times; {away_leak:.2f} "
        f"<i>opp. leakiness</i> = <b style='color:#e6e9ef'>{exp.away_xg:.2f}</b></div>",
        unsafe_allow_html=True,
    )
    st.altair_chart(_strength_bars(exp), width="stretch")
    st.caption("Attack and defence are each team's rate vs the league average (higher is better).")


def _render_ev_calculator(slate) -> None:
    """Model probabilities vs the odds a bookmaker is actually offering -> edge, EV, Kelly."""
    from soccer.models.value import expected_value, implied_probabilities, kelly_fraction, overround

    st.markdown("**Value calculator** — enter the odds you can get")
    st.caption(
        "There is no free feed of odds for upcoming matches, so bring your bookmaker's "
        "decimal odds. Edge compares the model to the vig-free market; treat it sceptically."
    )
    result = {m.name: m.probability for m in slate.result}
    names = [m.name for m in slate.result]  # [home, draw, away]
    defaults = [round(1 / result[n], 2) if result[n] else 2.0 for n in names]

    cols = st.columns(3)
    odds = [
        cols[i].number_input(f"{n} odds", min_value=1.01, value=float(defaults[i]), step=0.05)
        for i, n in enumerate(names)
    ]
    market = implied_probabilities(*odds)

    body = ""
    for i, name in enumerate(names):
        model_p = result[name]
        ev = expected_value(model_p, odds[i])
        kelly = kelly_fraction(model_p, odds[i])
        colour = "#16a34a" if ev > 0 else "#8b8b8b"
        cells = [
            name,
            f"{model_p:.0%}",
            f"{market[i]:.0%}",
            f'<span style="color:{colour}">{ev * 100:+.1f}%</span>',
            f'<span style="color:{colour}">{kelly * 100:.1f}%</span>' if kelly > 0 else "—",
        ]
        tds = "".join(f'<td style="padding:3px 14px 3px 0">{c}</td>' for c in cells)
        body += f'<tr style="border-top:1px solid #33333322">{tds}</tr>'
    head = "".join(
        f'<th style="text-align:left;padding:4px 14px 4px 0">{h}</th>'
        for h in ["Outcome", "Model", "Market", "Edge (EV)", "Kelly"]
    )
    st.markdown(
        f'<table style="font-size:0.88rem;border-collapse:collapse">{head}{body}</table>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"Bookmaker margin (overround): {overround(*odds) * 100:.1f}%. "
        "Edge = model probability * odds - 1. Kelly = fraction of bankroll at that edge."
    )


def _render_market_edge(report) -> None:
    """Honest 'does the model beat the closing line' summary for the league."""
    beats = report.beats_market
    verdict = "beats" if beats else "does not beat"
    colour = "#16a34a" if beats else "#dc2626"
    yld_colour = "#16a34a" if report.yield_pct > 0 else "#dc2626"
    st.markdown(
        f"<b>Model vs market</b> (closing line, {report.n_matches} past matches) — "
        f'the model <span style="color:{colour}">{verdict}</span> the market on log loss '
        f"({report.model_log_loss:.3f} vs {report.market_log_loss:.3f}); flat-staking its "
        f'positive-edge picks yielded <span style="color:{yld_colour}">'
        f"{report.yield_pct:+.1f}%</span> over {report.n_bets} bets.",
        unsafe_allow_html=True,
    )


def _blend_curve_chart(curve):
    frame = pl.DataFrame(
        {"weight": [p.weight for p in curve], "log_loss": [p.log_loss for p in curve]}
    ).to_pandas()
    best = min(curve, key=lambda p: p.log_loss)
    line = (
        alt.Chart(frame)
        .mark_line(color="#3b82f6")
        .encode(
            x=alt.X(
                "weight:Q",
                title="model weight (0 = market, 1 = model)",
                axis=alt.Axis(format="%"),
            ),
            y=alt.Y("log_loss:Q", title="log loss", scale=alt.Scale(zero=False)),
        )
    )
    mark = (
        alt.Chart(pl.DataFrame({"weight": [best.weight], "log_loss": [best.log_loss]}).to_pandas())
        .mark_point(color="#16c784", size=90, filled=True)
        .encode(x="weight:Q", y="log_loss:Q")
    )
    return (line + mark).properties(height=240)


def _calibration_chart(bins):
    frame = pl.DataFrame(
        {
            "predicted": [b.mean_predicted for b in bins],
            "observed": [b.observed_rate for b in bins],
            "count": [b.count for b in bins],
        }
    ).to_pandas()
    diag = (
        alt.Chart(pl.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]}).to_pandas())
        .mark_line(strokeDash=[4, 4], color="#8b95a1")
        .encode(x="x:Q", y="y:Q")
    )
    pts = (
        alt.Chart(frame)
        .mark_circle(color="#16c784", opacity=0.8)
        .encode(
            x=alt.X(
                "predicted:Q",
                title="predicted home-win",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%"),
            ),
            y=alt.Y(
                "observed:Q",
                title="actual home-win",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%"),
            ),
            size=alt.Size("count:Q", legend=None),
        )
    )
    return (diag + pts).properties(height=240)


def _render_report_card(report, division: str) -> None:
    """Honest model-vs-market scorecard: proper scores, the blend curve, and calibration."""
    st.caption(
        f"{division_name(division)} — each match forecast from a model fit only on earlier "
        f"matches ({report.n} scored), then judged against the vig-free closing line. Lower is "
        "better. RPS is the standard 1X2 metric; the blend is a log-opinion pool of the two."
    )
    metrics = [
        ("Baseline (base rates)", report.baseline),
        ("Model (goals only)", report.model_goals),
        ("Model (shots-on-target)", report.model),
        ("Market (closing line)", report.market),
        (f"Blend ({report.best_weight:.0%} model)", report.blend),
    ]
    best_ll = min(s.log_loss for _n, s in metrics)
    best_rps = min(s.rps for _n, s in metrics)
    rows = []
    for name, s in metrics:
        ll = (
            f"<b style='color:{_YES}'>{s.log_loss:.4f}</b>"
            if abs(s.log_loss - best_ll) < 1e-9
            else f"{s.log_loss:.4f}"
        )
        rp = (
            f"<b style='color:{_YES}'>{s.rps:.4f}</b>"
            if abs(s.rps - best_rps) < 1e-9
            else f"{s.rps:.4f}"
        )
        rows.append(
            {"Forecaster": _esc(name), "Log loss": ll, "RPS": rp, "Brier": f"{s.brier:.4f}"}
        )
    st.markdown(_html_table(rows), unsafe_allow_html=True)

    w = report.best_weight
    verdict = (
        "the model adds essentially nothing to the market — the closing line is the benchmark"
        if w <= 0.1
        else "the model carries real weight alongside the market"
        if w >= 0.4
        else "the model adds a little on top of the market"
    )
    st.info(
        f"Best blend: **{w:.0%} model / {1 - w:.0%} market** over {report.n} matches — {verdict}.",
        icon=":material/insights:",
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Blend-weight curve** — log loss as the mix shifts model↔market")
        st.altair_chart(_blend_curve_chart(report.blend_curve), width="stretch")
    with c2:
        st.markdown("**Calibration** — model home-win probability vs how often it happened")
        st.altair_chart(_calibration_chart(report.model_calibration), width="stretch")

    st.markdown(
        "**Biggest model↔market disagreements** — the market is usually right, so these are "
        "the model's boldest departures, not value picks"
    )
    labels = ("Home", "Draw", "Away")
    div_rows = [
        {
            "Match": f"{_esc(d.home)} v {_esc(d.away)}",
            "Model H/D/A": f"{d.model[0]:.0%} / {d.model[1]:.0%} / {d.model[2]:.0%}",
            "Market H/D/A": f"{d.market[0]:.0%} / {d.market[1]:.0%} / {d.market[2]:.0%}",
            "Result": labels[d.actual],
        }
        for d in report.divergences
    ]
    st.markdown(_html_table(div_rows), unsafe_allow_html=True)
    st.caption(
        "Closing 1X2 odds from football-data.co.uk (Pinnacle preferred), de-vigged. Model = "
        "Poisson fit on a shots-on-target expected-goals blend (the app's live forecast model)."
    )


def _render_players(rows) -> None:
    st.caption(
        "StatsBomb shots across all ingested matches. G-xG > 0 = clinical finishing. "
        "Points above the diagonal outscored their chances."
    )
    c = st.columns(3)
    c[0].metric("Players", len(rows), border=True)
    c[1].metric("Top xG", f"{rows[0].xg:.1f}", rows[0].player.split()[-1], border=True)
    clinical = max(rows, key=lambda r: r.xg_diff)
    c[2].metric(
        "Best finisher (G-xG)", f"{clinical.xg_diff:+.1f}", clinical.player.split()[-1], border=True
    )

    frame = pl.DataFrame(
        {
            "player": [r.player for r in rows],
            "team": [r.team for r in rows],
            "xg": [round(r.xg, 2) for r in rows],
            "goals": [r.goals for r in rows],
            "shots": [r.shots for r in rows],
        }
    ).to_pandas()

    scatter_base = alt.Chart(frame).encode(
        x=alt.X("xg:Q", title="Expected goals (xG)"),
        y=alt.Y("goals:Q", title="Goals"),
        tooltip=["player", "team", "xg", "goals", "shots"],
    )
    top_val = max(float(frame["goals"].max()), float(frame["xg"].max()))
    diagonal = (
        alt.Chart(pl.DataFrame({"v": [0.0, top_val]}).to_pandas())
        .mark_line(color="#9aa0a6", strokeDash=[4, 4])
        .encode(x="v", y="v")
    )
    points = scatter_base.mark_circle(size=90, color="#2563eb", opacity=0.75)
    st.altair_chart((diagonal + points).properties(height=360), width="stretch")

    table_rows = [
        {
            "#": i,
            "Player": _esc(r.player),
            "Team": f'<span style="color:#8b8b8b">{_esc(r.team)}</span>',
            "xG": f"{r.xg:.1f}",
            "npxG": f"{r.npxg:.1f}",
            "G": f"<b>{r.goals}</b>",
            "Sh": str(r.shots),
            "G-xG": _diff_span(r.xg_diff),
        }
        for i, r in enumerate(rows, 1)
    ]
    st.markdown(_html_table(table_rows), unsafe_allow_html=True)
    st.caption(ATTRIBUTION_STATSBOMB)


def _diff_span(diff: float) -> str:
    colour = "#16a34a" if diff > 0.5 else "#dc2626" if diff < -0.5 else "#8b8b8b"
    return f'<span style="color:{colour}">{diff:+.1f}</span>'


# Leaderboard columns: (header, profile attribute, is a per-90 rate, number format).
_LEADER_COLUMNS = [
    ("G", "goals", True, "%.2f"),
    ("xG", "xg", True, "%.2f"),
    ("npxG", "npxg", True, "%.2f"),
    ("Ast", "assists", True, "%.2f"),
    ("xA", "xa", True, "%.2f"),
    ("Key pass", "key_passes", True, "%.2f"),
    ("Prg pass", "progressive_passes", True, "%.1f"),
    ("Prg carry", "progressive_carries", True, "%.1f"),
    ("Dribbles", "dribbles_completed", True, "%.1f"),
    ("Tackles", "tackles", True, "%.1f"),
    ("Int", "interceptions", True, "%.1f"),
    ("Recov", "ball_recoveries", True, "%.1f"),
]

_RANK_OPTIONS = {
    "Goal contributions": "contributions",
    "Goals": "goals",
    "Expected (xG+xA)": "xgxa",
    "Assists": "assists",
    "xA": "xa",
    "Progression": "progressive",
    "Defending": "defensive",
    "Passes": "passes",
    "Minutes": "minutes",
}


def _render_player_leaderboard(profiles, *, per90: bool, pool_label: str = "") -> None:
    mode = "per 90 minutes" if per90 else "totals"
    scope = f"{pool_label} · " if pool_label and pool_label != "All competitions" else ""
    st.caption(
        f"{scope}StatsBomb event data across all ingested matches, shown as {mode}. "
        "Sort any column by clicking its header."
    )

    top = profiles[0]
    kpi = st.columns(4)
    kpi[0].metric("Players", len(profiles), border=True)
    kpi[1].metric("Most G+A", top.goal_contributions, top.player.split()[-1], border=True)
    clinical = max(profiles, key=lambda p: p.xg_diff)
    kpi[2].metric(
        "Best finisher (G-xG)", f"{clinical.xg_diff:+.1f}", clinical.player.split()[-1], border=True
    )
    creator = max(profiles, key=lambda p: p.xa)
    kpi[3].metric("Top xA", f"{creator.xa:.1f}", creator.player.split()[-1], border=True)

    data = {
        "Player": [p.player for p in profiles],
        "Team": [p.team for p in profiles],
        "Pos": [(p.position or "")[:14] for p in profiles],
        "Min": [p.minutes for p in profiles],
    }
    for header, attr, _is_rate, _fmt in _LEADER_COLUMNS:
        data[header] = [
            round(p.per90(getattr(p, attr)), 2) if per90 else getattr(p, attr) for p in profiles
        ]
    data["Pass %"] = [round(p.pass_pct, 1) for p in profiles]
    frame = pl.DataFrame(data).to_pandas()

    column_config = {
        "Min": st.column_config.NumberColumn("Min", format="%d", help="Minutes played"),
        "Pass %": st.column_config.NumberColumn("Pass %", format="%.1f%%"),
    }
    for header, _attr, _is_rate, fmt in _LEADER_COLUMNS:
        column_config[header] = st.column_config.NumberColumn(header, format=fmt if per90 else "%d")

    st.dataframe(
        frame,
        width="stretch",
        hide_index=True,
        column_config=column_config,
        height=min(560, 60 + 35 * len(profiles)),
    )
    st.caption(ATTRIBUTION_STATSBOMB)


_CATEGORY_COLOURS = {
    "Attacking": "#dc2626",
    "Possession": "#2563eb",
    "Defending": "#16a34a",
}


def _percentile_bars_chart(rows: list[dict]):
    """FBref-style percentile fingerprint from [{metric, category, percentile, value}, ...]."""
    frame = pl.DataFrame(rows).to_pandas()
    order = [r["metric"] for r in rows]
    domain = list(_CATEGORY_COLOURS)
    rng = [_CATEGORY_COLOURS[c] for c in domain]
    bars = (
        alt.Chart(frame)
        .mark_bar(cornerRadiusEnd=3, height=14)
        .encode(
            x=alt.X("percentile:Q", scale=alt.Scale(domain=[0, 100]), title="Percentile"),
            y=alt.Y("metric:N", sort=order, title=None),
            color=alt.Color(
                "category:N",
                scale=alt.Scale(domain=domain, range=rng),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                "category",
                "metric",
                alt.Tooltip("value", title="per 90"),
                alt.Tooltip("percentile", title="pct"),
            ],
        )
    )
    labels = bars.mark_text(align="left", dx=3, color="#8b8b8b").encode(text="value:Q")
    midline = (
        alt.Chart(pl.DataFrame({"v": [50]}).to_pandas())
        .mark_rule(color="#9aa0a6", strokeDash=[3, 3])
        .encode(x="v:Q")
    )
    return (bars + labels + midline).properties(height=22 * len(rows) + 10)


def _render_player_profile(profile, percentiles, *, pool_label: str = "") -> None:
    pos = f" · {profile.position}" if profile.position else ""
    st.subheader(f"{profile.player}", anchor=False)
    st.caption(f"{profile.team}{pos} · {profile.matches} matches · {profile.minutes} minutes")

    row = st.columns(5)
    row[0].metric("Goals", profile.goals, f"xG {profile.xg:.1f}", border=True)
    row[1].metric("Assists", profile.assists, f"xA {profile.xa:.1f}", border=True)
    row[2].metric("Shots", profile.shots, f"{profile.per90(profile.shots):.1f}/90", border=True)
    row[3].metric("Pass %", f"{profile.pass_pct:.0f}%", f"{profile.passes} att", border=True)
    row[4].metric(
        "Prog. actions", profile.progressive_passes + profile.progressive_carries, border=True
    )

    if not percentiles:
        st.info("Not enough minutes for a percentile fingerprint at this threshold.")
        return

    pool_note = pool_label if pool_label and pool_label != "All competitions" else "all loaded"
    st.markdown(f"**Percentile rank** — vs {pool_note} players (per 90)")
    rows = [
        {
            "metric": mp.label,
            "category": mp.category,
            "percentile": mp.percentile,
            "value": mp.value,
        }
        for mp in percentiles
    ]
    st.altair_chart(_percentile_bars_chart(rows), width="stretch")
    st.caption(
        "Bar = percentile within the pool; number = the player's per-90 value. Dashed line "
        "is the median (50th). " + ATTRIBUTION_STATSBOMB
    )


def _render_fixtures(fixtures) -> None:
    comps = sorted({f.competition for f in fixtures})
    if len(comps) > 1:
        choice = st.selectbox(
            "Competition", ["All competitions", *comps], key="fixtures_competition"
        )
        if choice != "All competitions":
            fixtures = [f for f in fixtures if f.competition == choice]
    forecastable = [f for f in fixtures if f.slate is not None]
    uncovered = [f for f in fixtures if f.slate is None]
    st.caption(
        f"{len(forecastable)} of {len(fixtures)} upcoming matches forecast from last "
        "season's Dixon-Coles model. Predictions are a preseason projection, directional."
    )
    if not forecastable:
        st.info(
            "No forecastable fixtures here yet. Go to **Home** to update fixtures and add a "
            "league, then check back.",
            icon=":material/event_upcoming:",
        )
        _render_uncovered_fixtures(uncovered)
        return

    hdr = ["Date (UTC)", "Competition", "Match", "Pred", "1", "X", "2", "O2.5", "BTTS", "Favourite"]
    body = ""
    for f in forecastable:
        s = f.slate
        x, y, _ = s.most_likely_score
        home_p, draw_p, away_p = (m.probability for m in s.result)
        over25 = next(o for o in s.over_under if o.line == 2.5).over
        btts_yes = next(m.probability for m in s.btts if m.name == "Yes")
        fav = f.home if home_p >= away_p else f.away
        cells = [
            f.kickoff_utc.strftime("%m-%d %H:%M"),
            f'<span style="color:#8b8b8b">{_esc(f.competition)}</span>',
            f"{_esc(f.home)} v {_esc(f.away)}",
            f"<b>{x}-{y}</b>",
            _pct_cell(home_p, home_p >= max(draw_p, away_p)),
            _pct_cell(draw_p, draw_p >= max(home_p, away_p)),
            _pct_cell(away_p, away_p >= max(home_p, draw_p)),
            _pct_cell(over25, over25 >= 0.5),
            _pct_cell(btts_yes, btts_yes >= 0.5),
            f'<span style="color:{_YES}">{_esc(fav)}</span>',
        ]
        tds = "".join(f'<td style="padding:3px 12px 3px 0">{c}</td>' for c in cells)
        body += f'<tr style="border-top:1px solid #33333322">{tds}</tr>'
    head = "".join(f'<th style="text-align:left;padding:4px 12px 4px 0">{h}</th>' for h in hdr)
    table_style = "width:100%;border-collapse:collapse;font-size:0.86rem"
    st.markdown(
        f'<table style="{table_style}">{head}{body}</table>',
        unsafe_allow_html=True,
    )
    st.caption("1/X/2 = home / draw / away win. O2.5 = over 2.5 goals. BTTS = both teams score.")
    _render_uncovered_fixtures(uncovered)


def _render_uncovered_fixtures(uncovered) -> None:
    if not uncovered:
        return
    with st.expander(f"{len(uncovered)} upcoming fixtures without a forecast"):
        st.caption(
            "No loaded model covers these -- a promoted team absent from last season, "
            "or a competition without loaded history. Listed honestly, not hidden."
        )
        st.markdown(
            "".join(
                f'<div style="font-size:0.85rem;padding:1px 0">'
                f'<span style="color:#8b8b8b">{f.kickoff_utc.strftime("%m-%d")} · '
                f"{_esc(f.competition)}</span> &nbsp; {_esc(f.home)} v {_esc(f.away)}</div>"
                for f in uncovered
            ),
            unsafe_allow_html=True,
        )


def _pct_cell(p: float, is_max: bool) -> str:
    """Outcome probability as a Kalshi cent price; leader in Yes-green, tabular figures."""
    color = _YES if is_max else _MUTE
    return f"<b style='color:{color};font-variant-numeric:tabular-nums'>{_cents(p)}</b>"


# Per-page identity: a Material Symbol icon and a one-line description, for a consistent
# header on every page and cleaner navigation.
# Navigation: (routing key, sidebar label, icon, one-line description). The routing keys
# stay stable so the page dispatch is untouched; only the plain-English labels and the
# captions the user reads change. Ordered as a journey: start -> this season -> explore.
_NAV = [
    ("Home", "Home", ":material/home:", "Set up and quick actions"),
    ("Assistant", "Ask a question", ":material/chat:", "Chat about your data in plain English"),
    ("Live Centre", "Live scores", ":material/bolt:", "Today's and recent results"),
    ("Predictor", "Predictions", ":material/insights:", "Fixtures, matchups and season odds"),
    ("Analytics", "League tables", ":material/table_chart:", "Standings, form and title odds"),
    ("Team", "Teams", ":material/shield:", "One club, everything at a glance"),
    ("Records", "Records", ":material/military_tech:", "Streaks and standout results"),
    ("Analysis", "Analysis", ":material/analytics:", "Match xG and player scouting"),
    ("Data Health", "About & sources", ":material/health_and_safety:", "Where the data comes from"),
]
_NAV_LABELS = [label for _k, label, _i, _c in _NAV]
_NAV_CAPTIONS = [caption for _k, _l, _i, caption in _NAV]
_KEY_BY_LABEL = {label: key for key, label, _i, _c in _NAV}
_LABEL_BY_KEY = {key: label for key, label, _i, _c in _NAV}
_HEADER = {key: (icon, label, caption) for key, label, icon, caption in _NAV}


def _page_header(page: str) -> None:
    icon, label, blurb = _HEADER.get(page, (":material/dashboard:", page, ""))
    st.header(f"{icon} {label}", anchor=False)
    if blurb:
        st.caption(blurb)


def _render_match_analysis(settings) -> None:
    """Analysis > Match tab: xG timeline and shot map for a chosen StatsBomb match."""
    from collections import Counter

    matches = shot_matches(settings.analytics_db)
    if not matches:
        st.info(
            "No match data yet. Go to **About & sources** → **Add player data** to load some.",
            icon=":material/groups:",
        )
        return
    st.caption(
        "Shot data is StatsBomb open data: a fixed historical set, not the live season. "
        "Fullest coverage is 2015/16 (Premier League, La Liga, Serie A, Ligue 1), World "
        "Cups and Euros. Match counts are shown in brackets."
    )
    comp_counts = Counter(comp for _mid, _lbl, comp, _s in matches)
    comps = [c for c, _ in comp_counts.most_common()]  # richest competition first
    fcols = st.columns([1, 1, 2])
    if len(comps) > 1:
        clabels = {f"{c} ({comp_counts[c]})": c for c in comps}
        chosen = clabels[fcols[0].selectbox("Competition", list(clabels), key="ma_comp")]
        matches = [m for m in matches if m[2] == chosen]
    season_counts = Counter(s for _mid, _lbl, _c, s in matches if s)
    seasons = sorted(season_counts, key=lambda s: (season_counts[s], s), reverse=True)
    if len(seasons) > 1:
        slabels = {f"{s} ({season_counts[s]})": s for s in seasons}
        chosen_season = slabels[fcols[1].selectbox("Season", list(slabels), key="ma_season")]
        matches = [m for m in matches if m[3] == chosen_season]
    labels = {f"{lbl}": mid for mid, lbl, _comp, _s in matches}
    picked = fcols[2].selectbox("Match", list(labels), key="ma_match")
    data = shot_map(settings.analytics_db, labels[picked])
    if data is None:
        st.info("No shots for that match.")
    else:
        _render_shot_map(data)


def _render_players_page(settings) -> None:
    """Analysis > Players tab: full-event leaderboard and per-player scouting profiles."""
    if not has_player_events(settings.analytics_db):
        # No full-event stats -- fall back to the shots-only board, or prompt.
        rows = player_board(settings.analytics_db, top=25, min_shots=3, order="xg")
        if rows:
            _render_players(rows)
        else:
            st.info(
                "No player data yet. Go to **About & sources** → **Add player data** to load some.",
                icon=":material/groups:",
            )
        return

    comps = player_competitions(settings.analytics_db)
    competition, season, scope = None, None, "All competitions"
    row = st.columns(4)
    slot = 0
    if len(comps) > 1:
        clabels = {"All competitions": None} | {f"{c}  ({n})": c for c, n in comps}
        comp_choice = row[slot].selectbox("Competition", list(clabels), key="pl_comp")
        slot += 1
        competition = clabels[comp_choice]
        if competition:
            scope = competition
    if competition:
        seasons = player_seasons(settings.analytics_db, competition)
        if len(seasons) > 1:
            slabels = {"All seasons": None} | {f"{s}  ({n})": s for s, n in seasons}
            season = slabels[row[slot].selectbox("Season", list(slabels), key="pl_season")]
            slot += 1
            if season:
                scope = f"{competition} {season}"
    view = row[slot].segmented_control(
        "View", ["Leaderboard", "Player profile"], default="Leaderboard"
    )
    slot += 1
    min_minutes = row[slot].slider("Min minutes", 90, 900, 270, step=30)

    if view == "Player profile":
        pool = player_profiles(
            settings.analytics_db,
            top=100_000,
            min_minutes=min_minutes,
            order="contributions",
            competition=competition,
            season=season,
        )
        if not pool:
            st.info("No players clear that minutes threshold. Lower it above.")
            return
        names = [p.player for p in pool]
        picked = st.selectbox("Player", names)
        profile = player_profile(
            settings.analytics_db, picked, competition=competition, season=season
        )
        pcts = player_percentiles(
            settings.analytics_db,
            picked,
            min_minutes=min_minutes,
            competition=competition,
            season=season,
        )
        if profile is None:
            st.info("No profile for that player.")
        else:
            _render_player_profile(profile, pcts, pool_label=scope)
    else:
        rc = st.columns([3, 1])
        rank_label = rc[0].selectbox("Rank by", list(_RANK_OPTIONS))
        per90 = rc[1].toggle("Per 90 minutes", value=False)
        profiles = player_profiles(
            settings.analytics_db,
            top=40,
            min_minutes=min_minutes,
            order=_RANK_OPTIONS[rank_label],
            competition=competition,
            season=season,
        )
        if not profiles:
            st.info("No players clear that minutes threshold. Lower it above.")
        else:
            _render_player_leaderboard(profiles, per90=per90, pool_label=scope)


def _require_password() -> None:
    """Gate the whole app behind SOCCER_DASHBOARD_PASSWORD when it is set.

    Unset (ordinary local use) -> open, unchanged. Set (e.g. when exposing the app over a
    public Cloudflare tunnel) -> a password is required before anything renders, which also
    protects the data-download actions on the Home page from the open internet.
    """
    import os

    password = os.environ.get("SOCCER_DASHBOARD_PASSWORD")
    if not password or st.session_state.get("_authed"):
        return
    st.title("Soccer Analytics")
    st.caption("This dashboard is password-protected.")
    entered = st.text_input(
        "Password", type="password", label_visibility="collapsed", placeholder="Password"
    )
    if entered and entered == password:
        st.session_state._authed = True
        st.rerun()
    elif entered:
        st.error("Incorrect password.")
    st.stop()


def main() -> None:
    st.set_page_config(
        page_title="Soccer Analytics",
        page_icon=":material/sports_soccer:",
        layout="wide",
    )
    _require_password()

    settings = get_settings()
    with st.sidebar:
        st.markdown("### :material/sports_soccer: Soccer Analytics")
        st.caption("Local-first football intelligence")
        st.space("small")
        # Apply a pending navigation request (from a call-to-action button) before the
        # radio is created -- setting its key afterwards is not allowed.
        if "_nav_to" in st.session_state:
            st.session_state.nav = st.session_state.pop("_nav_to")
        selected = st.radio(
            "Menu",
            _NAV_LABELS,
            captions=_NAV_CAPTIONS,
            label_visibility="collapsed",
            key="nav",
        )
    page = _KEY_BY_LABEL[selected]

    _page_header(page)

    if page == "Home":
        _render_home(settings)
        return

    if page == "Assistant":
        _render_assistant(settings)
        return

    if page == "Analysis":
        match_tab, players_tab = st.tabs(["Match", "Players"])
        with match_tab:
            _render_match_analysis(settings)
        with players_tab:
            _render_players_page(settings)
        return

    if page == "Predictor":
        available = analytics_available(settings.analytics_db)
        upcoming_tab, match_tab, season_tab, card_tab = st.tabs(
            ["Upcoming", "Matchup", "Season", "Scorecard"]
        )

        with upcoming_tab:
            _render_fixtures(
                _cached_fixture_forecasts(
                    str(settings.live_db), str(settings.analytics_db), 5000
                )
            )

        with match_tab:
            if not available:
                st.info(
                    "No league data yet. Go to **About & sources** → **Add a league**.",
                    icon=":material/download:",
                )
            else:
                season, division = _league_season_pickers(
                    available, key="pred_match", latest_only=True
                )
                st.caption("Odds and a likely score for any upcoming matchup this season.")
                teams = forecast_teams(settings.analytics_db, season, division)
                fc = st.columns([2, 2, 1])
                home = fc[0].selectbox("Home", teams, index=0)
                away = fc[1].selectbox("Away", teams, index=min(1, len(teams) - 1))
                mle = fc[2].toggle("Dixon-Coles instead", value=False)
                if home == away:
                    st.info("Pick two different teams.")
                else:
                    slate = forecast_slate(
                        settings.analytics_db, season, division, home, away, mle=mle
                    )
                    if slate is None:
                        st.info("Could not forecast that matchup.")
                    else:
                        _render_forecast(slate)
                        st.divider()
                        exp = forecast_explanation(
                            settings.analytics_db, season, division, home, away
                        )
                        if exp is not None:
                            _render_forecast_explanation(exp)
                            st.divider()
                        _render_ev_calculator(slate)
                        report = _cached_market_edge(str(settings.analytics_db), season, division)
                        if report is not None:
                            st.divider()
                            _render_market_edge(report)

        with season_tab:
            if not available:
                st.info(
                    "No league data yet. Go to **About & sources** → **Add a league**.",
                    icon=":material/download:",
                )
            else:
                season, division = _league_season_pickers(
                    available, key="pred_season", latest_only=True
                )
                st.caption(
                    "Title, top-four and relegation odds from simulating the rest of the season."
                )
                briefing = _cached_season_briefing(str(settings.analytics_db), season, division)
                if briefing is None:
                    st.info("No results for that selection.")
                else:
                    _render_season(briefing)

        with card_tab:
            if not available:
                st.info(
                    "No league data yet. Go to **About & sources** → **Add a league**.",
                    icon=":material/download:",
                )
            else:
                st.caption(
                    "How good are these forecasts, really? Measured over the last few seasons "
                    "against the bookmaker's closing line — the honest benchmark."
                )
                by_league = {division_name(d): d for _s, d, _n in available}
                league = st.selectbox("League", sorted(by_league), key="card_league")
                report = _cached_forecast_report(str(settings.analytics_db), by_league[league], 6)
                if report is None:
                    st.info("No closing odds loaded for that league yet.")
                else:
                    _render_report_card(report, by_league[league])
        return

    if page == "Analytics":
        available = analytics_available(settings.analytics_db)
        if not available:
            st.info(
                "No league data yet. Go to **Home** → **Add a league** to download some.",
                icon=":material/download:",
            )
            return
        season, division, extra = _league_season_pickers(available, extra=1)
        last_n = extra[0].slider("Form window (matches)", 3, 10, 5)
        snap = _cached_analytics(str(settings.analytics_db), season, division)
        if snap is None:
            st.info("No results for that selection.")
            return
        forms = team_form(settings.analytics_db, season, division, last_n=last_n)
        table_tab, form_tab, under_tab = st.tabs(["Table", "Form guide", "Underlying"])
        with table_tab:
            _render_analytics(snap, forms=forms, last_n=last_n)
        with form_tab:
            if forms:
                _render_trends(forms, last_n=last_n)
            else:
                st.info("No results for that selection.")
        with under_tab:
            under = underlying_table(settings.analytics_db, season, division)
            if under is None:
                st.info("No shot data for this season, so no underlying-performance view.")
            else:
                _render_underlying(under)
        return

    if page == "Team":
        available = analytics_available(settings.analytics_db)
        if not available:
            st.info(
                "No league data yet. Go to **About & sources** → **Add a league**.",
                icon=":material/download:",
            )
            return
        season, division, extra = _league_season_pickers(available, extra=1)
        teams = forecast_teams(settings.analytics_db, season, division)
        if not teams:
            st.info("No teams for that selection.")
            return
        team = extra[0].selectbox("Team", teams)
        dossier = _cached_team_dossier(str(settings.analytics_db), division, season, team)
        if dossier is None:
            st.info("No data for that team in the chosen season.")
        else:
            _render_team(dossier)
        return

    if page == "Records":
        available = analytics_available(settings.analytics_db)
        if not available:
            st.info(
                "No league data yet. Go to **Home** → **Add a league** to download some.",
                icon=":material/download:",
            )
            return
        season, division = _league_season_pickers(available)
        season_tab, all_time_tab = st.tabs(["This season", "All-time"])
        with season_tab:
            records = season_records(settings.analytics_db, season, division)
            if records is None:
                st.info("No results for that selection.")
            else:
                _render_records(records)
        with all_time_tab:
            hist = _cached_league_history(str(settings.analytics_db), division)
            if hist is None:
                st.info("No history loaded for that league.")
            else:
                _render_league_history(hist)
        return

    if not settings.live_db.exists():
        st.info(
            "No data yet. Go to **Home** → **Refresh live scores** to get started.",
            icon=":material/bolt:",
        )
        return

    with LiveDB(settings.live_db) as db:
        if page == "Live Centre":
            snap = live_snapshot(db)  # unfiltered KPIs; filters applied below
            comps = sorted({c for c, _ in snap.competition_counts})
            fcols = st.columns([1, 2])
            only_live = fcols[0].toggle("In play only", value=False)
            chosen = fcols[1].selectbox("Competition", ["All", *comps])
            _render_live(
                live_snapshot(
                    db,
                    in_play_only=only_live,
                    competition=None if chosen == "All" else chosen,
                )
            )
        else:
            _render_health(health_snapshot(settings, db))
            st.divider()
            _render_data_manager(settings)


if __name__ == "__main__":
    main()
