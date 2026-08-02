"""Streamlit dashboard -- a thin render over `dashboard/data.py`.

Read-only. It shows only surfaces the ingested data can actually fill (a Live Centre and
a Data Health panel); there is deliberately no Match Centre or Player Hub, because
nothing ingests lineups or player events yet -- an empty surface reads as broken.

Status is shown as a reserved palette that always carries a text label, never colour
alone: live is green, delayed/postponed amber, cancelled red, concluded a muted grey.
Run with `soccer dashboard` (or `streamlit run src/soccer/dashboard/app.py`).
"""

from __future__ import annotations

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
    forecast_slate,
    forecast_teams,
    has_player_events,
    health_snapshot,
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
    team_form,
)
from soccer.domain.match_state import MatchStatus, MatchView
from soccer.sources.football_data_co_uk import division_name, season_label
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


def _render_live(snap: LiveSnapshot) -> None:
    k = snap.kpis
    st.subheader("Live Centre")

    cols = st.columns(5)
    cols[0].metric("Matches", k.total)
    cols[1].metric("In play", k.in_play)
    cols[2].metric("Competitions", k.competitions)
    cols[3].metric("Sources", k.sources)
    cols[4].metric("Updated", k.freshness_label)

    if k.any_stale:
        st.warning(
            "⚠ Some rows are **stale** — served from cache after a source failed. "
            "They are labelled in the Source column.",
            icon="⚠️",
        )
    if k.total == 0:
        st.info("No matches stored yet. Run `soccer ingest`, then refresh.")
        return

    rows = []
    for v in snap.matches:
        colour, text = _marker(v)
        source = v.source.replace("_", "-") + (" ⚠" if v.is_stale else "")
        rows.append(
            {
                "": f'<span style="color:{colour};font-weight:600">{text}</span>',
                "Home": v.home,
                "Score": f"<b>{v.score}</b>",
                "Away": v.away,
                "Competition": f'<span style="color:#8b8b8b">{v.competition}</span>',
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
    st.subheader("Data Health")
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


@st.cache_data(show_spinner="Simulating the season…")
def _cached_season_briefing(db_path: str, season: str, division: str):
    from pathlib import Path

    return season_briefing(Path(db_path), season, division)


def _render_season(briefing) -> None:
    st.subheader(
        f"Season oracle — {division_name(briefing.division)} {season_label(briefing.season)}"
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
    k[0].metric("Title favourite", names.get(fav.team, fav.team), f"{fav.title_pct:.0%}")
    k[1].metric("Most at risk", names.get(drop.team, drop.team), f"{drop.relegation_pct:.0%} down")
    k[2].metric(
        "On the bubble", names.get(tightest.team, tightest.team), f"{tightest.title_pct:.0%} title"
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


def _league_season_options(available: list[tuple[str, str, int]]) -> dict:
    """Selector labels for league/season slices: grouped by league, newest first, and the
    most-recent season of each league tagged 'latest' so it never reads as stale."""
    latest: dict[str, str] = {}
    for season, division, _n in available:
        if division not in latest or season > latest[division]:
            latest[division] = season
    ordered = sorted(available, key=lambda t: t[0], reverse=True)
    ordered = sorted(ordered, key=lambda t: division_name(t[1]))
    labels: dict = {}
    for season, division, _n in ordered:
        tag = " · latest" if season == latest[division] else ""
        labels[f"{division_name(division)} - {season_label(season)}{tag}"] = (season, division)
    return labels


def _render_records(records) -> None:
    st.subheader("Records & streaks")
    st.caption(
        "Active runs counted back from each team's most recent match, plus the season's "
        "standout results. Streaks are the records to watch."
    )
    streaks = records.streaks
    on_fire = max(streaks, key=lambda s: s.winning)
    unbeaten = streaks[0]  # already sorted by active unbeaten
    struggling = max(streaks, key=lambda s: s.winless)
    k = st.columns(3)
    k[0].metric("Longest unbeaten", unbeaten.team, f"{unbeaten.unbeaten} games")
    k[1].metric("Hot streak", on_fire.team, f"{on_fire.winning} wins")
    k[2].metric("Winless run", struggling.team, f"{struggling.winless} games")

    st.markdown("**Active streaks**")
    rows = [
        {
            "Team": s.team,
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
            f"{m.home} <b>{m.score}</b> {m.away} "
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
    st.subheader("Form & trends")
    st.caption(
        f"Last {last_n} matches vs the season baseline. ▲ rising, ▼ sliding. "
        "O2.5 = share of a team's games with over 2.5 goals; BTTS = both teams scored."
    )
    hottest, coldest = forms[0], forms[-1]
    goalfest = max(forms, key=lambda f: f.over25_rate)
    k = st.columns(3)
    k[0].metric("Hottest", hottest.team, f"{hottest.recent_form} · {hottest.recent_ppg:.2f} ppg")
    k[1].metric("Coldest", coldest.team, f"{coldest.recent_form} · {coldest.recent_ppg:.2f} ppg")
    k[2].metric("Most goals", goalfest.team, f"{goalfest.over25_rate:.0%} over 2.5")

    rows = [
        {
            "Team": f.team,
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


def _render_analytics(snap: AnalyticsSnapshot) -> None:
    st.subheader(f"Analytics — {division_name(snap.division)} {season_label(snap.season)}")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**League table**")
        rows = [
            {
                "#": r.position,
                "Team": r.team,
                "P": r.played,
                "GD": f"{r.goal_difference:+d}",
                "Pts": f"<b>{r.points}</b>",
            }
            for r in snap.table
        ]
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
            {"#": r.position, "Team": snap.names.get(r.team, r.team), "Elo": f"{r.rating:.0f}"}
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


def _render_shot_map(data) -> None:
    st.subheader(f"Match centre — {data.label}")
    st.caption("StatsBomb event data. Circle size ∝ xG; filled = goal. Both teams attack →")

    cols = st.columns(len(data.team_xg) or 1)
    for col, row in zip(cols, data.team_xg, strict=False):
        col.metric(f"{row.name} xG", f"{row.xg:.2f}", f"{row.goals} goals")

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


def _market_table(title: str, markets, *, odds: bool = True) -> str:
    rows = []
    for m in markets:
        cells = f"<td>{m.name}</td><td style='text-align:right'>{m.probability:.0%}</td>"
        if odds:
            cells += f"<td style='text-align:right;color:#8b8b8b'>{m.fair_odds:.2f}</td>"
        rows.append(f"<tr>{cells}</tr>")
    head = f"<b>{title}</b>"
    return f"{head}<table style='width:100%;font-size:0.88rem'>{''.join(rows)}</table>"


def _render_forecast(slate) -> None:
    st.subheader(f"{slate.home} vs {slate.away}")
    c = st.columns(3)
    c[0].metric(f"{slate.home} xG", f"{slate.home_expected:.2f}")
    c[1].metric(f"{slate.away} xG", f"{slate.away_expected:.2f}")
    x, y, p = slate.most_likely_score
    c[2].metric("Most likely score", f"{x}-{y}", f"{p:.0%}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(_market_table("Result (1X2)", slate.result), unsafe_allow_html=True)
        st.markdown(_market_table("Double chance", slate.double_chance), unsafe_allow_html=True)
        st.markdown(_market_table("Both teams to score", slate.btts), unsafe_allow_html=True)
    with col2:
        ou_rows = "".join(
            f"<tr><td>Over/Under {ou.line}</td>"
            f"<td style='text-align:right'>{ou.over:.0%}</td>"
            f"<td style='text-align:right'>{ou.under:.0%}</td></tr>"
            for ou in slate.over_under
        )
        st.markdown(
            f"<b>Total goals lines</b><table style='width:100%;font-size:0.88rem'>"
            f"<tr><td></td><td style='text-align:right'>Over</td>"
            f"<td style='text-align:right'>Under</td></tr>{ou_rows}</table>",
            unsafe_allow_html=True,
        )
        st.markdown(_market_table("Clean sheet", slate.clean_sheet), unsafe_allow_html=True)
        st.markdown(_market_table("Win to nil", slate.win_to_nil), unsafe_allow_html=True)
    with col3:
        st.markdown(
            _market_table("Total goals", slate.total_goals, odds=False), unsafe_allow_html=True
        )
        cs_rows = "".join(
            f"<tr><td>{x}-{y}</td><td style='text-align:right'>{p:.0%}</td></tr>"
            for x, y, p in slate.correct_scores
        )
        st.markdown(
            f"<b>Correct score</b><table style='width:100%;font-size:0.88rem'>{cs_rows}</table>",
            unsafe_allow_html=True,
        )
    st.caption(
        "Fair prices (no margin) from a Dixon-Coles model fit on the last ~3 seasons, "
        "re-fit as new results land. Directional, not advice."
    )


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


def _render_players(rows) -> None:
    st.subheader("Player leaderboard")
    st.caption(
        "StatsBomb shots across all ingested matches. G-xG > 0 = clinical finishing. "
        "Points above the diagonal outscored their chances."
    )
    c = st.columns(3)
    c[0].metric("Players", len(rows))
    c[1].metric("Top xG", f"{rows[0].xg:.1f}", rows[0].player.split()[-1])
    clinical = max(rows, key=lambda r: r.xg_diff)
    c[2].metric("Best finisher (G-xG)", f"{clinical.xg_diff:+.1f}", clinical.player.split()[-1])

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
            "Player": r.player,
            "Team": f'<span style="color:#8b8b8b">{r.team}</span>',
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
    st.subheader("Player leaderboard")
    mode = "per 90 minutes" if per90 else "totals"
    scope = f"{pool_label} · " if pool_label and pool_label != "All competitions" else ""
    st.caption(
        f"{scope}StatsBomb event data across all ingested matches, shown as {mode}. "
        "Sort any column by clicking its header."
    )

    top = profiles[0]
    kpi = st.columns(4)
    kpi[0].metric("Players", len(profiles))
    kpi[1].metric("Most G+A", top.goal_contributions, top.player.split()[-1])
    clinical = max(profiles, key=lambda p: p.xg_diff)
    kpi[2].metric("Best finisher (G-xG)", f"{clinical.xg_diff:+.1f}", clinical.player.split()[-1])
    creator = max(profiles, key=lambda p: p.xa)
    kpi[3].metric("Top xA", f"{creator.xa:.1f}", creator.player.split()[-1])

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


def _render_player_profile(profile, percentiles, *, pool_label: str = "") -> None:
    pos = f" · {profile.position}" if profile.position else ""
    st.subheader(f"{profile.player}")
    st.caption(f"{profile.team}{pos} · {profile.matches} matches · {profile.minutes} minutes")

    row = st.columns(5)
    row[0].metric("Goals", profile.goals, f"xG {profile.xg:.1f}")
    row[1].metric("Assists", profile.assists, f"xA {profile.xa:.1f}")
    row[2].metric("Shots", profile.shots, f"{profile.per90(profile.shots):.1f}/90")
    row[3].metric("Pass %", f"{profile.pass_pct:.0f}%", f"{profile.passes} att")
    row[4].metric("Prog. actions", profile.progressive_passes + profile.progressive_carries)

    if not percentiles:
        st.info("Not enough minutes for a percentile fingerprint at this threshold.")
        return

    pool_note = pool_label if pool_label and pool_label != "All competitions" else "all loaded"
    st.markdown(f"**Percentile rank** — vs {pool_note} players (per 90)")
    order = [mp.label for mp in percentiles]
    frame = pl.DataFrame(
        {
            "metric": [mp.label for mp in percentiles],
            "category": [mp.category for mp in percentiles],
            "percentile": [mp.percentile for mp in percentiles],
            "value": [mp.value for mp in percentiles],
        }
    ).to_pandas()

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
    st.altair_chart(
        (bars + labels + midline).properties(height=22 * len(percentiles) + 10), width="stretch"
    )
    st.caption(
        "Bar = percentile within the pool; number = the player's per-90 value. Dashed line "
        "is the median (50th). " + ATTRIBUTION_STATSBOMB
    )


def _render_fixtures(fixtures) -> None:
    st.subheader("Fixtures & forecasts")
    forecastable = [f for f in fixtures if f.slate is not None]
    uncovered = [f for f in fixtures if f.slate is None]
    st.caption(
        f"{len(forecastable)} of {len(fixtures)} upcoming matches forecast from last "
        "season's Dixon-Coles model. Predictions are a preseason projection, directional."
    )
    if not forecastable:
        st.info(
            "No forecastable fixtures yet. Run `soccer ingest` for fixtures and "
            "`soccer ingest-history` to fit the league models."
        )
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
            f'<span style="color:#8b8b8b">{f.competition}</span>',
            f"{f.home} v {f.away}",
            f"<b>{x}-{y}</b>",
            _pct_cell(home_p, home_p >= max(draw_p, away_p)),
            _pct_cell(draw_p, draw_p >= max(home_p, away_p)),
            _pct_cell(away_p, away_p >= max(home_p, draw_p)),
            f"{over25:.0%}",
            f"{btts_yes:.0%}",
            f'<span style="color:#16a34a">{fav}</span>',
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

    if uncovered:
        with st.expander(f"{len(uncovered)} upcoming fixtures without a forecast"):
            st.caption(
                "No loaded model covers these -- a promoted team absent from last season, "
                "or a competition without loaded history. Listed honestly, not hidden."
            )
            st.markdown(
                "".join(
                    f'<div style="font-size:0.85rem;padding:1px 0">'
                    f'<span style="color:#8b8b8b">{f.kickoff_utc.strftime("%m-%d")} · '
                    f"{f.competition}</span> &nbsp; {f.home} v {f.away}</div>"
                    for f in uncovered
                ),
                unsafe_allow_html=True,
            )


def _pct_cell(p: float, is_max: bool) -> str:
    """Outcome probability, bolded when it is the likeliest of the three."""
    return f"<b>{p:.0%}</b>" if is_max else f"{p:.0%}"


def main() -> None:
    st.set_page_config(page_title="Soccer Analytics", page_icon="⚽", layout="wide")
    st.title("⚽ Soccer Analytics")

    settings = get_settings()
    with st.sidebar:
        st.header("View")
        page = st.radio(
            "Page",
            [
                "Live Centre",
                "Fixtures",
                "Season",
                "Analytics",
                "Trends",
                "Records",
                "Forecast",
                "Shot Map",
                "Players",
                "Data Health",
            ],
            label_visibility="collapsed",
        )
        st.button("↻ Refresh")

    if page == "Fixtures":
        _render_fixtures(fixture_forecasts(settings.live_db, settings.analytics_db))
        return

    if page == "Players":
        if not has_player_events(settings.analytics_db):
            # No full-event stats -- fall back to the shots-only board, or prompt.
            rows = player_board(settings.analytics_db, top=25, min_shots=3, order="xg")
            if rows:
                _render_players(rows)
            else:
                st.info(
                    "No event data yet. Run `soccer ingest-events --competition 43 --season 106`, "
                    "then `soccer ingest-events --from-raw`, and reload."
                )
            return

        with st.sidebar:
            comps = player_competitions(settings.analytics_db)
            competition, season, scope = None, None, "All competitions"
            if len(comps) > 1:
                clabels = {"All competitions": None} | {f"{c}  ({n})": c for c, n in comps}
                comp_choice = st.selectbox("Competition", list(clabels))
                competition = clabels[comp_choice]
                if competition:
                    scope = competition
            if competition:
                seasons = player_seasons(settings.analytics_db, competition)
                if len(seasons) > 1:
                    slabels = {"All seasons": None} | {f"{s}  ({n})": s for s, n in seasons}
                    season = slabels[st.selectbox("Season", list(slabels))]
                    if season:
                        scope = f"{competition} {season}"
            view = st.segmented_control(
                "View", ["Leaderboard", "Player profile"], default="Leaderboard"
            )
            min_minutes = st.slider("Min minutes", 90, 900, 270, step=30)

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
                st.info("No players clear that minutes threshold. Lower it in the sidebar.")
                return
            with st.sidebar:
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
            with st.sidebar:
                rank_label = st.selectbox("Rank by", list(_RANK_OPTIONS))
                per90 = st.toggle("Per 90 minutes", value=False)
            profiles = player_profiles(
                settings.analytics_db,
                top=40,
                min_minutes=min_minutes,
                order=_RANK_OPTIONS[rank_label],
                competition=competition,
                season=season,
            )
            if not profiles:
                st.info("No players clear that minutes threshold. Lower it in the sidebar.")
            else:
                _render_player_leaderboard(profiles, per90=per90, pool_label=scope)
        return

    if page == "Forecast":
        available = analytics_available(settings.analytics_db)
        if not available:
            st.info("No analytics data yet. Run `soccer ingest-history`, then reload.")
            return
        st.caption(
            "Explore any matchup from a completed season (the latest is the freshest data). "
            "For the upcoming season's real fixtures, see the **Fixtures** page."
        )
        with st.sidebar:
            labels = _league_season_options(available)
            picked = st.selectbox("League / season", list(labels))
            season, division = labels[picked]
            teams = forecast_teams(settings.analytics_db, season, division)
            home = st.selectbox("Home", teams, index=0)
            away = st.selectbox("Away", teams, index=min(1, len(teams) - 1))
            mle = st.toggle("Dixon-Coles model", value=True)
        if home == away:
            st.info("Pick two different teams.")
        else:
            slate = forecast_slate(settings.analytics_db, season, division, home, away, mle=mle)
            if slate is None:
                st.info("Could not forecast that matchup.")
            else:
                _render_forecast(slate)
                st.divider()
                _render_ev_calculator(slate)
                report = _cached_market_edge(str(settings.analytics_db), season, division)
                if report is not None:
                    st.divider()
                    _render_market_edge(report)
        return

    if page == "Analytics":
        available = analytics_available(settings.analytics_db)
        if not available:
            st.info("No analytics data yet. Run `soccer ingest-history`, then reload.")
            return
        with st.sidebar:
            labels = _league_season_options(available)
            picked = st.selectbox("League / season", list(labels))
        season, division = labels[picked]
        snap = _cached_analytics(str(settings.analytics_db), season, division)
        if snap is None:
            st.info("No results for that selection.")
        else:
            _render_analytics(snap)
        return

    if page == "Trends":
        available = analytics_available(settings.analytics_db)
        if not available:
            st.info("No analytics data yet. Run `soccer ingest-history`, then reload.")
            return
        with st.sidebar:
            labels = _league_season_options(available)
            picked = st.selectbox("League / season", list(labels))
            last_n = st.slider("Form window (matches)", 3, 10, 5)
        season, division = labels[picked]
        forms = team_form(settings.analytics_db, season, division, last_n=last_n)
        if not forms:
            st.info("No results for that selection.")
        else:
            _render_trends(forms, last_n=last_n)
        return

    if page == "Records":
        available = analytics_available(settings.analytics_db)
        if not available:
            st.info("No analytics data yet. Run `soccer ingest-history`, then reload.")
            return
        with st.sidebar:
            labels = _league_season_options(available)
            picked = st.selectbox("League / season", list(labels))
        season, division = labels[picked]
        records = season_records(settings.analytics_db, season, division)
        if records is None:
            st.info("No results for that selection.")
        else:
            _render_records(records)
        return

    if page == "Season":
        available = analytics_available(settings.analytics_db)
        if not available:
            st.info("No analytics data yet. Run `soccer ingest-history`, then reload.")
            return
        with st.sidebar:
            # Only leagues (round-robin) simulate sensibly; every loaded slice is offered.
            labels = _league_season_options(available)
            picked = st.selectbox("League / season", list(labels))
        season, division = labels[picked]
        briefing = _cached_season_briefing(str(settings.analytics_db), season, division)
        if briefing is None:
            st.info("No results for that selection.")
        else:
            _render_season(briefing)
        return

    if page == "Shot Map":
        matches = shot_matches(settings.analytics_db)
        if not matches:
            st.info("No event data yet. Run `soccer ingest-events --match <id>`, then reload.")
            return
        with st.sidebar:
            comps = sorted({comp for _mid, _lbl, comp, _s in matches})
            if len(comps) > 1:
                chosen = st.selectbox("Competition", comps)
                matches = [m for m in matches if m[2] == chosen]
            seasons = sorted({s for _mid, _lbl, _c, s in matches if s}, reverse=True)
            if len(seasons) > 1:
                chosen_season = st.selectbox("Season", seasons)
                matches = [m for m in matches if m[3] == chosen_season]
            labels = {f"{lbl}": mid for mid, lbl, _comp, _s in matches}
            picked = st.selectbox("Match", list(labels))
        data = shot_map(settings.analytics_db, labels[picked])
        if data is None:
            st.info("No shots for that match.")
        else:
            _render_shot_map(data)
        return

    if not settings.live_db.exists():
        st.info("No live database yet. Run `soccer ingest` to populate it, then reload.")
        return

    with LiveDB(settings.live_db) as db:
        if page == "Live Centre":
            snap = live_snapshot(db)  # unfiltered KPIs; filters applied below
            with st.sidebar:
                comps = sorted({c for c, _ in snap.competition_counts})
                only_live = st.toggle("In play only", value=False)
                chosen = st.selectbox("Competition", ["All", *comps])
            _render_live(
                live_snapshot(
                    db,
                    in_play_only=only_live,
                    competition=None if chosen == "All" else chosen,
                )
            )
        else:
            _render_health(health_snapshot(settings, db))


if __name__ == "__main__":
    main()
