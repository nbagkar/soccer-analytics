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
    health_snapshot,
    live_snapshot,
    shot_map,
    shot_matches,
)
from soccer.domain.match_state import MatchStatus, MatchView
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


def _render_analytics(snap: AnalyticsSnapshot) -> None:
    st.subheader(f"Analytics — {snap.division} {snap.season}")

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
    st.subheader(f"Shot map — {data.label}")
    st.caption("StatsBomb event data. Circle size ∝ xG; filled = goal. Both teams attack →")

    cols = st.columns(len(data.team_xg) or 1)
    for col, row in zip(cols, data.team_xg, strict=False):
        col.metric(f"{row.name} xG", f"{row.xg:.2f}", f"{row.goals} goals")

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
    st.caption(ATTRIBUTION_STATSBOMB)


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


def main() -> None:
    st.set_page_config(page_title="Soccer Analytics", page_icon="⚽", layout="wide")
    st.title("⚽ Soccer Analytics")

    settings = get_settings()
    with st.sidebar:
        st.header("View")
        page = st.radio(
            "Page",
            ["Live Centre", "Analytics", "Shot Map", "Data Health"],
            label_visibility="collapsed",
        )
        st.button("↻ Refresh")

    if page == "Analytics":
        available = analytics_available(settings.analytics_db)
        if not available:
            st.info("No analytics data yet. Run `soccer ingest-history`, then reload.")
            return
        with st.sidebar:
            labels = {f"{s} / {d}  ({n})": (s, d) for s, d, n in available}
            picked = st.selectbox("Season / division", list(labels))
        season, division = labels[picked]
        snap = _cached_analytics(str(settings.analytics_db), season, division)
        if snap is None:
            st.info("No results for that selection.")
        else:
            _render_analytics(snap)
        return

    if page == "Shot Map":
        matches = shot_matches(settings.analytics_db)
        if not matches:
            st.info("No event data yet. Run `soccer ingest-events --match <id>`, then reload.")
            return
        with st.sidebar:
            labels = {f"{lbl} ({mid})": mid for mid, lbl in matches}
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
