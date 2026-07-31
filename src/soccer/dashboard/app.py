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
    HealthSnapshot,
    LiveSnapshot,
    health_snapshot,
    live_snapshot,
)
from soccer.domain.match_state import MatchStatus, MatchView
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


def main() -> None:
    st.set_page_config(page_title="Soccer Analytics", page_icon="⚽", layout="wide")
    st.title("⚽ Soccer Analytics")

    settings = get_settings()
    if not settings.live_db.exists():
        st.info("No database yet. Run `soccer ingest` to populate it, then reload.")
        return

    with st.sidebar:
        st.header("View")
        page = st.radio("Page", ["Live Centre", "Data Health"], label_visibility="collapsed")
        st.button("↻ Refresh")  # any interaction reruns and re-reads the DB

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
