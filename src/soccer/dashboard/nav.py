"""Dashboard page registry -- plain data, no Streamlit import.

Shared by the app (sidebar + routing) and `soccer usage` (to list pages that were never
visited), so it must stay importable without the dashboard extra.
"""

# Per-page identity: (routing key, sidebar label, Material icon, one-line description). The
# routing keys stay stable -- the usage log records them -- so only labels and captions change.
NAV = [
    ("Home", "Home", ":material/home:", "Set up and quick actions"),
    ("Data Health", "About & sources", ":material/health_and_safety:", "Where the data comes from"),
    ("Predictor", "Predictions", ":material/insights:", "Fixtures, matchups and season odds"),
    (
        "Players",
        "Player scouting",
        ":material/person_search:",
        "Leaderboards, profiles, similar players",
    ),
    ("Market", "Bookmakers' lines", ":material/candlestick_chart:", "The market vs the model"),
    (
        "Analytics",
        "League tables",
        ":material/table_chart:",
        "Standings, form and underlying numbers",
    ),
    ("Team", "Teams", ":material/shield:", "One club, everything at a glance"),
    ("Records", "Records", ":material/military_tech:", "Streaks and standout results"),
    ("Analysis", "Match xG", ":material/sports:", "Shot maps and xG timelines"),
    ("Assistant", "Ask a question", ":material/chat:", "Chat about your data in plain English"),
]

# The six sections of the sidebar (the upgrade plan's dashboard layout), in order.
SECTIONS: list[tuple[str, list[str]]] = [
    ("Home", ["Home", "Data Health"]),
    ("Predictions", ["Predictor"]),
    ("Player scouting", ["Players"]),
    ("Market intelligence", ["Market"]),
    ("Team & match analysis", ["Analytics", "Team", "Records", "Analysis"]),
    ("AI research", ["Assistant"]),
]

PAGE_KEYS = [key for key, _label, _icon, _caption in NAV]

# Pages taken out of the sidebar but kept in the code, so one can come back by moving its
# row into NAV. Archived 2026-10-20: Live scores ("Live Centre") had 1 visit in its first
# 15 days of usage tracking while Home had 31; the assistant still answers "what's on now"
# from the live feed and refreshes it itself.
ARCHIVED = [
    ("Live Centre", "Live scores", ":material/bolt:", "Today's and recent results"),
]


def url_path(key: str) -> str:
    """A page's URL segment: "Data Health" -> "data-health"."""
    return key.lower().replace(" ", "-")
