"""Dashboard page registry -- plain data, no Streamlit import.

Shared by the app (sidebar + routing) and `soccer usage` (to list pages that were never
visited), so it must stay importable without the dashboard extra.
"""

# Per-page identity: a Material Symbol icon and a one-line description, for a consistent
# header on every page and cleaner navigation.
# Navigation: (routing key, sidebar label, icon, one-line description). The routing keys
# stay stable so the page dispatch is untouched; only the plain-English labels and the
# captions the user reads change. Ordered as a journey: start -> this season -> explore.
NAV = [
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

PAGE_KEYS = [key for key, _label, _icon, _caption in NAV]
