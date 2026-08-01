"""MCP server -- exposes the platform to LLM clients over stdio.

Thin by design: every tool is a one-line wrapper over `mcp/tools.py`, and the tool
docstrings are what an LLM reads to decide when to call them, so they describe intent,
not implementation. Guided prompts orchestrate several tools into a briefing, mirroring
the cricket-mcp pattern that inspired this project.

Run with `soccer mcp` (stdio). Tools are read-only -- the server never ingests or
mutates; that stays with the CLI.
"""

from __future__ import annotations

from mcp.server import MCPServer

from soccer.config import get_settings
from soccer.mcp import tools

mcp = MCPServer(
    "soccer-analytics",
    instructions=(
        "Local-first football analytics over free data. Use the tools to answer "
        "questions about live matches, league tables, power rankings, match forecasts "
        "and season simulations. Historical analytics come from completed seasons "
        "(default 2025/26 Premier League, division code E0); call list_available_data "
        "to see what is loaded. Forecasts are modestly skilful and somewhat "
        "over-confident on medium-strong home favourites -- present them as directional."
    ),
)


@mcp.tool()
def get_live_matches(in_play_only: bool = True, limit: int = 50) -> dict:
    """Current live and recently-finished matches with score, status and minute.

    Set in_play_only=False to include finished/scheduled matches. Requires `soccer
    ingest` to have been run.
    """
    return tools.live_matches(get_settings(), in_play_only=in_play_only, limit=limit)


@mcp.tool()
def get_league_table(season: str = "2526", division: str = "E0") -> dict:
    """League table computed from historical results. season like '2526', division 'E0'."""
    return tools.league_table(get_settings(), season, division)


@mcp.tool()
def get_power_rankings(season: str = "2526", division: str = "E0", top: int = 0) -> dict:
    """Elo power rankings from historical results. top=0 returns all teams."""
    return tools.power_rankings(get_settings(), season, division, top)


@mcp.tool()
def forecast_match(home: str, away: str, season: str = "2526", division: str = "E0") -> dict:
    """Forecast one match: outcome probabilities, expected goals and likely scorelines.

    Fits a Poisson model on the given season/division. If a team name is not found the
    result lists the available teams.
    """
    return tools.forecast_match(get_settings(), season, division, home, away)


@mcp.tool()
def simulate_league(season: str = "2526", division: str = "E0", sims: int = 5000) -> dict:
    """Monte Carlo season simulation: title, top-4 and relegation probabilities per team."""
    return tools.simulate_league(get_settings(), season, division, sims)


@mcp.tool()
def search_teams(query: str, season: str = "2526", division: str = "E0") -> dict:
    """Find team names matching a query within a season/division."""
    return tools.search_teams(get_settings(), query, season, division)


@mcp.tool()
def get_data_health() -> dict:
    """Source availability, live/delayed status, licensing and capability coverage."""
    return tools.data_health(get_settings())


@mcp.tool()
def list_available_data() -> dict:
    """What data is loaded: live match count and the historical (season, division) slices."""
    return tools.available_data(get_settings())


@mcp.prompt()
def pre_match_briefing(home: str, away: str, season: str = "2526", division: str = "E0") -> str:
    """A briefing for an upcoming match, orchestrating the forecast and form tools."""
    return (
        f"Give a concise pre-match briefing for {home} vs {away} ({division} {season}). "
        f"Call forecast_match for the outcome probabilities, expected goals and likely "
        f"scorelines, and get_power_rankings for both sides' Elo standing. Then summarise: "
        f"who is favoured and by how much, the most likely scorelines, and one caveat "
        f"about the model's reliability."
    )


@mcp.prompt()
def title_race_report(season: str = "2526", division: str = "E0") -> str:
    """A report on the title and relegation races from a season simulation."""
    return (
        f"Report on the {division} {season} title race. Call simulate_league and "
        f"get_league_table, then describe the title favourites with their probabilities, "
        f"the top-four picture, and who is most at risk of relegation. Note that this is a "
        f"full-season replay from fitted strengths, not a mid-season projection."
    )


@mcp.prompt()
def daily_digest() -> str:
    """A digest of what is happening live right now."""
    return (
        "Give a short digest of live football right now. Call get_live_matches and "
        "highlight the closest games, any notable scorelines, and how many matches are in "
        "play. Mention if the data is stale."
    )


def run() -> None:
    mcp.run(transport="stdio")
