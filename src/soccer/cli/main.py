"""CLI entry point.

Command set follows the plan's cricket-mcp-derived shape: `doctor` first, because
the most common failure in a multi-source free-data pipeline is not a crash but a
source quietly degrading. `doctor` answers "what do I actually have right now".
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from soccer.config import get_settings
from soccer.domain.aliases import AliasStore, suggest_duplicates
from soccer.domain.match_state import MatchStateStore
from soccer.ingest.pipeline import IngestPipeline
from soccer.sources.football_data_co_uk import FootballDataCoUk
from soccer.sources.football_data_org import FootballDataOrg
from soccer.sources.registry import SOURCES, Capability, SourceId, Trust, attributions, sources_for
from soccer.sources.thesportsdb import (
    ATTRIBUTION,
    LiveResult,
    SourceUnavailableError,
    TheSportsDB,
)
from soccer.storage.analytics_db import AnalyticsDB
from soccer.storage.live_db import LiveDB
from soccer.storage.raw import RawStore

app = typer.Typer(
    name="soccer",
    help="Local-first football intelligence centre built on free and open data.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


def _status_text(*, enabled: bool, reason: str | None = None) -> Text:
    if enabled:
        return Text("enabled", style="green")
    return Text(f"disabled ({reason})" if reason else "disabled", style="yellow")


@app.command()
def doctor() -> None:
    """Report configuration, source availability and known coverage gaps."""
    settings = get_settings()

    console.print()
    console.print("[bold]Configuration[/bold]")
    console.print(f"  data dir      {settings.data_dir}")
    console.print(f"  live db       {settings.live_db}")
    console.print(f"  analytics db  {settings.analytics_db}")
    console.print(
        f"  raw snapshots {settings.raw_dir} "
        f"[dim]({'present' if settings.raw_dir.exists() else 'not yet created'})[/dim]"
    )

    table = Table(title="\nSources", title_justify="left", header_style="bold")
    table.add_column("Source")
    table.add_column("Status")
    table.add_column("Trust")
    table.add_column("Latency")
    table.add_column("Licence")

    for source in SOURCES.values():
        enabled = settings.is_enabled(source.id)
        reason = None
        if not enabled:
            reason = "no token" if source.id.value == "football_data_org" else "off by default"

        if source.latency_seconds is None:
            latency = "static"
        elif source.latency_seconds <= 120:
            latency = f"~{source.latency_seconds}s [green](live)[/green]"
        elif source.latency_seconds < 86_400:
            latency = f"~{source.latency_seconds // 3600}h"
        else:
            latency = f"~{source.latency_seconds // 86_400}d"

        licence = source.licence
        if "NO EXPLICIT" in licence or "UNCONFIRMED" in licence:
            licence = f"[yellow]{licence}[/yellow]"

        trust_style = {
            Trust.PRIMARY: "green",
            Trust.CORROBORATING: "cyan",
            Trust.EXPERIMENTAL: "yellow",
        }[source.trust]

        table.add_row(
            source.name,
            _status_text(enabled=enabled, reason=reason),
            Text(source.trust.value, style=trust_style),
            latency,
            licence,
        )

    console.print(table)

    # Capability coverage is the part that matters most: it is where the gap between
    # what the product promises and what free data delivers becomes visible.
    coverage = Table(title="\nCapability coverage", title_justify="left", header_style="bold")
    coverage.add_column("Capability")
    coverage.add_column("Served by")

    for capability in Capability:
        providers = [
            source
            for source in sources_for(capability, include_disabled=True)
            if settings.is_enabled(source.id)
        ]
        if providers:
            names = ", ".join(source.name for source in providers)
            coverage.add_row(capability.value, Text(names, style="green"))
        else:
            coverage.add_row(capability.value, Text("not available", style="red"))

    console.print(coverage)

    unresolved = [source for source in SOURCES.values() if "UNCONFIRMED" in source.licence]
    if unresolved:
        console.print("\n[bold yellow]Unresolved licensing[/bold yellow]")
        for source in unresolved:
            console.print(f"  {source.name}: {source.licence}")

    if not settings.football_data_org_token:
        console.print(
            "\n[yellow]No football-data.org token.[/yellow] Fixtures, results and "
            "standings are unavailable. Register free at "
            "https://www.football-data.org/client/register and set "
            "SOCCER_FOOTBALL_DATA_ORG_TOKEN."
        )

    console.print("\n[bold]Attribution required[/bold]")
    for line in attributions():
        console.print(f"  {line}")
    console.print()


@app.command()
def sources() -> None:
    """Show what each source provides, and the caveats attached to it."""
    for source in SOURCES.values():
        console.print(f"\n[bold]{source.name}[/bold]  [dim]{source.base_url}[/dim]")
        console.print(f"  capabilities  {', '.join(sorted(c.value for c in source.capabilities))}")
        console.print(f"  licence       {source.licence}")
        if source.rate_limit_per_minute:
            console.print(f"  rate limit    {source.rate_limit_per_minute}/min")
        if source.mutable_history:
            console.print(
                "  [yellow]history is retroactively mutable -- re-check past results[/yellow]"
            )
        for caveat in source.caveats:
            console.print(f"  [dim]- {caveat}[/dim]")
    console.print()


@app.command()
def init() -> None:
    """Create the data directories."""
    settings = get_settings()
    settings.ensure_dirs()
    console.print(f"[green]Ready.[/green] Data directory: {settings.data_dir}")


@app.command()
def live(
    in_play_only: bool = typer.Option(
        True, "--in-play/--all", help="Only matches currently being played."
    ),
    league: str | None = typer.Option(None, help="Filter by league name (substring)."),
) -> None:
    """Show current live scores."""
    settings = get_settings()
    settings.ensure_dirs()

    async def run() -> LiveResult:
        async with TheSportsDB(
            RawStore(settings.raw_dir),
            api_key=settings.thesportsdb_key,
            rate_limit_per_minute=settings.thesportsdb_rpm,
        ) as source:
            return await source.livescore()

    try:
        result = asyncio.run(run())
    except SourceUnavailableError as exc:
        # Deliberately not an empty table: a dead feed and a quiet night must not
        # look alike.
        console.print(f"[red]Live feed unavailable.[/red] {exc}")
        console.print(
            "[dim]Free access to TheSportsDB's livescore endpoint is undocumented "
            "and may have been withdrawn. Delayed sources are unaffected.[/dim]"
        )
        raise typer.Exit(code=1) from exc

    matches = result.in_play if in_play_only else result.matches
    if league:
        matches = [m for m in matches if league.lower() in m.league.lower()]

    if result.is_stale:
        age = datetime.now(UTC) - result.fetched_at
        console.print(
            f"[yellow]STALE[/yellow] live fetch failed; showing cached data from "
            f"{int(age.total_seconds() // 60)} min ago.\n"
        )

    if not matches:
        scope = "in play" if in_play_only else "returned by the feed"
        console.print(f"[dim]No matches {scope} right now.[/dim]")
        console.print(f"\n[dim]{ATTRIBUTION}[/dim]")
        return

    table = Table(header_style="bold", title_justify="left", expand=False)
    table.add_column("", width=6, justify="right", no_wrap=True)
    table.add_column("Home", width=20, no_wrap=True, overflow="ellipsis")
    table.add_column("", width=5, justify="center", no_wrap=True)
    table.add_column("Away", width=20, no_wrap=True, overflow="ellipsis")
    table.add_column("League", width=24, no_wrap=True, overflow="ellipsis")

    for match in sorted(matches, key=lambda m: (m.league, m.home_team)):
        score = f"{match.home_score}-{match.away_score}" if match.home_score is not None else "-"
        minute = (
            f"{match.display_minute}'"
            if match.status.is_in_play and match.display_minute
            else match.raw_status
        )
        style = "green" if match.status.is_in_play else "dim"
        table.add_row(
            Text(minute, style=style),
            match.home_team,
            Text(score, style="bold"),
            match.away_team,
            Text(match.league, style="dim"),
        )

    console.print(table)
    console.print(f"\n[dim]{ATTRIBUTION}[/dim]")


@app.command()
def prune(
    keep_days: int = typer.Option(7, help="Days of live snapshots to retain."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Report without deleting."),
) -> None:
    """Delete old live-feed snapshots.

    Only the high-frequency live endpoint is pruned. Slow endpoints (competitions,
    standings, matches) are the provenance record -- they are cheap and are kept.
    """
    settings = get_settings()
    store = RawStore(settings.raw_dir)

    live_dir = settings.raw_dir / SourceId.THESPORTSDB.value / "livescore"
    before = len(list(live_dir.rglob("*.json.gz"))) if live_dir.is_dir() else 0

    if dry_run:
        console.print(
            f"[dim]Dry run:[/dim] {before} live snapshot(s) on disk, retaining {keep_days} days."
        )
        return

    removed = store.prune(SourceId.THESPORTSDB, "livescore", keep_days=keep_days)
    console.print(f"[green]Pruned[/green] {removed} snapshot(s); {before - removed} retained.")


@app.command()
def ingest(
    days: int = typer.Option(3, help="Days ahead to ingest fixtures for."),
) -> None:
    """Fetch from enabled sources, resolve to canonical matches, and persist state."""
    settings = get_settings()
    settings.ensure_dirs()
    raw = RawStore(settings.raw_dir)

    async def run() -> list[str]:
        lines: list[str] = []
        with LiveDB(settings.live_db) as db:
            pipeline = IngestPipeline(db)

            if settings.football_data_org_token:
                today = datetime.now(UTC).date()
                async with FootballDataOrg(
                    settings.football_data_org_token,
                    raw,
                    rate_limit_per_minute=settings.football_data_org_rpm,
                ) as fd:
                    summary = await pipeline.ingest_football_data(
                        fd, today, today + timedelta(days=days)
                    )
                lines.append(str(summary))
            else:
                lines.append("football_data_org: skipped (no token)")

            try:
                async with TheSportsDB(
                    raw,
                    api_key=settings.thesportsdb_key,
                    rate_limit_per_minute=settings.thesportsdb_rpm,
                ) as tsdb:
                    summary = await pipeline.ingest_thesportsdb(tsdb)
                lines.append(str(summary))
            except SourceUnavailableError as exc:
                lines.append(f"thesportsdb: unavailable ({exc})")
        return lines

    for line in asyncio.run(run()):
        console.print(f"  {line}")
    console.print("\n[green]Ingest complete.[/green] View with [bold]soccer matches[/bold].")


@app.command()
def matches(
    in_play: bool = typer.Option(False, "--in-play", help="Only matches in play."),
) -> None:
    """Show canonical match state ingested into the local database."""
    settings = get_settings()
    if not settings.live_db.exists():
        console.print("[yellow]No data yet.[/yellow] Run [bold]soccer ingest[/bold] first.")
        raise typer.Exit(code=1)

    with LiveDB(settings.live_db) as db:
        views = MatchStateStore(db).list_current(in_play_only=in_play)

    if not views:
        console.print("[dim]No matches stored.[/dim]")
        return

    table = Table(header_style="bold", expand=False)
    table.add_column("Kickoff (UTC)", no_wrap=True)
    table.add_column("", width=6, justify="right", no_wrap=True)
    table.add_column("Home", width=20, no_wrap=True, overflow="ellipsis")
    table.add_column("", width=5, justify="center", no_wrap=True)
    table.add_column("Away", width=20, no_wrap=True, overflow="ellipsis")
    table.add_column("Competition", width=18, no_wrap=True, overflow="ellipsis")
    table.add_column("Source", no_wrap=True)

    for view in views:
        if view.status.is_in_play:
            marker = Text(view.minute or "LIVE", style="green")
        elif view.status.is_concluded:
            marker = Text("FT", style="dim")
        else:
            marker = Text(view.kickoff_utc.strftime("%H:%M"), style="dim")

        provenance = view.source.replace("_", "-")
        if view.is_stale:
            provenance += " (stale)"

        table.add_row(
            view.kickoff_utc.strftime("%m-%d %H:%M"),
            marker,
            view.home,
            Text(view.score, style="bold"),
            view.away,
            Text(view.competition, style="dim"),
            Text(provenance, style="yellow" if view.is_stale else "dim"),
        )

    console.print(table)
    console.print(f"\n[dim]{ATTRIBUTION}[/dim]")


@app.command()
def aliases() -> None:
    """List curated name aliases the resolver applies."""
    settings = get_settings()
    if not settings.live_db.exists():
        console.print("[yellow]No database yet.[/yellow] Run [bold]soccer ingest[/bold] first.")
        raise typer.Exit(code=1)

    with LiveDB(settings.live_db) as db:
        entries = AliasStore(db).all()

    if not entries:
        console.print(
            "[dim]No aliases defined.[/dim] Find candidates with "
            "[bold]soccer aliases-suggest[/bold]."
        )
        return

    table = Table(header_style="bold")
    table.add_column("Type")
    table.add_column("Alias (normalized)")
    table.add_column("Canonical")
    table.add_column("Country")
    for entry in entries:
        table.add_row(
            entry.entity_type,
            entry.alias_name,
            entry.canonical_name,
            entry.country or "[dim]any[/dim]",
        )
    console.print(table)


@app.command("alias-add")
def alias_add(
    variant: str = typer.Argument(..., help="The spelling to route (e.g. 'PSV Eindhoven')."),
    canonical: str = typer.Argument(..., help="The canonical form (e.g. 'PSV')."),
    entity_type: str = typer.Option("team", help="team or competition."),
    country: str | None = typer.Option(None, help="Restrict the alias to one country."),
) -> None:
    """Declare that two names refer to the same entity.

    Applies to entities seen after this point. Existing splits are not retro-merged.
    """
    settings = get_settings()
    settings.ensure_dirs()
    with LiveDB(settings.live_db) as db:
        try:
            AliasStore(db).add(entity_type, variant, canonical, country=country)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc
    scope = f" in {country}" if country else ""
    console.print(
        f"[green]Added.[/green] {entity_type} '{variant}' resolves to "
        f"'{canonical}'{scope} from now on."
    )


@app.command("aliases-suggest")
def aliases_suggest(
    entity_type: str = typer.Option("team", help="team or competition."),
) -> None:
    """Surface probable duplicate entities that may need an alias."""
    settings = get_settings()
    if not settings.live_db.exists():
        console.print("[yellow]No database yet.[/yellow] Run [bold]soccer ingest[/bold] first.")
        raise typer.Exit(code=1)

    with LiveDB(settings.live_db) as db:
        candidates = suggest_duplicates(db, entity_type)

    if not candidates:
        console.print(
            f"[green]No probable {entity_type} duplicates found.[/green] "
            "(Translations like Köln/Cologne cannot be auto-detected -- add those "
            "directly with [bold]soccer alias-add[/bold].)"
        )
        return

    console.print(f"[bold]{len(candidates)} probable duplicate(s):[/bold]\n")
    for candidate in candidates:
        where = f" ({candidate.country})" if candidate.country else ""
        console.print(f"  {candidate.name_a!r} ~ {candidate.name_b!r}{where}")
        console.print(f"    [dim]{candidate.reason}[/dim]")
        console.print(f"    [cyan]{candidate.alias_command()}[/cyan]\n")


@app.command()
def rebuild() -> None:
    """Re-derive all live state from the immutable raw snapshots, applying aliases.

    Use after adding aliases (to retro-apply them) or after any resolver/mapper fix.
    Aliases are preserved; everything else is rebuilt from raw. Atomic: a failure
    leaves the previous state intact.
    """
    settings = get_settings()
    if not settings.raw_dir.exists():
        console.print("[yellow]No raw snapshots.[/yellow] Run [bold]soccer ingest[/bold] first.")
        raise typer.Exit(code=1)

    raw = RawStore(settings.raw_dir)
    with LiveDB(settings.live_db) as db:
        try:
            summary = IngestPipeline(db).replay_from_raw(raw)
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc
    console.print(f"  {summary}")
    console.print("\n[green]Rebuilt.[/green] View with [bold]soccer matches[/bold].")


@app.command()
def dashboard(port: int = typer.Option(8501, help="Port to serve on.")) -> None:
    """Launch the read-only Streamlit dashboard over the local database."""
    import subprocess
    import sys
    from importlib.resources import files
    from importlib.util import find_spec

    if find_spec("streamlit") is None:
        console.print(
            "[yellow]Streamlit is not installed.[/yellow] Install the dashboard extra:\n"
            "  pip install -e '.[dashboard]'"
        )
        raise typer.Exit(code=1)

    # files() imports only the (empty) package __init__, not app.py, so this does not
    # require streamlit to be importable at CLI load time.
    app_path = str(files("soccer.dashboard") / "app.py")
    console.print(f"[green]Starting dashboard[/green] on http://localhost:{port}")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", app_path, "--server.port", str(port)],
        check=False,
    )


@app.command("ingest-history")
def ingest_history(
    seasons: str = typer.Option("2526", help="Comma-separated seasons, e.g. 2526,2425."),
    divisions: str = typer.Option("E0", help="Comma-separated divisions, e.g. E0,E1,SP1."),
) -> None:
    """Download historical results from football-data.co.uk into the analytics store.

    A batch source: static season CSVs, no rate limit. Missing (season, division)
    combinations are skipped, not errors.
    """
    settings = get_settings()
    settings.ensure_dirs()
    season_list = [s.strip() for s in seasons.split(",") if s.strip()]
    division_list = [d.strip() for d in divisions.split(",") if d.strip()]

    raw = RawStore(settings.raw_dir)
    total = 0
    with FootballDataCoUk(raw) as source, AnalyticsDB(settings.analytics_db) as adb:
        for season in season_list:
            for division in division_list:
                results = source.fetch_division(season, division)
                if not results:
                    console.print(f"  [dim]{season}/{division}: not available[/dim]")
                    continue
                adb.load_results(results)
                total += len(results)
                console.print(f"  {season}/{division}: {len(results)} results")

    console.print(
        f"\n[green]Loaded {total} results.[/green] View a table with [bold]soccer table[/bold]."
    )


@app.command()
def table(
    season: str = typer.Option("2526", help="Season, e.g. 2526 for 2025/26."),
    division: str = typer.Option("E0", help="Division, e.g. E0 for the Premier League."),
) -> None:
    """Show a league table computed from ingested historical results."""
    settings = get_settings()
    if not settings.analytics_db.exists():
        console.print(
            "[yellow]No analytics data yet.[/yellow] Run [bold]soccer ingest-history[/bold] first."
        )
        raise typer.Exit(code=1)

    with AnalyticsDB(settings.analytics_db) as adb:
        rows = adb.league_table(season, division)

    if not rows:
        console.print(
            f"[dim]No results for {season}/{division}.[/dim] "
            f"Ingest it with [bold]soccer ingest-history --seasons {season} "
            f"--divisions {division}[/bold]."
        )
        return

    tbl = Table(title=f"{division} {season}", header_style="bold", title_justify="left")
    tbl.add_column("#", justify="right")
    tbl.add_column("Team")
    for col in ("P", "W", "D", "L", "GF", "GA", "GD", "Pts"):
        tbl.add_column(col, justify="right")

    for r in rows:
        style = "bold green" if r.position == 1 else ""
        tbl.add_row(
            str(r.position),
            r.team,
            str(r.played),
            str(r.won),
            str(r.drawn),
            str(r.lost),
            str(r.goals_for),
            str(r.goals_against),
            f"{r.goal_difference:+d}",
            Text(str(r.points), style=style),
        )
    console.print(tbl)


if __name__ == "__main__":
    app()
