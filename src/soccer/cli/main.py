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
    new_leagues: str = typer.Option(
        "", help="Comma-separated 'new league' country codes, e.g. BRA,ARG,USA."
    ),
    recent: int = typer.Option(3, help="For new leagues: keep the N most-recent seasons."),
) -> None:
    """Download historical results from football-data.co.uk into the analytics store.

    A batch source: static season CSVs, no rate limit. Missing (season, division)
    combinations are skipped, not errors. `--new-leagues` pulls the extra-country files
    (Brazil, Argentina, ...) which use a single-file, all-seasons schema.
    """
    settings = get_settings()
    settings.ensure_dirs()
    season_list = [s.strip() for s in seasons.split(",") if s.strip()]
    division_list = [d.strip() for d in divisions.split(",") if d.strip()]
    new_list = [c.strip().upper() for c in new_leagues.split(",") if c.strip()]

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
        for code in new_list:
            results = source.fetch_new_league(code, recent_seasons=recent)
            if not results:
                console.print(f"  [dim]{code}: not available[/dim]")
                continue
            adb.load_results(results)
            total += len(results)
            n_seasons = len({r.season for r in results})
            console.print(f"  {code}: {len(results)} results ({n_seasons} seasons)")

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


@app.command("power-rankings")
def power_rankings(
    season: str = typer.Option("2526", help="Season, e.g. 2526."),
    division: str = typer.Option("E0", help="Division, e.g. E0."),
    top: int = typer.Option(0, help="Show only the top N (0 = all)."),
) -> None:
    """Elo power rankings computed from historical results."""
    from soccer.models.elo import power_ranking

    outcomes = _load_outcomes(season, division)
    ranking = power_ranking(outcomes)
    names = _display_names(outcomes)
    if top > 0:
        ranking = ranking[:top]

    tbl = Table(title=f"Elo power rankings — {division} {season}", header_style="bold")
    tbl.add_column("#", justify="right")
    tbl.add_column("Team")
    tbl.add_column("Elo", justify="right")
    tbl.add_column("P", justify="right")
    for r in ranking:
        style = "bold green" if r.position == 1 else ""
        tbl.add_row(
            str(r.position),
            names.get(r.team, r.team),
            Text(f"{r.rating:.0f}", style=style),
            str(r.played),
        )
    console.print(tbl)


@app.command()
def forecast(
    home: str = typer.Argument(..., help="Home team, e.g. Arsenal."),
    away: str = typer.Argument(..., help="Away team, e.g. Chelsea."),
    season: str = typer.Option("2526", help="Season to fit the model on."),
    division: str = typer.Option("E0", help="Division to fit the model on."),
    mle: bool = typer.Option(False, "--mle", help="Use the Dixon-Coles MLE model."),
) -> None:
    """Forecast a match: outcome probabilities, expected goals, and likely scorelines."""
    from soccer.domain.names import normalize_name
    from soccer.models.elo import EloConfig, compute_ratings, expected_score
    from soccer.models.poisson import fit_poisson

    outcomes = _load_outcomes(season, division)
    names = _display_names(outcomes)
    if mle:
        from soccer.models.dixon_coles import fit_dixon_coles

        model = fit_dixon_coles(outcomes)
    else:
        model = fit_poisson(outcomes)

    home_norm, away_norm = normalize_name(home), normalize_name(away)
    missing = [
        original
        for original, norm in ((home, home_norm), (away, away_norm))
        if norm not in model.strengths
    ]
    if missing:
        console.print(f"[red]Unknown team(s):[/red] {', '.join(missing)}")
        console.print(f"[dim]Available: {', '.join(sorted(names.values()))}[/dim]")
        raise typer.Exit(code=1)

    fc = model.forecast(home_norm, away_norm)
    ratings = compute_ratings(outcomes)
    elo_home = expected_score(ratings[home_norm], ratings[away_norm], EloConfig())
    hn, an = names[home_norm], names[away_norm]

    console.print(f"\n[bold]{hn}[/bold] vs [bold]{an}[/bold]  [dim]({division} {season})[/dim]\n")
    console.print(
        f"  Expected goals   {hn} [bold]{fc.home_expected:.2f}[/bold]  -  "
        f"[bold]{fc.away_expected:.2f}[/bold] {an}"
    )
    console.print(
        f"  Outcome          {hn} [green]{fc.prob_home:.0%}[/green]   "
        f"Draw [yellow]{fc.prob_draw:.0%}[/yellow]   "
        f"{an} [cyan]{fc.prob_away:.0%}[/cyan]"
    )
    scores = "   ".join(f"{x}-{y} ({p:.0%})" for x, y, p in fc.top_scores)
    console.print(f"  Likely scores    {scores}")
    console.print(
        f"  Elo              {hn} {ratings[home_norm]:.0f}  vs  {ratings[away_norm]:.0f} {an}"
        f"   [dim](home expected {elo_home:.2f})[/dim]\n"
    )


@app.command()
def backtest(
    season: str = typer.Option("2526", help="Season to backtest on."),
    division: str = typer.Option("E0", help="Division to backtest on."),
    min_history: int = typer.Option(60, help="Matches of warmup before the first forecast."),
    model: str = typer.Option("ratio", help="Forecast model: 'ratio' or 'dc' (Dixon-Coles MLE)."),
    half_life: float = typer.Option(
        0.0, help="Time-decay half-life in days for the DC model (0 = no decay)."
    ),
    history: str = typer.Option(
        "",
        help="Comma-separated earlier seasons to also train on, e.g. 2425,2324. "
        "Eliminates the warmup so early-season matches can be forecast.",
    ),
) -> None:
    """Walk-forward backtest of the forecast: log loss, Brier, and calibration."""
    from soccer.models.backtest import backtest_dixon_coles, backtest_poisson

    outcomes = _load_outcomes(season, division)
    prior: list = []
    if history:
        with AnalyticsDB(get_settings().analytics_db) as adb:
            for s in history.split(","):
                prior.extend(adb.outcomes_for(s.strip(), division))
        # With prior seasons providing the teams and warmup, predict from match one.
        min_history = 0
        if model != "dc" or half_life <= 0:
            console.print(
                "[yellow]Tip:[/yellow] older seasons weighted equally can drag the fit "
                "toward stale form. Pair --history with [bold]--model dc --half-life 140[/bold] "
                "so recency is weighted."
            )

    try:
        if model == "dc":
            import math as _math

            xi = _math.log(2) / half_life if half_life > 0 else 0.0
            console.print(
                "[dim]Fitting Dixon-Coles by MLE before each match; this is slower...[/dim]"
            )
            result = backtest_dixon_coles(
                outcomes, min_history=min_history, time_decay=xi, prior=prior
            )
        else:
            result = backtest_poisson(outcomes, min_history=min_history, prior=prior)
    except ValueError as exc:
        console.print(f"[yellow]{exc}[/yellow]")
        raise typer.Exit(code=1) from exc

    skill = result.log_loss_skill
    skill_style = "green" if skill > 0 else "red"
    model_label = "Dixon-Coles MLE" if model == "dc" else "ratio-method Poisson"
    console.print(
        f"\n[bold]Backtest - {division} {season}[/bold]  "
        f"[dim]({result.n_predictions} forecasts, {model_label}, walk-forward)[/dim]\n"
    )
    console.print(
        f"  Log loss   model [bold]{result.log_loss:.3f}[/bold]  vs  "
        f"baseline {result.baseline_log_loss:.3f}   "
        f"([{skill_style}]{skill:+.1%} skill[/{skill_style}])"
    )
    console.print(
        f"  Brier      model [bold]{result.brier:.3f}[/bold]  vs  "
        f"baseline {result.baseline_brier:.3f}"
    )

    cal = Table(
        title="\nCalibration (home-win probability)",
        title_justify="left",
        header_style="bold",
    )
    cal.add_column("Predicted band")
    cal.add_column("Mean pred", justify="right")
    cal.add_column("Observed", justify="right")
    cal.add_column("N", justify="right")
    for b in result.calibration:
        # Colour observed vs predicted agreement: close = good calibration.
        gap = abs(b.mean_predicted - b.observed_rate)
        style = "green" if gap < 0.1 else "yellow" if gap < 0.2 else "red"
        cal.add_row(
            f"{b.lower:.0%}-{b.upper:.0%}",
            f"{b.mean_predicted:.0%}",
            Text(f"{b.observed_rate:.0%}", style=style),
            str(b.count),
        )
    console.print(cal)
    console.print(
        "\n[dim]Lower log loss is better; positive skill beats predicting the league's "
        "base rates. Well-calibrated = observed tracks predicted.[/dim]\n"
    )


@app.command()
def simulate(
    season: str = typer.Option("2526", help="Season to simulate."),
    division: str = typer.Option("E0", help="Division to simulate."),
    after: str | None = typer.Option(
        None,
        help="Only simulate matches on/after this date (YYYY-MM-DD); earlier matches "
        "set the current table. Omit to replay the whole season.",
    ),
    sims: int = typer.Option(10000, help="Number of Monte Carlo runs."),
    top: int = typer.Option(4, help="Size of the 'top' bucket (e.g. 4 for top four)."),
    seed: int | None = typer.Option(None, help="Seed for reproducible runs."),
) -> None:
    """Monte Carlo league simulation: title, top-N and relegation probabilities."""
    from datetime import date as _date

    from soccer.models.poisson import fit_poisson
    from soccer.models.simulation import simulate_season

    outcomes = _load_outcomes(season, division)
    names = _display_names(outcomes)

    cutoff = _date.fromisoformat(after) if after else None
    played = [o for o in outcomes if cutoff and o.match_date < cutoff]
    remaining_matches = [o for o in outcomes if not cutoff or o.match_date >= cutoff]

    # Fit on what has been played; before any cutoff, use the whole season's strengths.
    model = fit_poisson(played or outcomes)
    points_start, gd_start = _standings_from(played)
    remaining = [(o.home_norm, o.away_norm) for o in remaining_matches]

    result = simulate_season(
        model,
        remaining,
        points_start=points_start,
        goal_diff_start=gd_start,
        teams=list(names),
        n_sims=sims,
        top_n=top,
        seed=seed,
    )

    scope = f"rest of season from {after}" if cutoff else "full-season replay"
    tbl = Table(
        title=f"Simulation - {division} {season} ({scope}, {sims:,} runs)",
        header_style="bold",
    )
    tbl.add_column("Team")
    tbl.add_column("Title", justify="right")
    tbl.add_column(f"Top {top}", justify="right")
    tbl.add_column("Releg", justify="right")
    tbl.add_column("xPts", justify="right")
    for p in result.projections:
        title_style = "bold green" if p.title_pct >= 0.5 else ""
        tbl.add_row(
            names.get(p.team, p.team),
            Text(f"{p.title_pct:.0%}", style=title_style),
            f"{p.top_pct:.0%}",
            f"{p.relegation_pct:.0%}" if p.relegation_pct >= 0.005 else "-",
            f"{p.expected_points:.0f}",
        )
    console.print(tbl)


def _standings_from(played: list) -> tuple[dict[str, int], dict[str, int]]:
    points: dict[str, int] = {}
    goal_diff: dict[str, int] = {}
    for o in played:
        hp = 3 if o.fthg > o.ftag else 1 if o.fthg == o.ftag else 0
        ap = 3 if o.ftag > o.fthg else 1 if o.fthg == o.ftag else 0
        points[o.home_norm] = points.get(o.home_norm, 0) + hp
        points[o.away_norm] = points.get(o.away_norm, 0) + ap
        goal_diff[o.home_norm] = goal_diff.get(o.home_norm, 0) + (o.fthg - o.ftag)
        goal_diff[o.away_norm] = goal_diff.get(o.away_norm, 0) + (o.ftag - o.fthg)
    return points, goal_diff


def _load_outcomes(season: str, division: str) -> list:
    """Shared loader for the forecasting commands."""
    settings = get_settings()
    if not settings.analytics_db.exists():
        console.print(
            "[yellow]No analytics data yet.[/yellow] Run [bold]soccer ingest-history[/bold] first."
        )
        raise typer.Exit(code=1)
    with AnalyticsDB(settings.analytics_db) as adb:
        outcomes = adb.outcomes_for(season, division)
    if not outcomes:
        console.print(
            f"[dim]No results for {season}/{division}.[/dim] Ingest with "
            f"[bold]soccer ingest-history --seasons {season} --divisions {division}[/bold]."
        )
        raise typer.Exit(code=1)
    return outcomes


def _display_names(outcomes: list) -> dict[str, str]:
    names: dict[str, str] = {}
    for o in outcomes:
        names[o.home_norm] = o.home
        names[o.away_norm] = o.away
    return names


@app.command()
def mcp() -> None:
    """Run the MCP server (stdio), exposing the platform to LLM clients."""
    from importlib.util import find_spec

    if find_spec("mcp") is None:
        console.print(
            "[yellow]The MCP SDK is not installed.[/yellow] Install the extra:\n"
            "  pip install -e '.[mcp]'"
        )
        raise typer.Exit(code=1)
    from soccer.mcp.server import run

    run()


@app.command("ingest-events")
def ingest_events(
    match: int = typer.Option(0, help="A single StatsBomb match_id."),
    competition: int = typer.Option(0, help="Competition id (with --season) for a whole season."),
    season: int = typer.Option(0, help="Season id (with --competition)."),
    from_raw: bool = typer.Option(
        False, "--from-raw", help="Re-parse already-snapshotted matches offline (no fetch)."
    ),
) -> None:
    """Ingest StatsBomb open-data events into the analytics store: shots (xG) and full
    per-player match stats (passing, carrying, defending, discipline).

    StatsBomb open data is under a proprietary EULA (no redistribution, no commercial
    use, logo attribution) -- an explicit opt-in. Data stays in the local gitignored
    store. `--from-raw` re-parses every match already snapshotted, no network needed --
    the way to backfill richer stats after a parser change.
    """
    from soccer.sources.registry import SourceId
    from soccer.sources.statsbomb import ATTRIBUTION as SB_ATTRIBUTION
    from soccer.sources.statsbomb import (
        StatsBomb,
        parse_match_meta,
        parse_player_stats,
        parse_shots,
    )
    from soccer.storage.analytics_db import MatchMeta

    settings = get_settings()
    settings.ensure_dirs()
    raw = RawStore(settings.raw_dir)

    with StatsBomb(raw) as sb, AnalyticsDB(settings.analytics_db) as adb:
        metas: list[MatchMeta] = []
        if from_raw:
            match_ids = adb.matches_with_shots()
            if not match_ids:
                console.print("[yellow]Nothing snapshotted yet.[/yellow] Ingest a match first.")
                raise typer.Exit(code=1)
            console.print(f"[dim]Re-parsing {len(match_ids)} snapshotted match(es)...[/dim]")
        elif match:
            match_ids = [match]
        elif competition and season:
            listing = sb.matches(competition, season)
            match_ids = [m["match_id"] for m in listing]
            metas = [MatchMeta(**parse_match_meta(m)) for m in listing if m.get("match_id")]
            console.print(f"[dim]{len(match_ids)} matches in {competition}/{season}...[/dim]")
        else:
            console.print("[yellow]Provide --match, or --competition and --season.[/yellow]")
            raise typer.Exit(code=1)

        if metas:
            adb.load_match_meta(metas)

        total_shots, total_players = 0, 0
        for match_id in match_ids:
            snapshot = raw.latest(SourceId.STATSBOMB, f"events_{match_id}")
            if from_raw:
                events = snapshot.payload if snapshot else []
            else:
                # Static open data: reuse a snapshot if we have one, only fetch if not.
                events = snapshot.payload if snapshot else sb.fetch_events(match_id)
            if not events:
                continue
            shots = parse_shots(events, match_id)
            stats = parse_player_stats(events, match_id)
            if shots:
                adb.load_shots(shots)
                total_shots += len(shots)
            if stats:
                adb.load_player_stats(stats)
                total_players += len(stats)
            if match or len(match_ids) <= 10:
                console.print(f"  match {match_id}: {len(shots)} shots, {len(stats)} players")

    console.print(
        f"\n[green]Loaded {total_shots} shots and {total_players} player-match rows[/green] "
        f"across {len(match_ids)} match(es)."
    )
    console.print(f"[dim]{SB_ATTRIBUTION}[/dim]")


@app.command()
def xg(match: int = typer.Argument(..., help="StatsBomb match_id.")) -> None:
    """Expected-goals summary for a match: team xG and the top shooters."""
    from soccer.sources.statsbomb import ATTRIBUTION as SB_ATTRIBUTION

    settings = get_settings()
    if not settings.analytics_db.exists():
        console.print(
            "[yellow]No analytics data.[/yellow] Run [bold]soccer ingest-events[/bold] first."
        )
        raise typer.Exit(code=1)

    with AnalyticsDB(settings.analytics_db) as adb:
        teams = adb.team_xg(match)
        shooters = adb.top_shooters(match, 8)

    if not teams:
        console.print(
            f"[dim]No shots for match {match}.[/dim] Ingest it: "
            f"[bold]soccer ingest-events --match {match}[/bold]."
        )
        return

    team_tbl = Table(title=f"xG — match {match}", header_style="bold", title_justify="left")
    team_tbl.add_column("Team")
    team_tbl.add_column("xG", justify="right")
    team_tbl.add_column("Goals", justify="right")
    team_tbl.add_column("Shots", justify="right")
    for r in teams:
        team_tbl.add_row(r.name, f"{r.xg:.2f}", str(r.goals), str(r.shots))
    console.print(team_tbl)

    shooter_tbl = Table(title="\nTop shooters by xG", header_style="bold", title_justify="left")
    shooter_tbl.add_column("Player")
    shooter_tbl.add_column("Team", style="dim")
    shooter_tbl.add_column("xG", justify="right")
    shooter_tbl.add_column("G", justify="right")
    for r in shooters:
        style = "bold green" if r.goals > 0 else ""
        shooter_tbl.add_row(r.name, r.team or "", Text(f"{r.xg:.2f}", style=style), str(r.goals))
    console.print(shooter_tbl)
    console.print(f"\n[dim]{SB_ATTRIBUTION}[/dim]")


@app.command()
def serve(
    live_interval: int = typer.Option(60, help="Seconds between live-score ingests."),
    fixtures_interval: int = typer.Option(3600, help="Seconds between fixture ingests."),
    once: bool = typer.Option(False, "--once", help="Run all due jobs once and exit."),
) -> None:
    """Run ingestion unattended on a cadence: live scores, fixtures, and housekeeping.

    A failing source is logged and retried next interval, never crashes the loop. Stop
    with Ctrl-C.
    """
    import time
    from datetime import timedelta

    from soccer.ingest.scheduler import Job, JobResult, Scheduler

    settings = get_settings()
    settings.ensure_dirs()
    raw = RawStore(settings.raw_dir)

    def live_job() -> str:
        async def run_it() -> str:
            with LiveDB(settings.live_db) as db:
                async with TheSportsDB(
                    raw,
                    api_key=settings.thesportsdb_key,
                    rate_limit_per_minute=settings.thesportsdb_rpm,
                ) as tsdb:
                    return str(await IngestPipeline(db).ingest_thesportsdb(tsdb))

        return asyncio.run(run_it())

    def fixtures_job() -> str:
        if not settings.football_data_org_token:
            return "skipped (no football-data.org token)"

        async def run_it() -> str:
            today = datetime.now(UTC).date()
            with LiveDB(settings.live_db) as db:
                async with FootballDataOrg(
                    settings.football_data_org_token,
                    raw,
                    rate_limit_per_minute=settings.football_data_org_rpm,
                ) as fd:
                    summary = await IngestPipeline(db).ingest_football_data(
                        fd, today, today + timedelta(days=2)
                    )
            return str(summary)

        return asyncio.run(run_it())

    def prune_job() -> str:
        removed = raw.prune(SourceId.THESPORTSDB, "livescore", keep_days=7)
        return f"pruned {removed} old live snapshot(s)"

    scheduler = Scheduler(
        jobs=[
            Job("live", timedelta(seconds=live_interval), live_job),
            Job("fixtures", timedelta(seconds=fixtures_interval), fixtures_job),
            Job("prune", timedelta(days=1), prune_job),
        ]
    )

    def report(result: JobResult) -> None:
        stamp = datetime.now(UTC).strftime("%H:%M:%S")
        if result.ok:
            console.print(f"[dim]{stamp}[/dim] [green]{result.name}[/green]  {result.message}")
        else:
            console.print(f"[dim]{stamp}[/dim] [red]{result.name} failed[/red]  {result.message}")

    if once:
        for result in scheduler.run_due():
            report(result)
        return

    console.print(
        f"[green]Serving[/green] — live every {live_interval}s, fixtures every "
        f"{fixtures_interval}s. Ctrl-C to stop.\n"
    )
    tick = max(1, min(live_interval, 15))
    try:
        while True:
            for result in scheduler.run_due():
                report(result)
            time.sleep(tick)
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped.[/yellow]")


@app.command()
def players(
    top: int = typer.Option(20, help="Number of players to show."),
    min_shots: int = typer.Option(3, help="Minimum shots to qualify."),
    by: str = typer.Option("xg", help="Rank by: xg, goals, or npxg (non-penalty xG)."),
) -> None:
    """Player leaderboard from ingested StatsBomb shots: xG, goals, and finishing."""
    from soccer.sources.statsbomb import ATTRIBUTION as SB_ATTRIBUTION

    settings = get_settings()
    if not settings.analytics_db.exists():
        console.print(
            "[yellow]No event data.[/yellow] Run [bold]soccer ingest-events[/bold] first."
        )
        raise typer.Exit(code=1)

    with AnalyticsDB(settings.analytics_db) as adb:
        rows = adb.player_leaderboard(limit=top, min_shots=min_shots, order=by)

    if not rows:
        console.print(
            "[dim]No player data.[/dim] Ingest a competition, e.g. "
            "[bold]soccer ingest-events --competition 43 --season 106[/bold] (World Cup 2022)."
        )
        return

    tbl = Table(title=f"Player leaderboard (by {by})", header_style="bold")
    tbl.add_column("#", justify="right")
    tbl.add_column("Player")
    tbl.add_column("Team", style="dim")
    tbl.add_column("xG", justify="right")
    tbl.add_column("npxG", justify="right")
    tbl.add_column("G", justify="right")
    tbl.add_column("Sh", justify="right")
    tbl.add_column("G-xG", justify="right")
    for i, r in enumerate(rows, 1):
        diff = r.xg_diff
        diff_style = "green" if diff > 0.5 else "red" if diff < -0.5 else "dim"
        tbl.add_row(
            str(i),
            r.player,
            r.team,
            f"{r.xg:.1f}",
            f"{r.npxg:.1f}",
            str(r.goals),
            str(r.shots),
            Text(f"{diff:+.1f}", style=diff_style),
        )
    console.print(tbl)
    console.print("[dim]G-xG: goals minus expected goals — positive = clinical finishing.[/dim]")
    console.print(f"[dim]{SB_ATTRIBUTION}[/dim]")


if __name__ == "__main__":
    app()
