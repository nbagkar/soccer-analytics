# soccer-analytics

A local-first "football intelligence centre" built under a hard **$0 constraint** — no
paid APIs, no subscriptions, no cloud hosting, no new hardware. It runs on your own
machine; when it's off, ingestion pauses and catches up next time.

The distinguishing idea isn't the feature list — it's the discipline. **Every data
source was verified by direct fetch before any code depended on it.** That process
corrected several load-bearing assumptions (see
[`docs/source-verification.md`](docs/source-verification.md)), and the corrections are
encoded as tests so they can't quietly regress.

## The honest reality of $0

| Genuinely free | Not reliably free |
|---|---|
| Near-live scores (one narrow source) | Live scores across major leagues |
| Fixtures, results, standings (12 competitions) | Lineups, scorers, squads |
| Historical event analytics (open datasets) | Live xG / passing networks |
| Elo / Dixon-Coles style forecasting | Comprehensive injuries & transfers |
| Local dashboards, alerts, provenance | Guaranteed uptime or latency |

Concretely: football-data.org's free tier gives **delayed** scores for 12 competitions
at 10 requests/min (no lineups). TheSportsDB is currently the only free source with
genuine live scores — and that access is undocumented, so the design degrades to
delayed rather than breaking when it disappears. Full detail, with quoted terms, is in
[`docs/source-verification.md`](docs/source-verification.md).

## Status

Working end-to-end today: source adapters → immutable raw snapshots → canonical
entity/match resolution with a source crosswalk → SQLite live state → curated aliases →
replay-from-raw, plus historical results (football-data.co.uk → DuckDB) with computed
league tables, Elo power rankings, ratio-method and Dixon-Coles-MLE match forecasting,
Monte Carlo league simulations, walk-forward forecast backtesting, StatsBomb event
analytics (real xG, shot data), and an MCP server that makes it all queryable in natural
language. The read-only Streamlit dashboard spans a Live Centre, upcoming **Fixtures with
forecasts**, a **Season** oracle (Monte Carlo title/top-4/relegation odds), league
Analytics, **Trends** (form), **Records** (streaks), a full **Forecast** market slate with
a value/EV calculator, a **Match centre** (xG timeline, shot map, shot log), and a
**Players** view — a full-event leaderboard filterable by competition and season, with
per-player **percentile scouting fingerprints**. Forecasts fit on a rolling multi-season
window and re-fit as results land. **353 passing tests.**

The forecasting is evaluated honestly rather than assumed good. On real 2025/26 Premier
League data the walk-forward backtest shows the ratio-method Poisson beats a base-rate
baseline by only ~3% log-loss skill and is somewhat over-confident on medium-strong home
favourites. The "proper" fix — full Dixon-Coles maximum-likelihood fitting — does **not**
improve on it on a single season (measured, not assumed); only adding time-decay
weighting of recent form nudges it ahead. The real levers are more data and better
features (xG), not a fancier fitting method — a finding worth more than a hidden
disappointment.

This realizes the original 12-week plan's full scope. Remaining ideas are optional
extensions: Wyscout event data (a second, CC-BY event source) and richer visualizations.

On betting value: there is no free source of odds for *upcoming* matches, so rather than
fake a live edge, `soccer value` measures a real one against history — betting the model's
positive-EV picks at the closing 1X2 odds already in the football-data.co.uk files
(Pinnacle's close preferred). On 2024/25 Premier League the model beats the base-rate
baseline but **loses to the closing line on log loss** (≈0.99 vs ≈0.97), and flat-stake
yields swing either side of zero with the model and warmup — i.e. noise, not edge. The
dashboard turns the same primitives into a what-if calculator: enter the odds you can
actually get and see the model's edge, EV, and Kelly stake.

## Quickstart

Requires **Python 3.12** (capped below 3.13 by the `socceraction` dependency).

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e .

cp .env.example .env
# Add a free football-data.org token (https://www.football-data.org/client/register)
# to SOCCER_FOOTBALL_DATA_ORG_TOKEN. TheSportsDB works with the default free key.

soccer doctor        # what's configured and what each source can actually do
soccer ingest        # fetch, resolve to canonical matches, persist state
soccer matches       # the ingested live centre, read from SQLite
```

`.env` is gitignored — never commit your token.

## Commands

| Command | What it does |
|---|---|
| `soccer doctor` | Configuration, source availability, capability coverage, licensing flags |
| `soccer sources` | What each source provides and the caveats attached |
| `soccer ingest` | Fetch enabled sources → resolve → persist canonical match state |
| `soccer matches` | Show ingested match state (add `--in-play`) |
| `soccer live` | Ad-hoc live scores straight from TheSportsDB |
| `soccer ingest-history` | Download historical results (football-data.co.uk) into DuckDB. `--new-leagues BRA,ARG` adds extra-country files (Brazil, ...) |
| `soccer table` | League table computed from historical results |
| `soccer power-rankings` | Elo power rankings from historical results |
| `soccer forecast` | Match forecast — outcome probabilities, expected goals, likely scores |
| `soccer simulate` | Monte Carlo league simulation — title, top-N, relegation odds |
| `soccer backtest` | Walk-forward forecast evaluation — log loss, Brier, calibration |
| `soccer value` | Market-edge analysis — does the model beat the closing line? Yield vs Pinnacle odds |
| `soccer ingest-events` | Ingest StatsBomb open-data events — shots (xG) and full per-player stats. `--from-raw` re-parses snapshots offline. Proprietary EULA, opt-in |
| `soccer xg` | Expected-goals summary for a match — team xG and top shooters |
| `soccer players` | Player leaderboard — xG, non-penalty xG, goals, finishing (G-xG) |
| `soccer dashboard` | Launch the read-only Streamlit dashboard |
| `soccer serve` | Run ingestion unattended on a cadence (live, fixtures, current-season history auto-refresh, housekeeping) |
| `soccer mcp` | Run the MCP server (stdio) — the platform as LLM tools + prompts |
| `soccer aliases-suggest` | Surface probable duplicate entities to review |
| `soccer alias-add` | Declare two names refer to the same entity |
| `soccer aliases` | List curated aliases |
| `soccer rebuild` | Re-derive all state from raw snapshots (applies aliases retroactively) |
| `soccer prune` | Delete old live-feed snapshots |
| `soccer init` | Create data directories |

## Architecture

```
free APIs / open datasets
        │
   source adapters ── immutable raw JSON snapshots (provenance + replay)
        │
   normalize + resolve ── canonical ids, source crosswalk, aliases
        │
   SQLite live state (WAL) ── current score/status per canonical match
```

- `src/soccer/sources/` — adapters + the capability/licence registry
- `src/soccer/storage/` — immutable raw snapshots, SQLite live DB
- `src/soccer/domain/` — name normalization, entity/match resolution, aliases, state
- `src/soccer/ingest/` — mappers, pipeline, rate limiting
- `src/soccer/cli/` — the `soccer` command

Two design commitments worth calling out. **Identity is never assumed from names** — a
provider's stable id is trusted for its own records; linking across sources by name is
a recorded, lower-confidence inference, never a silent merge. And **a failed source is
never an empty success** — it degrades to flagged stale data or a clear error, because
an empty fixture list and a broken API must not look alike.

## Data sources & licensing

This project reads documented and open sources only; no website scraping. Source terms
vary and are enforced by design:

- **football-data.org** — free tier, fixtures/results/standings
- **TheSportsDB** — live scores; attribution required, resale prohibited
- **StatsBomb Open Data** — historical events, but a **proprietary EULA**: no
  redistribution, no commercial use, logo attribution required
- **Wyscout** (CC BY 4.0) and **SkillCorner** (MIT) — the genuinely permissive datasets

The `data/` directory (raw snapshots, local database) is gitignored — partly for size,
partly because some sources' terms forbid redistributing their payloads.

## Development

```bash
pip install -e ".[dev]"
pytest            # 353 tests
ruff check .      # lint
ruff format .     # format
```

Tests lean toward failure paths — rate-limit exhaustion, stale-cache fallback,
malformed payloads, cross-source identity edge cases — because that's where a
multi-source free-data pipeline actually breaks.
