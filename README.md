# soccer-analytics

[![CI](https://github.com/nbagkar/soccer-analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/nbagkar/soccer-analytics/actions/workflows/ci.yml)

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
| Fixtures, results, standings, squads (12 competitions) | Confirmed lineups, scorers |
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
Monte Carlo league simulations, walk-forward forecast backtesting, and StatsBomb event
analytics (real xG, shot data). The read-only Streamlit dashboard has:

- **Ask a question** — the built-in assistant (below), at parity with the pages
- **Live scores** — what's in play now, else the last week's results
- **Predictions** — upcoming fixtures with forecasts (Champions League ties via an
  experimental cross-league model: domestic ratings plus league strengths fitted on past
  cup results, +3.5% ± 2% skill over base rates in a walk-forward backtest); any **Matchup** (market slate,
  correct-score grid, team-news adjustment, attribution); the **Season** projection
  (Monte Carlo title / top-four / relegation odds); and the **Scorecard** — model vs the
  bookmaker's closing line, with calibration
- **League tables** — standings, form guide, and underlying (xG) performance
- **Teams** — one club at a glance, with a season-on-season trajectory
- **Records** — this season's streaks and standout results, plus all-time honours
- **Analysis** — StatsBomb match xG (timeline, shot map, shot log) and player scouting
  (leaderboard, percentile fingerprints, match logs, similar players)
- **About & sources** — data health and optional data top-ups

Forecasts fit on a rolling multi-season window and re-fit as results land.

The forecasting is evaluated honestly rather than assumed good. On real 2025/26 Premier
League data the walk-forward backtest shows the ratio-method Poisson beats a base-rate
baseline by only ~3% log-loss skill and is somewhat over-confident on medium-strong home
favourites. The "proper" fix — full Dixon-Coles maximum-likelihood fitting — does **not**
improve on it on a single season (measured, not assumed); only adding time-decay
weighting of recent form nudges it ahead. The real levers are more data and better
features (xG), not a fancier fitting method — a finding worth more than a hidden
disappointment.

The biggest single gain since came from the market itself. Team ratings now use each
*played* match's closing odds, inverted into the market's implied expected goals (no odds
for upcoming matches are needed or used). Walk-forward against the closing line, that closed
about a third of the model's log-loss gap — +0.0194 → +0.0127 on 10,021 matches across nine
leagues it was never tuned on, with all 18 leagues tested improving. The model still trails
the closing line; it is a better-informed forecaster, not a source of betting edge.

This realizes the original 12-week plan's full scope. Remaining ideas are optional
extensions: Wyscout event data (a second, CC-BY event source) and richer visualizations.

The dashboard is also usable without the terminal: a **Home** page shows data status and
loads/refreshes data with buttons (scores, fixtures, a league's history, a StatsBomb
player-data pack), and a built-in **Assistant** answers plain-English questions
("who's top of the Premier League?", "Arsenal vs Chelsea?", "how many goals did Messi
score?") entirely offline — rule-based, no LLM, nothing leaves the machine.

The dashboard keeps a local usage log (page visits, data actions, which kind of question
the assistant answered, and the text of questions it couldn't). `soccer usage` reports what
gets used and what never does, so features are kept or cut on evidence. It never leaves the
machine; set `SOCCER_USAGE_TRACKING=false` to turn it off.

Data stays current without the terminal. When the dashboard opens, anything stale is
refreshed in the background — results and fixtures after a day, injury news after a day
(with FPL enabled), squads after a week — and the Predictions page is pre-computed so its
first visit is fast. Live scores refresh when you look at them, once they're 30 minutes
old. Every page warns plainly if data is stale or a refresh failed, and adjusted forecasts
show the date of the team news they use. **Home → Keep it current** does the same on
demand, and `soccer serve` runs the very same refreshes unattended.

`soccer check` self-checks the loaded data (season ordering, duplicates, split club names,
future-dated results) and runs in `soccer doctor` and the weekly review — so a data defect
announces itself instead of waiting to be spotted on a page.

On betting value: there is no free source of odds for *upcoming* matches, so rather than
fake a live edge, `soccer value` measures a real one against history — betting the model's
positive-EV picks at the closing 1X2 odds already in the football-data.co.uk files
(Pinnacle's close preferred). On 2024/25 Premier League the model beats the base-rate
baseline but **loses to the closing line on log loss** (≈0.99 vs ≈0.97), and flat-stake
yields swing either side of zero with the model and warmup — i.e. noise, not edge. The
dashboard's Scorecard shows the same comparison; it deliberately has no betting
calculators or bet tracker, since the model has no demonstrated edge to bet on.

## Quickstart

Requires **Python 3.12+** (CI tests 3.12).

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
| `soccer serve` | Run the dashboard's refreshes unattended on a cadence (live scores, fixtures, current-season results, injury news and squads when stale, housekeeping) |
| `soccer check` | Self-check the loaded data; exits non-zero on errors |
| `soccer aliases-suggest` | Surface probable duplicate entities to review |
| `soccer alias-add` | Declare two names refer to the same entity |
| `soccer aliases` | List curated aliases |
| `soccer rebuild` | Re-derive all state from raw snapshots (applies aliases retroactively) |
| `soccer prune` | Delete old live-feed snapshots |
| `soccer init` | Create data directories |
| `soccer usage` | What actually gets used: pages, data actions, assistant intents, unanswered questions |

## Share the dashboard (free, public)

The dashboard is a Streamlit **server**, so it can't run on serverless hosts (Vercel,
Netlify) — and free PaaS tiers (Streamlit Cloud, Render, HF Spaces) use ephemeral disks, so
they'd re-download the data on every cold start rather than use your loaded-up store.

The free way to expose *this* instance — with all your data, nothing to re-fetch — is a
Cloudflare quick tunnel to your own machine:

```bash
brew install cloudflared
./scripts/serve_public.sh          # prompts for a password, prints a public https URL
```

It runs the dashboard locally and hands back a `https://<random>.trycloudflare.com` URL. The
tunnel is up only while the script runs (while your machine is on). Because the Home page can
trigger data downloads, the script requires a password: set `SOCCER_DASHBOARD_PASSWORD` (or
enter one when prompted) and the app gates on it. Ctrl-C stops the tunnel and the app.

Note the data licences: fetch-at-runtime for personal analysis is the permitted use — be
mindful before exposing a public instance widely.

## Architecture

```
free APIs / open datasets
        │
   source adapters ── immutable raw JSON snapshots (provenance + replay)
        │
   normalize + resolve ── canonical ids, source crosswalk, aliases
        │
   SQLite live state (WAL) ── current score/status per canonical match, team news, refresh log
        │
   DuckDB analytics ── historical results, shots, player stats, squads
        │
   models ── Poisson / Dixon-Coles, Elo, Monte Carlo, walk-forward evaluation
        │
   CLI · Streamlit dashboard · built-in assistant
```

- `src/soccer/sources/` — adapters + the capability/licence registry
- `src/soccer/storage/` — immutable raw snapshots, SQLite live DB, DuckDB analytics, data checks
- `src/soccer/domain/` — name normalization, entity/match resolution, aliases, state,
  availability, data freshness, usage log
- `src/soccer/ingest/` — mappers, pipeline, rate limiting, scheduler
- `src/soccer/models/` — forecasting, simulation, evaluation, market comparison
- `src/soccer/dashboard/` — pages, the assistant, refresh actions, background auto-refresh
- `src/soccer/cli/` — the `soccer` command

The `data/` directory holds `raw/` (snapshots), `live.sqlite`, `analytics.duckdb` and
`usage.sqlite` (the local usage log, kept separate so logging never invalidates data caches).

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
pip install -e ".[dashboard,dev]"
pytest              # full suite (~40s)
ruff check src tests
ruff format src tests
mypy src            # strict
```

CI runs all four on every push and pull request.

Tests lean toward failure paths — rate-limit exhaustion, stale-cache fallback,
malformed payloads, cross-source identity edge cases — because that's where a
multi-source free-data pipeline actually breaks.
