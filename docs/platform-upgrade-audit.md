# Phase 0 audit — "Football Intelligence Platform" upgrade plan

Audited 2026-11-16 against `main` @ `fd22db0`. Nothing was changed in `src/` or `tests/`.
Every number below was measured on this machine today unless marked *unverified*.

## 1. Headline

The plan describes a platform that **mostly already exists**. Of its ~45 named
capabilities, roughly 30 are built and backtested, ~8 are partial, and ~7 are genuinely
missing. The missing pieces cluster around one gap: **no current-season player data and
no structured odds for upcoming matches**. API-Football would close the first; a free
source already closes most of the second.

Four parts of the plan reverse decisions this project made deliberately, on evidence.
They need your call before implementation (section 7).

## 2. Current state (measured)

| Area | State |
|---|---|
| Code | 20.2k lines `src/`, 10.2k lines `tests/`, 165 commits, CI (ruff, mypy strict, pytest) |
| Tests | **699 passed** in 101 s (README says ~40 s; it has grown) |
| Results store | 275,240 matches, 34 divisions, 1993-07 → **2026-11-15**; 99.2% of post-2020 rows carry closing 1X2 odds; O/U 2.5 closing from 2019/20 |
| Event data | StatsBomb: 3,961 matches, 89k shots, 99.6k player-match rows, 6.18M actions, xT grid. Newest season 2025; **no current-season events** (catalog ceiling) |
| Live state | SQLite: 1,349 canonical teams, 4,308 canonical matches, 672 FPL availability rows (PL only), 5,058 squad players / 176 clubs |
| Freshness | Results/fixtures/injuries/squads all refreshed today; live scores last 2026-10-20 (page archived) |
| Raw snapshots | 903 MB immutable, replayable (`soccer rebuild`) |

**Usage evidence (`soccer usage`, 42 days):** Predictions 13 visits, Home 12, League
tables 6, Assistant 3, Teams 2. **Analysis (player scouting / shot maps) never visited.
The assistant answered one question in 42 days.** This should weigh on phase order.

## 3. Plan vs. reality, section by section

### 3.1 Data infrastructure (plan §3)

| Plan item | Exists? | Where / notes |
|---|---|---|
| football-data.org, TheSportsDB, StatsBomb adapters | ✅ | `sources/*.py`, plus football-data.co.uk (the main history + odds source the plan omits) and FPL |
| Capability + licence registry | ✅ | `sources/registry.py` — every source's real capabilities, verified by fetch, encoded as tests |
| Raw JSON snapshots, provenance, replay | ✅ | `storage/raw.py` |
| Rate limiting | ✅ | `ingest/ratelimit.py` (token bucket, 429 back-off) |
| "Never treat failure as empty success" | ✅ | Design commitment in README; `sources/errors.py` |
| Canonical schema for teams/matches | ✅ partial | `domain/crosswalk.py`, `matches.py` (live). Analytics side keys on normalized names via the audited name bridge (`soccer audit-names`). **Two identity systems, not one.** |
| Canonical **player** identity | ❌ | No player crosswalk anywhere. StatsBomb/FPL names never join. Needed before any cross-source scouting. |
| Data-health system | ✅ mostly | `domain/freshness.py`, `refresh_log`, `storage/integrity.py` (`soccer check`), About & sources page, stale warnings on every page |
| API quota / usage / cost tracking | ❌ | Nothing counts requests against a daily quota — never needed at $0 |
| API-Football adapter | ❌ | New |

### 3.2 Prediction engine (plan §4)

| Plan item | Exists? | Notes |
|---|---|---|
| Elo, Poisson, Dixon-Coles (MLE + time decay), Monte Carlo | ✅ | `models/elo.py`, `poisson.py`, `dixon_coles.py`, `simulation.py` |
| Market-informed ratings | ✅ | Closing 1X2 + O/U inverted to implied xG — biggest gain to date (gap to close +0.0194 → +0.0127 on 10k untuned matches) |
| Goal-space calibration | ✅ | `calibrate_model(spread, home_shift, rho)`; post-hoc 1X2 calibration rejected because scorelines/totals would disagree |
| Walk-forward backtest, log loss, Brier, RPS, calibration | ✅ | `models/backtest.py`, `evaluation.py`, Scorecard tab |
| Base-rate and market baselines, blend-weight curve | ✅ | `evaluate_forecasts` already scores model / closing line / blend / base rate on the same walk-forward |
| Season backtest | ✅ | `models/season_backtest.py` with bootstrap ranges |
| Prediction output (1X2, xG, scorelines, totals, factors, freshness) | ✅ mostly | Missing an explicit **model version** stamp and a **confidence** field |
| By-competition / by-odds-range / by-season breakdowns | ✅ partial | Per league and per season yes; per odds band only as calibration bands |
| XGBoost / LightGBM | ❌ | Not installed. See 4.2 for why expectations should be low |
| Learned ensemble | ⚠️ | The blend-weight curve *is* a learned model+market ensemble; optimal model weight measured ≈ 0.00–0.05. A model-only stack doesn't exist |
| Isotonic / Platt | ❌ | Deliberately not done (above) |

### 3.3 Player intelligence (plan §5)

Exists, on **historical StatsBomb data only**: leaderboard, per-90 percentiles vs
positional peers (200+ min pool), similar-player search, match logs, xT (validated:
split-half r 0.88), npxG, G−xG, xA, progressive passes/carries, pressures. Lives in
`dashboard/data.py` (`player_board`, `player_percentiles`, `player_similarity`…).

Missing: anything **current-season**, age/U23 (no birthdates), trajectories across
seasons, market value. API-Football can supply current-season counting stats and age,
but **not progressive actions, xA or (probably) xG** — the plan is right that those must
not be fabricated. Current-season scouting would therefore be a *shallower* metric set
than the historical one.

### 3.4 Market intelligence (plan §6)

| Plan item | Exists? | Notes |
|---|---|---|
| Implied probabilities, margin removal | ✅ | `models/value.py` |
| Closing-line comparison, log loss/Brier vs close | ✅ | Scorecard |
| Historical returns | ✅ | `soccer value` (yields swing around 0 — noise) |
| Opening vs closing / line movement | ❌ — **but data is already on disk** | football-data.co.uk files carry pre-match *and* closing odds per bookmaker (Bet365, Betfair Exchange, Max, Avg) for 1X2, O/U 2.5 and Asian handicap. Only closing is parsed today |
| Odds for **upcoming** matches | ❌ — **free source found today** | `football-data.co.uk/fixtures.csv` (+ `new_league_fixtures.csv`): pre-match odds for the coming round, same columns. One snapshot per round, not a tick series. README's "no free source of upcoming odds" claim is wrong |
| BTTS odds | ❌ | Not in the free CSVs; would need API-Football odds |
| Max drawdown, staking | ❌ by design | Removed 2026-10-04 (see 7.3) |

Small finding: the 2026/27 CSVs have **no Pinnacle columns**, so `_best_close_odds`
now falls back to market average for this season. Worth checking that the Scorecard's
"sharpest close" wording still holds.

### 3.5 Team intelligence (plan §7)

✅ Team page (season trajectory vs last season), form, home/away, xG over/under-performance
(xP), records/streaks, H2H, league style profiles, FPL injuries (PL only) folded into
forecasts. ❌ Rest days / fixture congestion — **computable for free** from `match_date`
(league games only; cup games missing for most leagues). ❌ Injuries outside the PL.

### 3.6 AI assistant (plan §8)

✅ 31 intents, rule-based, offline, multi-turn context, inline charts, a question-bank
test with a depth tier. Every answer routes to the same tested `data.py` functions the
pages use, so it already satisfies "controlled tools, no free SQL, LLM explains computed
numbers" — minus the LLM. ❌ No LLM provider layer. ❌ No automated reports.

### 3.7 Dashboard (plan §9)

8 pages today map onto the plan's six sections with little change:

| Plan section | Today |
|---|---|
| Home / Live Center | Home (+ archived Live scores) |
| Predictions | Predictions (Fixtures, Matchup, Season, Scorecard) |
| Player Scouting | Analysis → scouting |
| Market Intelligence | Scorecard (inside Predictions) |
| Team & Match Analysis | League tables, Teams, Records, Analysis → match xG |
| AI Research | Ask a question |

### 3.8 Automation (plan §10)

✅ `soccer serve` scheduler + background auto-refresh on dashboard open, per-job
isolation, stale banners. Needs a quota-aware job class for API-Football.

## 4. Verified external facts

### 4.1 API-Football

- **Official pricing page is behind a Cloudflare bot check** — I did not get past it and
  did not try to. Third-party listings (*unverified*): Free 100 req/day; Pro **$19/mo,
  7,500 req/day**; Ultra $29 (75k); Mega $39 (150k); 7-day trial; no overage, requests
  stop at quota. One source lists a conflicting euro price ladder.
- **Probed the API directly** (`v3.football.api-sports.io/status`, no key): HTTP 403 with
  body `{"errors": {"token": "Missing application key…"}, "results": 0, "response": []}`.
  So **an error comes back with an empty `response` array** — exactly the empty-success
  trap. The adapter must treat a non-empty `errors` as failure regardless of HTTP status
  (it's reportedly 200 for some error types — verify with a real key).
- Free-plan season restrictions are reported but unconfirmed. **If the free plan excludes
  the current season, coverage can only be verified on the trial or one paid month.**
- Budget sanity: 6 competitions × ~10 fixtures/round, ~4 endpoints per fixture after
  full time, plus daily injuries/odds per league ≈ a few hundred requests/day. Pro's
  7,500/day is ample; Free's 100/day is enough for a coverage probe only.

### 4.2 Model headroom (from this repo's own measurements)

Every model change since August moved the log-loss gap to the closing line in the 3rd–4th
decimal, and the optimal blend weight on the model stays ≈ 0. Gradient boosting on the
plan's feature list mostly re-learns what market-implied ratings already encode.
Realistic expectation: **a small gain at best, possibly none**. Worth one controlled
experiment in the existing harness, not a new model stack. The one genuinely new free
feature is rest days/congestion.

### 4.3 LLM APIs (*unverified*, aggregator sources, prices conflict)

DeepSeek's flash model ~$0.30 in / $1.20 out per M tokens at peak; Gemini flash-class
~$0.75 / $3.75 (promotional, doubling 2027-01-01); OpenAI's cheapest tier ~$0.10 / $0.50.
At this app's usage (1 assistant question in 42 days) any of them costs cents per month.
Cost is not the constraint — privacy and value are (7.4).

## 5. Architectural weaknesses

1. **Business logic lives in the dashboard package.** `dashboard/data.py` (2,486 lines),
   `assistant.py` (2,879) and `app.py` (2,917) hold scouting, market and forecasting
   orchestration. The plan's `scouting/`, `market/`, `intelligence/` packages are the right
   direction, but move code **when a phase touches it**, not as a big-bang refactor.
2. **Two identity systems** (canonical-id crosswalk for live data; normalized-name bridge
   for analytics) and **no player identity**. API-Football adds a third id space; it should
   map into the existing crosswalk, not start a fourth scheme.
3. **No request accounting.** Fine at $0; a hard requirement for a quota'd paid API.
4. **Dead tables** `bet`, `confirmed_lineup` (0 rows) left in `live.sqlite` after the cuts.
5. **Test suite time** is 2.5× the README figure — fine now, but watch it as adapters grow.

## 6. Conflicts between sources (for the adapter design)

- API-Football fixtures overlap football-data.org (fixtures/standings) and
  football-data.co.uk (results). Proposal: football-data.co.uk stays the **results and
  odds-history** source of record (30 years, closing odds, backtests depend on it);
  API-Football becomes the source for **current-season player stats, lineups, injuries
  beyond the PL, events, and upcoming odds incl. BTTS**; football-data.org stays for
  fixtures/standings and as a cross-check. No duplicate ingestion of results.
- FPL injuries vs API-Football injuries for the PL: keep FPL (free, has minutes/price
  used by the availability weighting) and use API-Football for the other leagues.

## 7. Decisions only you can make

1. **Budget.** README and the project's stated constraint is **$0**. The plan sets $50.
   Lifting it is your call; I won't subscribe to anything either way — you would.
2. **API-Football account.** Registration needs you (account creation + their bot check).
   Recommend: free key first, run the coverage probe, then decide on Pro with real data.
3. **Betting ROI / drawdown / "profitable opportunity" (plan §6.4).** Staking tools were
   removed on 2026-10-04 because the model has no edge over the close and nobody used
   them. The plan's own caveat agrees. Recommend: build line-movement and model-vs-market
   *analysis*, not returns/drawdown dashboards, unless a backtest ever shows an edge.
4. **Cloud LLM.** The README promises the assistant is offline and "nothing leaves the
   machine". An LLM narrator would send questions plus computed numbers to a provider.
   Given 1 use in 42 days, recommend deferring (or opt-in, off by default, narrating only
   tool outputs).

## 8. Proposed implementation sequence

Reordered from the plan to put **free wins and verification before spending**, and
**the most-used page (Predictions) before the never-visited one (Analysis)**.

| # | Phase | Cost | Files (new = ★) | Exit test |
|---|---|---|---|---|
| A | **Free odds:** parse pre-match odds already in the CSVs + ingest `fixtures.csv` upcoming odds. Live model-vs-market on Fixtures/Matchup; opening→closing movement on Scorecard. Fix README claim; check Pinnacle fallback wording | $0 | `sources/football_data_co_uk.py`, `storage/analytics_db.py` (odds_observation table), ★`market/` (implied, movement), `dashboard/data.py`, `app.py`, registry + tests | Upcoming PL fixture shows model vs market prob; walk-forward CLV of model computed |
| B | **API-Football probe:** adapter with `errors`-is-failure, quota ledger (request count from headers), raw snapshots, registry entry; `soccer probe-api-football` writes a coverage matrix (6 comps × endpoints × current season) to `docs/source-verification.md` | $0 (free key) | ★`sources/api_football.py`, ★`ingest/quota.py`, `config.py`, `registry.py`, `.env.example`, `cli/main.py`, tests with respx | Coverage matrix from real responses; decision memo for Pro |
| C | **Rest-days feature + one boosted-model experiment** in `evaluate_forecasts`; ship only if the gap to close improves on untuned leagues. Add model-version + confidence to forecast output | $0 (+ scikit-learn, one dep: HistGradientBoosting + isotonic) | `models/evaluation.py`, ★`models/features.py`, ★`models/boosted.py`, `dashboard/data.py` | Scorecard row; shipped or documented-as-rejected |
| D | **API-Football ingestion** (if B + your approval): current-season player stats, lineups, injuries (non-PL), upcoming odds incl. BTTS; player crosswalk | $19/mo *unverified* | adapter, mappers, `domain/crosswalk.py` (player ids), `analytics_db.py`, `actions.py`, scheduler | Quota under budget for 7 days; freshness on About page |
| E | **Current-season scouting** on D's data: U23 filter, cross-season trajectories, honest metric-availability labels | $0 | ★`scouting/` (moved out of `data.py`), `app.py` | Percentiles match raw source totals |
| F | **Dashboard regroup** into the six sections (labels/navigation; no page rewrites) | $0 | `dashboard/nav.py`, `app.py` | AppTest + preview check |
| G | **LLM narrator** — only if you choose it in 7.4 | ~$1/mo | ★`intelligence/llm_provider.py`, `assistant.py` | Works with key unset; numbers in prose match tool output |

Each phase ends with the plan's §14 report (files, tests, results, limits, API usage).

**Recommended first step: Phase A** — it needs no money or account, uses data already on
disk, and improves the page people actually use.
