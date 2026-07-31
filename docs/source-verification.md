# Source Verification Record

Ground truth for every data source considered, verified by direct fetch rather than
documentation or recall. This exists because the original plan's source assumptions
turned out to be wrong in ways that change the product.

**Verified:** 2026-07-31. Re-verify before relying on any of this; free tiers move.

---

## Headline corrections to the original plan

Three assumptions were load-bearing and wrong:

1. **football-data.org's free tier does not provide live scores.** It provides
   *delayed* scores. Livescores begin at the EUR 12/mo tier. The plan's Live Centre,
   its 60-second active-match polling cadence, and its Match Centre lineups/scorers
   all assumed access the free tier does not grant.

2. **OpenLigaDB is not a live feed.** The plan lists it as the "live-data fallback."
   It is a fast *post-match* source, settling roughly 1-1.5h after full time.

3. **TheSportsDB is the only confirmed free live source.** The plan files it under
   "optional team metadata and artwork." It is currently the single most important
   source for the live surface — and the least contractually secure.

The plan's competition tiers are also close to inverted: Eredivisie, Primeira Liga and
Brasileirao sit in "later expansion" but are free, while Europa League, Conference
League, MLS, Copa Libertadores, AFCON and Copa America sit in "initial live priority"
and are not.

---

## Summary

| Source | Live? | License | Verdict |
|---|---|---|---|
| football-data.org | No — delayed | Free tier, 12 comps | Default. Fixtures/standings only |
| TheSportsDB | **Yes, ~60s** | Attribution, no resale | Default, **livescore only** |
| OpenLigaDB | No — ~186 min lag | ODbL (unconfirmed) | Opt-in, German football |
| football-data.co.uk | No — 1-3 days | **None found** | Default. Analytics backbone |
| openfootball (.TXT repos) | No — ~1 day | CC0-1.0 | Default, historical |
| openfootball/football.json | No — 2 months stale | CC0-1.0 | Superseded by .TXT repos |
| ESPN JSON | Yes | Terms forbid it | **Drop** |
| FPL API | Yes, PL only | Terms forbid DB-building | Opt-in, off by default |
| StatsBomb / Wyscout / SkillCorner | n/a | *pending* | *verification in flight* |

---

## football-data.org

Free tier, verified on the coverage and pricing pages.

- **Rate limit: 10 calls/minute.** Lowest of any tier.
- **Scores and schedules are marked "delayed."** Live scoring starts at EUR 12/mo.
- **Lineups, subs, goalscorers, bookings and squads start at EUR 29/mo** — none are free.
- **12 competitions, no domestic cups:** Champions League, Premier League, La Liga,
  Bundesliga, Serie A (Italy), Ligue 1, Championship (England), Eredivisie,
  Primeira Liga, Serie A (Brazil), World Cup, European Championships.
- Not free: Europa League, Conference League, MLS, Copa Libertadores, AFCON,
  Copa America, all domestic cups, all women's competitions.

**Design consequence:** poll the date-batched `/matches` endpoint, never per-match.
At 10 req/min, per-match polling of a Saturday slate exhausts the budget by itself.

### Behaviours found by driving the live API

Four constraints that the documentation does not state, each confirmed by request:

**1. Hard 10-day cap on `/matches`.** Exceeding it returns
`400 {"message":"Specified period must not exceed 10 days."}`. Enforced client-side so
an oversized range costs zero requests; `matches_over_range()` chunks explicitly,
because at 10 req/min the difference between one request and nine must be a deliberate
choice rather than a hidden one.

**2. The `plan` label does not predict access.** Copa Libertadores reports
`plan: TIER_FOUR`, yet:

| Request | Result |
|---|---|
| `/competitions/CLI/standings` | **200, fully populated** — 8 groups, real tables |
| `/matches` unfiltered date sweep | **CLI fixtures included** |
| `/matches?competitions=CLI` | **200 with `count: 0`** |

So a competition above the tier is partially reachable, and the one request that fails
does so *silently* — an empty 200, indistinguishable from a quiet week. Never infer
access from the tier label; probe and observe. Never build a feature on it either:
this is undocumented and can be closed without notice.

**3. `/competitions` returns 13, not the advertised 12.** The thirteenth is CLI.
Filtering on `plan == TIER_ONE` yields exactly the documented twelve.

**4. Standings default to the last completed season.** A request with no `season`
parameter in July 2026 returns the *finished* 2025/26 table (Arsenal, 38 played,
85 points) — not an empty current one. Any UI showing standings must display the
season, or a finished table reads as a live one.

The `competitions` filter accepts both codes and numeric IDs (`DED` and `2003` behave
identically).

## TheSportsDB

The free tier is far more truncated than commonly documented — but its live endpoint
works, which is the opposite of what its own pricing page implies.

- **`livescore.php` is genuinely live.** Polled twice 71s apart: match minute
  incremented 15 -> 16, `updated` advanced exactly 60s. Returns match minute, status,
  score across ~40 leagues, including some women's football.
- **Bulk endpoints are crippled:** `all_leagues` returns 5 leagues; a full season query
  returns 15 events of 380. Useless for backfill.
- Free key is `123` (documented). Legacy key `3` still works but is undocumented.
- 30 requests/minute.
- Terms permit storage explicitly: *"You can scrape, copy and modify any content
  returned from the API, as long as you use the official end points."* Attribution with
  a link back is required. Resale prohibited. App-store publishing requires a paid tier.

**Risk:** the pricing page frames livescore as a paid feature ("2 min livescore"), so
free access appears to be undocumented generosity. Do not make it load-bearing without
a fallback.

**Status-code note:** `strStatus` uses `1H`/`2H`/`HT`/`FT` (all with a numeric
`strProgress`) and `P` (empty progress). `P` is a **penalty shootout in progress**, not
postponement — verified against a 3-3 MLS Next Pro match (a competition that decides
draws by shootout). Postponement has its own codes. Mapping `P` to postponed would show
a live match as called off.

## OpenLigaDB

- **Empirically not live.** All 306 Bundesliga 2025/26 matches: zero updated during the
  0-105 min in-play window. Minimum lag 172 min, median 186 — the signature of a
  scheduled post-match import. A live Swiss fixture still read `matchIsFinished=false,
  goals=[]` three hours after kickoff.
- Goal-level detail is good once it lands: `matchMinute`, scorer, penalty/own-goal flags.
- An MQTT push channel exists in the samples repo, but its npm client `@wgd/oldb`
  returns 404 on the registry. Not installable.
- **Results are retroactively mutable: any logged-in user can edit any result for six
  days after the match.** Store `lastUpdateDateTime` and re-check.
- **League shortcuts are a trap.** 816 all-time entries include test junk
  (`BastardLeague`, `Moisty Mire League`) and duplicate shortcuts for the same
  competition — the 2026 World Cup exists as `wm26`, `wm2026`, `WC2026` and more, most
  dead. Pin known-good shortcuts (`bl1`, `bl2`, `bl3`, `dfb`, `ffb1`); never trust
  `getavailableleagues` blindly.
- No documented rate limit. Poll `getlastchangedate` before `getmatchdata`.
- **License unconfirmed.** Homepage claims ODbL, but `openligadb.de/lizenz` is a Blazor
  SPA that would not render. If ODbL genuinely applies it carries share-alike
  obligations on derived databases — materially different from CC0. **Verify in a real
  browser before shipping.**

## football-data.co.uk

The strongest analytics source available free, and the one with the murkiest terms.

- 22 main divisions across 11 countries, plus 16 extra top divisions.
- Results to 1993/94; stats and odds from 2000/01.
- **132 columns** on the current season's E0.csv: shots, shots on target, corners,
  fouls, cards, referee, attendance — plus ~108 columns of betting odds from 17
  bookmakers, including closing lines. No xG anywhere.
- Updated at least twice weekly (Sunday and Wednesday nights). 1-3 day latency.
- **No explicit license, no copyright grant, no redistribution clause found.** The only
  usage text is a purpose statement: data is *"made available for the purposes of league
  match prediction only."* Downloading and analysing is plainly intended; there is no
  affirmative permission to republish. **Do not redistribute the CSVs or expose them
  verbatim.**
- Men's football only. No women's coverage.
- Itself an aggregation layer — compiled from Flashscore, BBC, ESPN and others.

## openfootball

- **`football.json` is a stale downstream artifact.** Last data commit 2026-05-30,
  ~2 months old, and there is no 2026-27 directory. The README's weekly-update promise
  is commented out in the source.
- **The upstream Football.TXT repos are current to today** and already carry 2026-27:
  `openfootball/england`, `deutschland`, `espana`, `italy`, `europe`, `austria`.
- **Use the .TXT repos, not the JSON.** Costs a Football.TXT parser; buys 2 months of
  freshness and the current season.
- Coverage: 20 divisions, 2010-11 through 2025-26. Results only — no shots, cards,
  xG or odds.
- **CC0-1.0, verified.** Cleanest license of any source here. No attribution required.
- ~1 day latency during season; stops entirely out of season. Not a live fallback.

## ESPN public JSON — dropped

Rich and live, but Disney's terms bar users from accessing "using a robot, spider,
script, or other automated means" and from "compiling, building, creating or
contributing to any collection of data, data set or database." Scripted polling into a
local database is precisely what is named. Also undocumented, unversioned, and
Cloudflare-fronted. The data it offers is duplicated by sources with better terms.

## FPL API — opt-in, off by default

Uniquely valuable and genuinely grey.

- **Free Opta-derived xG:** `expected_goals`, `expected_assists`,
  `expected_goal_involvements`, `expected_goals_conceded`, plus BPS, ICT, per-90
  variants, tackles, recoveries, clearances/blocks/interceptions.
- Injury and availability data: `status`, `news`, `chance_of_playing_next_round`.
- Live during gameweeks via `/api/fixtures/` and `/api/event/{n}/live/`.
- **Premier League only.**
- Premier League terms contemplate private personal use but expressly bar "creating a
  database (electronic or otherwise) that includes material downloaded or otherwise
  obtained from the Website or App."
- Undocumented, no published rate limit; aggressive polling has historically drawn blocks.

Record the tension rather than treating it as unrestricted. Cache hard, poll gently,
never redistribute.

---

---

# Historical datasets

| Dataset | Coverage | Licence | Redistribute? | Commercial? |
|---|---|---|---|---|
| StatsBomb | 24 comps, 4,235 matches, 426 w/ 360 | **Proprietary EULA** | **No** | **No** |
| Wyscout (figshare) | 1,941 matches, 5 leagues + WC18 + Euro16 | CC BY 4.0 | Yes | Yes |
| SkillCorner | 10 A-League 2024/25 matches | MIT | Yes | Yes |

## StatsBomb — NOT open data

The single most consequential correction. The README's soft "please credit us" wording
is **not** the licence. The binding document is `LICENSE.pdf`, a 5-page "StatsBomb
Public Data User Agreement" (updated 2023-09-08), governed by England and Wales law.
GitHub classifies it `NOASSERTION`.

> **1.2. The User may not:**
> **1.2.1.** edit, distort, distribute, reproduce, sell or in any way provide the data
> to any external or third party;
> **1.2.2.** commercially exploit the data or any analysis derived from the use of the
> Service;

> **1.4.** The User is required to accredit any publication of analysis formed from
> StatsBomb Data with the StatsBomb brand logo.

**Consequences:**
- Building analysis tools is permitted (§1.1 covers "analysis, research").
  Redistributing the data those tools ingest is not.
- Never ship a derived dataset, public API, or served cache. Users must pull from the
  repo themselves.
- **If this platform is ever monetized, StatsBomb data cannot be part of it** — and the
  bar extends to "any analysis derived" from it, not just the raw data.
- Attribution requires the **logo**, not a text credit.

**Coverage:** 24 competitions, 80 competition-seasons, 4,235 matches. Canonical repo is
`hudl/open-data`; the old `statsbomb/open-data` URL 301-redirects.

**360 data: 12 competition-seasons** — Bundesliga 23/24, AFCON 2023, World Cup 2022,
La Liga 20/21, Ligue 1 21/22 and 22/23, MLS 2023, Euro 2020, Euro 2024, Women's Euro
2022 and 2025, Women's World Cup 2023.

> **Trap:** key on `match_available_360`, not `match_updated_360`. Several rows carry a
> non-null `match_updated_360` while having no 360 data at all.

**Women's football is a genuine strength:** 7 competitions, 13 competition-seasons,
including both Women's Euros with 360.

**Size: 16.13 GB working tree, ~23.5 GB cloned** (events 12.82 GB, 360 3.21 GB).
Do not clone. Use kloppy's `load_open_data()` to fetch per-match from raw URLs and
cache only what is used. Events-only is ~12.9 GB.

## Wyscout — the most permissive dataset

**CC BY 4.0, verified per-article via the figshare API.** Events (7770599), Matches
(7770422), Players (7765196). Commercial use and redistribution both permitted.

> The `koenvo/wyscout-soccer-match-event-dataset` repo has **no LICENSE file** (404) and
> the GitHub API reports `license: null`. Its CC BY 4.0 claim is inherited, not granted.
> Pull data from the repo (it is reshaped for kloppy), but **attribute to figshare** as
> the licensing source of record.

Cite: Pappalardo, L., Cintia, P., Rossi, A. et al. *A public data set of spatio-temporal
match events in soccer competitions.* Sci Data 6, 236 (2019).
https://doi.org/10.1038/s41597-019-0247-7

**Coverage:** 1,941 matches — Ligue 1, Premier League, Serie A, La Liga (380 each),
Bundesliga (306), all 2017/18, plus World Cup 2018 (64) and Euro 2016 (51).
Static; last commit 2023-12-04. Being unmaintained is fine — it reformats a frozen
dataset. Use `processed-v2/` for kloppy >= 3.14.

## SkillCorner — broadcast tracking, and that matters

**MIT licensed** — genuinely open, commercial use and redistribution allowed.
Attribution is requested, not required.

**10 matches, Australian A-League 2024/25** (Nov 2024 - May 2025). If a plan cites
9 matches or European fixtures, it is out of date.

**The analytical trap.** This is broadcast tracking derived from video, and the file is
named `*_tracking_extrapolated.jsonl`. Off-camera players are **extrapolated, not
omitted** — every frame looks complete, so naive code silently treats guesses as
observations.

- Each player record carries `is_detected`.
- **Biased unless filtered on `is_detected`:** pitch control, packing / line-breaking,
  off-ball defensive shape, team centroid, spatial dominance surfaces.
- **Valid:** on-ball and near-ball analysis, physical/running metrics over detected
  spans, anything explicitly restricted to detected players.

10 fps, metres, origin at pitch centre. SkillCorner state ~97% player-identity accuracy
and recommend smoothing speed/acceleration. ~164 MB.

## Library constraints

**kloppy 3.19.0** (2026-06-07) — Python 3.9-3.13. Parsers for all three datasets
natively, including StatsBomb 360 via the `three_sixty_data` argument. Light
dependencies.

**socceraction 1.5.3** (2024-08-15) — **the binding constraint on the whole stack.**
- `requires_python = "<3.13,>=3.9"` — **Python 3.12 is the ceiling.**
- Pins `numpy<2.0.0`, which constrains every other package in the environment.
- Nearly two years without a release.
- Event-only. **No tracking support** — SkillCorner work is kloppy plus custom code.
- Prefer the `socceraction.spadl.kloppy` bridge: kloppy parses, socceraction values.

---

## Cross-source identity (how the crosswalk joins sources)

The two live sources share no identifiers, so entities are joined by normalized name.
Observed alignment between football-data.org and TheSportsDB for the Premier League:

| football-data.org | TheSportsDB | Join |
|---|---|---|
| `name` "Arsenal FC" / `shortName` "Arsenal" / `tla` "ARS" | `strTeam` "Arsenal" / "ARS" | `shortName` ≈ `strTeam`; TLA equal |
| `name` "Aston Villa FC" / "Aston Villa" / "AVL" | "Aston Villa" / "AVL" | same |

Two practical findings:

- **Full names never match; normalized names do.** "Arsenal FC" → "arsenal" ==
  "Arsenal" → "arsenal". The `FC`/`CF`/`AC` affix stripping in `domain/names.py` is
  what closes the gap. football-data.org's `shortName` aligns with TheSportsDB's
  `strTeam` more directly than its `name` does.
- **The TLA is a strong secondary key** (ARS=ARS, AVL=AVL) — available for future
  disambiguation, though not globally unique across leagues.

Verified end-to-end: resolving 20 football-data.org PL teams then 10 TheSportsDB teams
produced **20 canonical entities, not 30** — every TheSportsDB team linked to its
football-data.org counterpart (`exact_name`, confidence 0.9), zero false merges.

What normalization deliberately cannot join, left for manual linking:
`1. FC Köln` vs `FC Cologne` (translation), `Bayern München` vs `Bayern Munich`.

### Match reconciliation reality

- **The two live sources have near-zero coverage overlap at any moment.**
  football-data.org is 12 European majors (often pre-season / not live);
  TheSportsDB's live feed skews to South American and minor leagues because it is the
  opposite side of the clock from European primetime. Checked live: zero genuine
  overlap. So cross-source *match* reconciliation is designed-for and correct, but rare
  in practice. The everyday value of match identity is a stable id that collapses the
  same match across many polls (verified: 27 real fixtures re-resolve to the same 27
  ids on a second poll, zero duplicates).
- **Timestamp skew is unmeasured** — no overlapping fixture existed to compare. The
  match kickoff tolerance (6h) is set from reasoning: the same (competition, home,
  away) triple never recurs within a day, so a wide window is safe.
- **An alias layer is the main gap.** Affix-stripping bridges English divergences
  ("Arsenal FC"/"Arsenal") but not appended-city ones ("PSV"/"PSV Eindhoven") or
  translations ("Köln"/"Cologne"). Robust cross-source reconciliation needs a curated
  alias/known-equivalents table — future work. Until then those link manually.

## Still unverified

- API-Football free tier — pricing page is Cloudflare bot-gated, needs a manual
  browser check. Deprioritized: TheSportsDB already covers free live; API-Football
  only matters if a documented-free live tier would be contractually safer.
- OpenLigaDB licence — page is a Blazor SPA that would not render for any fetcher.
  Deprioritized: source is off by default and share-alike only binds *redistribution*,
  which this local-first platform does not do. Resolve before enabling + publishing.
- Goalserve, Entity Sport, official league/federation APIs, Wikidata
- Other open datasets (Metrica, DFL, Opta, Second Spectrum, PFF). kloppy ships parsers
  for these formats, which suggests open samples may exist, but neither their existence
  nor their licensing is confirmed
