# API-Football: what a key gets, and whether to pay

Measured 2026-11-17 with a free key (`soccer probe-api-football`). The matrices are in
[`api-football-coverage.md`](api-football-coverage.md) (2026/27) and
[`api-football-coverage-2024.md`](api-football-coverage-2024.md) (2024/25). Field details
below were read from the stored raw responses, not from documentation.

## What the free plan gives

- **No current season, for any of the six competitions.** Every 2026/27 query returned
  `errors.plan: "Free plans do not have access to this season, try from 2022 to 2024."`
- **2022–2024 only.** For 2024/25 every endpoint answered for all six competitions.
- **No bulk queries.** `/fixtures?ids=…` (up to 20 matches per request) is refused on the
  free plan: `"Free plans do not have access to the Ids parameter."` Match-level data costs
  one request per match.
- 100 requests a day by the server's own count (requests refused by the per-minute rate
  limit did not count), 10 a minute; the per-minute window is shared across processes.

So the free plan can't feed the dashboard. It's only useful for testing the schema and for a
slow backfill of past seasons.

## What the data contains (2024/25, all six competitions)

| Data | Present | Notes |
|---|---|---|
| Fixtures, events, lineups, standings | ✅ | |
| Team match statistics | ✅ | Includes **`expected_goals`** (team xG per match) and `goals_prevented`, plus shots in/out of the box, possession, passes |
| Player match stats | ✅ | Minutes, rating, goals, assists, shots, **key passes**, passes and accuracy, **tackles, interceptions**, duels, dribbles, cards. Many fields null for low-minute players |
| Player season stats | ✅ | Same families, aggregated; `birth`/`age` present, so **U23 filters are possible** |
| Player xG / xA / progressive passes or carries | ❌ | Not in any response; must not be derived |
| Injuries | ✅ | 1,687–3,168 rows per competition-season, all six (we only have PL via FPL today) |
| Odds | ◻️ | Empty for 2024/25, and the API declares `odds=False` for past seasons. **Current-season odds coverage is still unverified**; it can only be confirmed on a paid plan |

## What paying (~$19/mo Pro, price unverified) would add over what's free today

1. **Current-season team xG for the top five leagues and the Champions League.** The project
   has none today (StatsBomb stops at 2025). This is the one candidate that could move the
   **model**: the earlier StatsBomb xG experiment was mixed, but it had one season per league
   and an 80–90% name-match rate.
2. **Injuries beyond the Premier League**, feeding the existing team-news adjustment (that
   adjustment is a prior, not backtested).
3. **Current-season player stats** for scouting. The usage log shows the Analysis page never
   visited in 42 days.
4. Lineups, events and possibly BTTS odds. Upcoming 1X2 and O/U odds are already free (Phase A).

## Recommendation

Don't subscribe yet. The best first step is free:

- **Backfill one competition's 2024/25 team xG on the free plan**: about 380 requests, so ~4
  days at ~95 a day. Then test it in the existing Scorecard harness: does an xG-blend
  rating move the gap to the closing line, or the line-movement direction share, off its
  current value? That answers whether item 1 is worth paying for, before paying.
- Subscribe for **one month** only if that test shows a real gain. Then confirm odds and
  injuries for the current season with the probe, and ingest.

Items 2–4 alone don't clear the bar: the most-used page (Predictions) gains little from them,
and the pages they would feed aren't being used.
