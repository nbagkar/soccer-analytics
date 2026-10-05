"""A fixed bank of realistic assistant questions, scored as a number -- the assistant's
quality metric.

Why it exists: fixes were being checked against the same ad-hoc probe that found the
failures, so a re-run of that probe always looked perfect ("teaching to the test"); a fresh
probe of 30 new questions then got only ~12 right. The bank makes quality measurable and
keeps it honest:

* DEV cases are what fixes are designed against.
* HOLDOUT cases were written at the same time, before any fix, in different phrasings, and
  are only ever SCORED -- never used to design a fix. If dev improves and holdout doesn't,
  the fix was overfit.

Each case states what a correct answer must look like: acceptable intents, text it must
contain, and text that would make it wrong. "fallback" is an acceptable intent only where the
loaded data genuinely cannot answer -- an honest "I can't" beats a confident wrong answer.

Everything runs against a deterministic seeded store (`build_store`), so the score is
reproducible in CI and doesn't drift with live data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path


@dataclass(frozen=True)
class Case:
    question: str
    intents: tuple[str, ...]
    contains: tuple[str, ...] = ()
    excludes: tuple[str, ...] = ()
    note: str = ""


@dataclass
class Result:
    case: Case
    intent: str | None
    text: str
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


# Seeded leagues: `seed_results` makes the FIRST-listed club the strongest (wins every home
# game 2-0, draws 1-1 away), so table order = list order. Last season's Premier League order
# differs from this season's, so "last season" questions have a checkable answer.
PL_NOW = [
    "Man City",
    "Arsenal",
    "Liverpool",
    "Chelsea",
    "Tottenham",
    "Brighton",
    "Fulham",
    "Brentford",
]
PL_LAST = [
    "Arsenal",
    "Liverpool",
    "Man City",
    "Brighton",
    "Chelsea",
    "Tottenham",
    "Brentford",
    "Fulham",
]
OTHER_LEAGUES = {
    "E1": ["Leeds", "Derby", "Sunderland", "Wrexham"],
    "SP1": ["Real Madrid", "Barcelona", "Ath Madrid", "Sevilla"],
    "D1": ["Bayern Munich", "Dortmund", "Leverkusen", "Stuttgart"],
    "F1": ["Paris SG", "Marseille", "Lyon", "Lille"],
    "I1": ["Inter", "Milan", "Juventus", "Napoli"],
    "N1": ["PSV Eindhoven", "Ajax", "Feyenoord", "Twente"],
    "SC0": ["Celtic", "Rangers", "Hearts", "Aberdeen"],
}


def build_store(tmp_path: Path) -> tuple[Path, Path]:
    """(analytics_db, live_db) seeded for the bank."""
    from soccer.domain.match_state import MatchStatus
    from soccer.storage.live_db import LiveDB
    from tests.test_dashboard_data import add_match, seed_player_events, seed_results

    analytics, live = tmp_path / "analytics.duckdb", tmp_path / "live.sqlite"
    seed_results(analytics, division="E0", teams=PL_LAST, season="2425")
    seed_results(analytics, division="E0", teams=PL_NOW, season="2526")
    for division, teams in OTHER_LEAGUES.items():
        seed_results(analytics, division=division, teams=teams, season="2526")
    seed_player_events(analytics)  # Messi + Otamendi: the historical player archive

    soon = datetime.now(UTC)
    with LiveDB(live) as db:
        for i, (home, away, comp, hours) in enumerate(
            [
                ("Arsenal FC", "Tottenham Hotspur FC", "Premier League", 3),  # tonight
                ("Chelsea FC", "Liverpool FC", "Premier League", 27),
                ("Derby County FC", "Wrexham AFC", "Championship", 28),
                ("AFC Ajax", "PSV", "Eredivisie", 50),
                ("Celtic FC", "Rangers FC", "Scottish Premiership", 52),
            ]
        ):
            add_match(
                db,
                match_id=f"bank{i}",
                home=home,
                away=away,
                competition=comp,
                status=MatchStatus.NOT_STARTED,
                observed_at=soon + timedelta(hours=hours),
            )
    return analytics, live


T = "team"
DEV: list[Case] = [
    # -- the fresh 30-question probe (2026-10-05) --
    Case("who is the best team in europe", ("fallback", "league_compare", "power")),
    Case("how many points does liverpool have", ("standings", T), ("Liverpool", "points")),
    Case(
        "when is the next north london derby",
        ("fixtures", "h2h"),
        ("Arsenal", "Tottenham"),
        ("Derby",),
    ),
    Case("who has won the most premier league titles", ("honours",), ("Premier League",)),
    Case(
        "what was the score when arsenal played chelsea last",
        ("h2h", "recent_results"),
        ("Arsenal", "Chelsea"),
        ("predicted",),
    ),
    Case("is haaland injured", ("availability",)),
    Case("who is the top scorer for arsenal", ("top_scorers", "fallback"), (), ("Messi",)),
    Case(
        "what's the most likely score for real madrid vs barcelona",
        ("forecast",),
        ("Real Madrid", "Barcelona"),
    ),
    Case("which teams are in the relegation zone", ("standings",), ("Brentford",)),
    Case("how many goals has messi scored", ("player",), ("Messi", "historical")),
    Case("man city's home record this season", (T,), ("Man City", "home")),
    Case("who are the most improved teams", ("improvement", "season_compare"), ("Man City",)),
    Case("tottenham vs arsenal odds", ("forecast",), ("Tottenham", "Arsenal")),
    Case("show me the championship table", ("standings",), ("Championship", "Leeds")),
    Case(
        "how did brighton do last season",
        (T, "standings", "season_compare"),
        ("Brighton", "2024/25"),
        ("Their last",),
    ),
    Case("will chelsea make top four", ("title_odds",), ("Chelsea", "top four")),
    Case("who's on a winning streak", ("records", "form")),
    Case("lowest scoring team in serie a", ("scoring",), ("Napoli", "Serie A")),
    Case("predict the bundesliga winner", ("title_odds",), ("Bundesliga",), ("name two teams",)),
    Case("what time does the champions league final start", ("fallback",)),
    Case("psg vs bayern prediction", ("forecast",), ("Bayern",), ("name two teams",)),
    Case("who plays tonight", ("fixtures",), ("Arsenal",)),
    Case("compare arsenal and spurs", ("team_compare",), ("Arsenal", "Tottenham")),
    Case("how good is the model at predicting draws", ("model",)),
    Case("ajax next match", ("fixtures",), ("Ajax", "PSV")),
    Case("which league has the most draws", ("league_compare",)),
    Case("who scored in the last arsenal game", ("recent_results", "fallback"), (), ("predicted",)),
    Case("rangers vs celtic", ("forecast",), ("Rangers", "Celtic")),
    Case("which team concedes the most at home", ("defence",), ("home",)),
    Case("top of the table clash this weekend", ("fixtures", "standings")),
    # -- more of the same categories --
    Case("how many points do arsenal have", ("standings", T), ("Arsenal", "points")),
    Case("where do tottenham sit in the table", ("standings",), ("Tottenham",)),
    Case("which team scores the most goals in la liga", ("scoring",), ("Real Madrid",)),
    Case("best attack in the bundesliga", ("scoring",), ("Bayern Munich",)),
    Case("liverpool away record", (T,), ("Liverpool", "away")),
    Case("compare chelsea and liverpool", ("team_compare",), ("Chelsea", "Liverpool")),
    Case("spurs next game", ("fixtures",), ("Tottenham",)),
    Case("derby county next match", ("fixtures",), ("Derby", "Wrexham")),
    Case(
        "how did arsenal do in 2024/25", (T, "standings", "season_compare"), ("Arsenal", "2024/25")
    ),
    Case("who won the league last season", ("standings", "honours"), ("Arsenal", "2024/25")),
    Case(
        "who is the top scorer in the premier league this season",
        ("top_scorers",),
        ("team",),
        ("Messi",),
    ),
    Case("games today", ("fixtures",), ("Arsenal",)),
    Case("what was the result of the last chelsea game", ("recent_results",), ("Chelsea",)),
    Case("messi stats", ("player",), ("Messi",)),
    Case("who is top of the premier league", ("standings",), ("Man City",)),
]

# Written alongside DEV, before any fix. Score only -- do not design fixes from these.
HOLDOUT: list[Case] = [
    Case("how many points have man city got", ("standings", T), ("Man City", "points")),
    Case("who are spurs playing next", ("fixtures",), ("Tottenham", "Arsenal")),
    Case("compare man city and liverpool", ("team_compare",), ("Man City", "Liverpool")),
    Case("lowest scoring side in the bundesliga", ("scoring",), ("Stuttgart",)),
    Case("which team has the leakiest home defence", ("defence",), ("home",)),
    Case(
        "what was the score last time liverpool played man city",
        ("h2h", "recent_results"),
        ("Liverpool", "Man City"),
        ("predicted",),
    ),
    Case(
        "how did chelsea do last season",
        (T, "standings", "season_compare"),
        ("Chelsea", "2024/25"),
        ("Their last",),
    ),
    Case("predict who wins la liga", ("title_odds",), ("La Liga",), ("name two teams",)),
    Case(
        "is the north london derby this weekend",
        ("fixtures", "h2h"),
        ("Arsenal", "Tottenham"),
        ("Derby",),
    ),
    Case("barcelona's home record", (T,), ("Barcelona", "home")),
    Case("who's playing today", ("fixtures",), ("Arsenal",)),
    Case(
        "most goals scored in serie a this season",
        ("scoring", "top_scorers"),
        ("Inter",),
        ("Messi",),
    ),
    Case("which teams are going down", ("title_odds", "standings"), ("relegation",)),
    Case("how many goals has otamendi scored", ("player",), ("Otamendi", "historical")),
    Case("who is arsenal's top scorer this season", ("top_scorers", "fallback"), (), ("Messi",)),
    Case("celtic v rangers prediction", ("forecast",), ("Celtic", "Rangers")),
    Case("bayern vs psg who wins", ("forecast",), ("Bayern",), ("name two teams",)),
    Case("where is derby in the table", ("standings",), ("Derby", "Championship")),
    Case("show me the eredivisie standings", ("standings",), ("Eredivisie",)),
    Case("what's tottenham's goal difference", ("standings", T), ("Tottenham",)),
]


def score(cases: list[Case], analytics: Path, live: Path) -> list[Result]:
    from soccer.dashboard.assistant import answer

    results = []
    for case in cases:
        reply = answer(case.question, analytics, live)
        text = reply.text
        problems = []
        if reply.intent not in case.intents:
            problems.append(f"intent {reply.intent!r}, expected one of {case.intents}")
        problems += [f"missing {s!r}" for s in case.contains if s.lower() not in text.lower()]
        problems += [f"contains {s!r}" for s in case.excludes if s.lower() in text.lower()]
        results.append(Result(case, reply.intent, text, problems))
    return results


def report(name: str, results: list[Result]) -> str:
    passed = sum(r.ok for r in results)
    lines = [f"{name}: {passed}/{len(results)} ({passed / len(results):.0%})"]
    lines += [f"  FAIL {r.case.question!r}: {'; '.join(r.problems)}" for r in results if not r.ok]
    return "\n".join(lines)
