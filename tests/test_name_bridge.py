"""Cross-source club-name bridging: the shared matcher, stored-key migrations, the audit."""

from __future__ import annotations

import sqlite3
from datetime import date

from soccer.dashboard.data import match_team, resolve_canonical_name


class TestMatchTeam:
    def test_one_league_keeps_a_namesake_elsewhere_out(self) -> None:
        # Portugal's "Vitória SC" is Guimaraes in football-data.co.uk; Brazil has a "Vitoria".
        everywhere = {"vitoria": "Vitoria", "guimaraes": "Guimaraes", "porto": "Porto"}
        portugal = {"guimaraes": "Guimaraes", "porto": "Porto"}
        brazil = {"vitoria": "Vitoria", "flamengo": "Flamengo"}
        assert match_team("Vitória SC", portugal) == "guimaraes"
        assert match_team("EC Vitória", brazil) == "vitoria"
        # the curated alias wins over an exact namesake, even across leagues
        assert match_team("Vitória SC", everywhere) == "guimaraes"

    def test_verbose_names_and_ambiguity(self) -> None:
        teams = {"dortmund", "bayern munich", "man city", "man united"}
        assert match_team("Borussia Dortmund", teams) == "dortmund"
        assert match_team("FC Bayern München", teams) == "bayern munich"  # curated alias
        assert match_team("Man", teams) is None  # two candidates -> no guess

    def test_unresolved_keeps_the_source_name(self) -> None:
        assert resolve_canonical_name("Sabah FK", {"porto": "Porto"}) == ("Sabah FK", "sabah fk")


def test_live_db_rekeys_names_stored_before_transliteration(tmp_path) -> None:
    from soccer.storage.live_db import LiveDB

    path = tmp_path / "live.sqlite"
    LiveDB(path).close()
    conn = sqlite3.connect(path)
    conn.execute(
        "INSERT INTO canonical_entity VALUES ('t1', 'team', 'Brøndby IF', 'br ndby if', 'DK', 'x')"
    )
    conn.execute("PRAGMA user_version=10")  # as written by the old normalizer
    conn.commit()
    conn.close()

    LiveDB(path).close()  # migrates to v11
    conn = sqlite3.connect(path)
    assert conn.execute("SELECT normalized_name FROM canonical_entity").fetchone() == (
        "brondby if",
    )
    assert conn.execute("PRAGMA user_version").fetchone()[0] >= 11


def test_analytics_db_rekeys_results_once(tmp_path) -> None:
    from dataclasses import replace

    from soccer.storage.analytics_db import NAMES_VERSION, AnalyticsDB
    from tests.test_integrity import _row

    path = tmp_path / "a.duckdb"
    row = replace(
        _row("2526", "UCL", date(2026, 1, 3), "FC København", "Arsenal"), home_norm="k benhavn"
    )
    with AnalyticsDB(path) as adb:
        adb.load_results([row])
        adb._con.execute("DELETE FROM store_meta")  # as if opened before NAMES_VERSION existed
        adb._con.execute("UPDATE results SET home_norm='k benhavn'")
    with AnalyticsDB(path) as adb:
        (out,) = adb.outcomes_for("2526", "UCL")
        version = adb._con.execute("SELECT value FROM store_meta").fetchone()
    assert out.home_norm == "kobenhavn"
    assert version == (str(NAMES_VERSION),)


def test_audit_flags_a_squad_filed_under_the_wrong_club(tmp_path) -> None:
    from soccer.dashboard.data import name_audit
    from soccer.storage.analytics_db import AnalyticsDB, SquadMember
    from tests.test_dashboard_data import seed_results

    analytics = tmp_path / "analytics.duckdb"
    seed_results(analytics, division="P1", teams=["Porto", "Guimaraes"])
    seed_results(analytics, division="BRA", teams=["Vitoria", "Flamengo"], season="2026")

    def member(team: str, norm: str) -> SquadMember:
        return SquadMember("PPL", team, norm, None, "A Player", "a player", None, None, None, "x")

    with AnalyticsDB(analytics) as adb:
        adb.load_squads([member("Porto", "porto"), member("Vitoria", "vitoria")])
    links = {link.link: link for link in name_audit(tmp_path / "none.sqlite", analytics)}
    squads = links["Squads (PPL) → results"]
    assert (squads.resolved, squads.total, squads.unresolved) == (1, 2, ["Vitoria"])
    assert not squads.healthy
