"""Name normalization tests.

The heart of the crosswalk, so tested hardest -- and split deliberately into two
groups: matches that MUST happen (real spellings of one club) and matches that MUST
NOT (distinct clubs, and same-club-different-language). A false merge corrupts data
silently, so the negative cases matter as much as the positive ones.
"""

from __future__ import annotations

import pytest

from soccer.domain.names import (
    name_tokens,
    names_match,
    normalize_name,
    strip_diacritics,
)


class TestMustMatch:
    """Real divergences between football-data.org and TheSportsDB that must resolve."""

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            ("Arsenal FC", "Arsenal"),
            ("Aston Villa FC", "Aston Villa"),
            ("Chelsea FC", "Chelsea"),
            ("Liverpool FC", "Liverpool"),
            ("Brighton & Hove Albion FC", "Brighton and Hove Albion"),
            ("Wolverhampton Wanderers FC", "Wolverhampton Wanderers"),
            # Diacritics
            ("1. FC Köln", "FC Koln"),
            ("Atlético de Madrid", "Atletico Madrid"),
            # Affix position and case
            ("SSC Napoli", "Napoli"),
            ("AC Milan", "Milan"),
        ],
    )
    def test_pairs_resolve_to_same_name(self, a: str, b: str) -> None:
        assert names_match(a, b), f"{a!r} and {b!r} should match"


class TestMustNotMatch:
    """False merges corrupt data silently. These distinct clubs must stay distinct."""

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            # Rivals distinguished only by a token an aggressive normalizer might strip.
            ("Manchester United FC", "Manchester City FC"),
            ("Sheffield United", "Sheffield Wednesday"),
            # First team vs a distinct reserve/second side.
            ("Atlanta United", "Atlanta United II"),
            # Same club under a language difference -- normalization cannot bridge this,
            # and must not pretend to. The resolver records it unresolved instead.
            ("1. FC Köln", "FC Cologne"),
            ("Bayern München", "Bayern Munich"),
            # Entirely different clubs.
            ("Arsenal FC", "Aston Villa FC"),
        ],
    )
    def test_distinct_clubs_do_not_match(self, a: str, b: str) -> None:
        assert not names_match(a, b), f"{a!r} and {b!r} must NOT match"


class TestNormalization:
    def test_strips_diacritics(self) -> None:
        assert strip_diacritics("Köln") == "Koln"
        assert strip_diacritics("Atlético") == "Atletico"

    def test_leading_ordinal_removed(self) -> None:
        # "1." dropped, then leading "fc" affix dropped too.
        assert normalize_name("1. FC Köln") == "koln"

    def test_latin_connectors_dropped_as_whole_tokens_only(self) -> None:
        assert normalize_name("Atlético de Madrid") == "atletico madrid"
        # ...but a token that merely starts with a connector survives intact.
        assert normalize_name("Deportivo") == "deportivo"

    def test_embedded_digits_survive(self) -> None:
        # 1899 is part of Hoffenheim's identity; only a leading ordinal is dropped.
        assert "1899" in normalize_name("TSG 1899 Hoffenheim")

    def test_affix_never_consumes_whole_name(self) -> None:
        # A name that is only an affix must not normalize to "".
        assert normalize_name("FC") == "fc"

    def test_empty_and_punctuation_only_yield_empty(self) -> None:
        assert normalize_name("") == ""
        assert normalize_name("  ") == ""
        assert normalize_name("...") == ""

    def test_empty_names_never_match(self) -> None:
        # An empty normalization must not become a wildcard join key.
        assert not names_match("", "")
        assert not names_match("FC", "SC")  # both would strip toward empty-ish

    def test_tokens_expose_overlap(self) -> None:
        assert name_tokens("Arsenal FC") == {"arsenal"}
        assert "united" in name_tokens("Manchester United FC")
