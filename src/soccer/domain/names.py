"""Team and competition name normalization.

This is the join mechanism between sources that share no common identifiers, and it is
where identity bugs hide -- so it is deliberately conservative. The rule the whole
crosswalk depends on: **normalization may bring two spellings of the same club
together, but it must never collapse two genuinely different clubs.** When in doubt,
leave names distinct and let the resolver record an unresolved link for review.

Grounded in real divergences observed between football-data.org and TheSportsDB:

    "Arsenal FC"                vs "Arsenal"                  -> both "arsenal"
    "Aston Villa FC"            vs "Aston Villa"              -> both "aston villa"
    "Brighton & Hove Albion FC" vs "Brighton and Hove Albion"-> both "brighton hove albion"
    "1. FC Köln"                vs "FC Cologne"               -> "koln" vs "cologne" (no match)

That last case is the point: Köln and Cologne are the same club under a translation,
which normalization cannot and must not paper over. Forcing a match there would be a
silent data-corruption bug.
"""

from __future__ import annotations

import re
import unicodedata

# Club-type affixes stripped only when they appear as whole tokens at the start or end
# of a name. Kept small on purpose: an aggressive list risks turning distinct clubs
# into collisions. These are the safe, high-frequency ones seen across the free sources.
#
# NOT included, and deliberately so:
#   "united"/"city"/"town"/"rovers" -- these distinguish clubs ("Man United" vs
#   "Man City"), so stripping them would merge rivals.
_AFFIXES = frozenset(
    {
        "fc",
        "afc",
        "cf",
        "sc",
        "ac",
        "as",
        "ss",
        "ssc",
        "us",
        "cd",
        "sd",
        "rc",
        "ol",  # only stripped as a bare leading/trailing token; "Lyon" survives anyway
        "club",
        "calcio",
    }
)

# Connector tokens dropped when they stand alone, so "Brighton & Hove" == "Brighton
# and Hove" and "Atlético de Madrid" == "Atlético Madrid". Removed only as whole
# tokens, so "Deportivo" (starts with "de") and similar are untouched.
_CONNECTORS = frozenset({"and", "de", "di", "do", "da", "of"})

# "1." in "1. FC Köln", "1899" is kept (part of Hoffenheim's identity) -- so only a
# leading ordinal like "1." is dropped, not embedded digits.
_LEADING_ORDINAL = re.compile(r"^\d+\.\s*")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def strip_diacritics(text: str) -> str:
    """Köln -> Koln, Atlético -> Atletico. Decompose then drop combining marks."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize_name(raw: str) -> str:
    """Reduce a club or competition name to a comparable canonical form.

    The transform, in order: drop a leading ordinal, lowercase, strip diacritics,
    map '&' to 'and', split on non-alphanumerics, drop the 'and' stopword, drop
    club-type affixes only at the ends, and rejoin on single spaces.

    Returns "" only for input that is empty or entirely punctuation -- callers treat
    an empty normalization as "unnameable", never as a match key.
    """
    if not raw:
        return ""

    text = _LEADING_ORDINAL.sub("", raw.strip())
    text = strip_diacritics(text).lower()
    text = text.replace("&", " and ")
    tokens = [t for t in _NON_ALNUM.split(text) if t]

    # Drop connectors so "Brighton and Hove" == "Brighton & Hove" and
    # "Atlético de Madrid" == "Atlético Madrid". Whole-token only.
    tokens = [t for t in tokens if t not in _CONNECTORS]

    # Strip affixes only at the boundaries, and never strip away the entire name --
    # "FC" alone must not normalize to "".
    while len(tokens) > 1 and tokens[0] in _AFFIXES:
        tokens.pop(0)
    while len(tokens) > 1 and tokens[-1] in _AFFIXES:
        tokens.pop()

    return " ".join(tokens)


def name_tokens(raw: str) -> frozenset[str]:
    """Token set of a normalized name, for overlap heuristics above the exact match."""
    normalized = normalize_name(raw)
    return frozenset(normalized.split()) if normalized else frozenset()


def names_match(a: str, b: str) -> bool:
    """Exact match on normalized names. The only automatic-merge criterion.

    Deliberately strict: equality after normalization, nothing fuzzier. Anything this
    rejects is left for the resolver to record as unresolved rather than guessed.
    """
    na, nb = normalize_name(a), normalize_name(b)
    return bool(na) and na == nb
