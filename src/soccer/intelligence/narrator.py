"""Narrate an answer the assistant has already computed -- and reject any invented number.

The rule-based assistant (dashboard/assistant.py) routes the question, queries the local
stores through the same tested functions the pages use, and produces a Reply. This module
hands the LLM only that Reply (its text and table rows) plus the question and the data date,
asks for a short conversational rewrite, and then checks the result: every number in the
narration must appear in the facts it was given. One that doesn't means the model added
something it was never told, so the narration is discarded and the computed answer stands.

Never narrated: questions the assistant couldn't answer (there is nothing computed to
explain, and the LLM must not answer football questions from its own memory).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from soccer.intelligence.llm_provider import LLMClient, LLMError

SYSTEM_PROMPT = (
    "You narrate answers for a football analytics app. You get a user's question and the "
    "app's computed answer (FACTS). Rewrite it as a short, clear reply of at most about 120 "
    "words.\n"
    "Rules:\n"
    "- Use only what FACTS say. Every number you write must appear in FACTS exactly as written "
    "there (same digits, same format, e.g. 62% stays 62%). Do not compute, round, convert or "
    "add numbers.\n"
    "- Do not add names, dates, statistics, predictions or opinions that FACTS don't contain.\n"
    "- If FACTS say data is missing or limited, say so. Keep any caveat FACTS give about the "
    "model's accuracy or the bookmakers.\n"
    "- No betting advice. Plain prose; a few short bullet points are fine."
)

# Never narrate these: nothing was computed, or the answer is already the plain help text.
SKIP_INTENTS = frozenset({"fallback", "help", None})
MAX_TABLE_ROWS = 20

_NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")


def _canonical(token: str) -> str:
    """'1,931' -> '1931', '0.50' -> '0.5', '2.0' -> '2', so equal values compare equal."""
    plain = token.replace(",", "")
    if "." in plain:
        plain = plain.rstrip("0").rstrip(".")
    return plain or "0"


def numbers_in(text: str) -> set[str]:
    return {_canonical(m) for m in _NUMBER.findall(text)}


def facts_from_reply(text: str, table: list[dict[str, Any]] | None) -> str:
    lines = [text.strip()]
    if table:
        lines.append("")
        lines.append("Table:")
        for row in table[:MAX_TABLE_ROWS]:
            lines.append("; ".join(f"{k}: {v}" for k, v in row.items()))
        if len(table) > MAX_TABLE_ROWS:
            lines.append(f"(first {MAX_TABLE_ROWS} of {len(table)} rows)")
    return "\n".join(lines)


@dataclass(frozen=True)
class Narration:
    text: str | None
    """The narration, or None when it was rejected or failed (show the computed answer)."""
    model: str
    note: str = ""
    invented: list[str] = field(default_factory=list)
    """Numbers the narration used that the facts don't contain -- why it was rejected."""

    @property
    def accepted(self) -> bool:
        return self.text is not None


def should_narrate(intent: str | None) -> bool:
    return intent not in SKIP_INTENTS


def narrate(
    client: LLMClient,
    question: str,
    text: str,
    table: list[dict[str, Any]] | None,
    *,
    data_date: date | None,
) -> Narration:
    """Ask the LLM to explain a computed answer; keep it only if it adds no number."""
    facts = facts_from_reply(text, table)
    as_of = data_date.isoformat() if data_date else "unknown"
    prompt = f"Question: {question}\nData through: {as_of}\n\nFACTS:\n{facts}"
    model = f"{client.provider}/{client.model}"
    try:
        out = client.complete(SYSTEM_PROMPT, prompt)
    except LLMError as exc:
        return Narration(None, model, note=f"narrator unavailable ({exc})")
    allowed = numbers_in(facts) | numbers_in(question) | numbers_in(as_of)
    invented = sorted(numbers_in(out.text) - allowed)
    if invented:
        return Narration(
            None,
            model,
            note="narration discarded: it used numbers the computed answer doesn't contain",
            invented=invented,
        )
    return Narration(out.text, model)
