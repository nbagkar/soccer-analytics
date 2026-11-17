"""Provider-independent chat client: one OpenAI-compatible `/chat/completions` call.

OpenAI, DeepSeek and Google Gemini all serve the OpenAI chat-completions shape, so a single
small httpx client covers the three providers the upgrade plan names. No vendor SDK. The model
id is always the operator's choice (SOCCER_LLM_MODEL): model names and prices change too often
to hard-code, and this project only states what it has verified.

Spending is bounded locally: every call's token usage (from the response's `usage` object) is
added to a per-month ledger in `quota.sqlite`, and a call that would start once the month's cap
is spent is refused before it is sent. Set a hard limit on the provider's own dashboard too --
that is the only cap the provider enforces.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from soccer.config import Settings

# OpenAI-compatible endpoints per provider. Base URLs only; models are configured.
PROVIDER_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
}

_LEDGER = """
CREATE TABLE IF NOT EXISTS llm_usage (
    month          TEXT NOT NULL,   -- UTC, YYYY-MM
    provider       TEXT NOT NULL,
    model          TEXT NOT NULL,
    calls          INTEGER NOT NULL,
    input_tokens   INTEGER NOT NULL,
    output_tokens  INTEGER NOT NULL,
    PRIMARY KEY (month, provider, model)
);
"""


class LLMError(RuntimeError):
    """The provider failed, refused, or answered in an unexpected shape."""


class LLMBudgetError(LLMError):
    """This month's token cap is spent -- refused before sending."""


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    model: str


@dataclass(frozen=True)
class MonthUsage:
    month: str
    calls: int
    input_tokens: int
    output_tokens: int
    cap: int
    price_in_per_m: float | None
    price_out_per_m: float | None

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def estimated_cost(self) -> float | None:
        """USD at the operator's configured prices, or None when they aren't set."""
        if self.price_in_per_m is None or self.price_out_per_m is None:
            return None
        return (
            self.input_tokens * self.price_in_per_m + self.output_tokens * self.price_out_per_m
        ) / 1_000_000


class LLMClient:
    def __init__(
        self,
        *,
        provider: str,
        model: str,
        api_key: str,
        ledger_path: Path,
        monthly_token_cap: int,
        base_url: str | None = None,
        price_in_per_m: float | None = None,
        price_out_per_m: float | None = None,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        url = base_url or PROVIDER_BASE_URLS.get(provider)
        if not url:
            raise ValueError(
                f"Unknown LLM provider {provider!r}: use one of {sorted(PROVIDER_BASE_URLS)} "
                "or set SOCCER_LLM_BASE_URL"
            )
        if not model or not api_key:
            raise ValueError("An LLM needs both SOCCER_LLM_MODEL and SOCCER_LLM_API_KEY")
        self.provider = provider
        self.model = model
        self._url = url.rstrip("/") + "/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._client = client or httpx.Client(timeout=timeout)
        self._ledger = Path(ledger_path)
        self._cap = monthly_token_cap
        self._prices = (price_in_per_m, price_out_per_m)
        self._clock = clock
        self._ledger.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._ledger) as con:
            con.executescript(_LEDGER)

    def _month(self) -> str:
        return self._clock().astimezone(UTC).strftime("%Y-%m")

    def usage(self) -> MonthUsage:
        with sqlite3.connect(self._ledger) as con:
            row = con.execute(
                "SELECT COALESCE(SUM(calls),0), COALESCE(SUM(input_tokens),0), "
                "COALESCE(SUM(output_tokens),0) FROM llm_usage WHERE month=?",
                (self._month(),),
            ).fetchone()
        return MonthUsage(self._month(), row[0], row[1], row[2], self._cap, *self._prices)

    def _record(self, input_tokens: int, output_tokens: int) -> None:
        with sqlite3.connect(self._ledger) as con:
            con.execute(
                "INSERT INTO llm_usage VALUES (?,?,?,1,?,?) "
                "ON CONFLICT(month, provider, model) DO UPDATE SET calls=calls+1, "
                "input_tokens=input_tokens+excluded.input_tokens, "
                "output_tokens=output_tokens+excluded.output_tokens",
                (self._month(), self.provider, self.model, input_tokens, output_tokens),
            )

    def complete(self, system: str, user: str, *, max_tokens: int = 400) -> Completion:
        """One chat completion. Raises LLMBudgetError before sending if the cap is spent."""
        spent = self.usage()
        if spent.tokens >= self._cap:
            raise LLMBudgetError(
                f"LLM token cap for {spent.month} reached ({spent.tokens:,}/{self._cap:,})"
            )
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }
        try:
            response = self._client.post(self._url, json=body, headers=self._headers)
        except httpx.HTTPError as exc:
            raise LLMError(f"{self.provider}: {type(exc).__name__}") from exc
        try:
            data = response.json()
        except ValueError as exc:
            raise LLMError(f"{self.provider}: HTTP {response.status_code}, not JSON") from exc
        usage = (data.get("usage") or {}) if isinstance(data, dict) else {}
        input_tokens = int(usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or 0)
        if input_tokens or output_tokens:
            self._record(input_tokens, output_tokens)
        if response.status_code >= 400 or not isinstance(data, dict) or data.get("error"):
            detail = data.get("error") if isinstance(data, dict) else None
            message = detail.get("message") if isinstance(detail, dict) else detail
            raise LLMError(f"{self.provider}: HTTP {response.status_code}: {message}")
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"{self.provider}: response has no message content") from exc
        if not isinstance(text, str) or not text.strip():
            raise LLMError(f"{self.provider}: empty completion")
        return Completion(
            text.strip(), input_tokens, output_tokens, str(data.get("model") or self.model)
        )


def client_from_settings(settings: Settings) -> LLMClient | None:
    """The configured client, or None when the narrator is off (any setting missing)."""
    if not (settings.llm_provider and settings.llm_model and settings.llm_api_key):
        return None
    return LLMClient(
        provider=settings.llm_provider,
        model=settings.llm_model,
        api_key=settings.llm_api_key.get_secret_value(),
        ledger_path=settings.quota_db,
        monthly_token_cap=settings.llm_monthly_token_cap,
        base_url=settings.llm_base_url,
        price_in_per_m=settings.llm_price_in_per_m,
        price_out_per_m=settings.llm_price_out_per_m,
        timeout=settings.http_timeout_seconds,
    )
