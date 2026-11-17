"""LLM narrator: the provider client (mocked HTTP), the invented-number guard, and the chat page.

The non-negotiables under test: the app works with no LLM configured; a provider failure
or a spent budget never breaks an answer; and a narration that mentions a number the
computed answer doesn't contain is discarded.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from soccer.config import Settings
from soccer.intelligence.llm_provider import (
    PROVIDER_BASE_URLS,
    Completion,
    LLMBudgetError,
    LLMClient,
    LLMError,
    client_from_settings,
)
from soccer.intelligence.narrator import (
    facts_from_reply,
    narrate,
    numbers_in,
    should_narrate,
)


def completion(text: str, prompt_tokens: int = 100, completion_tokens: int = 40) -> dict[str, Any]:
    return {
        "model": "test-model",
        "choices": [{"message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


def make_client(tmp_path: Path, handler: Any, *, cap: int = 10_000, **kw: Any) -> LLMClient:
    return LLMClient(
        provider="deepseek",
        model="test-model",
        api_key="sk-test",
        ledger_path=tmp_path / "quota.sqlite",
        monthly_token_cap=cap,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        **kw,
    )


class TestClient:
    def test_sends_openai_compatible_request_and_records_usage(self, tmp_path: Path) -> None:
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(200, json=completion("Arsenal lead."))

        c = make_client(tmp_path, handler, price_in_per_m=0.3, price_out_per_m=1.2)
        out = c.complete("system", "user")
        assert out == Completion("Arsenal lead.", 100, 40, "test-model")
        req = seen[0]
        assert str(req.url) == PROVIDER_BASE_URLS["deepseek"] + "/chat/completions"
        assert req.headers["authorization"] == "Bearer sk-test"
        body = json.loads(req.content)
        assert body["model"] == "test-model" and body["messages"][0]["role"] == "system"
        usage = c.usage()
        assert (usage.calls, usage.tokens) == (1, 140)
        assert usage.estimated_cost == pytest.approx((100 * 0.3 + 40 * 1.2) / 1e6)

    def test_errors_raise_llm_error(self, tmp_path: Path) -> None:
        bad = {"error": {"message": "invalid api key"}}
        c = make_client(tmp_path, lambda r: httpx.Response(401, json=bad))
        with pytest.raises(LLMError, match="invalid api key"):
            c.complete("s", "u")
        c = make_client(tmp_path, lambda r: httpx.Response(200, text="<html>"))
        with pytest.raises(LLMError):
            c.complete("s", "u")
        c = make_client(tmp_path, lambda r: httpx.Response(200, json={"choices": []}))
        with pytest.raises(LLMError):
            c.complete("s", "u")

    def test_network_failure_is_an_llm_error(self, tmp_path: Path) -> None:
        def boom(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("down")

        with pytest.raises(LLMError):
            make_client(tmp_path, boom).complete("s", "u")

    def test_spent_cap_refuses_before_sending_and_resets_monthly(self, tmp_path: Path) -> None:
        sent: list[int] = []
        now = [datetime(2026, 11, 30, 23, 0, tzinfo=UTC)]

        def handler(request: httpx.Request) -> httpx.Response:
            sent.append(1)
            return httpx.Response(200, json=completion("ok", 100, 50))

        c = make_client(tmp_path, handler, cap=150, clock=lambda: now[0])
        c.complete("s", "u")
        with pytest.raises(LLMBudgetError):
            c.complete("s", "u")
        assert len(sent) == 1
        now[0] = datetime(2026, 12, 1, 0, 30, tzinfo=UTC)
        c.complete("s", "u")  # a new month
        assert len(sent) == 2

    def test_unknown_provider_without_base_url_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError):
            LLMClient(
                provider="acme",
                model="m",
                api_key="k",
                ledger_path=tmp_path / "q.sqlite",
                monthly_token_cap=10,
            )


class TestSettings:
    def test_off_unless_fully_configured_and_key_hidden(self, tmp_path: Path) -> None:
        assert client_from_settings(Settings(data_dir=tmp_path)) is None
        half = Settings(data_dir=tmp_path, llm_provider="openai", llm_model="m")
        assert client_from_settings(half) is None
        full = Settings(
            data_dir=tmp_path,
            llm_provider="openai",
            llm_model="m",
            llm_api_key="sk-secret-value",  # type: ignore[arg-type]
        )
        client = client_from_settings(full)
        assert client is not None and client.provider == "openai"
        assert "sk-secret-value" not in repr(full)


class Fake:
    """A stand-in client: returns fixed text, or raises."""

    provider, model = "fake", "m1"

    def __init__(self, text: str | None = None, error: Exception | None = None) -> None:
        self.text, self.error = text, error
        self.prompts: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, *, max_tokens: int = 400) -> Completion:
        self.prompts.append((system, user))
        if self.error:
            raise self.error
        assert self.text is not None
        return Completion(self.text, 10, 10, self.model)


COMPUTED = "**Premier League 2026/27 — Arsenal** lead on 27 points, 5 clear of Brighton (22)."
TABLE = [{"Team": "Arsenal", "Pts": 27}, {"Team": "Brighton", "Pts": 22}]


class TestNarrator:
    def test_numbers_are_canonical(self) -> None:
        assert numbers_in("1,931 matches, 62%, xG 2.50, 2026/27") == {
            "1931",
            "62",
            "2.5",
            "2026",
            "27",
        }

    def test_facts_carry_the_table(self) -> None:
        facts = facts_from_reply(COMPUTED, TABLE)
        assert "Team: Arsenal; Pts: 27" in facts and COMPUTED in facts

    def test_a_faithful_narration_is_kept(self) -> None:
        fake = Fake("Arsenal are top with 27 points, 5 ahead of Brighton on 22.")
        n = narrate(fake, "who is top?", COMPUTED, TABLE, data_date=date(2026, 11, 15))  # type: ignore[arg-type]
        assert n.accepted and n.text and "27 points" in n.text
        assert "Data through: 2026-11-15" in fake.prompts[0][1]

    def test_an_invented_number_discards_the_narration(self) -> None:
        fake = Fake("Arsenal are top with 27 points and have won 9 of their 10 games.")
        n = narrate(fake, "who is top?", COMPUTED, TABLE, data_date=None)  # type: ignore[arg-type]
        assert not n.accepted and n.invented == ["10", "9"]

    def test_provider_failure_falls_back_to_the_computed_answer(self) -> None:
        fake = Fake(error=LLMBudgetError("cap reached"))
        n = narrate(fake, "q", COMPUTED, None, data_date=None)  # type: ignore[arg-type]
        assert not n.accepted and "cap reached" in n.note

    def test_unanswered_questions_are_never_sent(self) -> None:
        assert not should_narrate("fallback") and not should_narrate(None)
        assert not should_narrate("help")
        assert should_narrate("standings")


class TestChatPage:
    def _run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text: str) -> Any:
        pytest.importorskip("streamlit")
        from importlib.resources import files

        import soccer.config as config
        from soccer.intelligence import llm_provider
        from tests.test_dashboard_data import TestAnalyticsSnapshot

        TestAnalyticsSnapshot()._seed_results(tmp_path / "analytics.duckdb")
        monkeypatch.setenv("SOCCER_DATA_DIR", str(tmp_path))
        monkeypatch.setenv("SOCCER_AUTO_REFRESH", "false")
        monkeypatch.setenv("SOCCER_LLM_PROVIDER", "deepseek")
        monkeypatch.setenv("SOCCER_LLM_MODEL", "m1")
        monkeypatch.setenv("SOCCER_LLM_API_KEY", "sk-test")
        monkeypatch.setattr(
            llm_provider.LLMClient,
            "complete",
            lambda self, system, user, max_tokens=400: Completion(text, 10, 10, "m1"),
        )
        config._settings = None
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(files("soccer.dashboard") / "app.py"), default_timeout=120)
        at.session_state["_pin_page"] = "Assistant"
        at.run()
        assert not at.exception
        at.chat_input[0].set_value("who is top of the premier league?").run()
        assert not at.exception, at.exception
        return at

    def test_narration_shown_with_the_computed_answer_kept(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import soccer.config as config

        try:
            at = self._run(tmp_path, monkeypatch, "Here is the table you asked for.")
            said = [m.value for m in at.markdown]
            assert "Here is the table you asked for." in said
            assert any("Explained by deepseek/m1" in (c.value or "") for c in at.caption)
            assert at.expander, "the computed answer stays one click away"
        finally:
            config._settings = None

    def test_invented_numbers_fall_back_to_the_computed_answer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import soccer.config as config

        try:
            at = self._run(tmp_path, monkeypatch, "They have 999 points.")
            assert "They have 999 points." not in [m.value for m in at.markdown]
            assert any("narration discarded" in (c.value or "") for c in at.caption)
        finally:
            config._settings = None
