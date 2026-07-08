"""Tests for the llama.cpp model adapter (no real GGUF file or model download used)."""

from __future__ import annotations

from typing import Any

import pytest

from toolsmith.env.llamacpp_model import LlamaCppModel


class _FakeEngine:
    """Stand-in for llama_cpp.Llama, injected via LlamaCppModel(engine=...) for tests."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict[str, Any]] = []

    def create_chat_completion(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {"choices": [{"message": {"content": self.content}}]}


def test_generate_returns_content() -> None:
    engine = _FakeEngine("Paris is sunny today.")
    model = LlamaCppModel(engine=engine)

    result = model.generate([{"role": "user", "content": "hi"}], [])

    assert result == "Paris is sunny today."


def test_generate_passes_messages_through() -> None:
    engine = _FakeEngine("ok")
    model = LlamaCppModel(engine=engine)
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]

    model.generate(messages, [])

    assert engine.calls[0]["messages"] == messages


def test_generate_uses_greedy_temperature_by_default() -> None:
    engine = _FakeEngine("ok")
    model = LlamaCppModel(engine=engine)

    model.generate([{"role": "user", "content": "hi"}], [])

    assert engine.calls[0]["temperature"] == 0.0


def test_generate_respects_custom_max_tokens_and_temperature() -> None:
    engine = _FakeEngine("ok")
    model = LlamaCppModel(engine=engine, max_tokens=128, temperature=0.7)

    model.generate([{"role": "user", "content": "hi"}], [])

    assert engine.calls[0]["max_tokens"] == 128
    assert engine.calls[0]["temperature"] == 0.7


def test_generate_none_content_returns_empty_string() -> None:
    engine = _FakeEngine(None)  # type: ignore[arg-type]
    model = LlamaCppModel(engine=engine)

    assert model.generate([{"role": "user", "content": "hi"}], []) == ""


def test_missing_model_path_and_engine_raises() -> None:
    with pytest.raises(ValueError, match="model_path"):
        LlamaCppModel()
