"""Tests for the Anthropic Messages API model adapter (network calls always mocked)."""

from __future__ import annotations

import pytest

from toolsmith.env import anthropic_model as anthropic_model_module
from toolsmith.env.anthropic_model import (
    AnthropicModel,
    AnthropicRequestError,
    _to_anthropic_messages,
)


def test_to_anthropic_messages_splits_system_and_folds_tool_turns() -> None:
    messages = [
        {"role": "system", "content": "You are a travel assistant."},
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": '{"tool": "weather_lookup", "args": {}}'},
        {"role": "tool", "tool_name": "weather_lookup", "content": '{"temp_c": 20}'},
    ]

    system, converted = _to_anthropic_messages(messages)

    assert system == "You are a travel assistant."
    assert [m["role"] for m in converted] == ["user", "assistant", "user"]
    assert "temp_c" in converted[-1]["content"]


def test_generate_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(
        anthropic_model_module,
        "_post_anthropic",
        lambda body, api_key: {"content": [{"type": "text", "text": "Sunny in Paris."}]},
    )

    model = AnthropicModel()
    result = model.generate(
        [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}], []
    )

    assert result == "Sunny in Paris."


def test_generate_concatenates_multiple_text_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(
        anthropic_model_module,
        "_post_anthropic",
        lambda body, api_key: {
            "content": [{"type": "text", "text": "Hello "}, {"type": "text", "text": "world"}]
        },
    )

    model = AnthropicModel()
    result = model.generate([{"role": "user", "content": "hi"}], [])

    assert result == "Hello world"


def test_generate_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    model = AnthropicModel()
    with pytest.raises(AnthropicRequestError):
        model.generate([{"role": "user", "content": "hi"}], [])


def test_generate_unexpected_response_shape_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setattr(
        anthropic_model_module, "_post_anthropic", lambda body, api_key: {"unexpected": True}
    )

    model = AnthropicModel()
    with pytest.raises(AnthropicRequestError):
        model.generate([{"role": "user", "content": "hi"}], [])


def test_generate_request_body_uses_configured_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    captured: dict = {}

    def fake_post(body: dict, api_key: str) -> dict:
        captured.update(body)
        return {"content": [{"type": "text", "text": "ok"}]}

    monkeypatch.setattr(anthropic_model_module, "_post_anthropic", fake_post)

    model = AnthropicModel(model="claude-sonnet-5", max_tokens=512)
    model.generate([{"role": "user", "content": "hi"}], [])

    assert captured["model"] == "claude-sonnet-5"
    assert captured["max_tokens"] == 512
