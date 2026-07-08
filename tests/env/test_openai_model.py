"""Tests for the OpenAI Chat Completions model adapter (network calls always mocked)."""

from __future__ import annotations

import pytest

from toolsmith.env import openai_model as openai_model_module
from toolsmith.env.openai_model import OpenAIModel, OpenAIRequestError, _to_openai_messages

SAMPLE_TOOLS = [
    {
        "type": "function",
        "function": {"name": "geocode_city", "description": "...", "parameters": {}},
    }
]


def test_to_openai_messages_folds_tool_turns_into_user_text() -> None:
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": '{"tool": "geocode_city", "args": {}}'},
        {"role": "tool", "tool_name": "geocode_city", "content": '{"lat": 48.8}'},
    ]

    converted = _to_openai_messages(messages)

    assert [m["role"] for m in converted] == ["system", "user", "assistant", "user"]
    assert "lat" in converted[-1]["content"]


def test_generate_returns_tool_call_as_our_json_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(
        openai_model_module,
        "_post_openai",
        lambda body, api_key: {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "geocode_city",
                                    "arguments": '{"city": "Paris"}',
                                }
                            }
                        ],
                    }
                }
            ]
        },
    )

    model = OpenAIModel()
    result = model.generate([{"role": "user", "content": "geocode paris"}], SAMPLE_TOOLS)

    assert result == '{"tool": "geocode_city", "args": {"city": "Paris"}}'


def test_generate_returns_plain_content_as_final_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(
        openai_model_module,
        "_post_openai",
        lambda body, api_key: {"choices": [{"message": {"content": "Sunny in Paris."}}]},
    )

    model = OpenAIModel()
    result = model.generate([{"role": "user", "content": "hi"}], SAMPLE_TOOLS)

    assert result == "Sunny in Paris."


def test_generate_request_body_passes_tools_through_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    captured: dict = {}

    def fake_post(body: dict, api_key: str) -> dict:
        captured.update(body)
        return {"choices": [{"message": {"content": "ok"}}]}

    monkeypatch.setattr(openai_model_module, "_post_openai", fake_post)

    OpenAIModel(model="gpt-4o-mini").generate([{"role": "user", "content": "hi"}], SAMPLE_TOOLS)

    assert captured["tools"] is SAMPLE_TOOLS
    assert captured["model"] == "gpt-4o-mini"


def test_generate_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    model = OpenAIModel()
    with pytest.raises(OpenAIRequestError):
        model.generate([{"role": "user", "content": "hi"}], SAMPLE_TOOLS)


def test_generate_unexpected_response_shape_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(openai_model_module, "_post_openai", lambda body, api_key: {"bad": True})

    model = OpenAIModel()
    with pytest.raises(OpenAIRequestError):
        model.generate([{"role": "user", "content": "hi"}], SAMPLE_TOOLS)
