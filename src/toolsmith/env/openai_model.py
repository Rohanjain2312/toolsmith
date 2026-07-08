"""OpenAI Chat Completions adapter implementing the Model ABC, via native function-calling."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from toolsmith.env.model import Model

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_MAX_TOKENS = 1024


class OpenAIRequestError(RuntimeError):
    """Raised when the OpenAI API request fails or returns an unusable response."""


def _to_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert our internal message log to OpenAI chat format.

    History is kept as plain system/user/assistant text turns, folding "tool" role results
    into a user turn (same approach as env/anthropic_model.py). This sidesteps OpenAI's
    tool_call_id pairing requirement for native tool_calls in history, while the CURRENT
    generation step below still uses real native function-calling.
    """
    converted = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "tool":
            converted.append({"role": "user", "content": f"Tool result: {content}"})
        else:
            converted.append({"role": role, "content": content})
    return converted


def _post_openai(body: dict[str, Any], api_key: str) -> dict[str, Any]:
    """POST a request body to the OpenAI Chat Completions API and return the parsed response."""
    request = urllib.request.Request(
        OPENAI_API_URL,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={"content-type": "application/json", "authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read())
    except (OSError, urllib.error.URLError) as exc:
        raise OpenAIRequestError(f"OpenAI API request failed: {exc}") from exc


class OpenAIModel(Model):
    """Model adapter that calls the OpenAI Chat Completions API using native function-calling.

    Unlike env/anthropic_model.py (which relies solely on our plain-text tool-call protocol),
    this adapter passes `tools` (already in OpenAI's own format via env/openai_export.py)
    natively, so GPT-4o-mini is evaluated the way it's normally used. The native tool_calls
    response is converted back to our {"tool", "args"} JSON-text protocol so it plugs into the
    same parser/episode runner as every other adapter.
    """

    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = DEFAULT_MAX_TOKENS) -> None:
        self._model = model
        self._max_tokens = max_tokens

    def generate(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
        """Call the OpenAI API with native tools; return raw text in our tool-call protocol."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise OpenAIRequestError("OPENAI_API_KEY is not set")

        body = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": _to_openai_messages(messages),
            "tools": tools,
        }
        payload = _post_openai(body, api_key)

        try:
            message = payload["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise OpenAIRequestError(f"unexpected OpenAI response shape: {payload}") from exc

        tool_calls = message.get("tool_calls")
        if tool_calls:
            call = tool_calls[0]["function"]
            return json.dumps({"tool": call["name"], "args": json.loads(call["arguments"])})
        return message.get("content") or ""
