"""Anthropic Messages API adapter implementing the Model ABC, for gold-trajectory generation."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from toolsmith.env.model import Model

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_MAX_TOKENS = 1024


class AnthropicRequestError(RuntimeError):
    """Raised when the Anthropic API request fails or returns an unusable response."""


def _to_anthropic_messages(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Split our internal message log into an Anthropic `system` string + `messages` list.

    The episode protocol represents tool results as plain-text "tool" role turns (not native
    tool_use/tool_result blocks) so every model backend shares one uniform text protocol; this
    folds "tool" role turns into a user turn instead of using Anthropic's native tool_result
    content blocks.
    """
    system_parts = []
    converted = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "system":
            system_parts.append(content)
        elif role == "tool":
            converted.append({"role": "user", "content": f"Tool result: {content}"})
        else:
            converted.append({"role": role, "content": content})
    return "\n\n".join(system_parts), converted


def _post_anthropic(body: dict[str, Any], api_key: str) -> dict[str, Any]:
    """POST a request body to the Anthropic Messages API and return the parsed JSON response."""
    request = urllib.request.Request(
        ANTHROPIC_API_URL,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read())
    except (OSError, urllib.error.URLError) as exc:
        raise AnthropicRequestError(f"Anthropic API request failed: {exc}") from exc


class AnthropicModel(Model):
    """Model adapter that calls the Anthropic Messages API, for gold-trajectory generation."""

    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = DEFAULT_MAX_TOKENS) -> None:
        self._model = model
        self._max_tokens = max_tokens

    def generate(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
        """Call the Anthropic Messages API and return its raw text response.

        `tools` is unused: the episode protocol's tool list is already embedded as plain text
        in the system prompt (see env/runner.py's build_system_prompt), so no native tool-use
        parameter is needed here.
        """
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise AnthropicRequestError("ANTHROPIC_API_KEY is not set")

        system, anthropic_messages = _to_anthropic_messages(messages)
        body = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": system,
            "messages": anthropic_messages,
        }
        payload = _post_anthropic(body, api_key)

        try:
            return "".join(
                block["text"] for block in payload["content"] if block["type"] == "text"
            )
        except (KeyError, TypeError) as exc:
            raise AnthropicRequestError(f"unexpected Anthropic response shape: {payload}") from exc
