"""Bound oversized tool-result text before it re-enters the model's context."""

from __future__ import annotations

import json

MAX_RESULT_TOKENS = 512
# chars/4 is a standard rough estimate for token count; avoids a tokenizer dependency here.
CHARS_PER_TOKEN_APPROX = 4

_TRUNCATION_MARKER = "...[truncated]"


def truncate_result_text(text: str, max_tokens: int = MAX_RESULT_TOKENS) -> str:
    """Truncate `text` to roughly `max_tokens`, appending a marker if cut."""
    max_chars = max_tokens * CHARS_PER_TOKEN_APPROX
    if len(text) <= max_chars:
        return text
    # Truncate to leave room for the marker so the total stays close to max_chars.
    body_budget = max_chars - len(_TRUNCATION_MARKER)
    return text[:body_budget] + _TRUNCATION_MARKER


def truncate_tool_result(result: dict, max_tokens: int = MAX_RESULT_TOKENS) -> dict:
    """Serialize a tool-result dict and truncate it if it exceeds the budget."""
    serialized = json.dumps(result)
    truncated = truncate_result_text(serialized, max_tokens=max_tokens)
    if truncated == serialized:
        return result
    return {"truncated": True, "text": truncated}
