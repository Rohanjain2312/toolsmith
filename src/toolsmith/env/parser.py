"""Parse raw LLM text output into a tool call or a final answer."""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedToolCall:
    """A successfully parsed tool-call request extracted from model text."""

    name: str
    args: dict


@dataclass(frozen=True)
class ParsedFinalAnswer:
    """Plain text treated as the model's final answer (no tool call found)."""

    text: str


class ToolCallParseError(Exception):
    """Raised when text looks like a tool-call attempt but fails to parse."""


def find_balanced_brace_spans(text: str) -> list[str]:
    """Return substrings of `text` for each top-level balanced `{...}` span.

    Public: also reused by eval/bfcl_adapter.py, which (unlike parse_model_output) needs ALL
    spans in a response, not just the first, to support BFCL's "parallel" multi-call category.
    """
    spans: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escape = False
    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    spans.append(text[start : index + 1])
                    start = -1
    if depth > 0 and start != -1:
        # Unterminated brace span: keep the tail so malformed tool-call
        # attempts (e.g. truncated JSON) can still be detected below.
        spans.append(text[start:])
    return spans


def parse_model_output(text: str) -> ParsedToolCall | ParsedFinalAnswer:
    """Extract a tool call or a final answer from raw model text.

    A tool call is signaled by a JSON object containing exactly the keys
    "tool" (string) and "args" (object). Any candidate `{...}` span that
    contains the literal substring '"tool"' but does not parse into a valid
    tool call is treated as a failed tool-call attempt and raises
    `ToolCallParseError`. If no such span exists at all, the stripped text
    is returned as a `ParsedFinalAnswer`.
    """
    stripped = text.strip()
    if "{" not in stripped:
        return ParsedFinalAnswer(text=stripped)

    candidates = find_balanced_brace_spans(stripped)
    looked_like_attempt = False

    for candidate in candidates:
        looks_like_tool_attempt = '"tool"' in candidate
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            if looks_like_tool_attempt:
                looked_like_attempt = True
            continue

        if isinstance(parsed, dict) and "tool" in parsed and "args" in parsed:
            name = parsed["tool"]
            args = parsed["args"]
            if not isinstance(name, str) or not isinstance(args, dict):
                looked_like_attempt = True
                continue
            return ParsedToolCall(name=name, args=args)

        if looks_like_tool_attempt:
            looked_like_attempt = True

    if looked_like_attempt:
        raise ToolCallParseError(
            "Text contains a malformed or incomplete tool-call JSON object."
        )

    return ParsedFinalAnswer(text=stripped)
