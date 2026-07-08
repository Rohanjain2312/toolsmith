"""Format reward components R1-R4, scored directly on a single candidate model action."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from toolsmith.env.parser import (
    ParsedFinalAnswer,
    ParsedToolCall,
    ToolCallParseError,
    parse_model_output,
)
from toolsmith.tools.schemas import ToolNotFoundError, registry

R1_VALID_PARSE = 1.0
R2_TOOL_EXISTS = 0.5
R3_ARGS_VALID = 1.0
R4_NO_DUPLICATE = 0.5


def reward_valid_parse(raw_text: str) -> float:
    """R1: +1.0 if raw_text parses as a valid tool call or final answer, else 0.0."""
    try:
        parse_model_output(raw_text)
    except ToolCallParseError:
        return 0.0
    return R1_VALID_PARSE


def reward_tool_exists(raw_text: str) -> float:
    """R2: +0.5 if the parsed action is a tool call naming a registered tool."""
    import toolsmith.tools.sandbox  # noqa: F401  (registers all 12 sandbox tools)

    try:
        parsed = parse_model_output(raw_text)
    except ToolCallParseError:
        return 0.0
    if not isinstance(parsed, ParsedToolCall):
        return 0.0
    try:
        registry.get(parsed.name)
    except ToolNotFoundError:
        return 0.0
    return R2_TOOL_EXISTS


def reward_args_valid(raw_text: str) -> float:
    """R3: +1.0 if the parsed tool call's args pass that tool's Pydantic validation."""
    import toolsmith.tools.sandbox  # noqa: F401  (registers all 12 sandbox tools)

    try:
        parsed = parse_model_output(raw_text)
    except ToolCallParseError:
        return 0.0
    if not isinstance(parsed, ParsedToolCall):
        return 0.0
    try:
        spec = registry.get(parsed.name)
    except ToolNotFoundError:
        return 0.0
    try:
        spec.args_model.model_validate(parsed.args)
    except ValidationError:
        return 0.0
    return R3_ARGS_VALID


def _prior_tool_calls(prefix_history: list[dict[str, Any]]) -> set[tuple[str, str]]:
    """Extract (tool_name, canonical_args_json) pairs from prior assistant turns in the prefix."""
    seen = set()
    for message in prefix_history:
        if message.get("role") != "assistant":
            continue
        try:
            parsed = parse_model_output(message["content"])
        except ToolCallParseError:
            continue
        if isinstance(parsed, ParsedToolCall):
            seen.add((parsed.name, json.dumps(parsed.args, sort_keys=True)))
    return seen


def reward_no_duplicate(raw_text: str, prefix_history: list[dict[str, Any]]) -> float:
    """R4: +0.5 unless this exact (tool, args) call already appears earlier in the prefix."""
    try:
        parsed = parse_model_output(raw_text)
    except ToolCallParseError:
        return 0.0
    if isinstance(parsed, ParsedFinalAnswer):
        return R4_NO_DUPLICATE
    candidate = (parsed.name, json.dumps(parsed.args, sort_keys=True))
    if candidate in _prior_tool_calls(prefix_history):
        return 0.0
    return R4_NO_DUPLICATE
