"""Tests for prep_sft_data.py's pure row-conversion logic (no dataset downloads)."""

from __future__ import annotations

import json

from scripts.prep_sft_data import convert_glaive_row, convert_xlam_row, filter_and_cap


def test_convert_xlam_row_happy_path() -> None:
    row = {
        "query": "What's the weather in Paris tomorrow?",
        "answers": json.dumps([{"name": "weather_lookup", "arguments": {"lat": 48.8, "lon": 2.3}}]),
        "tools": json.dumps(
            [{"name": "weather_lookup", "description": "Look up weather.", "parameters": {}}]
        ),
    }

    converted = convert_xlam_row(row)

    assert converted is not None
    assert converted["source"] == "xlam"
    roles = [m["role"] for m in converted["messages"]]
    assert roles == ["system", "user", "assistant"]
    assistant_content = json.loads(converted["messages"][-1]["content"])
    assert assistant_content == {"tool": "weather_lookup", "args": {"lat": 48.8, "lon": 2.3}}


def test_convert_xlam_row_multiple_calls_produces_multiple_assistant_turns() -> None:
    row = {
        "query": "Geocode Paris then check its weather.",
        "answers": json.dumps(
            [
                {"name": "geocode_city", "arguments": {"city": "Paris"}},
                {"name": "weather_lookup", "arguments": {"lat": 48.8, "lon": 2.3}},
            ]
        ),
        "tools": json.dumps([]),
    }

    converted = convert_xlam_row(row)

    assert converted is not None
    assistant_turns = [m for m in converted["messages"] if m["role"] == "assistant"]
    assert len(assistant_turns) == 2


def test_convert_xlam_row_malformed_json_returns_none() -> None:
    row = {"query": "x", "answers": "not json", "tools": "[]"}

    assert convert_xlam_row(row) is None


def test_convert_xlam_row_missing_field_returns_none() -> None:
    row = {"query": "x", "answers": json.dumps([])}

    assert convert_xlam_row(row) is None


def test_convert_xlam_row_empty_answers_returns_none() -> None:
    row = {"query": "x", "answers": json.dumps([]), "tools": json.dumps([])}

    assert convert_xlam_row(row) is None


def test_convert_glaive_row_happy_path() -> None:
    row = {
        "system": "SYSTEM: You have access to functions.",
        "chat": (
            "USER: What's the weather in Paris? ASSISTANT: "
            '<functioncall> {"name": "weather_lookup", "arguments": "{}"} <|endoftext|>'
        ),
    }

    converted = convert_glaive_row(row)

    assert converted is not None
    assert converted["source"] == "glaive"
    roles = [m["role"] for m in converted["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert "functioncall" in converted["messages"][-1]["content"]


def test_convert_glaive_row_without_functioncall_returns_none() -> None:
    row = {"system": "SYSTEM: chat.", "chat": "USER: hi ASSISTANT: hello there"}

    assert convert_glaive_row(row) is None


def test_convert_glaive_row_empty_assistant_turn_returns_none() -> None:
    row = {
        "system": "SYSTEM: chat.",
        "chat": (
            'USER: hi ASSISTANT: <functioncall> {"name": "x"} <|endoftext|> '
            "USER: thanks ASSISTANT: "
        ),
    }

    assert convert_glaive_row(row) is None


def test_filter_and_cap_drops_none_and_caps_length() -> None:
    rows = [{"id": 1}, None, {"id": 2}, None, {"id": 3}]

    result = filter_and_cap(rows, cap=2)

    assert result == [{"id": 1}, {"id": 2}]


def test_filter_and_cap_empty_input() -> None:
    assert filter_and_cap([], cap=10) == []
