"""Tests for the tool-call/final-answer parser."""

from __future__ import annotations

import pytest

from toolsmith.env.parser import (
    ParsedFinalAnswer,
    ParsedToolCall,
    ToolCallParseError,
    parse_model_output,
)


def test_clean_json_tool_call_parses() -> None:
    text = '{"tool": "geocode_city", "args": {"city": "Paris"}}'
    result = parse_model_output(text)
    assert result == ParsedToolCall(name="geocode_city", args={"city": "Paris"})


def test_code_fenced_json_tool_call_parses() -> None:
    text = '```json\n{"tool": "weather_lookup", "args": {"lat": 1, "lon": 2}}\n```'
    result = parse_model_output(text)
    assert result == ParsedToolCall(name="weather_lookup", args={"lat": 1, "lon": 2})


def test_plain_code_fenced_json_tool_call_parses() -> None:
    text = '```\n{"tool": "currency_convert", "args": {"amount": 10}}\n```'
    result = parse_model_output(text)
    assert result == ParsedToolCall(name="currency_convert", args={"amount": 10})


def test_leading_prose_before_json_parses() -> None:
    text = 'Let me check that.\n{"tool": "timezone_info", "args": {"city": "Rome"}}'
    result = parse_model_output(text)
    assert result == ParsedToolCall(name="timezone_info", args={"city": "Rome"})


def test_trailing_junk_after_json_parses() -> None:
    text = '{"tool": "distance_calc", "args": {"a": 1, "b": 2}}\nThanks!'
    result = parse_model_output(text)
    assert result == ParsedToolCall(name="distance_calc", args={"a": 1, "b": 2})


def test_multiple_json_objects_takes_first() -> None:
    text = (
        '{"tool": "flight_search", "args": {"origin": "DCA"}}\n'
        '{"tool": "poi_search", "args": {"city": "NYC"}}'
    )
    result = parse_model_output(text)
    assert result == ParsedToolCall(name="flight_search", args={"origin": "DCA"})


def test_malformed_truncated_json_raises_parse_error() -> None:
    text = '{"tool": "flight_search", "args": {'
    with pytest.raises(ToolCallParseError):
        parse_model_output(text)


def test_malformed_broken_syntax_raises_parse_error() -> None:
    text = '{"tool": "flight_search" "args": {}}'
    with pytest.raises(ToolCallParseError):
        parse_model_output(text)


def test_hallucinated_tool_name_parses_successfully() -> None:
    text = '{"tool": "make_coffee", "args": {"strength": "strong"}}'
    result = parse_model_output(text)
    assert result == ParsedToolCall(name="make_coffee", args={"strength": "strong"})


def test_plain_prose_with_no_json_is_final_answer() -> None:
    text = "The weather in Paris is sunny and 22 degrees."
    result = parse_model_output(text)
    assert result == ParsedFinalAnswer(text=text)


def test_prose_with_unrelated_braces_is_final_answer() -> None:
    text = "The cost is roughly {approx} 10 dollars, nothing fancy."
    result = parse_model_output(text)
    assert result == ParsedFinalAnswer(text=text)


def test_empty_string_is_final_answer() -> None:
    result = parse_model_output("")
    assert result == ParsedFinalAnswer(text="")


def test_whitespace_only_is_stripped_final_answer() -> None:
    result = parse_model_output("   \n\t  ")
    assert result == ParsedFinalAnswer(text="")


def test_tool_key_without_args_key_raises_parse_error() -> None:
    text = '{"tool": "geocode_city"}'
    with pytest.raises(ToolCallParseError):
        parse_model_output(text)


def test_final_answer_is_stripped_of_surrounding_whitespace() -> None:
    text = "  Here is the final answer.  \n"
    result = parse_model_output(text)
    assert result == ParsedFinalAnswer(text="Here is the final answer.")
