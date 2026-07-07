"""Tests for tool-result truncation."""

from __future__ import annotations

from toolsmith.env.truncation import (
    CHARS_PER_TOKEN_APPROX,
    MAX_RESULT_TOKENS,
    truncate_result_text,
    truncate_tool_result,
)


def test_short_text_passes_through_unchanged() -> None:
    text = "short result"
    assert truncate_result_text(text) == text


def test_oversized_text_is_truncated_with_marker() -> None:
    text = "x" * (MAX_RESULT_TOKENS * CHARS_PER_TOKEN_APPROX + 1000)
    result = truncate_result_text(text)
    assert len(result) < len(text)
    assert result.endswith("...[truncated]")


def test_boundary_exact_budget_not_truncated() -> None:
    text = "x" * (MAX_RESULT_TOKENS * CHARS_PER_TOKEN_APPROX)
    assert truncate_result_text(text) == text


def test_truncate_tool_result_small_dict_unchanged() -> None:
    result = {"distance_km": 123.4}
    assert truncate_tool_result(result) == result
    assert "truncated" not in truncate_tool_result(result)


def test_truncate_tool_result_oversized_dict_is_wrapped() -> None:
    result = {
        "flights": [
            {"id": f"FL{i}", "price": 100.0, "currency": "USD"} for i in range(500)
        ]
    }
    truncated = truncate_tool_result(result)
    assert truncated["truncated"] is True
    assert "text" in truncated
    assert truncated["text"].endswith("...[truncated]")


def test_determinism() -> None:
    text = "x" * (MAX_RESULT_TOKENS * CHARS_PER_TOKEN_APPROX + 50)
    assert truncate_result_text(text) == truncate_result_text(text)

    result = {"flights": [{"id": f"FL{i}"} for i in range(500)]}
    assert truncate_tool_result(result) == truncate_tool_result(result)
