"""Tests for the tool call executor."""

from __future__ import annotations

import toolsmith.tools.sandbox  # noqa: F401  (registers all 12 sandbox tools)
from toolsmith.env.executor import ExecutionResult, execute_tool_call


def test_successful_call_returns_ok_result() -> None:
    result = execute_tool_call(
        "distance_calc", {"lat1": 0, "lon1": 0, "lat2": 0, "lon2": 0}
    )

    assert result.ok is True
    assert result.error is None
    assert result.result is not None
    assert result.result["distance_km"] == 0.0


def test_unknown_tool_returns_error_without_raising() -> None:
    result = execute_tool_call("not_a_real_tool", {})

    assert result.ok is False
    assert result.result is None
    assert result.error is not None
    assert "not_a_real_tool" in result.error


def test_invalid_args_returns_error_without_raising() -> None:
    result = execute_tool_call("geocode_city", {"city": ""})

    assert result.ok is False
    assert result.result is None
    assert result.error is not None
    assert "geocode_city" in result.error


def test_domain_error_returns_error_without_raising() -> None:
    result = execute_tool_call("geocode_city", {"city": "Nonexistent Place"})

    assert result.ok is False
    assert result.result is None
    assert result.error is not None
    assert "Nonexistent Place" in result.error


def test_deterministic_repeat_calls_are_equal() -> None:
    args = {"lat1": 10.0, "lon1": 20.0, "lat2": 30.0, "lon2": 40.0}

    first = execute_tool_call("distance_calc", args)
    second = execute_tool_call("distance_calc", args)

    assert first == second
    assert isinstance(first, ExecutionResult)
