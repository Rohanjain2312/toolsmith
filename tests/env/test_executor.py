"""Tests for the tool call executor."""

from __future__ import annotations

import pytest

import toolsmith.tools.sandbox  # noqa: F401  (registers all 12 sandbox tools)
from toolsmith.env.executor import ExecutionResult, execute_tool_call
from toolsmith.tools.real import currency_convert as currency_convert_real_module
from toolsmith.tools.schemas import registry


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


def test_real_mode_dispatches_to_real_fn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TOOLSMITH_MODE", "real")
    monkeypatch.setattr(
        currency_convert_real_module,
        "_fetch_json",
        lambda url: {"amount": 1.0, "base": "USD", "date": "2026-09-01", "rates": {"EUR": 0.5}},
    )

    result = execute_tool_call(
        "currency_convert", {"amount": 100, "from_currency": "USD", "to_currency": "EUR"}
    )

    assert result.ok is True
    assert result.result is not None
    assert result.result["rate"] == 0.5  # from the mocked real API, not the sandbox FX table


def test_sandbox_mode_ignores_real_fn_even_when_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TOOLSMITH_MODE", raising=False)  # default: sandbox

    result = execute_tool_call(
        "currency_convert", {"amount": 100, "from_currency": "USD", "to_currency": "EUR"}
    )

    assert result.ok is True
    assert result.result is not None
    assert result.result["rate"] != 0.5  # sandbox's fixed FX table, not any real-mode rate


@pytest.mark.parametrize(
    "tool_name",
    [
        "weather_lookup",
        "geocode_city",
        "currency_convert",
        "country_info",
        "flight_search",
        "poi_search",
        "calendar_create_event",
    ],
)
def test_network_backed_tools_have_real_fn_wired(tool_name: str) -> None:
    assert registry.get(tool_name).real_fn is not None


@pytest.mark.parametrize(
    "tool_name",
    ["distance_calc", "packing_rules", "unit_convert", "datetime_math", "timezone_info"],
)
def test_pure_function_tools_have_no_real_fn(tool_name: str) -> None:
    # These are identical in both modes (pure computation / stdlib zoneinfo, no network) —
    # no separate tools/real implementation exists or is needed for them.
    assert registry.get(tool_name).real_fn is None
