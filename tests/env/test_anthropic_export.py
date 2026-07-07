"""Tests for the Anthropic tool-use exporter."""

from toolsmith.env.anthropic_export import (
    export_all_anthropic_tools,
    export_anthropic_tools,
)
from toolsmith.tools.sandbox.currency_convert import CurrencyConvertArgs


def test_exports_exactly_twelve_tools() -> None:
    tools = export_all_anthropic_tools()
    assert len(tools) == 12


def test_each_entry_has_anthropic_shaped_keys() -> None:
    tools = export_all_anthropic_tools()
    for tool in tools:
        assert set(tool.keys()) == {"name", "description", "input_schema"}
        assert "type" not in tool
        assert "function" not in tool


def test_currency_convert_input_schema_has_expected_fields() -> None:
    tools = export_all_anthropic_tools()
    currency_tool = next(tool for tool in tools if tool["name"] == "currency_convert")

    expected_fields = set(CurrencyConvertArgs.model_fields.keys())
    assert expected_fields <= set(currency_tool["input_schema"]["properties"].keys())
    assert {"amount", "from_currency", "to_currency"} <= set(
        currency_tool["input_schema"]["properties"].keys()
    )


def test_export_anthropic_tools_with_empty_list_returns_empty_list() -> None:
    assert export_anthropic_tools([]) == []


def test_exported_tool_names_are_unique() -> None:
    tools = export_all_anthropic_tools()
    names = [tool["name"] for tool in tools]
    assert len(names) == len(set(names))
