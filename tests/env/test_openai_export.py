"""Tests for the OpenAI function-calling exporter."""

from __future__ import annotations

from toolsmith.env.openai_export import export_all_openai_tools, export_openai_tools
from toolsmith.tools.sandbox.geocode_city import GeocodeCityArgs


def test_export_all_openai_tools_returns_twelve_entries() -> None:
    tools = export_all_openai_tools()
    assert len(tools) == 12


def test_each_entry_has_expected_shape() -> None:
    tools = export_all_openai_tools()
    for entry in tools:
        assert entry["type"] == "function"
        function = entry["function"]
        assert "name" in function
        assert "description" in function
        assert "parameters" in function


def test_geocode_city_parameters_contain_expected_field() -> None:
    tools = export_all_openai_tools()
    geocode_entry = next(t for t in tools if t["function"]["name"] == "geocode_city")
    parameters = geocode_entry["function"]["parameters"]
    expected_fields = set(GeocodeCityArgs.model_json_schema()["properties"])
    assert expected_fields <= set(parameters["properties"])
    assert "city" in parameters["properties"]


def test_export_openai_tools_with_empty_list_returns_empty_list() -> None:
    assert export_openai_tools([]) == []


def test_exported_names_are_unique() -> None:
    tools = export_all_openai_tools()
    names = [t["function"]["name"] for t in tools]
    assert len(names) == len(set(names))
