"""Export registered ToolSpecs into Anthropic tool-use JSON format."""

from __future__ import annotations

from typing import Any

from toolsmith.tools.schemas import ToolSpec, registry


def export_anthropic_tools(specs: list[ToolSpec]) -> list[dict[str, Any]]:
    """Convert a list of ToolSpecs into Anthropic tool-use tool definitions."""
    tools: list[dict[str, Any]] = []
    for spec in specs:
        input_schema = dict(spec.args_model.model_json_schema())
        input_schema.pop("title", None)
        tools.append(
            {
                "name": spec.name,
                "description": spec.description,
                "input_schema": input_schema,
            }
        )
    return tools


def export_all_anthropic_tools() -> list[dict[str, Any]]:
    """Trigger sandbox tool registration and export every registered tool."""
    import toolsmith.tools.sandbox  # noqa: F401 (import triggers registration)

    return export_anthropic_tools(registry.list())
