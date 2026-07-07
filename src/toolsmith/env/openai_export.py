"""Export registered ToolSpecs as OpenAI function-calling tool definitions."""

from __future__ import annotations

from toolsmith.tools.schemas import ToolSpec, registry


def export_openai_tools(specs: list[ToolSpec]) -> list[dict]:
    """Convert ToolSpecs into OpenAI's tool-calling JSON format."""
    exported = []
    for spec in specs:
        schema = dict(spec.args_model.model_json_schema())
        schema.pop("title", None)
        exported.append(
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": schema,
                },
            }
        )
    return exported


def export_all_openai_tools() -> list[dict]:
    """Export all registered sandbox tools as OpenAI tool-calling definitions."""
    import toolsmith.tools.sandbox  # noqa: F401  (triggers tool registration)

    return export_openai_tools(registry.list())
