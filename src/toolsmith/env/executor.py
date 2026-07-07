"""Dispatch a parsed tool call to the registry and run it, trapping all failure modes."""

from __future__ import annotations

import os
from dataclasses import dataclass

from pydantic import ValidationError

from toolsmith.tools.schemas import ToolNotFoundError, registry


@dataclass(frozen=True)
class ExecutionResult:
    """Outcome of executing a single tool call."""

    ok: bool
    result: dict | None = None
    error: str | None = None


def execute_tool_call(name: str, args: dict) -> ExecutionResult:
    """Look up, validate, and run a tool call, converting failures into an ExecutionResult."""
    try:
        spec = registry.get(name)
    except ToolNotFoundError:
        return ExecutionResult(ok=False, error=f"unknown tool: {name}")

    try:
        validated_args = spec.args_model.model_validate(args)
    except ValidationError as exc:
        return ExecutionResult(ok=False, error=f"invalid args for {name}: {exc}")

    mode = os.environ.get("TOOLSMITH_MODE", "sandbox")
    fn = spec.real_fn if mode == "real" and spec.real_fn is not None else spec.sandbox_fn

    try:
        result_model = fn(validated_args)
    except ValueError as exc:
        return ExecutionResult(ok=False, error=str(exc))

    return ExecutionResult(ok=True, result=result_model.model_dump(mode="json"))
