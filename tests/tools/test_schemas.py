"""Tests for the ToolSpec base and tool registry."""

import pytest
from pydantic import BaseModel

from toolsmith.tools.schemas import (
    ToolAlreadyRegisteredError,
    ToolNotFoundError,
    ToolRegistry,
    ToolSpec,
)


class DummyArgs(BaseModel):
    value: int


class DummyResult(BaseModel):
    doubled: int


def _dummy_sandbox_fn(args: DummyArgs) -> DummyResult:
    return DummyResult(doubled=args.value * 2)


def _make_spec(name: str = "dummy_tool") -> ToolSpec:
    return ToolSpec(
        name=name,
        description="Doubles a number.",
        args_model=DummyArgs,
        returns_model=DummyResult,
        sandbox_fn=_dummy_sandbox_fn,
    )


def test_register_and_get() -> None:
    registry = ToolRegistry()
    spec = _make_spec()
    registry.register(spec)
    assert registry.get("dummy_tool") is spec


def test_register_duplicate_raises() -> None:
    registry = ToolRegistry()
    registry.register(_make_spec())
    with pytest.raises(ToolAlreadyRegisteredError):
        registry.register(_make_spec())


def test_get_unknown_raises() -> None:
    registry = ToolRegistry()
    with pytest.raises(ToolNotFoundError):
        registry.get("nonexistent_tool")


def test_list_returns_all_registered() -> None:
    registry = ToolRegistry()
    registry.register(_make_spec("tool_a"))
    registry.register(_make_spec("tool_b"))
    names = {spec.name for spec in registry.list()}
    assert names == {"tool_a", "tool_b"}


def test_list_empty_registry() -> None:
    registry = ToolRegistry()
    assert registry.list() == []


def test_spec_invokes_sandbox_fn() -> None:
    spec = _make_spec()
    result = spec.sandbox_fn(DummyArgs(value=21))
    assert result == DummyResult(doubled=42)
