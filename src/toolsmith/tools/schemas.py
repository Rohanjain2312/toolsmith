"""ToolSpec base definition and the closed tool registry."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel


class ToolAlreadyRegisteredError(ValueError):
    """Raised when a tool name is registered twice."""


class ToolNotFoundError(KeyError):
    """Raised when looking up a tool name that isn't registered."""


@dataclass(frozen=True)
class ToolSpec:
    """Immutable definition of one tool: name, docs, arg/result schemas, and implementations."""

    name: str
    description: str
    args_model: type[BaseModel]
    returns_model: type[BaseModel]
    sandbox_fn: Callable[[BaseModel], BaseModel]
    real_fn: Callable[[BaseModel], BaseModel] | None = None


class ToolRegistry:
    """In-memory registry of ToolSpecs, keyed by name."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        """Register a tool spec, raising if its name is already taken."""
        if spec.name in self._tools:
            raise ToolAlreadyRegisteredError(f"tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        """Look up a tool spec by name, raising if it isn't registered."""
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolNotFoundError(name) from exc

    def list(self) -> list[ToolSpec]:
        """Return all registered tool specs."""
        return list(self._tools.values())


registry = ToolRegistry()
