"""Abstract model interface implemented by all LLM adapters (stub, llama.cpp, vLLM, API-based)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class StubModelExhaustedError(RuntimeError):
    """Raised when StubModel.generate is called more times than it has scripted responses."""


class Model(ABC):
    """Abstract base class for any component that turns conversation history into raw text."""

    @abstractmethod
    def generate(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
        """Generate the next raw text turn given conversation history and available tool defs."""


class StubModel(Model):
    """Deterministic test double that replays a fixed, pre-scripted sequence of responses."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._call_count = 0

    @property
    def call_count(self) -> int:
        """Number of times generate() has been called so far."""
        return self._call_count

    def generate(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
        """Ignore messages/tools and return the next scripted response, in order."""
        if self._call_count >= len(self._responses):
            raise StubModelExhaustedError(
                f"StubModel exhausted its {len(self._responses)} scripted responses"
            )
        response = self._responses[self._call_count]
        self._call_count += 1
        return response
