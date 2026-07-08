"""llama.cpp adapter implementing the Model ABC, for local GGUF inference (Space live tab)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from toolsmith.env.model import Model

DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.0  # greedy: this adapter backs eval + the demo, not exploration


class LlamaCppModel(Model):
    """Model adapter over llama-cpp-python, loading a local GGUF file for CPU inference.

    `engine` accepts a pre-built chat-completion object (e.g. an already-constructed
    `llama_cpp.Llama`, or a test double) so callers/tests never need a real GGUF file on disk;
    when omitted, a real `llama_cpp.Llama` is lazily constructed from `model_path`.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        n_ctx: int = 2048,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        engine: Any | None = None,
    ) -> None:
        if engine is None:
            if model_path is None:
                raise ValueError("model_path is required unless `engine` is provided")
            from llama_cpp import Llama  # lazy import: real GGUF loading only happens here

            engine = Llama(model_path=str(model_path), n_ctx=n_ctx, verbose=False)
        self._engine = engine
        self._max_tokens = max_tokens
        self._temperature = temperature

    def generate(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
        """Generate the next raw text turn via the loaded GGUF model's chat completion.

        `tools` is unused, matching env/anthropic_model.py: the tool list is already embedded
        as plain text in the system prompt (see env/runner.py's build_system_prompt).
        """
        response = self._engine.create_chat_completion(
            messages=messages, max_tokens=self._max_tokens, temperature=self._temperature
        )
        return response["choices"][0]["message"]["content"] or ""
