"""FastMCP server exposing one tool: run a ToolSmith task, return its trajectory summary."""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from toolsmith.env.model import Model
from toolsmith.env.runner import DEFAULT_TRAJECTORY_DIR, run_episode

mcp = FastMCP("ToolSmith")


def build_trajectory_summary(
    task_id: str, prompt: str, model: Model, trajectory_dir: Path = DEFAULT_TRAJECTORY_DIR
) -> dict[str, Any]:
    """Run one episode and return a JSON-serializable trajectory summary.

    Separated from the @mcp.tool-decorated handler below so it's directly unit-testable with a
    StubModel, without needing a running MCP server or a real local GGUF file.
    """
    state = run_episode(task_id, prompt, model, trajectory_dir=trajectory_dir)
    return {
        "final_answer": state.final_answer,
        "tool_calls": [
            {"tool_name": entry.tool_name, "args": entry.args, "ok": entry.ok}
            for entry in state.tool_calls
        ],
        "turns": state.turn,
        "status": state.status.value,
    }


def _default_model() -> Model:
    """Build the live model adapter: llama.cpp over a local GGUF named by TOOLSMITH_GGUF_PATH."""
    from toolsmith.env.llamacpp_model import LlamaCppModel

    model_path = os.environ.get("TOOLSMITH_GGUF_PATH")
    if not model_path:
        raise RuntimeError("TOOLSMITH_GGUF_PATH is not set — point it at a local GGUF file")
    return LlamaCppModel(model_path=model_path)


@mcp.tool
def run_toolsmith_task(prompt: str) -> dict:
    """Run a travel-ops task through the ToolSmith agent and return the trajectory summary."""
    task_id = f"mcp-{uuid.uuid4().hex[:12]}"
    return build_trajectory_summary(task_id, prompt, _default_model())


if __name__ == "__main__":
    mcp.run()  # defaults to stdio transport — matches the Claude Desktop config in the README
