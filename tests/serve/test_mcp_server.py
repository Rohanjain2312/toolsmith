"""Tests for the MCP server's handler logic (StubModel; no real MCP transport involved)."""

from __future__ import annotations

from pathlib import Path

import pytest

from toolsmith.env.model import StubModel
from toolsmith.serve.mcp_server import FastMCP, build_trajectory_summary, mcp, run_toolsmith_task


def test_build_trajectory_summary_final_answer(tmp_path: Path) -> None:
    model = StubModel(["The capital of France is Paris."])

    summary = build_trajectory_summary(
        "t-mcp-1", "What is the capital of France?", model, trajectory_dir=tmp_path
    )

    assert summary["final_answer"] == "The capital of France is Paris."
    assert summary["status"] == "final_answer"
    assert summary["turns"] == 0
    assert summary["tool_calls"] == []


def test_build_trajectory_summary_includes_tool_calls(tmp_path: Path) -> None:
    tool_call_text = '{"tool": "geocode_city", "args": {"city": "Paris"}}'
    model = StubModel([tool_call_text, "Paris is at 48.8N."])

    summary = build_trajectory_summary("t-mcp-2", "Geocode Paris.", model, trajectory_dir=tmp_path)

    expected_call = {"tool_name": "geocode_city", "args": {"city": "Paris"}, "ok": True}
    assert summary["tool_calls"] == [expected_call]
    assert summary["final_answer"] == "Paris is at 48.8N."


def test_build_trajectory_summary_is_json_serializable(tmp_path: Path) -> None:
    import json

    model = StubModel(["done"])

    summary = build_trajectory_summary("t-mcp-3", "hi", model, trajectory_dir=tmp_path)

    json.dumps(summary)  # must not raise


def test_mcp_server_is_a_fastmcp_instance() -> None:
    assert isinstance(mcp, FastMCP)


def test_run_toolsmith_task_has_docstring_used_as_mcp_description() -> None:
    # @mcp.tool leaves the function directly callable; its docstring is what FastMCP surfaces
    # as the tool description to MCP clients (see fastmcp usage note in the module docstring).
    assert run_toolsmith_task.__doc__ and "travel-ops task" in run_toolsmith_task.__doc__


def test_run_toolsmith_task_requires_gguf_path_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TOOLSMITH_GGUF_PATH", raising=False)

    with pytest.raises(RuntimeError, match="TOOLSMITH_GGUF_PATH"):
        run_toolsmith_task("What's the weather in Paris?")
