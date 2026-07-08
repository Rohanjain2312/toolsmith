"""Tests for the live tab's task-running logic (StubModel; no real llama.cpp/GGUF needed)."""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import pytest
from space.live_tab import build_live_tab, run_live_task

from space import live_tab
from toolsmith.env.model import StubModel


def test_run_live_task_yields_progressively_then_final(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tool_call_text = '{"tool": "geocode_city", "args": {"city": "Paris"}}'
    model = StubModel([tool_call_text, "Paris is at 48.8N."])
    monkeypatch.setattr(live_tab, "_build_model", lambda: model)

    updates = list(run_live_task("Geocode Paris.", real_mode=False, trajectory_dir=tmp_path))

    assert len(updates) == 2  # one per tool call revealed, then the final state
    assert updates[0]["status"] == "in_progress"
    expected_call = {"tool_name": "geocode_city", "args": {"city": "Paris"}, "ok": True}
    assert updates[0]["tool_calls"] == [expected_call]
    assert updates[-1]["status"] == "final_answer"
    assert updates[-1]["final_answer"] == "Paris is at 48.8N."


def test_run_live_task_final_answer_only_yields_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = StubModel(["No tool needed, the answer is 42."])
    monkeypatch.setattr(live_tab, "_build_model", lambda: model)

    updates = list(run_live_task("What is 6*7?", real_mode=False, trajectory_dir=tmp_path))

    assert len(updates) == 1
    assert updates[0]["tool_calls"] == []
    assert updates[0]["final_answer"] == "No tool needed, the answer is 42."


def test_run_live_task_sandbox_mode_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(live_tab, "_build_model", lambda: StubModel(["ok"]))
    # Establishes monkeypatch's ownership of this env var *before* run_live_task's own raw
    # os.environ[...] = ... write, so pytest restores the true original at teardown either
    # way — otherwise that write leaks TOOLSMITH_MODE into every later test in the session.
    monkeypatch.delenv("TOOLSMITH_MODE", raising=False)

    list(run_live_task("hi", real_mode=False, trajectory_dir=tmp_path))

    assert os.environ["TOOLSMITH_MODE"] == "sandbox"


def test_run_live_task_real_mode_sets_env_var(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(live_tab, "_build_model", lambda: StubModel(["ok"]))
    monkeypatch.delenv("TOOLSMITH_MODE", raising=False)  # see comment above

    list(run_live_task("hi", real_mode=True, trajectory_dir=tmp_path))

    assert os.environ["TOOLSMITH_MODE"] == "real"


def test_build_live_tab_builds_without_error() -> None:
    with gr.Blocks():
        build_live_tab()
