"""Tests for the replay tab's trajectory loading/rendering logic (no real replays required)."""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
from space.replay_tab import build_replay_tab, format_trajectory, load_curated_trajectories

_PAIR = {
    "task_id": "t1-001",
    "user_prompt": "Geocode Paris.",
    "sft": {
        "status": "final_answer",
        "tool_calls": [{"tool_name": "geocode_city", "args": {"city": "Paris"}, "ok": True}],
        "final_answer": "Paris is at 48.8N.",
    },
    "grpo": {
        "status": "final_answer",
        "tool_calls": [{"tool_name": "geocode_city", "args": {"city": "Paris"}, "ok": True}],
        "final_answer": "Paris is at 48.8N, 2.35E.",
    },
}


def test_load_curated_trajectories_reads_all_json_files(tmp_path: Path) -> None:
    (tmp_path / "t1-001.json").write_text(json.dumps(_PAIR))
    (tmp_path / "t1-002.json").write_text(json.dumps({**_PAIR, "task_id": "t1-002"}))

    loaded = load_curated_trajectories(tmp_path)

    assert {t["task_id"] for t in loaded} == {"t1-001", "t1-002"}


def test_load_curated_trajectories_missing_dir_returns_empty_list(tmp_path: Path) -> None:
    assert load_curated_trajectories(tmp_path / "does-not-exist") == []


def test_format_trajectory_includes_status_calls_and_answer() -> None:
    text = format_trajectory(_PAIR["sft"])

    assert "final_answer" in text
    assert "geocode_city" in text
    assert "Paris is at 48.8N." in text


def test_build_replay_tab_with_no_curated_trajectories_shows_placeholder(tmp_path: Path) -> None:
    with gr.Blocks():
        build_replay_tab(replays_dir=tmp_path / "empty")
    # must not raise even with zero curated trajectories available


def test_build_replay_tab_with_trajectories_builds_without_error(tmp_path: Path) -> None:
    (tmp_path / "t1-001.json").write_text(json.dumps(_PAIR))

    with gr.Blocks():
        build_replay_tab(replays_dir=tmp_path)
    # must not raise when curated trajectories are present
