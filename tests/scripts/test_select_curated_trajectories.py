"""Tests for the curated SFT-vs-GRPO trajectory pair selector, on fixture trajectories."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.select_curated_trajectories import (
    load_trajectories,
    select_curated_pairs,
    summarize,
    write_curated_pairs,
)

from toolsmith.env.state import EpisodeState, EpisodeStatus, ToolCallLogEntry


def _state(task_id: str, final_answer: str = "done") -> EpisodeState:
    return EpisodeState(
        task_id=task_id,
        messages=[],
        tool_calls=[
            ToolCallLogEntry(turn=0, tool_name="geocode_city", args={}, ok=True, result={})
        ],
        turn=1,
        status=EpisodeStatus.FINAL_ANSWER,
        final_answer=final_answer,
    )


def test_load_trajectories_keys_by_task_id(tmp_path: Path) -> None:
    states = [_state("t1"), _state("t2")]
    path = tmp_path / "trajectories.jsonl"
    path.write_text("\n".join(s.to_json() for s in states) + "\n")

    loaded = load_trajectories(path)

    assert set(loaded.keys()) == {"t1", "t2"}


def test_summarize_shape() -> None:
    summary = summarize(_state("t1", final_answer="Paris is sunny."))

    assert summary == {
        "status": "final_answer",
        "tool_calls": [{"tool_name": "geocode_city", "args": {}, "ok": True}],
        "final_answer": "Paris is sunny.",
    }


def test_select_curated_pairs_only_common_task_ids() -> None:
    sft_states = {"t1": _state("t1"), "t2": _state("t2")}
    grpo_states = {"t2": _state("t2"), "t3": _state("t3")}

    pairs = select_curated_pairs(sft_states, grpo_states, {}, count=20)

    assert [p["task_id"] for p in pairs] == ["t2"]


def test_select_curated_pairs_caps_at_count() -> None:
    sft_states = {f"t{i}": _state(f"t{i}") for i in range(30)}
    grpo_states = {f"t{i}": _state(f"t{i}") for i in range(30)}

    pairs = select_curated_pairs(sft_states, grpo_states, {}, count=20)

    assert len(pairs) == 20


def test_select_curated_pairs_is_deterministic() -> None:
    sft_states = {f"t{i}": _state(f"t{i}") for i in range(30)}
    grpo_states = {f"t{i}": _state(f"t{i}") for i in range(30)}

    first = select_curated_pairs(sft_states, grpo_states, {}, count=20)
    second = select_curated_pairs(sft_states, grpo_states, {}, count=20)

    assert [p["task_id"] for p in first] == [p["task_id"] for p in second]


def test_select_curated_pairs_includes_user_prompt() -> None:
    sft_states = {"t1": _state("t1")}
    grpo_states = {"t1": _state("t1")}

    pairs = select_curated_pairs(sft_states, grpo_states, {"t1": "Geocode Paris."}, count=20)

    assert pairs[0]["user_prompt"] == "Geocode Paris."


def test_write_curated_pairs_one_file_per_task(tmp_path: Path) -> None:
    pairs = [
        {"task_id": "t1", "user_prompt": "x", "sft": {}, "grpo": {}},
        {"task_id": "t2", "user_prompt": "y", "sft": {}, "grpo": {}},
    ]

    write_curated_pairs(pairs, tmp_path)

    assert json.loads((tmp_path / "t1.json").read_text())["task_id"] == "t1"
    assert json.loads((tmp_path / "t2.json").read_text())["task_id"] == "t2"
