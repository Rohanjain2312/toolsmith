"""Tests for generate_gold_trajectories.py's pure logic (StubModel, no live Anthropic calls)."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.generate_gold_trajectories import (
    generate_gold_trajectory,
    load_train_tasks,
    write_gold_sft_rows,
)

from toolsmith.data.taskspec import TaskSpec, ToolWasCalledWithCondition
from toolsmith.env.model import StubModel


def _spec(task_id: str, split: str) -> TaskSpec:
    return TaskSpec(
        id=task_id,
        tier="T1",
        user_prompt="Geocode Paris.",
        goal_spec=[ToolWasCalledWithCondition(tool_name="geocode_city", args={"city": "Paris"})],
        min_steps=1,
        split=split,
    )


def test_load_train_tasks_filters_by_split(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    specs = [_spec("t1-001", "train"), _spec("t1-002", "val"), _spec("t1-003", "train")]
    path.write_text("\n".join(s.model_dump_json() for s in specs) + "\n")

    loaded = load_train_tasks(path)

    assert [s.id for s in loaded] == ["t1-001", "t1-003"]


def test_load_train_tasks_respects_count_limit(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    specs = [_spec(f"t1-{i:03d}", "train") for i in range(5)]
    path.write_text("\n".join(s.model_dump_json() for s in specs) + "\n")

    loaded = load_train_tasks(path, count=2)

    assert len(loaded) == 2


def test_generate_gold_trajectory_passes_goal_check(tmp_path: Path) -> None:
    spec = _spec("t1-004", "train")
    responses = ['{"tool": "geocode_city", "args": {"city": "Paris"}}', "Paris is at 48.8N."]
    model = StubModel(responses)

    state = generate_gold_trajectory(spec, model, trajectory_dir=tmp_path)

    assert state is not None
    assert state.final_answer == "Paris is at 48.8N."


def test_generate_gold_trajectory_fails_goal_check_when_wrong_tool_called(tmp_path: Path) -> None:
    spec = _spec("t1-005", "train")
    responses = ['{"tool": "geocode_city", "args": {"city": "Tokyo"}}', "Tokyo is nice."]
    model = StubModel(responses)

    state = generate_gold_trajectory(spec, model, trajectory_dir=tmp_path)

    assert state is None


def test_write_gold_sft_rows_round_trip(tmp_path: Path) -> None:
    spec = _spec("t1-006", "train")
    responses = ['{"tool": "geocode_city", "args": {"city": "Paris"}}', "Paris is at 48.8N."]
    model = StubModel(responses)
    state = generate_gold_trajectory(spec, model, trajectory_dir=tmp_path)
    assert state is not None
    output_path = tmp_path / "gold.jsonl"

    write_gold_sft_rows([state], output_path)

    lines = output_path.read_text().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["source"] == "gold"
    assert row["messages"] == state.messages


def test_write_gold_sft_rows_empty_list(tmp_path: Path) -> None:
    output_path = tmp_path / "gold.jsonl"

    write_gold_sft_rows([], output_path)

    assert output_path.read_text() == ""
