"""Tests for collect_trajectories.py's pure logic (StubModel, no live GGUF load).

The round-trip test below is the direct regression test for BUGFIX-T02: before this script
existed, nothing in the repo produced the EpisodeState.to_json() JSONL format that
data/decision_points.py, scripts/audit_trajectories.py, and scripts/select_curated_trajectories.py
already read by default -- this proves that contract now actually holds.
"""

from __future__ import annotations

from pathlib import Path

from scripts.collect_trajectories import collect_trajectories, load_tasks, write_trajectories

from toolsmith.data.decision_points import load_trajectories
from toolsmith.data.taskspec import TaskSpec, ToolWasCalledWithCondition
from toolsmith.env.model import StubModel


def _spec(task_id: str, split: str = "train") -> TaskSpec:
    return TaskSpec(
        id=task_id,
        tier="T1",
        user_prompt="Geocode Paris.",
        goal_spec=[ToolWasCalledWithCondition(tool_name="geocode_city", args={"city": "Paris"})],
        min_steps=1,
        split=split,
    )


def test_load_tasks_reads_all_specs_regardless_of_split(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    specs = [_spec("t1-001", "train"), _spec("t1-002", "val")]
    path.write_text("\n".join(s.model_dump_json() for s in specs) + "\n")

    loaded = load_tasks(path)

    assert [s.id for s in loaded] == ["t1-001", "t1-002"]


def test_collect_trajectories_returns_full_episode_states(tmp_path: Path) -> None:
    spec = _spec("t1-003")
    responses = ['{"tool": "geocode_city", "args": {"city": "Paris"}}', "Paris is at 48.8N."]
    model = StubModel(responses)

    states = collect_trajectories([spec], model, trajectory_dir=tmp_path)

    assert len(states) == 1
    assert states[0].task_id == "t1-003"
    assert states[0].final_answer == "Paris is at 48.8N."


def test_write_trajectories_round_trips_through_decision_points_loader(tmp_path: Path) -> None:
    spec = _spec("t1-004")
    responses = ['{"tool": "geocode_city", "args": {"city": "Paris"}}', "Paris is at 48.8N."]
    model = StubModel(responses)
    states = collect_trajectories([spec], model, trajectory_dir=tmp_path)
    output_path = tmp_path / "sft_trajectories.jsonl"

    write_trajectories(states, output_path)
    loaded = load_trajectories(output_path)

    assert [s.task_id for s in loaded] == ["t1-004"]
    assert loaded[0].final_answer == "Paris is at 48.8N."
    assert loaded[0].tool_calls[0].tool_name == "geocode_city"


def test_write_trajectories_empty_list(tmp_path: Path) -> None:
    output_path = tmp_path / "empty.jsonl"

    write_trajectories([], output_path)

    assert output_path.read_text() == ""
