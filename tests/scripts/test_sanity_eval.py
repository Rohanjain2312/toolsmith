"""Tests for sanity_eval.py's aggregation logic, driven with StubModel (no real model needed)."""

from __future__ import annotations

from pathlib import Path

from scripts.sanity_eval import load_val_tasks, run_sanity_eval

from toolsmith.data.taskspec import TaskSpec, ToolWasCalledWithCondition
from toolsmith.env.model import StubModel


def _spec(task_id: str, split: str = "val") -> TaskSpec:
    return TaskSpec(
        id=task_id,
        tier="T1",
        user_prompt="Geocode Paris.",
        goal_spec=[ToolWasCalledWithCondition(tool_name="geocode_city", args={"city": "Paris"})],
        min_steps=1,
        split=split,
    )


def test_load_val_tasks_filters_by_split(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    specs = [_spec("t1-001", "val"), _spec("t1-002", "train"), _spec("t1-003", "val")]
    path.write_text("\n".join(s.model_dump_json() for s in specs) + "\n")

    loaded = load_val_tasks(path)

    assert [s.id for s in loaded] == ["t1-001", "t1-003"]


def test_load_val_tasks_respects_count_limit(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    specs = [_spec(f"t1-{i:03d}") for i in range(5)]
    path.write_text("\n".join(s.model_dump_json() for s in specs) + "\n")

    loaded = load_val_tasks(path, count=2)

    assert len(loaded) == 2


def test_run_sanity_eval_all_correct(tmp_path: Path) -> None:
    specs = [_spec("t1-a"), _spec("t1-b")]
    model = StubModel(
        [
            '{"tool": "geocode_city", "args": {"city": "Paris"}}',
            "Paris is at 48.8N.",
            '{"tool": "geocode_city", "args": {"city": "Paris"}}',
            "Paris is at 48.8N.",
        ]
    )

    report = run_sanity_eval(specs, model, trajectory_dir=tmp_path)

    assert report["task_count"] == 2
    assert report["json_valid_pct"] == 100.0
    assert report["correct_tool_pct"] == 100.0


def test_run_sanity_eval_wrong_tool_lowers_correct_tool_pct(tmp_path: Path) -> None:
    specs = [_spec("t1-a"), _spec("t1-b")]
    model = StubModel(
        [
            '{"tool": "weather_lookup", "args": {"lat": 0, "lon": 0, "date": "2026-09-03"}}',
            "It's fine out.",
            '{"tool": "geocode_city", "args": {"city": "Paris"}}',
            "Paris is at 48.8N.",
        ]
    )

    report = run_sanity_eval(specs, model, trajectory_dir=tmp_path)

    assert report["correct_tool_pct"] == 50.0
    assert report["json_valid_pct"] == 100.0


def test_run_sanity_eval_parse_failure_lowers_json_valid_pct(tmp_path: Path) -> None:
    specs = [_spec("t1-a")]
    model = StubModel(['{"tool": "geocode_city", "args": '])

    report = run_sanity_eval(specs, model, trajectory_dir=tmp_path)

    assert report["json_valid_pct"] == 0.0


def test_run_sanity_eval_empty_task_list(tmp_path: Path) -> None:
    report = run_sanity_eval([], StubModel([]), trajectory_dir=tmp_path)

    assert report == {"task_count": 0, "json_valid_pct": 0.0, "correct_tool_pct": 0.0}
