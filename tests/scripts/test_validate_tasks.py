"""Tests for validate_tasks.py: solver-based validation, JSONL round trip, and splitting."""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.validate_tasks import (
    TaskValidationError,
    load_tasks,
    stratified_split,
    validate_all,
    validate_task,
    write_jsonl,
)

from toolsmith.data.taskspec import TaskSpec, ToolWasCalledWithCondition


def _solvable_spec(task_id: str) -> TaskSpec:
    return TaskSpec(
        id=task_id,
        tier="T1",
        user_prompt="Geocode Paris.",
        goal_spec=[ToolWasCalledWithCondition(tool_name="geocode_city", args={"city": "Paris"})],
        min_steps=0,
        split="train",
    )


def _unsolvable_spec(task_id: str) -> TaskSpec:
    return TaskSpec(
        id=task_id,
        tier="T1",
        user_prompt="Geocode a place that doesn't exist.",
        goal_spec=[
            ToolWasCalledWithCondition(tool_name="geocode_city", args={"city": "Nowhereville"})
        ],
        min_steps=0,
        split="train",
    )


def test_validate_task_recomputes_min_steps() -> None:
    updated = validate_task(_solvable_spec("t1-001"))

    assert updated.min_steps == 1


def test_validate_task_raises_for_unsolvable_task() -> None:
    with pytest.raises(TaskValidationError):
        validate_task(_unsolvable_spec("t1-002"))


def test_validate_all_separates_accepted_and_rejected() -> None:
    specs = [_solvable_spec("t1-003"), _unsolvable_spec("t1-004")]

    accepted, rejections = validate_all(specs)

    assert [s.id for s in accepted] == ["t1-003"]
    assert len(rejections) == 1
    assert "t1-004" in rejections[0]


def test_write_and_load_jsonl_round_trip(tmp_path: Path) -> None:
    specs = [_solvable_spec("t1-005"), _solvable_spec("t1-006")]
    path = tmp_path / "tasks.jsonl"

    write_jsonl(specs, path)
    loaded = load_tasks(path)

    assert [s.id for s in loaded] == ["t1-005", "t1-006"]


def test_load_tasks_skips_malformed_lines(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    valid_line = _solvable_spec("t1-007").model_dump_json()
    path.write_text('{"not": "a valid task spec"}\n' + valid_line + "\n")

    loaded = load_tasks(path)

    assert [s.id for s in loaded] == ["t1-007"]


def test_stratified_split_respects_proportions_within_a_tier() -> None:
    specs = [_solvable_spec(f"t1-{i:03d}") for i in range(100)]

    split_specs = stratified_split(specs)
    counts: dict[str, int] = {}
    for spec in split_specs:
        counts[spec.split] = counts.get(spec.split, 0) + 1

    assert len(split_specs) == 100
    assert counts["train"] > counts["val"] > 0
    assert counts["val"] == counts["test"]


def test_stratified_split_is_deterministic() -> None:
    specs = [_solvable_spec(f"t1-{i:03d}") for i in range(50)]

    first = stratified_split(specs)
    second = stratified_split(specs)

    assert [(s.id, s.split) for s in first] == [(s.id, s.split) for s in second]
