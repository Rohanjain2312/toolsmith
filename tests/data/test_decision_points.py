"""Tests for the decision-point extractor, on hand-built fixture trajectories."""

from __future__ import annotations

from pathlib import Path

from toolsmith.data.decision_points import (
    DecisionPoint,
    extract_decision_points,
    load_trajectories,
    select_task_subset,
    write_decision_points,
)
from toolsmith.data.taskspec import TaskSpec, ToolWasCalledWithCondition
from toolsmith.env.state import EpisodeState, EpisodeStatus, ToolCallLogEntry


def _spec(task_id: str, tier: str) -> TaskSpec:
    return TaskSpec(
        id=task_id,
        tier=tier,
        user_prompt="x",
        goal_spec=[ToolWasCalledWithCondition(tool_name="geocode_city", args={"city": "Paris"})],
        min_steps=1,
        split="train",
    )


def _trajectory(task_id: str) -> EpisodeState:
    return EpisodeState(
        task_id=task_id,
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "geocode paris"},
            {"role": "assistant", "content": '{"tool": "geocode_city", "args": {"city": "Paris"}}'},
            {"role": "tool", "tool_name": "geocode_city", "content": "{}"},
            {"role": "assistant", "content": "Paris is at 48.8N."},
        ],
        tool_calls=[
            ToolCallLogEntry(
                turn=0, tool_name="geocode_city", args={"city": "Paris"}, ok=True, result={}
            )
        ],
        turn=2,
        status=EpisodeStatus.FINAL_ANSWER,
        final_answer="Paris is at 48.8N.",
    )


def test_load_trajectories_round_trip(tmp_path: Path) -> None:
    states = [_trajectory("t1-001"), _trajectory("t1-002")]
    path = tmp_path / "trajectories.jsonl"
    path.write_text("\n".join(s.to_json() for s in states) + "\n")

    loaded = load_trajectories(path)

    assert [s.task_id for s in loaded] == ["t1-001", "t1-002"]


def test_extract_decision_points_one_per_assistant_turn() -> None:
    state = _trajectory("t1-001")

    points = extract_decision_points([state])

    assert len(points) == 2
    assert all(isinstance(p, DecisionPoint) for p in points)
    assert points[0].task_id == "t1-001"
    # first decision point: right before the tool-call turn -> prefix is just system+user
    assert [m["role"] for m in points[0].prefix] == ["system", "user"]
    # second decision point: right before the final-answer turn -> prefix includes the tool turn
    assert [m["role"] for m in points[1].prefix] == ["system", "user", "assistant", "tool"]


def test_extract_decision_points_across_multiple_states() -> None:
    states = [_trajectory("t1-001"), _trajectory("t1-002")]

    points = extract_decision_points(states)

    assert len(points) == 4
    assert {p.task_id for p in points} == {"t1-001", "t1-002"}


def test_write_decision_points_round_trip(tmp_path: Path) -> None:
    points = extract_decision_points([_trajectory("t1-001")])
    path = tmp_path / "points.jsonl"

    write_decision_points(points, path)
    lines = path.read_text().splitlines()

    assert len(lines) == len(points)


def test_select_task_subset_full_set_returns_everything() -> None:
    specs = [_spec(f"t-{i}", "T1") for i in range(10)]

    subset = select_task_subset(specs, count=3, full_set=True)

    assert len(subset) == 10


def test_select_task_subset_returns_all_when_fewer_than_count() -> None:
    specs = [_spec(f"t-{i}", "T1") for i in range(5)]

    subset = select_task_subset(specs, count=700)

    assert len(subset) == 5


def test_select_task_subset_weights_toward_t1_t2() -> None:
    specs = (
        [_spec(f"t1-{i}", "T1") for i in range(200)]
        + [_spec(f"t2-{i}", "T2") for i in range(200)]
        + [_spec(f"t3-{i}", "T3") for i in range(200)]
        + [_spec(f"t4-{i}", "T4") for i in range(200)]
    )

    subset = select_task_subset(specs, count=100)
    tiers = [s.tier for s in subset]
    t1_t2_count = tiers.count("T1") + tiers.count("T2")
    t3_t4_count = tiers.count("T3") + tiers.count("T4")

    assert len(subset) <= 100
    assert t1_t2_count > t3_t4_count


def test_select_task_subset_is_deterministic() -> None:
    specs = [_spec(f"t-{i}", "T1") for i in range(200)]

    first = select_task_subset(specs, count=50)
    second = select_task_subset(specs, count=50)

    assert [s.id for s in first] == [s.id for s in second]
