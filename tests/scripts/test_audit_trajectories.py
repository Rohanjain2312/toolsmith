"""Tests for the trajectory audit script's flagging heuristics, on fixture trajectories."""

from __future__ import annotations

from pathlib import Path

from scripts.audit_trajectories import (
    audit_trajectory,
    flag_answer_without_required_tools,
    flag_repeated_stalling,
    flag_trivial_json_gaming,
    format_audit_record,
    load_trajectories,
    sample_trajectories,
)

from toolsmith.data.taskspec import TaskSpec, ToolWasCalledWithCondition
from toolsmith.env.state import EpisodeState, EpisodeStatus, ToolCallLogEntry


def _spec(task_id: str) -> TaskSpec:
    return TaskSpec(
        id=task_id,
        tier="T1",
        user_prompt="Geocode Paris.",
        goal_spec=[ToolWasCalledWithCondition(tool_name="geocode_city", args={"city": "Paris"})],
        min_steps=1,
        split="train",
    )


def _ok_call(turn: int, name: str = "geocode_city", args: dict | None = None) -> ToolCallLogEntry:
    return ToolCallLogEntry(turn=turn, tool_name=name, args=args or {}, ok=True, result={})


def _state(**kwargs: object) -> EpisodeState:
    defaults: dict[str, object] = {"task_id": "t"}
    defaults.update(kwargs)
    return EpisodeState(**defaults)  # type: ignore[arg-type]


def test_load_trajectories_round_trip(tmp_path: Path) -> None:
    states = [_state(task_id="t1"), _state(task_id="t2")]
    path = tmp_path / "trajectories.jsonl"
    path.write_text("\n".join(s.to_json() for s in states) + "\n")

    loaded = load_trajectories(path)

    assert [s.task_id for s in loaded] == ["t1", "t2"]


def test_sample_trajectories_caps_at_n_and_is_deterministic() -> None:
    states = [_state(task_id=f"t{i}") for i in range(10)]

    first = sample_trajectories(states, n=3)
    second = sample_trajectories(states, n=3)

    assert len(first) == 3
    assert [s.task_id for s in first] == [s.task_id for s in second]


def test_sample_trajectories_returns_all_when_fewer_than_n() -> None:
    states = [_state(task_id="t1")]

    assert len(sample_trajectories(states, n=20)) == 1


def test_flag_answer_without_required_tools_true_when_missing() -> None:
    state = _state(status=EpisodeStatus.FINAL_ANSWER, final_answer="Paris is nice.", tool_calls=[])

    assert flag_answer_without_required_tools(state, _spec("t1")) is True


def test_flag_answer_without_required_tools_false_when_called() -> None:
    state = _state(
        status=EpisodeStatus.FINAL_ANSWER,
        final_answer="Paris is at 48.8N.",
        tool_calls=[_ok_call(0, args={"city": "Paris"})],
    )

    assert flag_answer_without_required_tools(state, _spec("t1")) is False


def test_flag_answer_without_required_tools_false_without_final_answer() -> None:
    state = _state(status=EpisodeStatus.MAX_TURNS, tool_calls=[])

    assert flag_answer_without_required_tools(state, _spec("t1")) is False


def test_flag_trivial_json_gaming_true_on_duplicate_call() -> None:
    state = _state(
        tool_calls=[
            _ok_call(0, args={"city": "Paris"}),
            _ok_call(1, args={"city": "Paris"}),
        ]
    )

    assert flag_trivial_json_gaming(state) is True


def test_flag_trivial_json_gaming_false_on_distinct_calls() -> None:
    state = _state(
        tool_calls=[
            _ok_call(0, args={"city": "Paris"}),
            _ok_call(1, args={"city": "Tokyo"}),
        ]
    )

    assert flag_trivial_json_gaming(state) is False


def test_flag_repeated_stalling_true_on_max_turns() -> None:
    assert flag_repeated_stalling(_state(status=EpisodeStatus.MAX_TURNS)) is True


def test_flag_repeated_stalling_false_on_final_answer() -> None:
    assert flag_repeated_stalling(_state(status=EpisodeStatus.FINAL_ANSWER)) is False


def test_audit_trajectory_includes_all_flags_with_spec() -> None:
    state = _state(
        status=EpisodeStatus.FINAL_ANSWER,
        final_answer="done",
        tool_calls=[_ok_call(0, args={"city": "Paris"}), _ok_call(1, args={"city": "Paris"})],
    )

    record = audit_trajectory(state, _spec("t1"))

    assert record["flags"]["trivial_json_gaming"] is True
    assert record["flags"]["repeated_stalling"] is False
    assert "answer_without_required_tools" in record["flags"]


def test_audit_trajectory_without_spec_omits_that_flag() -> None:
    state = _state(status=EpisodeStatus.FINAL_ANSWER, final_answer="done")

    record = audit_trajectory(state, None)

    assert "answer_without_required_tools" not in record["flags"]


def test_format_audit_record_includes_flagged_section() -> None:
    record = {
        "task_id": "t1",
        "status": "final_answer",
        "turns": 2,
        "tool_calls": [("geocode_city", True)],
        "final_answer": "done",
        "flags": {"trivial_json_gaming": True, "repeated_stalling": False},
    }

    text = format_audit_record(record)

    assert "t1" in text
    assert "FLAGGED: trivial_json_gaming" in text


def test_format_audit_record_no_flags_section_when_clean() -> None:
    record = {
        "task_id": "t1",
        "status": "final_answer",
        "turns": 1,
        "tool_calls": [],
        "final_answer": "done",
        "flags": {"trivial_json_gaming": False, "repeated_stalling": False},
    }

    text = format_audit_record(record)

    assert "FLAGGED" not in text
