"""Tests for the R6 efficiency bonus and the hallucinated-tool / max-turns terminal penalties."""

from __future__ import annotations

from toolsmith.env.state import EpisodeState, EpisodeStatus, ToolCallLogEntry
from toolsmith.rewards.efficiency_rewards import (
    PENALTY_HALLUCINATED_TOOL,
    PENALTY_MAX_TURNS,
    R6_EFFICIENCY_BONUS,
    penalty_hallucinated_tool,
    penalty_max_turns,
    reward_efficiency,
)


def _state(**kwargs: object) -> EpisodeState:
    defaults: dict[str, object] = {"task_id": "t"}
    defaults.update(kwargs)
    return EpisodeState(**defaults)  # type: ignore[arg-type]


def _ok_call(turn: int, name: str = "geocode_city") -> ToolCallLogEntry:
    return ToolCallLogEntry(turn=turn, tool_name=name, args={}, ok=True, result={})


# --- R6: efficiency bonus ---


def test_efficiency_bonus_at_exact_min_steps() -> None:
    state = _state(status=EpisodeStatus.FINAL_ANSWER, tool_calls=[_ok_call(0)])

    assert reward_efficiency(state, min_steps=1) == R6_EFFICIENCY_BONUS


def test_efficiency_bonus_within_slack_of_one() -> None:
    state = _state(status=EpisodeStatus.FINAL_ANSWER, tool_calls=[_ok_call(0), _ok_call(1)])

    assert reward_efficiency(state, min_steps=1) == R6_EFFICIENCY_BONUS


def test_efficiency_bonus_zero_when_over_budget() -> None:
    state = _state(
        status=EpisodeStatus.FINAL_ANSWER,
        tool_calls=[_ok_call(0), _ok_call(1), _ok_call(2)],
    )

    assert reward_efficiency(state, min_steps=1) == 0.0


def test_efficiency_bonus_zero_without_final_answer() -> None:
    state = _state(status=EpisodeStatus.MAX_TURNS, tool_calls=[_ok_call(0)])

    assert reward_efficiency(state, min_steps=1) == 0.0


def test_efficiency_bonus_ignores_failed_calls() -> None:
    failed_call = ToolCallLogEntry(turn=0, tool_name="geocode_city", args={}, ok=False, error="x")
    state = _state(status=EpisodeStatus.FINAL_ANSWER, tool_calls=[failed_call, _ok_call(1)])

    assert reward_efficiency(state, min_steps=1) == R6_EFFICIENCY_BONUS


def test_efficiency_bonus_zero_steps_task() -> None:
    state = _state(status=EpisodeStatus.FINAL_ANSWER, tool_calls=[])

    assert reward_efficiency(state, min_steps=0) == R6_EFFICIENCY_BONUS


# --- hallucinated-tool penalty ---


def test_hallucinated_tool_penalty_applies() -> None:
    bad_call = ToolCallLogEntry(
        turn=0,
        tool_name="not_a_real_tool",
        args={},
        ok=False,
        error="unknown tool: not_a_real_tool",
    )
    state = _state(tool_calls=[bad_call])

    assert penalty_hallucinated_tool(state) == PENALTY_HALLUCINATED_TOOL


def test_hallucinated_tool_penalty_absent_when_no_unknown_tool() -> None:
    state = _state(tool_calls=[_ok_call(0)])

    assert penalty_hallucinated_tool(state) == 0.0


def test_hallucinated_tool_penalty_ignores_other_failure_reasons() -> None:
    invalid_args_call = ToolCallLogEntry(
        turn=0,
        tool_name="geocode_city",
        args={},
        ok=False,
        error="invalid args for geocode_city: x",
    )
    state = _state(tool_calls=[invalid_args_call])

    assert penalty_hallucinated_tool(state) == 0.0


def test_hallucinated_tool_penalty_no_calls() -> None:
    assert penalty_hallucinated_tool(_state()) == 0.0


# --- max-turns penalty ---


def test_max_turns_penalty_applies() -> None:
    state = _state(status=EpisodeStatus.MAX_TURNS)

    assert penalty_max_turns(state) == PENALTY_MAX_TURNS


def test_max_turns_penalty_absent_on_final_answer() -> None:
    state = _state(status=EpisodeStatus.FINAL_ANSWER)

    assert penalty_max_turns(state) == 0.0


def test_max_turns_penalty_absent_on_parse_failure() -> None:
    state = _state(status=EpisodeStatus.PARSE_FAILURE)

    assert penalty_max_turns(state) == 0.0
