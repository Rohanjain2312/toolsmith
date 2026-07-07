"""Tests for the episode state container."""

from __future__ import annotations

import json

from toolsmith.env.state import EpisodeState, EpisodeStatus, ToolCallLogEntry


def test_round_trip_with_tool_calls() -> None:
    entries = [
        ToolCallLogEntry(
            turn=0,
            tool_name="geocode_city",
            args={"city": "Paris"},
            ok=True,
            result={"lat": 48.8566, "lon": 2.3522},
        ),
        ToolCallLogEntry(
            turn=1,
            tool_name="flight_search",
            args={"origin": "CDG"},
            ok=False,
            error="missing destination",
        ),
    ]
    state = EpisodeState(
        task_id="task-001",
        messages=[{"role": "user", "content": "plan my trip"}],
        tool_calls=entries,
        turn=2,
        status=EpisodeStatus.MAX_TURNS,
        final_answer=None,
    )

    restored = EpisodeState.from_json(state.to_json())

    assert restored == state
    assert restored.status is EpisodeStatus.MAX_TURNS
    for original_entry, restored_entry in zip(state.tool_calls, restored.tool_calls, strict=True):
        assert restored_entry == original_entry


def test_round_trip_with_defaults() -> None:
    state = EpisodeState(task_id="task-002")

    restored = EpisodeState.from_json(state.to_json())

    assert restored == state
    assert restored.messages == []
    assert restored.tool_calls == []
    assert restored.turn == 0
    assert restored.status is EpisodeStatus.IN_PROGRESS
    assert restored.final_answer is None


def test_to_json_produces_valid_json() -> None:
    state = EpisodeState(
        task_id="task-003",
        tool_calls=[
            ToolCallLogEntry(turn=0, tool_name="weather_lookup", args={}, ok=True, result={})
        ],
        status=EpisodeStatus.FINAL_ANSWER,
        final_answer="done",
    )

    parsed = json.loads(state.to_json())

    assert parsed["task_id"] == "task-003"
    assert parsed["status"] == "final_answer"
    assert parsed["tool_calls"][0]["tool_name"] == "weather_lookup"


def test_episode_status_values() -> None:
    assert EpisodeStatus.IN_PROGRESS.value == "in_progress"
    assert EpisodeStatus.FINAL_ANSWER.value == "final_answer"
    assert EpisodeStatus.MAX_TURNS.value == "max_turns"
    assert EpisodeStatus.PARSE_FAILURE.value == "parse_failure"


def test_episode_state_is_mutable() -> None:
    state = EpisodeState(task_id="task-004")

    state.turn = 5
    state.status = EpisodeStatus.PARSE_FAILURE
    state.tool_calls.append(
        ToolCallLogEntry(turn=5, tool_name="unit_convert", args={"value": 1}, ok=True, result={})
    )

    assert state.turn == 5
    assert state.status == EpisodeStatus.PARSE_FAILURE
    assert len(state.tool_calls) == 1
