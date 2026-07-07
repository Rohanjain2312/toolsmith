"""Tests for the episode loop: final answer, max-turns, and parse-failure terminations."""

import json
from pathlib import Path

from toolsmith.env.model import StubModel
from toolsmith.env.runner import run_episode
from toolsmith.env.state import EpisodeStatus


def test_final_answer_termination(tmp_path: Path) -> None:
    model = StubModel(["The capital of France is Paris."])
    state = run_episode("t-final", "What is the capital of France?", model, trajectory_dir=tmp_path)

    assert state.status == EpisodeStatus.FINAL_ANSWER
    assert state.final_answer == "The capital of France is Paris."
    assert state.tool_calls == []


def test_max_turns_termination(tmp_path: Path) -> None:
    tool_call_text = (
        '{"tool": "distance_calc", "args": {"lat1": 0, "lon1": 0, "lat2": 1, "lon2": 1}}'
    )
    model = StubModel([tool_call_text] * 10)
    state = run_episode(
        "t-maxturns", "Compute something forever.", model, max_turns=8, trajectory_dir=tmp_path
    )

    assert state.status == EpisodeStatus.MAX_TURNS
    assert state.turn == 8
    assert len(state.tool_calls) == 8
    assert all(entry.ok for entry in state.tool_calls)


def test_parse_failure_termination(tmp_path: Path) -> None:
    model = StubModel(['{"tool": "distance_calc", "args": {'])  # truncated/malformed JSON
    state = run_episode("t-parsefail", "Do something.", model, trajectory_dir=tmp_path)

    assert state.status == EpisodeStatus.PARSE_FAILURE
    assert state.tool_calls == []


def test_tool_call_then_final_answer(tmp_path: Path) -> None:
    tool_call_text = (
        '{"tool": "distance_calc", "args": {"lat1": 0, "lon1": 0, "lat2": 0, "lon2": 0}}'
    )
    model = StubModel([tool_call_text, "The distance is 0 km."])
    state = run_episode(
        "t-roundtrip", "How far apart are these points?", model, trajectory_dir=tmp_path
    )

    assert state.status == EpisodeStatus.FINAL_ANSWER
    assert state.final_answer == "The distance is 0 km."
    assert len(state.tool_calls) == 1
    assert state.tool_calls[0].ok is True
    assert state.tool_calls[0].tool_name == "distance_calc"
    roles = [m["role"] for m in state.messages]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]


def test_trajectory_jsonl_is_written(tmp_path: Path) -> None:
    model = StubModel(["Final answer text."])
    run_episode("t-trajectory", "Say something.", model, trajectory_dir=tmp_path)

    trajectory_path = tmp_path / "t-trajectory.jsonl"
    assert trajectory_path.exists()
    lines = trajectory_path.read_text().strip().splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["event"] == "final_answer"
