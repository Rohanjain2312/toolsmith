"""Tests for the R5 outcome reward and its mandatory disk-persisted continuation cache."""

from __future__ import annotations

from pathlib import Path

import pytest

from toolsmith.data.taskspec import ToolWasCalledWithCondition
from toolsmith.env.model import StubModel, StubModelExhaustedError
from toolsmith.env.state import EpisodeStatus
from toolsmith.rewards.outcome_reward import (
    R5_GOAL_SATISFIED,
    ContinuationCache,
    compute_outcome_reward,
    get_or_build_final_state,
)

PREFIX = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "geocode paris"},
]
GOAL_SPEC = [ToolWasCalledWithCondition(tool_name="geocode_city", args={"city": "Paris"})]
TOOL_CALL = '{"tool": "geocode_city", "args": {"city": "Paris"}}'
WRONG_CALL = '{"tool": "geocode_city", "args": {"city": "Tokyo"}}'


# --- ContinuationCache ---


def test_cache_miss_then_hit(tmp_path: Path) -> None:
    cache = ContinuationCache(path=tmp_path / "cache.json")

    assert cache.get(PREFIX, TOOL_CALL) is None
    assert cache.misses == 1

    model = StubModel(["Paris, at 48.8N."])
    state = get_or_build_final_state("t1", PREFIX, TOOL_CALL, model, cache)
    cached = cache.get(PREFIX, TOOL_CALL)

    assert cached is not None
    assert cached.task_id == state.task_id
    assert cache.hits == 1


def test_cache_hit_rate() -> None:
    cache = ContinuationCache(path=Path("unused-in-this-test.json"))

    assert cache.hit_rate == 0.0
    cache.hits = 3
    cache.misses = 1
    assert cache.hit_rate == 0.75


def test_cache_persists_to_disk_and_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "cache.json"
    first_cache = ContinuationCache(path=path)
    get_or_build_final_state("t1", PREFIX, TOOL_CALL, StubModel(["Paris, at 48.8N."]), first_cache)

    second_cache = ContinuationCache(path=path)
    cached = second_cache.get(PREFIX, TOOL_CALL)

    assert cached is not None
    assert cached.task_id == "t1"
    assert second_cache.hits == 1
    assert second_cache.misses == 0


# --- compute_outcome_reward / get_or_build_final_state ---


def test_compute_outcome_reward_passes_goal(tmp_path: Path) -> None:
    cache = ContinuationCache(path=tmp_path / "cache.json")
    model = StubModel(["Paris, at 48.8N."])

    reward = compute_outcome_reward("t1", PREFIX, TOOL_CALL, GOAL_SPEC, model, cache)

    assert reward == R5_GOAL_SATISFIED


def test_compute_outcome_reward_fails_goal(tmp_path: Path) -> None:
    cache = ContinuationCache(path=tmp_path / "cache.json")
    model = StubModel(["Tokyo, nice city."])

    reward = compute_outcome_reward("t1", PREFIX, WRONG_CALL, GOAL_SPEC, model, cache)

    assert reward == 0.0


def test_compute_outcome_reward_parse_failure_never_invokes_frozen_model(tmp_path: Path) -> None:
    cache = ContinuationCache(path=tmp_path / "cache.json")
    model = StubModel([])  # any generate() call raises StubModelExhaustedError

    reward = compute_outcome_reward(
        "t1", PREFIX, '{"tool": "geocode_city", "args": ', GOAL_SPEC, model, cache
    )

    assert reward == 0.0


def test_compute_outcome_reward_final_answer_candidate_never_invokes_frozen_model(
    tmp_path: Path,
) -> None:
    cache = ContinuationCache(path=tmp_path / "cache.json")
    model = StubModel([])
    goal_spec = [ToolWasCalledWithCondition(tool_name="geocode_city", args={"city": "Paris"})]

    reward = compute_outcome_reward("t1", PREFIX, "I don't know.", goal_spec, model, cache)

    assert reward == 0.0


def test_compute_outcome_reward_uses_cache_on_second_call(tmp_path: Path) -> None:
    cache = ContinuationCache(path=tmp_path / "cache.json")
    model = StubModel(["Paris, at 48.8N."])  # only enough for ONE rollout

    first = compute_outcome_reward("t1", PREFIX, TOOL_CALL, GOAL_SPEC, model, cache)
    second = compute_outcome_reward("t1", PREFIX, TOOL_CALL, GOAL_SPEC, model, cache)

    assert first == second == R5_GOAL_SATISFIED
    with pytest.raises(StubModelExhaustedError):
        model.generate([], [])  # confirms the model was called exactly once, not twice


def test_get_or_build_final_state_executes_tool_call(tmp_path: Path) -> None:
    cache = ContinuationCache(path=tmp_path / "cache.json")
    model = StubModel(["Paris, at 48.8N."])

    state = get_or_build_final_state("t1", PREFIX, TOOL_CALL, model, cache)

    assert state.status == EpisodeStatus.FINAL_ANSWER
    assert len(state.tool_calls) == 1
    assert state.tool_calls[0].tool_name == "geocode_city"
    assert state.tool_calls[0].ok is True
