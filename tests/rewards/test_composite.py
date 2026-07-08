"""Tests for the composite TRL-compatible reward function, wiring R1-R6 end to end."""

from __future__ import annotations

from pathlib import Path

from toolsmith.data.taskspec import ToolWasCalledWithCondition
from toolsmith.env.model import StubModel
from toolsmith.rewards.composite import make_reward_func, score_completion
from toolsmith.rewards.outcome_reward import ContinuationCache

PREFIX = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "geocode paris"},
]
GOAL_SPEC = [ToolWasCalledWithCondition(tool_name="geocode_city", args={"city": "Paris"})]
GOOD_CALL = '{"tool": "geocode_city", "args": {"city": "Paris"}}'
WRONG_CALL = '{"tool": "geocode_city", "args": {"city": "Tokyo"}}'
MALFORMED = '{"tool": "geocode_city", "args": '


def _cache(tmp_path: Path) -> ContinuationCache:
    return ContinuationCache(path=tmp_path / "cache.json")


def test_score_completion_all_components_pass(tmp_path: Path) -> None:
    model = StubModel(["Paris, at 48.8N."])

    scored = score_completion("t1", PREFIX, GOOD_CALL, GOAL_SPEC, 1, model, _cache(tmp_path))

    assert scored["r1_valid_parse"] == 1.0
    assert scored["r2_tool_exists"] == 0.5
    assert scored["r3_args_valid"] == 1.0
    assert scored["r4_no_duplicate"] == 0.5
    assert scored["r5_goal_satisfied"] == 3.0
    assert scored["r6_efficiency"] == 0.5
    assert scored["penalty_hallucinated_tool"] == 0.0
    assert scored["penalty_max_turns"] == 0.0
    assert scored["total"] == 6.5


def test_score_completion_goal_not_satisfied(tmp_path: Path) -> None:
    model = StubModel(["Tokyo is nice."])

    scored = score_completion("t1", PREFIX, WRONG_CALL, GOAL_SPEC, 1, model, _cache(tmp_path))

    assert scored["r5_goal_satisfied"] == 0.0
    assert scored["r1_valid_parse"] == 1.0
    assert scored["total"] == 3.5


def test_score_completion_malformed_completion_scores_zero_and_skips_frozen_model(
    tmp_path: Path,
) -> None:
    model = StubModel([])  # generate() would raise if ever called

    scored = score_completion("t1", PREFIX, MALFORMED, GOAL_SPEC, 1, model, _cache(tmp_path))

    assert scored["total"] == 0.0
    assert scored["r1_valid_parse"] == 0.0
    assert scored["r5_goal_satisfied"] == 0.0


def test_score_completion_duplicate_call_zeroes_r4(tmp_path: Path) -> None:
    prefix_with_prior_call = [*PREFIX, {"role": "assistant", "content": GOOD_CALL}]
    model = StubModel(["Paris, at 48.8N."])

    scored = score_completion(
        "t1", prefix_with_prior_call, GOOD_CALL, GOAL_SPEC, 1, model, _cache(tmp_path)
    )

    assert scored["r4_no_duplicate"] == 0.0


def test_make_reward_func_returns_totals_aligned_with_completions(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    task_lookup = {"t1": (GOAL_SPEC, 1), "t2": (GOAL_SPEC, 1)}
    model = StubModel(["Paris, at 48.8N.", "Tokyo is nice."])
    reward_func = make_reward_func(task_lookup, model, cache)

    rewards = reward_func(
        prompts=[PREFIX, PREFIX],
        completions=[GOOD_CALL, WRONG_CALL],
        task_ids=["t1", "t2"],
    )

    assert len(rewards) == 2
    assert rewards[0] == 6.5
    assert rewards[1] == 3.5


def test_make_reward_func_logs_per_component_values(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    task_lookup = {"t1": (GOAL_SPEC, 1)}
    model = StubModel(["Paris, at 48.8N."])
    reward_func = make_reward_func(task_lookup, model, cache)

    reward_func(prompts=[PREFIX], completions=[GOOD_CALL], task_ids=["t1"])

    assert len(reward_func.last_component_log) == 1
    assert reward_func.last_component_log[0]["r5_goal_satisfied"] == 3.0
