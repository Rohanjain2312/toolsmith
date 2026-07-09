"""Composite GRPO reward: TRL-compatible callable wiring R1-R6 into one score per completion."""

from __future__ import annotations

from typing import Any, Protocol

from toolsmith.data.taskspec import GoalCondition
from toolsmith.env.model import Model
from toolsmith.rewards.efficiency_rewards import (
    penalty_hallucinated_tool,
    penalty_max_turns,
    reward_efficiency,
)
from toolsmith.rewards.format_rewards import (
    reward_args_valid,
    reward_no_duplicate,
    reward_tool_exists,
    reward_valid_parse,
)
from toolsmith.rewards.goalcheck import check_goal
from toolsmith.rewards.outcome_reward import (
    R5_GOAL_SATISFIED,
    ContinuationCache,
    get_or_build_final_state,
)


def _completion_to_text(completion: str | list[dict[str, Any]]) -> str:
    """Normalize a TRL reward-func completion to the raw assistant text the R1-R6 scorers expect.

    TRL passes completions as plain strings for standard datasets, but for conversational
    (message-list) prompts -- which this project's decision-point prefixes are -- it wraps each
    completion as a list of assistant message dicts, [{"role": "assistant", "content": ...}]
    (grpo_trainer.py). Collapse that back to the assistant text so the downstream parser, which
    takes a str, works in both cases.
    """
    if isinstance(completion, str):
        return completion
    return "".join(msg["content"] for msg in completion if msg.get("role") == "assistant")


def score_completion(
    task_id: str,
    prefix: list[dict[str, Any]],
    completion: str,
    goal_spec: list[GoalCondition],
    min_steps: int,
    frozen_model: Model,
    cache: ContinuationCache,
) -> dict[str, float]:
    """Score one candidate completion across all six reward components; return per-component
    values plus their sum under the "total" key."""
    state = get_or_build_final_state(task_id, prefix, completion, frozen_model, cache)
    components = {
        "r1_valid_parse": reward_valid_parse(completion),
        "r2_tool_exists": reward_tool_exists(completion),
        "r3_args_valid": reward_args_valid(completion),
        "r4_no_duplicate": reward_no_duplicate(completion, prefix),
        "r5_goal_satisfied": R5_GOAL_SATISFIED if check_goal(goal_spec, state) else 0.0,
        "r6_efficiency": reward_efficiency(state, min_steps),
        "penalty_hallucinated_tool": penalty_hallucinated_tool(state),
        "penalty_max_turns": penalty_max_turns(state),
    }
    components["total"] = sum(components.values())
    return components


class RewardFunc(Protocol):
    """TRL-compatible reward callable, with the last batch's per-component breakdown attached."""

    last_component_log: list[dict[str, float]]

    def __call__(
        self, prompts: list[Any], completions: list[Any], **kwargs: Any
    ) -> list[float]: ...


def make_reward_func(
    task_lookup: dict[str, tuple[list[GoalCondition], int]],
    frozen_model: Model,
    cache: ContinuationCache,
) -> RewardFunc:
    """Build a TRL-compatible reward_func(prompts, completions, **kwargs) -> list[float].

    `task_lookup` maps task_id -> (goal_spec, min_steps). GRPOTrainer forwards extra dataset
    columns as kwargs aligned with `completions`; `task_ids` is expected to be one such column,
    and `prompts[i]` is the decision point's message-list prefix for `completions[i]`.
    `reward_func.last_component_log` exposes the most recent batch's per-component values so
    the training loop can log R1-R6 separately to W&B.
    """

    def reward_func(prompts: list[Any], completions: list[Any], **kwargs: Any) -> list[float]:
        task_ids = kwargs["task_ids"]
        rewards = []
        component_log = []
        for prompt, completion, task_id in zip(prompts, completions, task_ids, strict=True):
            goal_spec, min_steps = task_lookup[task_id]
            scored = score_completion(
                task_id,
                prompt,
                _completion_to_text(completion),
                goal_spec,
                min_steps,
                frozen_model,
                cache,
            )
            component_log.append(scored)
            rewards.append(scored["total"])
        reward_func.last_component_log = component_log
        return rewards

    reward_func.last_component_log = []
    return reward_func
