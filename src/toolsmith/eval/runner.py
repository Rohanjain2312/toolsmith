"""Model-agnostic eval runner: any Model adapter x task list -> JSON-valid/tool/arg/completion %."""

from __future__ import annotations

from pathlib import Path

from toolsmith.data.taskspec import TaskSpec, ToolWasCalledWithCondition
from toolsmith.env.model import Model
from toolsmith.env.runner import DEFAULT_TRAJECTORY_DIR, run_episode
from toolsmith.env.state import EpisodeState, EpisodeStatus
from toolsmith.rewards.goalcheck import check_goal


def _is_json_valid(state: EpisodeState) -> bool:
    """A trajectory counts as JSON-valid iff the model never hit a parse failure."""
    return state.status is not EpisodeStatus.PARSE_FAILURE


def _required_tool_conditions(spec: TaskSpec) -> list[ToolWasCalledWithCondition]:
    return [c for c in spec.goal_spec if isinstance(c, ToolWasCalledWithCondition)]


def _called_correct_tool(spec: TaskSpec, state: EpisodeState) -> bool:
    """True iff every tool the goal spec requires was successfully called at least once
    (by name only — see `_arg_accuracy` for whether the args also matched)."""
    required = {c.tool_name for c in _required_tool_conditions(spec)}
    if not required:
        return True
    called = {entry.tool_name for entry in state.tool_calls if entry.ok}
    return required <= called


def _args_is_subset(expected: dict, actual: dict) -> bool:
    return all(key in actual and actual[key] == value for key, value in expected.items())


def _arg_accuracy(spec: TaskSpec, state: EpisodeState) -> float:
    """Fraction of goal-required tool calls whose args (not just tool name) were matched."""
    conditions = _required_tool_conditions(spec)
    if not conditions:
        return 1.0
    matched = sum(
        any(
            entry.ok
            and entry.tool_name == cond.tool_name
            and _args_is_subset(cond.args, entry.args)
            for entry in state.tool_calls
        )
        for cond in conditions
    )
    return matched / len(conditions)


def run_eval(
    specs: list[TaskSpec], model: Model, trajectory_dir: Path = DEFAULT_TRAJECTORY_DIR
) -> dict[str, float]:
    """Run each task through the episode runner and report aggregate eval metrics.

    Greedy decoding is enforced by the caller: pass a `model` already configured for
    temperature-0 / deterministic generation (StubModel is inherently deterministic; real
    adapters must set this themselves), so results are comparable across models.
    """
    if not specs:
        return {
            "task_count": 0,
            "json_valid_pct": 0.0,
            "correct_tool_pct": 0.0,
            "arg_accuracy_pct": 0.0,
            "task_completion_pct": 0.0,
        }

    json_valid = correct_tool = task_completion = 0
    arg_accuracy_sum = 0.0
    for spec in specs:
        state = run_episode(spec.id, spec.user_prompt, model, trajectory_dir=trajectory_dir)
        if _is_json_valid(state):
            json_valid += 1
        if _called_correct_tool(spec, state):
            correct_tool += 1
        arg_accuracy_sum += _arg_accuracy(spec, state)
        if check_goal(spec.goal_spec, state):
            task_completion += 1

    n = len(specs)
    return {
        "task_count": n,
        "json_valid_pct": 100.0 * json_valid / n,
        "correct_tool_pct": 100.0 * correct_tool / n,
        "arg_accuracy_pct": 100.0 * arg_accuracy_sum / n,
        "task_completion_pct": 100.0 * task_completion / n,
    }
