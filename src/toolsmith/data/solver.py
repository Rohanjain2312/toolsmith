"""BFS solver: prove a task's goal spec is achievable via sandbox tool calls, compute min_steps."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from toolsmith.data.taskspec import (
    CalendarEventExistsCondition,
    GoalCondition,
    NumericWithinToleranceCondition,
    ToolWasCalledWithCondition,
)
from toolsmith.env.executor import execute_tool_call

MAX_DEPTH = 6


class UnsolvableTaskError(ValueError):
    """Raised when no ordering of the required actions succeeds within MAX_DEPTH tool calls."""


class UnsupportedGoalConditionError(ValueError):
    """Raised when a goal spec can't be resolved into concrete, sandbox-executable tool calls.

    Specifically: a `numeric_within_tolerance(source="tool_result")` condition carries a tool
    name and a result field to check, but no args -- there's no way to derive which tool call
    would satisfy it. The intended pattern pairs it with a `tool_was_called_with` condition for
    the same tool (whose args the solver already counts); this error fires when that pairing is
    missing, instead of silently returning an undercounted min_steps.
    """


@dataclass(frozen=True)
class _RequiredAction:
    """One sandbox tool call that a task's goal spec demands has happened."""

    tool_name: str
    args: tuple[tuple[str, object], ...]

    def as_dict(self) -> dict:
        return dict(self.args)


def _extract_required_actions(goal_spec: list[GoalCondition]) -> list[_RequiredAction]:
    """Pull the concrete, sandbox-executable tool calls a goal spec demands.

    `answer_contains_fact` and `numeric_within_tolerance(source="final_answer")` depend on
    generated natural language, not sandbox facts, so a tool-call solver can't verify them —
    they're excluded here and simply don't add to min_steps.

    `numeric_within_tolerance(source="tool_result")` carries a tool name and result field but no
    args, so it can't independently contribute a required action either — it's expected to be
    paired with a `tool_was_called_with` condition for the same tool, which already counts the
    call. See `UnsupportedGoalConditionError` for the case where that pairing is missing.
    """
    actions: list[_RequiredAction] = []
    called_tool_names = {
        condition.tool_name
        for condition in goal_spec
        if isinstance(condition, ToolWasCalledWithCondition)
    }
    for condition in goal_spec:
        if isinstance(condition, ToolWasCalledWithCondition):
            actions.append(
                _RequiredAction(condition.tool_name, tuple(sorted(condition.args.items())))
            )
        elif isinstance(condition, CalendarEventExistsCondition):
            args = {
                "title": condition.title,
                "start": condition.start,
                "end": condition.end,
                "timezone": condition.timezone,
            }
            actions.append(_RequiredAction("calendar_create_event", tuple(sorted(args.items()))))
        elif (
            isinstance(condition, NumericWithinToleranceCondition)
            and condition.source == "tool_result"
            and condition.tool_name not in called_tool_names
        ):
            raise UnsupportedGoalConditionError(
                "numeric_within_tolerance(source='tool_result', "
                f"tool_name={condition.tool_name!r}) has no accompanying tool_was_called_with "
                "condition for the same tool_name in this goal spec, so the solver cannot "
                "determine which args would produce the result. Add a tool_was_called_with "
                "condition for the same tool."
            )
    return actions


def solve(goal_spec: list[GoalCondition]) -> int:
    """BFS over subsets of a goal spec's required tool calls; return the shortest solving depth."""
    import toolsmith.tools.sandbox  # noqa: F401  (registers all 12 sandbox tools)

    required = _extract_required_actions(goal_spec)
    if not required:
        return 0
    if len(required) > MAX_DEPTH:
        raise UnsolvableTaskError(
            f"goal spec requires {len(required)} tool calls, exceeding MAX_DEPTH={MAX_DEPTH}"
        )

    goal_state = frozenset(range(len(required)))
    start: frozenset[int] = frozenset()
    queue: deque[frozenset[int]] = deque([start])
    visited = {start}

    while queue:
        state = queue.popleft()
        if state == goal_state:
            return len(state)
        for index, action in enumerate(required):
            if index in state:
                continue
            result = execute_tool_call(action.tool_name, action.as_dict())
            if not result.ok:
                continue
            next_state = state | {index}
            if next_state not in visited:
                visited.add(next_state)
                queue.append(next_state)

    raise UnsolvableTaskError(
        "no ordering of required tool calls executed successfully in the sandbox"
    )
