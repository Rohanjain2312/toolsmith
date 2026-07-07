"""BFS solver: prove a task's goal spec is achievable via sandbox tool calls, compute min_steps."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from toolsmith.data.taskspec import (
    CalendarEventExistsCondition,
    GoalCondition,
    ToolWasCalledWithCondition,
)
from toolsmith.env.executor import execute_tool_call

MAX_DEPTH = 6


class UnsolvableTaskError(ValueError):
    """Raised when no ordering of the required actions succeeds within MAX_DEPTH tool calls."""


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
    """
    actions: list[_RequiredAction] = []
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
