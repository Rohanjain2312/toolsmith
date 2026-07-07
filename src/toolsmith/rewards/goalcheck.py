"""Evaluate a task's goal spec against a finished episode's tool log and final answer."""

from __future__ import annotations

import re

from toolsmith.data.taskspec import (
    AnswerContainsFactCondition,
    CalendarEventExistsCondition,
    GoalCondition,
    NumericWithinToleranceCondition,
    ToolWasCalledWithCondition,
)
from toolsmith.env.state import EpisodeState

_NUMBER_PATTERN = re.compile(r"-?\d+\.?\d*")


def check_goal(goal_spec: list[GoalCondition], state: EpisodeState) -> bool:
    """Return True iff every condition in the goal spec holds against the episode state."""
    return all(_check_condition(condition, state) for condition in goal_spec)


def _check_condition(condition: GoalCondition, state: EpisodeState) -> bool:
    if isinstance(condition, AnswerContainsFactCondition):
        return _check_answer_contains_fact(condition, state)
    if isinstance(condition, ToolWasCalledWithCondition):
        return _check_tool_was_called_with(condition, state)
    if isinstance(condition, CalendarEventExistsCondition):
        return _check_calendar_event_exists(condition, state)
    return _check_numeric_within_tolerance(condition, state)


def _check_answer_contains_fact(
    condition: AnswerContainsFactCondition, state: EpisodeState
) -> bool:
    if state.final_answer is None:
        return False
    return condition.fact.lower() in state.final_answer.lower()


def _args_is_subset(expected: dict, actual: dict) -> bool:
    return all(key in actual and actual[key] == value for key, value in expected.items())


def _check_tool_was_called_with(
    condition: ToolWasCalledWithCondition, state: EpisodeState
) -> bool:
    return any(
        entry.ok
        and entry.tool_name == condition.tool_name
        and _args_is_subset(condition.args, entry.args)
        for entry in state.tool_calls
    )


def _check_calendar_event_exists(
    condition: CalendarEventExistsCondition, state: EpisodeState
) -> bool:
    expected = {
        "title": condition.title,
        "start": condition.start,
        "end": condition.end,
        "timezone": condition.timezone,
    }
    return any(
        entry.ok
        and entry.tool_name == "calendar_create_event"
        and _args_is_subset(expected, entry.args)
        for entry in state.tool_calls
    )


def _extract_first_number(text: str) -> float | None:
    match = _NUMBER_PATTERN.search(text)
    return float(match.group()) if match else None


def _extract_tool_result_number(
    condition: NumericWithinToleranceCondition, state: EpisodeState
) -> float | None:
    for entry in reversed(state.tool_calls):
        if (
            entry.ok
            and entry.tool_name == condition.tool_name
            and entry.result is not None
            and condition.field in entry.result
        ):
            return float(entry.result[condition.field])
    return None


def _check_numeric_within_tolerance(
    condition: NumericWithinToleranceCondition, state: EpisodeState
) -> bool:
    if condition.source == "final_answer":
        value = _extract_first_number(state.final_answer) if state.final_answer else None
    else:
        value = _extract_tool_result_number(condition, state)

    if value is None:
        return False
    return abs(value - condition.expected) <= condition.tolerance
