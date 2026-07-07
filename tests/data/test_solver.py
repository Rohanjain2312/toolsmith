"""Tests for the BFS solver, on hand-built T1 and T2 tasks."""

from __future__ import annotations

import pytest

from toolsmith.data.solver import MAX_DEPTH, UnsolvableTaskError, solve
from toolsmith.data.taskspec import (
    AnswerContainsFactCondition,
    CalendarEventExistsCondition,
    ToolWasCalledWithCondition,
)


def test_t1_single_tool_task_solves_in_one_step() -> None:
    goal_spec = [
        ToolWasCalledWithCondition(tool_name="geocode_city", args={"city": "Paris"}),
    ]

    assert solve(goal_spec) == 1


def test_t2_two_tool_chain_solves_in_two_steps() -> None:
    goal_spec = [
        ToolWasCalledWithCondition(tool_name="geocode_city", args={"city": "Paris"}),
        ToolWasCalledWithCondition(
            tool_name="weather_lookup",
            args={"lat": 48.8566, "lon": 2.3522, "date": "2026-09-03"},
        ),
    ]

    assert solve(goal_spec) == 2


def test_calendar_event_condition_counts_as_one_step() -> None:
    goal_spec = [
        CalendarEventExistsCondition(
            title="Museum visit",
            start="2026-09-03T10:00:00",
            end="2026-09-03T12:00:00",
            timezone="Europe/Paris",
        ),
    ]

    assert solve(goal_spec) == 1


def test_goal_spec_with_no_sandbox_checkable_conditions_needs_zero_steps() -> None:
    goal_spec = [AnswerContainsFactCondition(fact="Paris")]

    assert solve(goal_spec) == 0


def test_unsolvable_task_raises_when_required_call_fails() -> None:
    goal_spec = [
        ToolWasCalledWithCondition(tool_name="geocode_city", args={"city": "Nonexistent Place"}),
    ]

    with pytest.raises(UnsolvableTaskError):
        solve(goal_spec)


def test_unsolvable_task_raises_when_exceeding_max_depth() -> None:
    goal_spec = [
        ToolWasCalledWithCondition(
            tool_name="distance_calc", args={"lat1": i, "lon1": 0, "lat2": 0, "lon2": 0}
        )
        for i in range(MAX_DEPTH + 1)
    ]

    with pytest.raises(UnsolvableTaskError):
        solve(goal_spec)


def test_mixed_answer_and_tool_conditions_only_counts_tool_steps() -> None:
    goal_spec = [
        AnswerContainsFactCondition(fact="Paris"),
        ToolWasCalledWithCondition(tool_name="geocode_city", args={"city": "Paris"}),
    ]

    assert solve(goal_spec) == 1
