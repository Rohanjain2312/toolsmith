"""Tests for the BFS solver, on hand-built T1 and T2 tasks."""

from __future__ import annotations

import pytest

from toolsmith.data.solver import (
    MAX_DEPTH,
    UnsolvableTaskError,
    UnsupportedGoalConditionError,
    solve,
)
from toolsmith.data.taskspec import (
    AnswerContainsFactCondition,
    CalendarEventExistsCondition,
    NumericWithinToleranceCondition,
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


def test_unaccompanied_tool_result_numeric_condition_raises_instead_of_undercounting() -> None:
    # Regression test for BUGFIX-T06: a tool_result-sourced numeric condition carries no args,
    # so the solver cannot derive which tool call would satisfy it. Silently excluding it (as
    # answer_contains_fact/final_answer-sourced numeric conditions are, correctly, since those
    # truly don't need a sandbox call) would undercount min_steps instead of failing loudly.
    goal_spec = [
        NumericWithinToleranceCondition(
            expected=5.5,
            tolerance=0.5,
            source="tool_result",
            tool_name="distance_calc",
            field="distance_km",
        ),
    ]

    with pytest.raises(UnsupportedGoalConditionError):
        solve(goal_spec)


def test_tool_result_numeric_condition_paired_with_tool_call_does_not_add_a_step() -> None:
    # The intended, generation-pipeline pattern: a tool_result-sourced numeric condition is
    # paired with a tool_was_called_with condition for the same tool, so the solver already
    # counts the one required call and the numeric condition just validates its result.
    goal_spec = [
        ToolWasCalledWithCondition(
            tool_name="distance_calc", args={"lat1": 0, "lon1": 0, "lat2": 0, "lon2": 1}
        ),
        NumericWithinToleranceCondition(
            expected=111.0,
            tolerance=5.0,
            source="tool_result",
            tool_name="distance_calc",
            field="distance_km",
        ),
    ]

    assert solve(goal_spec) == 1
