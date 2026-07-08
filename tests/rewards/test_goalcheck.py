"""Tests for the goal-spec checker: one condition type at a time, then composed."""

from __future__ import annotations

from toolsmith.data.taskspec import (
    AnswerContainsFactCondition,
    CalendarEventExistsCondition,
    NumericWithinToleranceCondition,
    ToolWasCalledWithCondition,
)
from toolsmith.env.state import EpisodeState, ToolCallLogEntry
from toolsmith.rewards.goalcheck import check_goal


def _state(**kwargs: object) -> EpisodeState:
    defaults: dict[str, object] = {"task_id": "t"}
    defaults.update(kwargs)
    return EpisodeState(**defaults)  # type: ignore[arg-type]


def test_answer_contains_fact_matches_case_insensitively() -> None:
    condition = AnswerContainsFactCondition(fact="paris")
    state = _state(final_answer="The capital of France is Paris.")

    assert check_goal([condition], state) is True


def test_answer_contains_fact_fails_when_missing() -> None:
    condition = AnswerContainsFactCondition(fact="Tokyo")
    state = _state(final_answer="The capital of France is Paris.")

    assert check_goal([condition], state) is False


def test_answer_contains_fact_fails_when_no_final_answer() -> None:
    condition = AnswerContainsFactCondition(fact="Paris")
    state = _state(final_answer=None)

    assert check_goal([condition], state) is False


def test_tool_was_called_with_matches_subset_of_args() -> None:
    condition = ToolWasCalledWithCondition(tool_name="geocode_city", args={"city": "Paris"})
    state = _state(
        tool_calls=[
            ToolCallLogEntry(
                turn=0,
                tool_name="geocode_city",
                args={"city": "Paris"},
                ok=True,
                result={"lat": 48.8566, "lon": 2.3522},
            )
        ]
    )

    assert check_goal([condition], state) is True


def test_tool_was_called_with_ignores_failed_calls() -> None:
    condition = ToolWasCalledWithCondition(tool_name="geocode_city", args={"city": "Paris"})
    state = _state(
        tool_calls=[
            ToolCallLogEntry(
                turn=0, tool_name="geocode_city", args={"city": "Paris"}, ok=False, error="boom"
            )
        ]
    )

    assert check_goal([condition], state) is False


def test_tool_was_called_with_mismatched_args_fails() -> None:
    condition = ToolWasCalledWithCondition(tool_name="geocode_city", args={"city": "Paris"})
    state = _state(
        tool_calls=[
            ToolCallLogEntry(
                turn=0, tool_name="geocode_city", args={"city": "Tokyo"}, ok=True, result={}
            )
        ]
    )

    assert check_goal([condition], state) is False


def test_calendar_event_exists_matches_exact_args() -> None:
    condition = CalendarEventExistsCondition(
        title="Museum visit",
        start="2026-09-03T10:00:00",
        end="2026-09-03T12:00:00",
        timezone="Europe/Paris",
    )
    state = _state(
        tool_calls=[
            ToolCallLogEntry(
                turn=0,
                tool_name="calendar_create_event",
                args={
                    "title": "Museum visit",
                    "start": "2026-09-03T10:00:00",
                    "end": "2026-09-03T12:00:00",
                    "timezone": "Europe/Paris",
                },
                ok=True,
                result={"event_id": "abc123", "status": "confirmed"},
            )
        ]
    )

    assert check_goal([condition], state) is True


def test_calendar_event_exists_fails_when_no_matching_call() -> None:
    condition = CalendarEventExistsCondition(
        title="Museum visit", start="2026-09-03T10:00:00", end="2026-09-03T12:00:00",
        timezone="Europe/Paris",
    )
    state = _state(tool_calls=[])

    assert check_goal([condition], state) is False


def test_numeric_within_tolerance_from_tool_result() -> None:
    condition = NumericWithinToleranceCondition(
        expected=5.5, tolerance=0.5, source="tool_result",
        tool_name="distance_calc", field="distance_km",
    )
    state = _state(
        tool_calls=[
            ToolCallLogEntry(
                turn=0,
                tool_name="distance_calc",
                args={},
                ok=True,
                result={"distance_km": 5.8},
            )
        ]
    )

    assert check_goal([condition], state) is True


def test_numeric_within_tolerance_from_tool_result_out_of_range() -> None:
    condition = NumericWithinToleranceCondition(
        expected=5.5, tolerance=0.1, source="tool_result",
        tool_name="distance_calc", field="distance_km",
    )
    state = _state(
        tool_calls=[
            ToolCallLogEntry(
                turn=0, tool_name="distance_calc", args={}, ok=True, result={"distance_km": 9.0}
            )
        ]
    )

    assert check_goal([condition], state) is False


def test_numeric_within_tolerance_from_final_answer() -> None:
    condition = NumericWithinToleranceCondition(expected=42.0, tolerance=1.0, source="final_answer")
    state = _state(final_answer="The converted amount is 42.3 EUR.")

    assert check_goal([condition], state) is True


def test_numeric_within_tolerance_from_final_answer_skips_hyphenated_id_prefix() -> None:
    # Regression test for BUGFIX-T04: a UTC offset like "UTC-5" earlier in the text must not be
    # mistaken for the intended numeric answer.
    condition = NumericWithinToleranceCondition(
        expected=200.0, tolerance=5.0, source="final_answer"
    )
    state = _state(final_answer="The timezone is UTC-5, and your flight costs 200 USD.")

    assert check_goal([condition], state) is True


def test_numeric_within_tolerance_from_final_answer_skips_digits_glued_to_letters() -> None:
    # Same underlying defect, broader trigger: digits glued directly to a letter (e.g. a flight
    # ID like "FL0001", this sandbox's actual ID format) must not be read as the answer either.
    condition = NumericWithinToleranceCondition(
        expected=200.0, tolerance=5.0, source="final_answer"
    )
    state = _state(final_answer="Your flight FL0001 costs 200 USD.")

    assert check_goal([condition], state) is True


def test_numeric_within_tolerance_from_final_answer_still_extracts_legitimate_negative() -> None:
    # A real negative number (preceded by whitespace, not glued to a letter) must still work.
    condition = NumericWithinToleranceCondition(expected=-5.0, tolerance=1.0, source="final_answer")
    state = _state(final_answer="The temperature is -5 degrees.")

    assert check_goal([condition], state) is True


def test_numeric_within_tolerance_missing_source_fails() -> None:
    condition = NumericWithinToleranceCondition(
        expected=5.5, tolerance=0.5, source="tool_result",
        tool_name="distance_calc", field="distance_km",
    )
    state = _state(tool_calls=[])

    assert check_goal([condition], state) is False


def test_check_goal_requires_all_conditions() -> None:
    conditions = [
        AnswerContainsFactCondition(fact="Paris"),
        ToolWasCalledWithCondition(tool_name="geocode_city", args={"city": "Paris"}),
    ]
    state = _state(
        final_answer="Paris is lovely.",
        tool_calls=[
            ToolCallLogEntry(
                turn=0, tool_name="geocode_city", args={"city": "Tokyo"}, ok=True, result={}
            )
        ],
    )

    assert check_goal(conditions, state) is False
