"""Tests for the TaskSpec model and its discriminated-union goal conditions."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from toolsmith.data.taskspec import (
    AnswerContainsFactCondition,
    CalendarEventExistsCondition,
    NumericWithinToleranceCondition,
    TaskSpec,
    ToolWasCalledWithCondition,
)


def _sample_spec() -> TaskSpec:
    return TaskSpec(
        id="t1-0001",
        tier="T1",
        user_prompt="What's the weather in Paris on 2026-09-03?",
        goal_spec=[
            ToolWasCalledWithCondition(
                tool_name="weather_lookup", args={"lat": 48.8566, "lon": 2.3522}
            )
        ],
        min_steps=1,
        split="train",
    )


def test_round_trip_single_condition() -> None:
    spec = _sample_spec()

    restored = TaskSpec.model_validate_json(spec.model_dump_json())

    assert restored == spec
    assert isinstance(restored.goal_spec[0], ToolWasCalledWithCondition)


def test_round_trip_all_condition_types() -> None:
    spec = TaskSpec(
        id="t3-0002",
        tier="T3",
        user_prompt="Plan a day trip and note the distance.",
        goal_spec=[
            AnswerContainsFactCondition(fact="Paris"),
            ToolWasCalledWithCondition(tool_name="geocode_city", args={"city": "Paris"}),
            CalendarEventExistsCondition(
                title="Museum visit",
                start="2026-09-03T10:00:00",
                end="2026-09-03T12:00:00",
                timezone="Europe/Paris",
            ),
            NumericWithinToleranceCondition(
                expected=5.5,
                tolerance=0.5,
                source="tool_result",
                tool_name="distance_calc",
                field="distance_km",
            ),
        ],
        min_steps=4,
        split="val",
    )

    restored = TaskSpec.model_validate_json(spec.model_dump_json())

    assert restored == spec
    assert [type(c) for c in restored.goal_spec] == [
        AnswerContainsFactCondition,
        ToolWasCalledWithCondition,
        CalendarEventExistsCondition,
        NumericWithinToleranceCondition,
    ]


def test_unknown_condition_type_rejected() -> None:
    with pytest.raises(ValidationError):
        TaskSpec.model_validate(
            {
                "id": "t1-0003",
                "tier": "T1",
                "user_prompt": "x",
                "goal_spec": [{"type": "not_a_real_condition"}],
                "min_steps": 0,
                "split": "train",
            }
        )


def test_numeric_condition_requires_tool_fields_when_source_is_tool_result() -> None:
    with pytest.raises(ValidationError):
        NumericWithinToleranceCondition(expected=1.0, tolerance=0.1, source="tool_result")


def test_numeric_condition_final_answer_source_does_not_require_tool_fields() -> None:
    condition = NumericWithinToleranceCondition(expected=1.0, tolerance=0.1, source="final_answer")

    assert condition.tool_name is None
    assert condition.field is None


def test_invalid_tier_rejected() -> None:
    with pytest.raises(ValidationError):
        TaskSpec.model_validate(
            {
                "id": "x",
                "tier": "T5",
                "user_prompt": "x",
                "goal_spec": [
                    {"type": "answer_contains_fact", "fact": "x"},
                ],
                "min_steps": 0,
                "split": "train",
            }
        )


def test_empty_goal_spec_rejected() -> None:
    with pytest.raises(ValidationError):
        TaskSpec(
            id="t1-0004",
            tier="T1",
            user_prompt="x",
            goal_spec=[],
            min_steps=0,
            split="train",
        )


def test_negative_min_steps_rejected() -> None:
    with pytest.raises(ValidationError):
        TaskSpec.model_validate(
            {
                "id": "t1-0005",
                "tier": "T1",
                "user_prompt": "x",
                "goal_spec": [{"type": "answer_contains_fact", "fact": "x"}],
                "min_steps": -1,
                "split": "train",
            }
        )
