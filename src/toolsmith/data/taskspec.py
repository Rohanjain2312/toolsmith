"""Pydantic task-spec model: prompt, machine-checkable goal spec, solver-computed min_steps."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator

Tier = Literal["T1", "T2", "T3", "T4"]
Split = Literal["train", "val", "test"]


class AnswerContainsFactCondition(BaseModel):
    """Goal condition: the episode's final answer text must contain this fact (case-insensitive)."""

    type: Literal["answer_contains_fact"] = "answer_contains_fact"
    fact: str = Field(min_length=1)


class ToolWasCalledWithCondition(BaseModel):
    """Goal condition: a successful call must match this tool name and a subset of these args."""

    type: Literal["tool_was_called_with"] = "tool_was_called_with"
    tool_name: str = Field(min_length=1)
    args: dict[str, Any]


class CalendarEventExistsCondition(BaseModel):
    """Goal condition: a calendar_create_event call with these exact fields must have succeeded."""

    type: Literal["calendar_event_exists"] = "calendar_event_exists"
    title: str = Field(min_length=1)
    start: str
    end: str
    timezone: str


class NumericWithinToleranceCondition(BaseModel):
    """Goal condition: a number from the final answer or a tool result must be near `expected`."""

    type: Literal["numeric_within_tolerance"] = "numeric_within_tolerance"
    expected: float
    tolerance: float = Field(ge=0)
    source: Literal["final_answer", "tool_result"]
    tool_name: str | None = None
    field: str | None = None

    @model_validator(mode="after")
    def _require_tool_result_fields(self) -> NumericWithinToleranceCondition:
        """When reading from a tool result, the tool name and result field must be specified."""
        if self.source == "tool_result" and (self.tool_name is None or self.field is None):
            raise ValueError("'tool_name' and 'field' are required when source is 'tool_result'")
        return self


GoalCondition = Annotated[
    AnswerContainsFactCondition
    | ToolWasCalledWithCondition
    | CalendarEventExistsCondition
    | NumericWithinToleranceCondition,
    Field(discriminator="type"),
]


class TaskSpec(BaseModel):
    """A single generated task: prompt, machine-checkable goal spec, min_steps, and split."""

    id: str = Field(min_length=1)
    tier: Tier
    user_prompt: str = Field(min_length=1)
    goal_spec: list[GoalCondition] = Field(min_length=1)
    min_steps: int = Field(ge=0)
    split: Split
