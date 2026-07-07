"""Deterministic sandbox implementation of the datetime_math tool."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Literal

from pydantic import BaseModel, model_validator

from toolsmith.tools.schemas import ToolSpec, registry

# Fixed sandbox "today". Never use date.today() here: the sandbox must be
# deterministic (same input -> identical output), so "today" is always this
# constant rather than the real wall clock. 2026-09-01 is a Tuesday.
SANDBOX_TODAY = date(2026, 9, 1)

_WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_WEEKDAY_INDEX = {name: idx for idx, name in enumerate(_WEEKDAY_NAMES)}

WeekdayName = Literal["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
Operation = Literal["add_days", "next_weekday", "weekday_of"]


class DatetimeMathArgs(BaseModel):
    """Arguments for a datetime_math call: an operation plus its operands."""

    operation: Operation
    base_date: date | None = None
    days: int | None = None
    weekday: WeekdayName | None = None

    @model_validator(mode="after")
    def _require_operands_for_operation(self) -> DatetimeMathArgs:
        """Require the operand(s) that the chosen operation needs."""
        if self.operation == "add_days" and self.days is None:
            raise ValueError("'days' is required for operation 'add_days'")
        if self.operation == "next_weekday" and self.weekday is None:
            raise ValueError("'weekday' is required for operation 'next_weekday'")
        return self


class DatetimeMathResult(BaseModel):
    """Result of a datetime_math call."""

    result_date: date
    weekday_name: str


def datetime_math(args: DatetimeMathArgs) -> DatetimeMathResult:
    """Perform a deterministic date computation relative to a base date."""
    base = args.base_date or SANDBOX_TODAY

    if args.operation == "add_days":
        assert args.days is not None  # enforced by model_validator
        result = base + timedelta(days=args.days)
    elif args.operation == "next_weekday":
        assert args.weekday is not None  # enforced by model_validator
        target_idx = _WEEKDAY_INDEX[args.weekday]
        # Strictly-after convention: if `base` itself falls on the requested
        # weekday, resolve to next week's occurrence (7 days later), not
        # today. `days_ahead` is computed in 1..7 so today is never returned.
        days_ahead = (target_idx - base.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        result = base + timedelta(days=days_ahead)
    else:  # "weekday_of"
        result = base

    weekday_name = _WEEKDAY_NAMES[result.weekday()]
    return DatetimeMathResult(result_date=result, weekday_name=weekday_name)


registry.register(
    ToolSpec(
        name="datetime_math",
        description=(
            "Perform deterministic date arithmetic: add days to a date, find "
            "the next occurrence of a weekday strictly after a date, or "
            "report which weekday a date falls on."
        ),
        args_model=DatetimeMathArgs,
        returns_model=DatetimeMathResult,
        sandbox_fn=datetime_math,
    )
)
