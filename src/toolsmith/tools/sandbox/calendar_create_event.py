"""Deterministic sandbox implementation of the calendar_create_event tool."""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field, model_validator

from toolsmith.tools.schemas import ToolSpec, registry


class InvalidTimezoneError(ValueError):
    """Raised when a timezone string is not a valid IANA zone name."""


class CalendarCreateEventArgs(BaseModel):
    """Arguments for a calendar_create_event call."""

    title: str = Field(min_length=1)
    start: datetime
    end: datetime
    timezone: str

    @model_validator(mode="after")
    def _check_end_after_start(self) -> CalendarCreateEventArgs:
        """Reject zero-duration or negative-duration events."""
        if self.end <= self.start:
            raise ValueError("end must be strictly after start")
        return self


class CalendarCreateEventResult(BaseModel):
    """Result of a calendar_create_event call."""

    event_id: str
    status: Literal["confirmed"]


def calendar_create_event(args: CalendarCreateEventArgs) -> CalendarCreateEventResult:
    """Create a deterministic calendar event, validating the timezone is a real IANA zone."""
    try:
        ZoneInfo(args.timezone)
    except ZoneInfoNotFoundError as exc:
        raise InvalidTimezoneError(f"unknown timezone: {args.timezone}") from exc

    digest_input = "|".join(
        (args.title, args.start.isoformat(), args.end.isoformat(), args.timezone)
    )
    event_id = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[:16]

    return CalendarCreateEventResult(event_id=event_id, status="confirmed")


# Deferred: tools.real.calendar_create_event imports these Args/Result classes back from this
# module, so importing it at the top would be circular.
from toolsmith.tools.real.calendar_create_event import calendar_create_event_real  # noqa: E402

registry.register(
    ToolSpec(
        name="calendar_create_event",
        description="Create a calendar event and return a deterministic confirmation id.",
        args_model=CalendarCreateEventArgs,
        returns_model=CalendarCreateEventResult,
        sandbox_fn=calendar_create_event,
        real_fn=calendar_create_event_real,
    )
)
