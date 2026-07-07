"""Tests for the calendar_create_event sandbox tool."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from toolsmith.tools.sandbox.calendar_create_event import (
    CalendarCreateEventArgs,
    InvalidTimezoneError,
    calendar_create_event,
)


def test_happy_path_returns_confirmed_status_and_event_id() -> None:
    result = calendar_create_event(
        CalendarCreateEventArgs(
            title="Flight to Tokyo",
            start=datetime(2026, 9, 1, 9, 0),
            end=datetime(2026, 9, 1, 10, 0),
            timezone="America/New_York",
        )
    )
    assert result.status == "confirmed"
    assert len(result.event_id) > 0


def test_deterministic_repeated_calls_produce_same_event_id() -> None:
    args = CalendarCreateEventArgs(
        title="Team Sync",
        start=datetime(2026, 9, 1, 9, 0),
        end=datetime(2026, 9, 1, 9, 30),
        timezone="Europe/London",
    )
    first = calendar_create_event(args)
    second = calendar_create_event(args)
    assert first.event_id == second.event_id


def test_zero_duration_event_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        CalendarCreateEventArgs(
            title="Instant Meeting",
            start=datetime(2026, 9, 1, 9, 0),
            end=datetime(2026, 9, 1, 9, 0),
            timezone="UTC",
        )


def test_empty_title_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        CalendarCreateEventArgs(
            title="",
            start=datetime(2026, 9, 1, 9, 0),
            end=datetime(2026, 9, 1, 10, 0),
            timezone="UTC",
        )


def test_invalid_timezone_raises_invalid_timezone_error() -> None:
    args = CalendarCreateEventArgs(
        title="Mystery Meeting",
        start=datetime(2026, 9, 1, 9, 0),
        end=datetime(2026, 9, 1, 10, 0),
        timezone="Not/AZone",
    )
    with pytest.raises(InvalidTimezoneError):
        calendar_create_event(args)


def test_different_titles_produce_different_event_ids() -> None:
    start = datetime(2026, 9, 1, 9, 0)
    end = datetime(2026, 9, 1, 10, 0)
    first = calendar_create_event(
        CalendarCreateEventArgs(
            title="Meeting A", start=start, end=end, timezone="UTC"
        )
    )
    second = calendar_create_event(
        CalendarCreateEventArgs(
            title="Meeting B", start=start, end=end, timezone="UTC"
        )
    )
    assert first.event_id != second.event_id


def test_negative_duration_event_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        CalendarCreateEventArgs(
            title="Backwards Meeting",
            start=datetime(2026, 9, 1, 10, 0),
            end=datetime(2026, 9, 1, 9, 0),
            timezone="UTC",
        )
