"""Tests for the timezone_info sandbox tool."""

import pytest
from pydantic import ValidationError

from toolsmith.tools.sandbox.timezone_info import (
    SANDBOX_NOW,
    TimezoneInfoArgs,
    UnknownTimezoneError,
    timezone_info,
)


def test_happy_path_named_timezone_negative_offset() -> None:
    args = TimezoneInfoArgs(timezone="America/New_York")
    result = timezone_info(args)
    assert result.timezone == "America/New_York"
    assert result.utc_offset_minutes < 0
    assert result.local_time == SANDBOX_NOW.astimezone(result.local_time.tzinfo)


def test_lat_lon_path_approximates_tokyo_positive_offset() -> None:
    args = TimezoneInfoArgs(lat=35.7, lon=139.0)
    result = timezone_info(args)
    assert result.utc_offset_minutes > 0
    assert result.timezone.startswith("UTC+")


def test_missing_timezone_and_coords_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        TimezoneInfoArgs()


def test_unknown_timezone_name_raises() -> None:
    with pytest.raises(UnknownTimezoneError):
        timezone_info(TimezoneInfoArgs(timezone="Not/AZone"))


def test_deterministic_repeated_calls_match() -> None:
    args = TimezoneInfoArgs(timezone="Asia/Tokyo")
    result_a = timezone_info(args)
    result_b = timezone_info(args)
    assert result_a == result_b


def test_deterministic_repeated_calls_match_lat_lon() -> None:
    args = TimezoneInfoArgs(lat=51.5, lon=-0.1)
    result_a = timezone_info(args)
    result_b = timezone_info(args)
    assert result_a == result_b
