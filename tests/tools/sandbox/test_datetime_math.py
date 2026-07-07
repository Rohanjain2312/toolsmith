"""Tests for the datetime_math sandbox tool."""

from datetime import date

import pytest
from pydantic import ValidationError

from toolsmith.tools.sandbox.datetime_math import (
    SANDBOX_TODAY,
    DatetimeMathArgs,
    datetime_math,
)


def test_add_days_happy_path_from_default_base() -> None:
    args = DatetimeMathArgs(operation="add_days", days=10)
    result = datetime_math(args)
    assert result.result_date == SANDBOX_TODAY + date.resolution * 10
    assert result.weekday_name == result.result_date.strftime("%a")


def test_next_weekday_when_base_already_on_that_weekday_skips_to_next_week() -> None:
    # SANDBOX_TODAY (2026-09-01) is a Tuesday.
    args = DatetimeMathArgs(operation="next_weekday", weekday="Tue")
    result = datetime_math(args)
    assert result.result_date == date(2026, 9, 8)
    assert result.result_date != SANDBOX_TODAY
    assert result.weekday_name == "Tue"


def test_next_weekday_a_few_days_after_base() -> None:
    # SANDBOX_TODAY is Tuesday 2026-09-01; the next Friday is 2026-09-04.
    args = DatetimeMathArgs(operation="next_weekday", weekday="Fri")
    result = datetime_math(args)
    assert result.result_date == date(2026, 9, 4)
    assert result.weekday_name == "Fri"


def test_weekday_of_reports_correct_weekday_for_sandbox_today() -> None:
    assert SANDBOX_TODAY.weekday() == 1  # Monday=0 ... Tuesday=1
    args = DatetimeMathArgs(operation="weekday_of", base_date=SANDBOX_TODAY)
    result = datetime_math(args)
    assert result.result_date == SANDBOX_TODAY
    assert result.weekday_name == "Tue"


def test_add_days_missing_days_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        DatetimeMathArgs(operation="add_days")


def test_next_weekday_missing_weekday_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        DatetimeMathArgs(operation="next_weekday")


def test_deterministic_repeated_calls_match() -> None:
    args = DatetimeMathArgs(operation="add_days", days=42, base_date=date(2026, 1, 1))
    result_a = datetime_math(args)
    result_b = datetime_math(args)
    assert result_a == result_b
