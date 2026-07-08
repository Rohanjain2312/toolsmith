"""Tests for the weather_lookup sandbox tool."""

from datetime import timedelta

import pytest
from pydantic import ValidationError

from toolsmith.tools.sandbox.weather_lookup import (
    SANDBOX_TODAY,
    WeatherLookupArgs,
    WeatherLookupOutOfRangeError,
    weather_lookup,
)


def test_happy_path_mid_window_date() -> None:
    args = WeatherLookupArgs(lat=38.9, lon=-77.0, date=SANDBOX_TODAY + timedelta(days=5))
    result = weather_lookup(args)
    assert isinstance(result.summary, str) and result.summary
    assert isinstance(result.temp_c, float)


def test_boundary_last_valid_day_succeeds() -> None:
    args = WeatherLookupArgs(lat=10.0, lon=10.0, date=SANDBOX_TODAY + timedelta(days=13))
    result = weather_lookup(args)
    assert result.date == SANDBOX_TODAY + timedelta(days=13)


def test_out_of_range_after_window_raises() -> None:
    args = WeatherLookupArgs(lat=10.0, lon=10.0, date=SANDBOX_TODAY + timedelta(days=14))
    with pytest.raises(WeatherLookupOutOfRangeError):
        weather_lookup(args)


def test_out_of_range_before_today_raises() -> None:
    args = WeatherLookupArgs(lat=10.0, lon=10.0, date=SANDBOX_TODAY - timedelta(days=1))
    with pytest.raises(WeatherLookupOutOfRangeError):
        weather_lookup(args)


def test_invalid_lat_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        WeatherLookupArgs(lat=999, lon=0, date=SANDBOX_TODAY)


def test_deterministic_repeated_calls_match() -> None:
    args = WeatherLookupArgs(lat=51.5, lon=-0.1, date=SANDBOX_TODAY + timedelta(days=3))
    result_a = weather_lookup(args)
    result_b = weather_lookup(args)
    assert result_a == result_b


def test_near_antipodal_query_point_does_not_raise() -> None:
    # Regression test for BUGFIX-T05: see test_distance_calc.py's equivalent test for the
    # underlying haversine floating-point domain-error mechanism (here, inside _nearest_city's
    # min() comparator).
    args = WeatherLookupArgs(
        lat=40.628064952348524, lon=144.24374679103528, date=SANDBOX_TODAY + timedelta(days=3)
    )

    result = weather_lookup(args)

    assert isinstance(result.summary, str) and result.summary
