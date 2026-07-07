"""Tests for the unit_convert sandbox tool."""

import pytest

from toolsmith.tools.sandbox.unit_convert import (
    PhysicallyInvalidTemperatureError,
    UnitConvertArgs,
    UnsupportedUnitError,
    unit_convert,
)


def test_happy_path_zero_celsius_to_fahrenheit() -> None:
    args = UnitConvertArgs(value=0.0, category="temperature", from_unit="C", to_unit="F")
    result = unit_convert(args)
    assert result.converted == pytest.approx(32.0)


def test_same_unit_conversion_returns_value_unchanged() -> None:
    args = UnitConvertArgs(value=42.0, category="distance", from_unit="km", to_unit="km")
    result = unit_convert(args)
    assert result.converted == pytest.approx(42.0)


def test_absolute_zero_kelvin_converts_correctly() -> None:
    to_celsius = UnitConvertArgs(value=0.0, category="temperature", from_unit="K", to_unit="C")
    to_fahrenheit = UnitConvertArgs(value=0.0, category="temperature", from_unit="K", to_unit="F")
    assert unit_convert(to_celsius).converted == pytest.approx(-273.15)
    assert unit_convert(to_fahrenheit).converted == pytest.approx(-459.67)


def test_below_absolute_zero_kelvin_raises() -> None:
    args = UnitConvertArgs(value=-1.0, category="temperature", from_unit="K", to_unit="C")
    with pytest.raises(PhysicallyInvalidTemperatureError):
        unit_convert(args)


def test_mismatched_category_and_unit_raises() -> None:
    args = UnitConvertArgs(value=10.0, category="distance", from_unit="C", to_unit="km")
    with pytest.raises(UnsupportedUnitError):
        unit_convert(args)


def test_deterministic_repeated_calls_match() -> None:
    args = UnitConvertArgs(value=100.0, category="weight", from_unit="kg", to_unit="lb")
    result_a = unit_convert(args)
    result_b = unit_convert(args)
    assert result_a == result_b


def test_distance_km_to_mi_round_trip() -> None:
    args = UnitConvertArgs(value=1.0, category="distance", from_unit="km", to_unit="mi")
    result = unit_convert(args)
    assert result.converted == pytest.approx(0.621371, rel=1e-4)
