"""Deterministic sandbox implementation of the unit_convert tool."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from toolsmith.tools.schemas import ToolSpec, registry

_ABSOLUTE_ZERO_C = -273.15
_ABSOLUTE_ZERO_K = 0.0
_ABSOLUTE_ZERO_F = -459.67

# Pivot-unit factors: multiply by factor to convert TO the pivot unit (meters/grams).
_DISTANCE_TO_METERS: dict[str, float] = {"km": 1000.0, "mi": 1609.344, "m": 1.0}
_WEIGHT_TO_GRAMS: dict[str, float] = {"kg": 1000.0, "lb": 453.59237, "g": 1.0}
_TEMPERATURE_UNITS = {"C", "F", "K"}


class UnsupportedUnitError(ValueError):
    """Raised when from_unit/to_unit isn't valid for the given category."""


class PhysicallyInvalidTemperatureError(ValueError):
    """Raised when a temperature is below absolute zero for its unit."""


class UnitConvertArgs(BaseModel):
    """Arguments for a unit_convert call."""

    value: float
    category: Literal["temperature", "distance", "weight"]
    from_unit: str
    to_unit: str


class UnitConvertResult(BaseModel):
    """Result of a unit_convert call."""

    value: float
    converted: float
    from_unit: str
    to_unit: str


def _to_celsius(value: float, unit: str) -> float:
    """Convert a temperature value in `unit` to Celsius."""
    if unit == "C":
        return value
    if unit == "F":
        return (value - 32) * 5 / 9
    return value - 273.15  # K


def _from_celsius(celsius: float, unit: str) -> float:
    """Convert a Celsius value to the target temperature `unit`."""
    if unit == "C":
        return celsius
    if unit == "F":
        return celsius * 9 / 5 + 32
    return celsius + 273.15  # K


def _check_temperature_bounds(value: float, unit: str) -> None:
    """Raise PhysicallyInvalidTemperatureError if value is below absolute zero for unit."""
    if unit == "K" and value < _ABSOLUTE_ZERO_K:
        raise PhysicallyInvalidTemperatureError(f"{value} K is below absolute zero")
    if unit == "C" and value < _ABSOLUTE_ZERO_C:
        raise PhysicallyInvalidTemperatureError(f"{value} C is below absolute zero")
    if unit == "F" and value < _ABSOLUTE_ZERO_F:
        raise PhysicallyInvalidTemperatureError(f"{value} F is below absolute zero")


def _convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    """Convert a temperature value between C/F/K via Celsius as the pivot."""
    if from_unit not in _TEMPERATURE_UNITS or to_unit not in _TEMPERATURE_UNITS:
        raise UnsupportedUnitError(
            f"unsupported temperature unit: {from_unit!r} or {to_unit!r}"
        )
    _check_temperature_bounds(value, from_unit)
    if from_unit == to_unit:
        return value
    celsius = _to_celsius(value, from_unit)
    return _from_celsius(celsius, to_unit)


def _convert_via_pivot(
    value: float, from_unit: str, to_unit: str, factors: dict[str, float], category: str
) -> float:
    """Convert value between units by scaling through a common pivot unit."""
    if from_unit not in factors or to_unit not in factors:
        raise UnsupportedUnitError(
            f"unsupported {category} unit: {from_unit!r} or {to_unit!r}"
        )
    if from_unit == to_unit:
        return value
    pivot_value = value * factors[from_unit]
    return pivot_value / factors[to_unit]


def unit_convert(args: UnitConvertArgs) -> UnitConvertResult:
    """Convert a value between units within temperature, distance, or weight categories."""
    if args.category == "temperature":
        converted = _convert_temperature(args.value, args.from_unit, args.to_unit)
    elif args.category == "distance":
        converted = _convert_via_pivot(
            args.value, args.from_unit, args.to_unit, _DISTANCE_TO_METERS, args.category
        )
    else:  # weight
        converted = _convert_via_pivot(
            args.value, args.from_unit, args.to_unit, _WEIGHT_TO_GRAMS, args.category
        )

    return UnitConvertResult(
        value=args.value,
        converted=converted,
        from_unit=args.from_unit,
        to_unit=args.to_unit,
    )


registry.register(
    ToolSpec(
        name="unit_convert",
        description="Convert a numeric value between units of temperature, distance, or weight.",
        args_model=UnitConvertArgs,
        returns_model=UnitConvertResult,
        sandbox_fn=unit_convert,
    )
)
