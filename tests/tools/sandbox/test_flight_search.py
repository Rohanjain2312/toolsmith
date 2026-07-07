"""Tests for the flight_search sandbox tool."""

import json
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from toolsmith.tools.sandbox.flight_search import (
    FlightSearchArgs,
    flight_search,
)

_WORLDDATA_DIR = Path("src/toolsmith/tools/sandbox/worlddata")
_FIRST_FLIGHT = json.loads((_WORLDDATA_DIR / "flights.json").read_text())[0]


def test_known_route_returns_expected_flights() -> None:
    result = flight_search(
        FlightSearchArgs(
            origin=_FIRST_FLIGHT["origin"], dest=_FIRST_FLIGHT["dest"], date=date(2026, 9, 10)
        )
    )
    assert len(result.flights) >= 1
    flight = next(f for f in result.flights if f.id == _FIRST_FLIGHT["id"])
    assert flight.price > 0
    assert len(flight.currency) == 3
    assert flight.arrive > flight.depart


def test_unmapped_route_returns_empty_list_not_error() -> None:
    result = flight_search(FlightSearchArgs(origin="ZZZ", dest="YYY", date=date(2026, 9, 10)))
    assert result.flights == []


def test_lowercase_origin_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        FlightSearchArgs(origin="jfk", dest="LHR", date=date(2026, 9, 10))


def test_wrong_length_origin_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        FlightSearchArgs(origin="JF", dest="LHR", date=date(2026, 9, 10))


def test_origin_equals_dest_returns_empty_list() -> None:
    # The generator never emits a route with origin == dest; the arg pattern
    # permits it, so this boundary case simply yields no matches (not an error).
    result = flight_search(
        FlightSearchArgs(origin="ZZZ", dest="ZZZ", date=date(2026, 9, 10))
    )
    assert result.flights == []


def test_deterministic_repeated_calls_match() -> None:
    args = FlightSearchArgs(
        origin=_FIRST_FLIGHT["origin"], dest=_FIRST_FLIGHT["dest"], date=date(2026, 9, 8)
    )
    first = flight_search(args)
    second = flight_search(args)
    assert first == second
    assert [f.id for f in first.flights] == [f.id for f in second.flights]
