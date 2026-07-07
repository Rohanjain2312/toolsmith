"""Tests for the flight_search sandbox tool."""

from datetime import date

import pytest
from pydantic import ValidationError

from toolsmith.tools.sandbox.flight_search import (
    FlightSearchArgs,
    flight_search,
)


def test_known_route_returns_expected_flights() -> None:
    result = flight_search(FlightSearchArgs(origin="JFK", dest="LHR", date=date(2026, 9, 10)))
    assert len(result.flights) >= 1
    flight = result.flights[0]
    assert flight.id
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
    # No fixture route has origin == dest; the pattern permits it, so we
    # document that this boundary case simply yields no matches (not an error).
    result = flight_search(FlightSearchArgs(origin="JFK", dest="JFK", date=date(2026, 9, 10)))
    assert result.flights == []


def test_deterministic_repeated_calls_match() -> None:
    args = FlightSearchArgs(origin="LAX", dest="JFK", date=date(2026, 9, 8))
    first = flight_search(args)
    second = flight_search(args)
    assert first == second
    assert [f.id for f in first.flights] == [f.id for f in second.flights]
