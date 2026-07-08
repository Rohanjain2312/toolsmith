"""Tests for the flight_search sandbox tool."""

import json
from datetime import date, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from toolsmith.tools.sandbox.flight_search import (
    FlightSearchArgs,
    flight_search,
)

_WORLDDATA_DIR = Path("src/toolsmith/tools/sandbox/worlddata")
_ALL_FLIGHTS = json.loads((_WORLDDATA_DIR / "flights.json").read_text())
_FIRST_FLIGHT = _ALL_FLIGHTS[0]
_FIRST_FLIGHT_DATE = datetime.fromisoformat(_FIRST_FLIGHT["depart"]).date()


def test_known_route_returns_expected_flights() -> None:
    result = flight_search(
        FlightSearchArgs(
            origin=_FIRST_FLIGHT["origin"], dest=_FIRST_FLIGHT["dest"], date=_FIRST_FLIGHT_DATE
        )
    )
    assert len(result.flights) >= 1
    flight = next(f for f in result.flights if f.id == _FIRST_FLIGHT["id"])
    assert flight.price > 0
    assert len(flight.currency) == 3
    assert flight.arrive > flight.depart


def test_date_filters_out_flights_on_other_dates() -> None:
    # TOK->AUC has two flights in the fixture data on genuinely different dates
    # (FL0008 on 2026-09-08, FL0043 on 2026-09-11); querying one date must not
    # return the other route's flight.
    same_route = [
        f for f in _ALL_FLIGHTS if f["origin"] == "TOK" and f["dest"] == "AUC"
    ]
    assert len(same_route) == 2
    first, second = same_route
    first_date = datetime.fromisoformat(first["depart"]).date()
    second_date = datetime.fromisoformat(second["depart"]).date()
    assert first_date != second_date

    result = flight_search(FlightSearchArgs(origin="TOK", dest="AUC", date=first_date))
    assert [f.id for f in result.flights] == [first["id"]]

    result = flight_search(FlightSearchArgs(origin="TOK", dest="AUC", date=second_date))
    assert [f.id for f in result.flights] == [second["id"]]


def test_date_with_no_matching_flight_returns_empty_list() -> None:
    result = flight_search(
        FlightSearchArgs(
            origin=_FIRST_FLIGHT["origin"],
            dest=_FIRST_FLIGHT["dest"],
            date=date(1999, 1, 1),
        )
    )
    assert result.flights == []


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
