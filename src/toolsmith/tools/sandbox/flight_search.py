"""Sandbox implementation of the flight_search tool: deterministic route-based flight lookup."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field

from toolsmith.tools.schemas import ToolSpec, registry

SANDBOX_TODAY = date(2026, 9, 1)


class FlightSearchArgs(BaseModel):
    """Input args for flight_search: origin/destination IATA codes and travel date."""

    origin: str = Field(pattern=r"^[A-Z]{3}$")
    dest: str = Field(pattern=r"^[A-Z]{3}$")
    date: date


class FlightOption(BaseModel):
    """A single bookable flight itinerary."""

    id: str
    depart: datetime
    arrive: datetime
    price: float
    currency: str


class FlightSearchResult(BaseModel):
    """Result of a flight search: zero or more matching flight options."""

    flights: list[FlightOption]


# Inline fixture; replaced by loading tools/sandbox/worlddata/flights.json (200 flights) in P1-T14.
_FLIGHT_FIXTURE: list[dict[str, str | float]] = [
    {
        "id": "FS001",
        "origin": "JFK",
        "dest": "LHR",
        "depart": "2026-09-10T19:30:00",
        "arrive": "2026-09-11T07:15:00",
        "price": 542.30,
        "currency": "USD",
    },
    {
        "id": "FS002",
        "origin": "JFK",
        "dest": "LHR",
        "depart": "2026-09-10T22:00:00",
        "arrive": "2026-09-11T09:50:00",
        "price": 489.99,
        "currency": "USD",
    },
    {
        "id": "FS003",
        "origin": "LHR",
        "dest": "CDG",
        "depart": "2026-09-12T08:05:00",
        "arrive": "2026-09-12T10:20:00",
        "price": 112.50,
        "currency": "GBP",
    },
    {
        "id": "FS004",
        "origin": "NRT",
        "dest": "SYD",
        "depart": "2026-09-15T23:10:00",
        "arrive": "2026-09-16T10:40:00",
        "price": 780.00,
        "currency": "USD",
    },
    {
        "id": "FS005",
        "origin": "LAX",
        "dest": "JFK",
        "depart": "2026-09-08T06:45:00",
        "arrive": "2026-09-08T15:10:00",
        "price": 325.75,
        "currency": "USD",
    },
    {
        "id": "FS006",
        "origin": "LAX",
        "dest": "JFK",
        "depart": "2026-09-08T13:20:00",
        "arrive": "2026-09-08T21:55:00",
        "price": 298.40,
        "currency": "USD",
    },
    {
        "id": "FS007",
        "origin": "CDG",
        "dest": "FCO",
        "depart": "2026-09-14T11:00:00",
        "arrive": "2026-09-14T13:05:00",
        "price": 145.20,
        "currency": "EUR",
    },
    {
        "id": "FS008",
        "origin": "FCO",
        "dest": "CDG",
        "depart": "2026-09-18T16:30:00",
        "arrive": "2026-09-18T18:35:00",
        "price": 152.00,
        "currency": "EUR",
    },
    {
        "id": "FS009",
        "origin": "SYD",
        "dest": "NRT",
        "depart": "2026-09-20T09:15:00",
        "arrive": "2026-09-20T18:00:00",
        "price": 812.60,
        "currency": "AUD",
    },
    {
        "id": "FS010",
        "origin": "JFK",
        "dest": "CDG",
        "depart": "2026-09-11T20:00:00",
        "arrive": "2026-09-12T09:30:00",
        "price": 601.15,
        "currency": "USD",
    },
]


def flight_search(args: FlightSearchArgs) -> FlightSearchResult:
    """Return fixture flights matching the exact origin+dest route (empty list if none match)."""
    flights = [
        FlightOption(
            id=str(entry["id"]),
            depart=datetime.fromisoformat(str(entry["depart"])),
            arrive=datetime.fromisoformat(str(entry["arrive"])),
            price=float(entry["price"]),
            currency=str(entry["currency"]),
        )
        for entry in _FLIGHT_FIXTURE
        if entry["origin"] == args.origin and entry["dest"] == args.dest
    ]
    return FlightSearchResult(flights=flights)


registry.register(
    ToolSpec(
        name="flight_search",
        description="Search for flights between two IATA airport codes on a given date.",
        args_model=FlightSearchArgs,
        returns_model=FlightSearchResult,
        sandbox_fn=flight_search,
    )
)
