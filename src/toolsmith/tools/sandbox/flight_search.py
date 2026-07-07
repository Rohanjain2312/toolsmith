"""Sandbox implementation of the flight_search tool: deterministic route-based flight lookup."""

from __future__ import annotations

import functools
import json
from datetime import date, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from toolsmith.tools.schemas import ToolSpec, registry

SANDBOX_TODAY = date(2026, 9, 1)
_WORLDDATA_DIR = Path(__file__).parent / "worlddata"


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


@functools.cache
def _load_flights() -> list[dict[str, str | float]]:
    """Load and cache the generated 200-flight table."""
    return json.loads((_WORLDDATA_DIR / "flights.json").read_text())


def flight_search(args: FlightSearchArgs) -> FlightSearchResult:
    """Return world-data flights matching the exact origin+dest route (empty list if none match)."""
    flights = [
        FlightOption(
            id=str(entry["id"]),
            depart=datetime.fromisoformat(str(entry["depart"])),
            arrive=datetime.fromisoformat(str(entry["arrive"])),
            price=float(entry["price"]),
            currency=str(entry["currency"]),
        )
        for entry in _load_flights()
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
