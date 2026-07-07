"""Sandbox implementation of the geocode_city tool: deterministic city name to lat/lon lookup."""

from __future__ import annotations

import functools
import json
from pathlib import Path

from pydantic import BaseModel, Field

from toolsmith.tools.schemas import ToolSpec, registry

_WORLDDATA_DIR = Path(__file__).parent / "worlddata"


class GeocodeCityArgs(BaseModel):
    """Input args for geocode_city: the city name to look up."""

    city: str = Field(min_length=1)


class GeocodeCityResult(BaseModel):
    """Result of geocoding a city: coordinates, country, and IANA timezone."""

    city: str
    lat: float
    lon: float
    country: str
    timezone: str


class CityNotFoundError(ValueError):
    """Raised when the requested city is not present in the sandbox world data."""


@functools.cache
def _load_cities() -> dict[str, dict[str, float | str]]:
    """Load and cache the generated city table, keyed by lowercase city name."""
    raw = json.loads((_WORLDDATA_DIR / "cities.json").read_text())
    return {entry["name"].lower(): entry for entry in raw}


def geocode_city(args: GeocodeCityArgs) -> GeocodeCityResult:
    """Look up a city's coordinates, country, and timezone (case-insensitive, trims whitespace)."""
    key = args.city.strip().lower()
    entry = _load_cities().get(key)
    if entry is None:
        raise CityNotFoundError(f"city not found in sandbox world data: {args.city!r}")
    return GeocodeCityResult(
        city=entry["name"],
        lat=entry["lat"],
        lon=entry["lon"],
        country=entry["country"],
        timezone=entry["timezone"],
    )


registry.register(
    ToolSpec(
        name="geocode_city",
        description="Resolve a city name to its latitude, longitude, country, and IANA timezone.",
        args_model=GeocodeCityArgs,
        returns_model=GeocodeCityResult,
        sandbox_fn=geocode_city,
    )
)
