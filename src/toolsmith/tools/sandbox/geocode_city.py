"""Sandbox implementation of the geocode_city tool: deterministic city name to lat/lon lookup."""

from __future__ import annotations

from pydantic import BaseModel, Field

from toolsmith.tools.schemas import ToolSpec, registry


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
    """Raised when the requested city is not present in the sandbox fixture."""


# Inline fixture; replaced by loading tools/sandbox/worlddata/cities.json in task P1-T14.
_CITY_FIXTURE: dict[str, dict[str, float | str]] = {
    "paris": {
        "city": "Paris",
        "lat": 48.8566,
        "lon": 2.3522,
        "country": "France",
        "timezone": "Europe/Paris",
    },
    "tokyo": {
        "city": "Tokyo",
        "lat": 35.6762,
        "lon": 139.6503,
        "country": "Japan",
        "timezone": "Asia/Tokyo",
    },
    "new york": {
        "city": "New York",
        "lat": 40.7128,
        "lon": -74.0060,
        "country": "United States",
        "timezone": "America/New_York",
    },
    "london": {
        "city": "London",
        "lat": 51.5074,
        "lon": -0.1278,
        "country": "United Kingdom",
        "timezone": "Europe/London",
    },
    "sydney": {
        "city": "Sydney",
        "lat": -33.8688,
        "lon": 151.2093,
        "country": "Australia",
        "timezone": "Australia/Sydney",
    },
    "cairo": {
        "city": "Cairo",
        "lat": 30.0444,
        "lon": 31.2357,
        "country": "Egypt",
        "timezone": "Africa/Cairo",
    },
    "rio de janeiro": {
        "city": "Rio de Janeiro",
        "lat": -22.9068,
        "lon": -43.1729,
        "country": "Brazil",
        "timezone": "America/Sao_Paulo",
    },
    "reykjavik": {
        "city": "Reykjavik",
        "lat": 64.1466,
        "lon": -21.9426,
        "country": "Iceland",
        "timezone": "Atlantic/Reykjavik",
    },
}


def geocode_city(args: GeocodeCityArgs) -> GeocodeCityResult:
    """Look up a city's coordinates, country, and timezone (case-insensitive, trims whitespace)."""
    key = args.city.strip().lower()
    entry = _CITY_FIXTURE.get(key)
    if entry is None:
        raise CityNotFoundError(f"city not found in sandbox fixture: {args.city!r}")
    return GeocodeCityResult(**entry)  # type: ignore[arg-type]


registry.register(
    ToolSpec(
        name="geocode_city",
        description="Resolve a city name to its latitude, longitude, country, and IANA timezone.",
        args_model=GeocodeCityArgs,
        returns_model=GeocodeCityResult,
        sandbox_fn=geocode_city,
    )
)
