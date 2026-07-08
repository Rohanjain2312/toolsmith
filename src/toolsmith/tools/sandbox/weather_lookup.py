"""Deterministic sandbox implementation of the weather_lookup tool."""

from __future__ import annotations

import functools
import json
from datetime import date, timedelta
from pathlib import Path

from pydantic import BaseModel, Field

from toolsmith.tools.schemas import ToolSpec, registry
from toolsmith.utils import haversine_km

SANDBOX_TODAY = date(2026, 9, 1)
FORECAST_WINDOW_DAYS = 13
_WORLDDATA_DIR = Path(__file__).parent / "worlddata"


class WeatherLookupOutOfRangeError(ValueError):
    """Raised when the requested date falls outside the sandbox's 14-day forecast window."""


class WeatherLookupArgs(BaseModel):
    """Arguments for a weather_lookup call."""

    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    date: date


class WeatherLookupResult(BaseModel):
    """Result of a weather_lookup call."""

    lat: float
    lon: float
    date: date
    summary: str
    temp_c: float


@functools.cache
def _load_world() -> tuple[list[dict[str, object]], dict[str, list[dict[str, object]]]]:
    """Load and cache the generated cities table and per-city weather table."""
    cities = json.loads((_WORLDDATA_DIR / "cities.json").read_text())
    weather = json.loads((_WORLDDATA_DIR / "weather.json").read_text())
    return cities, weather


def _nearest_city(lat: float, lon: float, cities: list[dict[str, object]]) -> dict[str, object]:
    return min(cities, key=lambda c: haversine_km(lat, lon, c["lat"], c["lon"]))


def weather_lookup(args: WeatherLookupArgs) -> WeatherLookupResult:
    """Return deterministic synthetic weather for a lat/lon and date within the sandbox window."""
    window_end = SANDBOX_TODAY + timedelta(days=FORECAST_WINDOW_DAYS)
    if not (SANDBOX_TODAY <= args.date <= window_end):
        raise WeatherLookupOutOfRangeError(
            f"date {args.date} is outside the sandbox forecast window "
            f"[{SANDBOX_TODAY}, {window_end}]"
        )

    cities, weather = _load_world()
    nearest = _nearest_city(args.lat, args.lon, cities)
    offset = (args.date - SANDBOX_TODAY).days
    day = weather[nearest["name"]][offset]

    return WeatherLookupResult(
        lat=args.lat,
        lon=args.lon,
        date=args.date,
        summary=day["summary"],
        temp_c=day["temp_c"],
    )


# Deferred: tools.real.weather_lookup imports these Args/Result classes back from this
# module, so importing it at the top would be circular.
from toolsmith.tools.real.weather_lookup import weather_lookup_real  # noqa: E402

registry.register(
    ToolSpec(
        name="weather_lookup",
        description="Look up a deterministic synthetic weather forecast for a lat/lon and date.",
        args_model=WeatherLookupArgs,
        returns_model=WeatherLookupResult,
        sandbox_fn=weather_lookup,
        real_fn=weather_lookup_real,
    )
)
