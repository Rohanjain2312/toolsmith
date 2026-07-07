"""Deterministic sandbox implementation of the weather_lookup tool."""

from __future__ import annotations

from datetime import date, timedelta

from pydantic import BaseModel, Field

from toolsmith.tools.schemas import ToolSpec, registry

SANDBOX_TODAY = date(2026, 9, 1)
FORECAST_WINDOW_DAYS = 13

# Fixed synthetic climate patterns: (summary, temp_c).
_CLIMATE_PATTERNS: list[tuple[str, float]] = [
    ("Sunny", 28.0),
    ("Partly cloudy", 22.5),
    ("Rain showers", 18.0),
    ("Overcast", 20.0),
    ("Clear skies", 25.0),
]


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


def weather_lookup(args: WeatherLookupArgs) -> WeatherLookupResult:
    """Return deterministic synthetic weather for a lat/lon and date within the sandbox window."""
    window_end = SANDBOX_TODAY + timedelta(days=FORECAST_WINDOW_DAYS)
    if not (SANDBOX_TODAY <= args.date <= window_end):
        raise WeatherLookupOutOfRangeError(
            f"date {args.date} is outside the sandbox forecast window "
            f"[{SANDBOX_TODAY}, {window_end}]"
        )

    # Deterministic index from lat/lon/day-offset; replaced by loading
    # src/toolsmith/tools/sandbox/worlddata/weather.json in task P1-T14.
    offset = (args.date - SANDBOX_TODAY).days
    index = (round(args.lat) + round(args.lon) + offset) % len(_CLIMATE_PATTERNS)
    summary, temp_c = _CLIMATE_PATTERNS[index]

    return WeatherLookupResult(
        lat=args.lat,
        lon=args.lon,
        date=args.date,
        summary=summary,
        temp_c=temp_c,
    )


registry.register(
    ToolSpec(
        name="weather_lookup",
        description="Look up a deterministic synthetic weather forecast for a lat/lon and date.",
        args_model=WeatherLookupArgs,
        returns_model=WeatherLookupResult,
        sandbox_fn=weather_lookup,
    )
)
