"""Real-mode weather_lookup implementation backed by the free Open-Meteo forecast API."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from toolsmith.tools.sandbox.weather_lookup import WeatherLookupArgs, WeatherLookupResult

_FORECAST_URL = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude={lat}&longitude={lon}"
    "&daily=temperature_2m_max,weather_code"
    "&start_date={date}&end_date={date}&timezone=UTC"
)

_WMO_SUMMARIES: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Fog",
    51: "Drizzle",
    53: "Drizzle",
    55: "Drizzle",
    61: "Rain",
    63: "Rain",
    65: "Rain",
    71: "Snow",
    73: "Snow",
    75: "Snow",
    80: "Rain showers",
    81: "Rain showers",
    82: "Rain showers",
    95: "Thunderstorm",
}


class OpenMeteoRequestError(RuntimeError):
    """Raised when the Open-Meteo API request fails or returns an unexpected payload."""


def _fetch_json(url: str) -> dict:
    """Fetch and parse a JSON payload from `url`. Isolated so tests can monkeypatch it."""
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.loads(response.read())


def _summary_for_code(code: int) -> str:
    """Map a WMO weather code to a short human-readable summary."""
    return _WMO_SUMMARIES.get(code, f"Weather code {code}")


def weather_lookup_real(args: WeatherLookupArgs) -> WeatherLookupResult:
    """Look up real-world weather for a lat/lon and date via the Open-Meteo forecast API."""
    url = _FORECAST_URL.format(lat=args.lat, lon=args.lon, date=args.date.isoformat())

    try:
        data = _fetch_json(url)
    except (OSError, urllib.error.URLError) as exc:
        raise OpenMeteoRequestError(f"failed to fetch weather data from Open-Meteo: {exc}") from exc

    try:
        temp_c = data["daily"]["temperature_2m_max"][0]
        code = data["daily"]["weather_code"][0]
    except (KeyError, IndexError) as exc:
        raise OpenMeteoRequestError(
            f"unexpected response shape from Open-Meteo: missing {exc}"
        ) from exc

    return WeatherLookupResult(
        lat=args.lat,
        lon=args.lon,
        date=args.date,
        summary=_summary_for_code(code),
        temp_c=temp_c,
    )
