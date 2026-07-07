"""Real-mode geocode_city implementation backed by the free Nominatim (OSM) geocoder."""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request

from toolsmith.tools.sandbox.geocode_city import GeocodeCityArgs, GeocodeCityResult

# Nominatim's usage policy requires a real, identifying User-Agent header on
# every request -- a generic/default one (or none at all) can get requests
# rejected or the IP blocked.
USER_AGENT = "toolsmith-agent/0.1 (https://github.com/Rohanjain2312/toolsmith)"

_SEARCH_URL = "https://nominatim.openstreetmap.org/search?q={query}&format=json&limit=1&addressdetails=1"

# Nominatim's usage policy caps free usage at 1 request/second; this module-level
# timestamp tracks the last request so `_respect_rate_limit` can self-throttle.
_last_request_at: float | None = None


class CityNotFoundError(ValueError):
    """Raised when Nominatim returns no results for the requested city."""


def _fetch_json(url: str, headers: dict[str, str]) -> dict:
    """Fetch and parse a JSON payload from `url`. Isolated so tests can monkeypatch it."""
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=10) as response:
        return json.loads(response.read())


def _respect_rate_limit() -> None:
    """Sleep if fewer than 1.0 seconds have passed since the last request, then record now."""
    global _last_request_at
    now = time.monotonic()
    if _last_request_at is not None:
        elapsed = now - _last_request_at
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
    _last_request_at = time.monotonic()


def geocode_city_real(args: GeocodeCityArgs) -> GeocodeCityResult:
    """Look up a city's coordinates, country, and approximate timezone via Nominatim."""
    _respect_rate_limit()

    query = urllib.parse.quote(args.city)
    url = _SEARCH_URL.format(query=query)
    results = _fetch_json(url, headers={"User-Agent": USER_AGENT})

    if not results:
        raise CityNotFoundError(f"city not found by Nominatim: {args.city!r}")

    entry = results[0]
    lat = float(entry["lat"])
    lon = float(entry["lon"])
    country = entry.get("address", {}).get("country", "Unknown")

    # Nominatim does not return IANA timezone names, and adding an offline
    # coordinate-to-timezone database is out of scope here, so we deliberately
    # approximate a fixed-offset zone from longitude alone (15 degrees per UTC
    # hour), matching the same approximation used in
    # `toolsmith.tools.sandbox.timezone_info`'s lat/lon path.
    offset_hours = round(lon / 15)
    timezone = f"UTC{offset_hours:+d}"

    return GeocodeCityResult(
        city=args.city,
        lat=lat,
        lon=lon,
        country=country,
        timezone=timezone,
    )
