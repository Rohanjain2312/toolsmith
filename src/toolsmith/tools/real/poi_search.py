"""Real-mode poi_search implementation backed by the free OpenTripMap API."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from toolsmith.tools.sandbox.poi_search import Poi, PoiSearchArgs, PoiSearchResult
from toolsmith.utils import haversine_km

BASE_URL = "https://api.opentripmap.com/0.1/en/places/radius"


class MissingApiKeyError(RuntimeError):
    """Raised when the OPENTRIPMAP_API_KEY environment variable is not set."""


class PoiSearchRequestError(RuntimeError):
    """Raised when the OpenTripMap API request fails or returns an unexpected payload."""


def _fetch_json(url: str) -> dict:
    """Fetch and parse a JSON payload from `url`. Isolated so tests can monkeypatch it."""
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.loads(response.read())


def poi_search_real(args: PoiSearchArgs) -> PoiSearchResult:
    """Search real-world points of interest near a lat/lon via the OpenTripMap API."""
    api_key = os.environ.get("OPENTRIPMAP_API_KEY")
    if not api_key:
        raise MissingApiKeyError("OPENTRIPMAP_API_KEY environment variable is not set")

    radius_m = int(args.radius_km * 1000)
    url = (
        f"{BASE_URL}?radius={radius_m}&lon={args.lon}&lat={args.lat}"
        f"&kinds={args.category}&apikey={api_key}&format=json"
    )

    try:
        data = _fetch_json(url)
    except (OSError, urllib.error.URLError) as exc:
        raise PoiSearchRequestError(
            f"failed to fetch POI data from OpenTripMap: {exc}"
        ) from exc

    try:
        pois: list[Poi] = []
        for entry in data:
            name = entry.get("name")
            if not name:
                continue
            point = entry["point"]
            lat, lon = float(point["lat"]), float(point["lon"])
            distance_km = haversine_km(args.lat, args.lon, lat, lon)
            pois.append(
                Poi(
                    name=name,
                    lat=lat,
                    lon=lon,
                    category=args.category,
                    distance_km=distance_km,
                )
            )
    except (KeyError, TypeError) as exc:
        raise PoiSearchRequestError(
            f"unexpected response shape from OpenTripMap: {exc}"
        ) from exc

    pois.sort(key=lambda poi: poi.distance_km)
    return PoiSearchResult(pois=pois)
