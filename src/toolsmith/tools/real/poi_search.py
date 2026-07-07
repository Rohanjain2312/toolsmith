"""Real-mode poi_search implementation backed by the free OpenTripMap API."""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request

from toolsmith.tools.sandbox.poi_search import Poi, PoiSearchArgs, PoiSearchResult

EARTH_RADIUS_KM = 6371.0
BASE_URL = "https://api.opentripmap.com/0.1/en/places/radius"


class MissingApiKeyError(RuntimeError):
    """Raised when the OPENTRIPMAP_API_KEY environment variable is not set."""


class PoiSearchRequestError(RuntimeError):
    """Raised when the OpenTripMap API request fails or returns an unexpected payload."""


def _fetch_json(url: str) -> dict:
    """Fetch and parse a JSON payload from `url`. Isolated so tests can monkeypatch it."""
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.loads(response.read())


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in km between two lat/lon points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_KM * c


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
            distance_km = _haversine_km(args.lat, args.lon, lat, lon)
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
