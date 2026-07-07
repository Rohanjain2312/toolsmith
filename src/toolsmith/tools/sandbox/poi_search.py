"""Deterministic sandbox implementation of the poi_search tool."""

from __future__ import annotations

import math

from pydantic import BaseModel, Field

from toolsmith.tools.schemas import ToolSpec, registry

EARTH_RADIUS_KM = 6371.0


class PoiSearchArgs(BaseModel):
    """Arguments for a poi_search call."""

    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)
    category: str = Field(min_length=1)
    radius_km: float = Field(gt=0)


class Poi(BaseModel):
    """A single point of interest with its distance from the search origin."""

    name: str
    lat: float
    lon: float
    category: str
    distance_km: float


class PoiSearchResult(BaseModel):
    """Result of a poi_search call."""

    pois: list[Poi]


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


# Fixed synthetic fixture of ~10 POIs near a handful of reference cities; replaced
# by loading src/toolsmith/tools/sandbox/worlddata/pois.json in task P1-T14.
_POIS: list[tuple[str, float, float, str]] = [
    ("Louvre Museum", 48.8606, 2.3376, "museum"),
    ("Jardin du Luxembourg", 48.8462, 2.3372, "park"),
    ("Le Petit Bistro", 48.8530, 2.3499, "restaurant"),
    ("Senso-ji Temple", 35.7148, 139.7967, "temple"),
    ("Tsukiji Market", 35.6654, 139.7707, "market"),
    ("Central Park", 40.7829, -73.9654, "park"),
    ("Statue of Liberty", 40.6892, -74.0445, "landmark"),
    ("British Museum", 51.5194, -0.1270, "museum"),
    ("Hyde Park", 51.5073, -0.1657, "park"),
    ("Colosseum", 41.8902, 12.4922, "landmark"),
]


def poi_search(args: PoiSearchArgs) -> PoiSearchResult:
    """Return fixture POIs matching category within radius_km, sorted by distance."""
    matches: list[Poi] = []
    for name, lat, lon, category in _POIS:
        if category.lower() != args.category.lower():
            continue
        distance_km = _haversine_km(args.lat, args.lon, lat, lon)
        if distance_km <= args.radius_km:
            matches.append(
                Poi(name=name, lat=lat, lon=lon, category=category, distance_km=distance_km)
            )

    matches.sort(key=lambda poi: poi.distance_km)
    return PoiSearchResult(pois=matches)


registry.register(
    ToolSpec(
        name="poi_search",
        description="Search deterministic fixture points of interest by category and radius.",
        args_model=PoiSearchArgs,
        returns_model=PoiSearchResult,
        sandbox_fn=poi_search,
    )
)
