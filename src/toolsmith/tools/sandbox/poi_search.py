"""Deterministic sandbox implementation of the poi_search tool."""

from __future__ import annotations

import functools
import json
import math
from pathlib import Path

from pydantic import BaseModel, Field

from toolsmith.tools.schemas import ToolSpec, registry

EARTH_RADIUS_KM = 6371.0
_WORLDDATA_DIR = Path(__file__).parent / "worlddata"


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


@functools.cache
def _load_pois() -> list[dict[str, object]]:
    """Load and cache the generated POI list."""
    return json.loads((_WORLDDATA_DIR / "pois.json").read_text())


def poi_search(args: PoiSearchArgs) -> PoiSearchResult:
    """Return world-data POIs matching category within radius_km, sorted by distance."""
    matches: list[Poi] = []
    for entry in _load_pois():
        if str(entry["category"]).lower() != args.category.lower():
            continue
        lat, lon = float(entry["lat"]), float(entry["lon"])
        distance_km = _haversine_km(args.lat, args.lon, lat, lon)
        if distance_km <= args.radius_km:
            matches.append(
                Poi(
                    name=str(entry["name"]),
                    lat=lat,
                    lon=lon,
                    category=str(entry["category"]),
                    distance_km=distance_km,
                )
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
