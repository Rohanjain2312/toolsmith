"""Deterministic sandbox implementation of the poi_search tool."""

from __future__ import annotations

import functools
import json
from pathlib import Path

from pydantic import BaseModel, Field

from toolsmith.tools.schemas import ToolSpec, registry
from toolsmith.utils import haversine_km

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
        distance_km = haversine_km(args.lat, args.lon, lat, lon)
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


# Deferred: tools.real.poi_search imports these Args/Result classes back from this
# module, so importing it at the top would be circular.
from toolsmith.tools.real.poi_search import poi_search_real  # noqa: E402

registry.register(
    ToolSpec(
        name="poi_search",
        description="Search deterministic fixture points of interest by category and radius.",
        args_model=PoiSearchArgs,
        returns_model=PoiSearchResult,
        sandbox_fn=poi_search,
        real_fn=poi_search_real,
    )
)
