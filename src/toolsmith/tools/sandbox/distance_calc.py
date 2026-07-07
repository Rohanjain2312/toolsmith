"""Deterministic sandbox implementation of the distance_calc tool."""

from __future__ import annotations

import math

from pydantic import BaseModel, Field

from toolsmith.tools.schemas import ToolSpec, registry

_EARTH_RADIUS_KM = 6371.0


class DistanceCalcArgs(BaseModel):
    """Arguments for a distance_calc call."""

    lat1: float = Field(ge=-90, le=90)
    lon1: float = Field(ge=-180, le=180)
    lat2: float = Field(ge=-90, le=90)
    lon2: float = Field(ge=-180, le=180)


class DistanceCalcResult(BaseModel):
    """Result of a distance_calc call."""

    distance_km: float


def distance_calc(args: DistanceCalcArgs) -> DistanceCalcResult:
    """Compute the great-circle distance in km between two lat/lon points via haversine."""
    lat1_rad = math.radians(args.lat1)
    lon1_rad = math.radians(args.lon1)
    lat2_rad = math.radians(args.lat2)
    lon2_rad = math.radians(args.lon2)

    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance_km = _EARTH_RADIUS_KM * c

    return DistanceCalcResult(distance_km=distance_km)


registry.register(
    ToolSpec(
        name="distance_calc",
        description="Compute the great-circle distance in km between two lat/lon points.",
        args_model=DistanceCalcArgs,
        returns_model=DistanceCalcResult,
        sandbox_fn=distance_calc,
    )
)
