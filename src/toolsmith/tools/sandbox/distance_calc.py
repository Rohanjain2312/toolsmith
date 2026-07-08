"""Deterministic sandbox implementation of the distance_calc tool."""

from __future__ import annotations

from pydantic import BaseModel, Field

from toolsmith.tools.schemas import ToolSpec, registry
from toolsmith.utils import haversine_km


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
    distance_km = haversine_km(args.lat1, args.lon1, args.lat2, args.lon2)
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
