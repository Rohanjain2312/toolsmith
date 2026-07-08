"""Shared helpers used across ToolSmith subpackages."""

from __future__ import annotations

import math

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two lat/lon points (degrees), via the haversine formula.

    Clamps the intermediate `a` term to [0, 1] before the sqrt: floating-point rounding can push
    it fractionally above 1.0 for near-antipodal points, which would otherwise raise
    `ValueError: math domain error` from `math.sqrt(1 - a)`.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    a = min(1.0, max(0.0, a))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_KM * c
