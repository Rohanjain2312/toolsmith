"""Tests for the poi_search sandbox tool."""

import pytest
from pydantic import ValidationError

from toolsmith.tools.sandbox.poi_search import (
    PoiSearchArgs,
    _haversine_km,
    poi_search,
)

LOUVRE = (48.8606, 2.3376)
BRITISH_MUSEUM = (51.5194, -0.1270)


def test_happy_path_finds_known_poi_nearby() -> None:
    lat, lon = LOUVRE
    args = PoiSearchArgs(lat=lat, lon=lon, category="museum", radius_km=5.0)
    result = poi_search(args)
    names = [poi.name for poi in result.pois]
    assert "Louvre Museum" in names


def test_category_filter_excludes_non_matching_pois() -> None:
    lat, lon = LOUVRE
    args = PoiSearchArgs(lat=lat, lon=lon, category="restaurant", radius_km=5.0)
    result = poi_search(args)
    names = [poi.name for poi in result.pois]
    assert "Le Petit Bistro" in names
    # Louvre Museum is at distance 0 but wrong category, so must be excluded.
    assert "Louvre Museum" not in names


def test_boundary_poi_just_outside_radius_excluded() -> None:
    distance = _haversine_km(*LOUVRE, *BRITISH_MUSEUM)
    lat, lon = LOUVRE
    args = PoiSearchArgs(lat=lat, lon=lon, category="museum", radius_km=distance - 0.5)
    result = poi_search(args)
    names = [poi.name for poi in result.pois]
    assert "British Museum" not in names
    assert "Louvre Museum" in names


def test_boundary_poi_just_inside_radius_included() -> None:
    distance = _haversine_km(*LOUVRE, *BRITISH_MUSEUM)
    lat, lon = LOUVRE
    args = PoiSearchArgs(lat=lat, lon=lon, category="museum", radius_km=distance + 0.5)
    result = poi_search(args)
    names = [poi.name for poi in result.pois]
    assert "British Museum" in names


def test_invalid_lat_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        PoiSearchArgs(lat=999, lon=0, category="museum", radius_km=5)


def test_zero_radius_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        PoiSearchArgs(lat=0, lon=0, category="museum", radius_km=0)


def test_empty_matches_returns_empty_list_not_error() -> None:
    args = PoiSearchArgs(lat=0.0, lon=0.0, category="museum", radius_km=1.0)
    result = poi_search(args)
    assert result.pois == []


def test_deterministic_repeated_calls_match() -> None:
    lat, lon = LOUVRE
    args = PoiSearchArgs(lat=lat, lon=lon, category="park", radius_km=5000.0)
    result_a = poi_search(args)
    result_b = poi_search(args)
    assert result_a == result_b
    assert [poi.name for poi in result_a.pois] == [poi.name for poi in result_b.pois]
