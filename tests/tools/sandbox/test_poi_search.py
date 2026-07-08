"""Tests for the poi_search sandbox tool."""

import json
from collections import defaultdict
from pathlib import Path

import pytest
from pydantic import ValidationError

from toolsmith.tools.sandbox.poi_search import PoiSearchArgs, poi_search
from toolsmith.utils import haversine_km

_POIS = json.loads(Path("src/toolsmith/tools/sandbox/worlddata/pois.json").read_text())


def _two_pois_same_category() -> tuple[dict[str, object], dict[str, object]]:
    """Find two distinct POIs sharing a category, for boundary-distance tests."""
    by_category: dict[str, list[dict[str, object]]] = defaultdict(list)
    for poi in _POIS:
        by_category[poi["category"]].append(poi)
    for pois in by_category.values():
        if len(pois) >= 2:
            return pois[0], pois[1]
    raise AssertionError("expected at least one category with 2+ generated POIs")


POI_A, POI_B = _two_pois_same_category()


def test_happy_path_finds_known_poi_nearby() -> None:
    args = PoiSearchArgs(
        lat=POI_A["lat"], lon=POI_A["lon"], category=POI_A["category"], radius_km=1.0
    )
    result = poi_search(args)
    names = [poi.name for poi in result.pois]
    assert POI_A["name"] in names


def test_category_filter_excludes_non_matching_pois() -> None:
    other_category = next(p["category"] for p in _POIS if p["category"] != POI_A["category"])
    args = PoiSearchArgs(
        lat=POI_A["lat"], lon=POI_A["lon"], category=other_category, radius_km=5000.0
    )
    result = poi_search(args)
    names = [poi.name for poi in result.pois]
    assert POI_A["name"] not in names


def test_boundary_poi_just_outside_radius_excluded() -> None:
    distance = haversine_km(POI_A["lat"], POI_A["lon"], POI_B["lat"], POI_B["lon"])
    args = PoiSearchArgs(
        lat=POI_A["lat"], lon=POI_A["lon"], category=POI_A["category"], radius_km=distance - 0.5
    )
    result = poi_search(args)
    names = [poi.name for poi in result.pois]
    assert POI_B["name"] not in names
    assert POI_A["name"] in names


def test_boundary_poi_just_inside_radius_included() -> None:
    distance = haversine_km(POI_A["lat"], POI_A["lon"], POI_B["lat"], POI_B["lon"])
    args = PoiSearchArgs(
        lat=POI_A["lat"], lon=POI_A["lon"], category=POI_A["category"], radius_km=distance + 0.5
    )
    result = poi_search(args)
    names = [poi.name for poi in result.pois]
    assert POI_B["name"] in names


def test_invalid_lat_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        PoiSearchArgs(lat=999, lon=0, category="museum", radius_km=5)


def test_zero_radius_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        PoiSearchArgs(lat=0, lon=0, category="museum", radius_km=0)


def test_empty_matches_returns_empty_list_not_error() -> None:
    args = PoiSearchArgs(lat=0.0, lon=0.0, category="nonexistent-category", radius_km=1.0)
    result = poi_search(args)
    assert result.pois == []


def test_deterministic_repeated_calls_match() -> None:
    args = PoiSearchArgs(
        lat=POI_A["lat"], lon=POI_A["lon"], category=POI_A["category"], radius_km=5000.0
    )
    result_a = poi_search(args)
    result_b = poi_search(args)
    assert result_a == result_b
    assert [poi.name for poi in result_a.pois] == [poi.name for poi in result_b.pois]


def test_near_antipodal_query_point_does_not_raise() -> None:
    # Regression test for BUGFIX-T05: see test_distance_calc.py's equivalent test for the
    # underlying haversine floating-point domain-error mechanism.
    args = PoiSearchArgs(
        lat=40.628064952348524, lon=144.24374679103528, category="museum", radius_km=1.0
    )

    result = poi_search(args)

    assert result.pois == []
