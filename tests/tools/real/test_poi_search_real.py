"""Tests for the real-mode poi_search OpenTripMap client (network calls always mocked)."""

from __future__ import annotations

import pytest

from toolsmith.tools.real import poi_search as poi_search_real_module
from toolsmith.tools.real.poi_search import MissingApiKeyError, poi_search_real
from toolsmith.tools.sandbox.poi_search import PoiSearchArgs


def _args() -> PoiSearchArgs:
    return PoiSearchArgs(lat=48.8566, lon=2.3522, category="museums", radius_km=5)


def test_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENTRIPMAP_API_KEY", "fake-test-key")
    monkeypatch.setattr(
        poi_search_real_module,
        "_fetch_json",
        lambda url: [
            {"name": "Test Museum", "point": {"lat": 48.86, "lon": 2.35}, "kinds": "museums"}
        ],
    )

    result = poi_search_real(_args())

    assert len(result.pois) == 1
    assert result.pois[0].name == "Test Museum"
    assert result.pois[0].distance_km > 0


def test_unnamed_entries_filtered_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENTRIPMAP_API_KEY", "fake-test-key")
    monkeypatch.setattr(
        poi_search_real_module,
        "_fetch_json",
        lambda url: [
            {"name": "", "point": {"lat": 48.861, "lon": 2.351}, "kinds": "museums"},
            {"point": {"lat": 48.862, "lon": 2.352}, "kinds": "museums"},
            {"name": "Test Museum", "point": {"lat": 48.86, "lon": 2.35}, "kinds": "museums"},
        ],
    )

    result = poi_search_real(_args())

    assert len(result.pois) == 1
    assert result.pois[0].name == "Test Museum"


def test_missing_api_key_raises_without_fetching(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENTRIPMAP_API_KEY", raising=False)

    def _fail_if_called(url: str) -> dict:
        raise AssertionError("_fetch_json should not be called when API key is missing")

    monkeypatch.setattr(poi_search_real_module, "_fetch_json", _fail_if_called)

    with pytest.raises(MissingApiKeyError):
        poi_search_real(_args())


def test_results_sorted_by_ascending_distance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENTRIPMAP_API_KEY", "fake-test-key")
    monkeypatch.setattr(
        poi_search_real_module,
        "_fetch_json",
        lambda url: [
            {"name": "Far Place", "point": {"lat": 48.90, "lon": 2.40}, "kinds": "museums"},
            {"name": "Near Place", "point": {"lat": 48.857, "lon": 2.353}, "kinds": "museums"},
        ],
    )

    result = poi_search_real(_args())

    assert [poi.name for poi in result.pois] == ["Near Place", "Far Place"]
    assert result.pois[0].distance_km < result.pois[1].distance_km


def test_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENTRIPMAP_API_KEY", "fake-test-key")
    monkeypatch.setattr(
        poi_search_real_module,
        "_fetch_json",
        lambda url: [
            {"name": "Test Museum", "point": {"lat": 48.86, "lon": 2.35}, "kinds": "museums"}
        ],
    )

    result_one = poi_search_real(_args())
    result_two = poi_search_real(_args())

    assert result_one == result_two
