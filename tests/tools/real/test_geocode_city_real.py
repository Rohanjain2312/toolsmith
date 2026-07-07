"""Tests for the real-mode geocode_city Nominatim client (network calls always mocked)."""

from __future__ import annotations

import urllib.parse

import pytest

from toolsmith.tools.real import geocode_city as geocode_city_real_module
from toolsmith.tools.real.geocode_city import (
    CityNotFoundError,
    geocode_city_real,
)
from toolsmith.tools.sandbox.geocode_city import GeocodeCityArgs


def _args(city: str = "Paris") -> GeocodeCityArgs:
    return GeocodeCityArgs(city=city)


def _paris_results() -> list[dict]:
    return [{"lat": "48.8566", "lon": "2.3522", "address": {"country": "France"}}]


def test_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(geocode_city_real_module, "_respect_rate_limit", lambda: None)
    monkeypatch.setattr(
        geocode_city_real_module,
        "_fetch_json",
        lambda url, headers: _paris_results(),
    )

    result = geocode_city_real(_args())

    assert result.lat == pytest.approx(48.8566)
    assert result.lon == pytest.approx(2.3522)
    assert result.country == "France"
    assert result.timezone.startswith("UTC")
    assert result.timezone[3] in ("+", "-")


def test_empty_results_raises_city_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(geocode_city_real_module, "_respect_rate_limit", lambda: None)
    monkeypatch.setattr(geocode_city_real_module, "_fetch_json", lambda url, headers: [])

    with pytest.raises(CityNotFoundError):
        geocode_city_real(_args("Nonexistentville"))


def test_url_and_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(geocode_city_real_module, "_respect_rate_limit", lambda: None)

    captured: dict[str, object] = {}

    def fake_fetch_json(url: str, headers: dict[str, str]) -> list[dict]:
        captured["url"] = url
        captured["headers"] = headers
        return [{"lat": "48.8566", "lon": "2.3522", "address": {"country": "France"}}]

    monkeypatch.setattr(geocode_city_real_module, "_fetch_json", fake_fetch_json)

    geocode_city_real(_args("Paris"))

    assert urllib.parse.quote("Paris") in captured["url"]
    headers = captured["headers"]
    assert isinstance(headers, dict)
    user_agent = headers.get("User-Agent", "")
    assert user_agent
    assert "python-requests" not in user_agent.lower()
    assert user_agent.lower() != "python-requests"


def test_rate_limit_invoked_once_per_call(monkeypatch: pytest.MonkeyPatch) -> None:
    call_count = {"n": 0}

    def fake_respect_rate_limit() -> None:
        call_count["n"] += 1

    monkeypatch.setattr(geocode_city_real_module, "_respect_rate_limit", fake_respect_rate_limit)
    monkeypatch.setattr(
        geocode_city_real_module,
        "_fetch_json",
        lambda url, headers: _paris_results(),
    )

    geocode_city_real(_args())
    geocode_city_real(_args())

    assert call_count["n"] == 2


def test_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(geocode_city_real_module, "_respect_rate_limit", lambda: None)
    monkeypatch.setattr(
        geocode_city_real_module,
        "_fetch_json",
        lambda url, headers: _paris_results(),
    )

    result_one = geocode_city_real(_args())
    result_two = geocode_city_real(_args())

    assert result_one == result_two
