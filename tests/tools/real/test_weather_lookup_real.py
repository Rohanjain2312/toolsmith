"""Tests for the real-mode weather_lookup Open-Meteo client (network calls always mocked)."""

from __future__ import annotations

from datetime import date

import pytest

from toolsmith.tools.real import weather_lookup as weather_lookup_real_module
from toolsmith.tools.real.weather_lookup import (
    OpenMeteoRequestError,
    weather_lookup_real,
)
from toolsmith.tools.sandbox.weather_lookup import WeatherLookupArgs


def _args() -> WeatherLookupArgs:
    return WeatherLookupArgs(lat=38.9, lon=-77.0, date=date(2026, 9, 3))


def test_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        weather_lookup_real_module,
        "_fetch_json",
        lambda url: {"daily": {"temperature_2m_max": [22.5], "weather_code": [1]}},
    )

    result = weather_lookup_real(_args())

    assert result.summary == "Mainly clear"
    assert result.temp_c == 22.5


def test_unmapped_weather_code_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        weather_lookup_real_module,
        "_fetch_json",
        lambda url: {"daily": {"temperature_2m_max": [10.0], "weather_code": [999]}},
    )

    result = weather_lookup_real(_args())

    assert result.summary == "Weather code 999"
    assert result.temp_c == 10.0


def test_missing_key_raises_open_meteo_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        weather_lookup_real_module,
        "_fetch_json",
        lambda url: {"daily": {}},
    )

    with pytest.raises(OpenMeteoRequestError):
        weather_lookup_real(_args())


def test_url_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_urls: list[str] = []

    def fake_fetch_json(url: str) -> dict:
        captured_urls.append(url)
        return {"daily": {"temperature_2m_max": [15.0], "weather_code": [0]}}

    monkeypatch.setattr(weather_lookup_real_module, "_fetch_json", fake_fetch_json)

    weather_lookup_real(_args())

    assert len(captured_urls) == 1
    url = captured_urls[0]
    assert "latitude=38.9" in url
    assert "longitude=-77.0" in url
    assert "start_date=2026-09-03" in url
    assert "end_date=2026-09-03" in url


def test_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        weather_lookup_real_module,
        "_fetch_json",
        lambda url: {"daily": {"temperature_2m_max": [18.2], "weather_code": [61]}},
    )

    result_one = weather_lookup_real(_args())
    result_two = weather_lookup_real(_args())

    assert result_one == result_two
