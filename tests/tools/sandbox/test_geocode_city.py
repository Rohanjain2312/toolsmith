"""Tests for the geocode_city sandbox tool."""

import pytest
from pydantic import ValidationError

from toolsmith.tools.sandbox.geocode_city import (
    CityNotFoundError,
    GeocodeCityArgs,
    geocode_city,
)


def test_known_city_returns_expected_result() -> None:
    result = geocode_city(GeocodeCityArgs(city="Paris"))
    assert result.city == "Paris"
    assert result.lat == pytest.approx(48.8566)
    assert result.lon == pytest.approx(2.3522)
    assert result.country == "France"
    assert result.timezone == "Europe/Paris"


def test_case_insensitive_lookup_matches_canonical() -> None:
    lower = geocode_city(GeocodeCityArgs(city="paris"))
    upper = geocode_city(GeocodeCityArgs(city="PARIS"))
    canonical = geocode_city(GeocodeCityArgs(city="Paris"))
    assert lower == upper == canonical


def test_empty_city_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        GeocodeCityArgs(city="")


def test_unknown_city_raises_city_not_found() -> None:
    with pytest.raises(CityNotFoundError):
        geocode_city(GeocodeCityArgs(city="Atlantis"))


def test_deterministic_repeated_calls_match() -> None:
    first = geocode_city(GeocodeCityArgs(city="Tokyo"))
    second = geocode_city(GeocodeCityArgs(city="Tokyo"))
    assert first == second


def test_whitespace_padded_city_still_resolves() -> None:
    result = geocode_city(GeocodeCityArgs(city="  London  "))
    assert result.city == "London"
    assert result.country == "United Kingdom"
