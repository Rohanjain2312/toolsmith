"""Tests for the country_info sandbox tool."""

import pytest
from pydantic import ValidationError

from toolsmith.tools.sandbox.country_info import (
    CountryInfoArgs,
    CountryNotFoundError,
    country_info,
)


def test_known_country_returns_expected_result() -> None:
    result = country_info(CountryInfoArgs(country="Japan"))
    assert result.country == "Japan"
    assert result.currency == "JPY"
    assert result.languages == ["Japanese"]
    assert result.plug_type == "Type A/B"


def test_case_insensitive_lookup_matches_canonical() -> None:
    lower = country_info(CountryInfoArgs(country="japan"))
    upper = country_info(CountryInfoArgs(country="JAPAN"))
    canonical = country_info(CountryInfoArgs(country="Japan"))
    assert lower == upper == canonical


def test_empty_country_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        CountryInfoArgs(country="")


def test_unknown_country_raises_country_not_found() -> None:
    with pytest.raises(CountryNotFoundError):
        country_info(CountryInfoArgs(country="Narnia"))


def test_deterministic_repeated_calls_match() -> None:
    first = country_info(CountryInfoArgs(country="Brazil"))
    second = country_info(CountryInfoArgs(country="Brazil"))
    assert first == second


def test_whitespace_padded_country_still_resolves() -> None:
    result = country_info(CountryInfoArgs(country="  Iceland  "))
    assert result.country == "Iceland"
    assert result.currency == "ISK"
