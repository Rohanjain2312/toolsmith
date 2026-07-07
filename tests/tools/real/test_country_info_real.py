"""Tests for the real-mode country_info REST Countries client (network calls always mocked)."""

from __future__ import annotations

import pytest

from toolsmith.tools.real import country_info as country_info_real_module
from toolsmith.tools.real.country_info import (
    CountryInfoRequestError,
    CountryNotFoundError,
    country_info_real,
)
from toolsmith.tools.sandbox.country_info import CountryInfoArgs


def _args() -> CountryInfoArgs:
    return CountryInfoArgs(country="Japan")


def test_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        country_info_real_module,
        "_fetch_json",
        lambda url: [
            {
                "name": {"common": "Japan"},
                "currencies": {"JPY": {"name": "Japanese yen", "symbol": "¥"}},
                "languages": {"jpn": "Japanese"},
            }
        ],
    )

    result = country_info_real(_args())

    assert result.currency == "JPY"
    assert "Japanese" in result.languages


def test_empty_list_raises_country_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(country_info_real_module, "_fetch_json", lambda url: [])

    with pytest.raises(CountryNotFoundError):
        country_info_real(_args())


def test_missing_key_raises_country_info_request_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        country_info_real_module,
        "_fetch_json",
        lambda url: [{"name": {"common": "Japan"}, "languages": {"jpn": "Japanese"}}],
    )

    with pytest.raises(CountryInfoRequestError):
        country_info_real(_args())


def test_placeholder_fields_are_fixed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        country_info_real_module,
        "_fetch_json",
        lambda url: [
            {
                "name": {"common": "Japan"},
                "currencies": {"JPY": {"name": "Japanese yen", "symbol": "¥"}},
                "languages": {"jpn": "Japanese"},
            }
        ],
    )

    result = country_info_real(_args())

    assert result.plug_type == "Not available from REST Countries API"
    assert result.visa_note == "Check destination visa requirements before travel."


def test_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        country_info_real_module,
        "_fetch_json",
        lambda url: [
            {
                "name": {"common": "Japan"},
                "currencies": {"JPY": {"name": "Japanese yen", "symbol": "¥"}},
                "languages": {"jpn": "Japanese"},
            }
        ],
    )

    result_one = country_info_real(_args())
    result_two = country_info_real(_args())

    assert result_one == result_two
