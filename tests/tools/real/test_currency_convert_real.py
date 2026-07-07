"""Tests for the real-mode currency_convert Frankfurter client (network calls always mocked)."""

from __future__ import annotations

import pytest

from toolsmith.tools.real import currency_convert as currency_convert_real_module
from toolsmith.tools.real.currency_convert import (
    CurrencyConvertRequestError,
    currency_convert_real,
)
from toolsmith.tools.sandbox.currency_convert import CurrencyConvertArgs


def test_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        currency_convert_real_module,
        "_fetch_json",
        lambda url: {"amount": 1.0, "base": "USD", "date": "2026-09-01", "rates": {"EUR": 0.92}},
    )

    result = currency_convert_real(
        CurrencyConvertArgs(amount=100, from_currency="USD", to_currency="EUR")
    )

    assert result.rate == 0.92
    assert result.converted == pytest.approx(92.0)


def test_same_currency_short_circuit(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_called(url: str) -> dict:
        raise AssertionError("_fetch_json should not be called for same-currency conversion")

    monkeypatch.setattr(currency_convert_real_module, "_fetch_json", fail_if_called)

    result = currency_convert_real(
        CurrencyConvertArgs(amount=50, from_currency="GBP", to_currency="GBP")
    )

    assert result.rate == 1.0
    assert result.converted == 50.0


def test_missing_currency_raises_currency_convert_request_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        currency_convert_real_module,
        "_fetch_json",
        lambda url: {"amount": 1.0, "base": "USD", "date": "2026-09-01", "rates": {}},
    )

    with pytest.raises(CurrencyConvertRequestError):
        currency_convert_real(
            CurrencyConvertArgs(amount=100, from_currency="USD", to_currency="EUR")
        )


def test_url_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_urls: list[str] = []

    def fake_fetch_json(url: str) -> dict:
        captured_urls.append(url)
        return {"amount": 1.0, "base": "USD", "date": "2026-09-01", "rates": {"EUR": 0.92}}

    monkeypatch.setattr(currency_convert_real_module, "_fetch_json", fake_fetch_json)

    currency_convert_real(
        CurrencyConvertArgs(amount=100, from_currency="USD", to_currency="EUR")
    )

    assert len(captured_urls) == 1
    url = captured_urls[0]
    assert "base=USD" in url
    assert "symbols=EUR" in url


def test_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        currency_convert_real_module,
        "_fetch_json",
        lambda url: {"amount": 1.0, "base": "USD", "date": "2026-09-01", "rates": {"EUR": 0.92}},
    )

    args = CurrencyConvertArgs(amount=100, from_currency="USD", to_currency="EUR")
    result_one = currency_convert_real(args)
    result_two = currency_convert_real(args)

    assert result_one == result_two
