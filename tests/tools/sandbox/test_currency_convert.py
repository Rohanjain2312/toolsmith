"""Tests for the currency_convert sandbox tool."""

import pytest
from pydantic import ValidationError

from toolsmith.tools.sandbox.currency_convert import (
    CurrencyConvertArgs,
    CurrencyNotFoundError,
    currency_convert,
)


def test_happy_path_usd_to_eur() -> None:
    args = CurrencyConvertArgs(amount=100.0, from_currency="USD", to_currency="EUR")
    result = currency_convert(args)
    assert result.rate == pytest.approx(0.92)
    assert result.converted == pytest.approx(92.0)


def test_same_currency_conversion_is_identity() -> None:
    args = CurrencyConvertArgs(amount=50.0, from_currency="USD", to_currency="USD")
    result = currency_convert(args)
    assert result.rate == 1.0
    assert result.converted == 50.0


def test_invalid_amount_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        CurrencyConvertArgs(amount=0, from_currency="USD", to_currency="EUR")


def test_lowercase_currency_code_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        CurrencyConvertArgs(amount=10.0, from_currency="usd", to_currency="EUR")


def test_unknown_currency_raises_currency_not_found_error() -> None:
    args = CurrencyConvertArgs(amount=10.0, from_currency="USD", to_currency="ZZZ")
    with pytest.raises(CurrencyNotFoundError):
        currency_convert(args)


def test_deterministic_repeated_calls_match() -> None:
    args = CurrencyConvertArgs(amount=250.0, from_currency="GBP", to_currency="JPY")
    result_a = currency_convert(args)
    result_b = currency_convert(args)
    assert result_a == result_b
