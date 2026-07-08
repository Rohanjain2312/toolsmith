"""Tests for the BFCL decode_ast/decode_execute conversion logic, on fixture responses."""

from __future__ import annotations

import pytest

from toolsmith.eval.bfcl_adapter import (
    SUPPORTED_CATEGORIES,
    UnsupportedBFCLCategoryError,
    decode_ast,
    decode_execute,
    validate_category,
)


def test_decode_ast_single_call() -> None:
    response = '{"tool": "geocode_city", "args": {"city": "Paris"}}'

    result = decode_ast(response)

    assert result == [{"geocode_city": {"city": "Paris"}}]


def test_decode_ast_preserves_native_arg_types() -> None:
    args = '{"lat": 48.8, "lon": 2.3, "radius_km": 5, "open_now": true}'
    response = f'{{"tool": "poi_search", "args": {args}}}'

    result = decode_ast(response)

    args = result[0]["poi_search"]
    assert args["lat"] == 48.8
    assert isinstance(args["radius_km"], int)
    assert args["open_now"] is True


def test_decode_ast_no_call_returns_empty_list() -> None:
    assert decode_ast("I don't know the answer.") == []


def test_decode_ast_malformed_json_returns_empty_list() -> None:
    assert decode_ast('{"tool": "geocode_city", "args": ') == []


def test_decode_ast_multiple_calls_for_parallel_category() -> None:
    response = (
        '{"tool": "geocode_city", "args": {"city": "Paris"}}'
        '{"tool": "geocode_city", "args": {"city": "Tokyo"}}'
    )

    result = decode_ast(response)

    assert result == [
        {"geocode_city": {"city": "Paris"}},
        {"geocode_city": {"city": "Tokyo"}},
    ]


def test_decode_execute_renders_callable_string_form() -> None:
    args = '{"amount": 100.0, "from_currency": "USD", "to_currency": "EUR"}'
    response = f'{{"tool": "currency_convert", "args": {args}}}'

    result = decode_execute(response)

    assert result == ["currency_convert(amount=100.0, from_currency='USD', to_currency='EUR')"]


def test_decode_execute_no_call_returns_empty_list() -> None:
    assert decode_execute("no tool call here") == []


def test_decode_execute_multiple_calls() -> None:
    response = (
        '{"tool": "geocode_city", "args": {"city": "Paris"}}'
        '{"tool": "geocode_city", "args": {"city": "Tokyo"}}'
    )

    result = decode_execute(response)

    assert result == ["geocode_city(city='Paris')", "geocode_city(city='Tokyo')"]


@pytest.mark.parametrize("category", ["simple", "multiple", "parallel"])
def test_validate_category_accepts_supported(category: str) -> None:
    validate_category(category)  # must not raise


def test_validate_category_rejects_unsupported() -> None:
    with pytest.raises(UnsupportedBFCLCategoryError):
        validate_category("exec_simple")


def test_supported_categories_matches_plan_scope() -> None:
    assert SUPPORTED_CATEGORIES == ("simple", "multiple", "parallel")
