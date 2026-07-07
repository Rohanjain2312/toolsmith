"""Tests for the packing_rules sandbox tool."""

import pytest
from pydantic import ValidationError

from toolsmith.tools.sandbox.packing_rules import (
    PackingRulesArgs,
    packing_rules,
)


def test_happy_path_tropical_medium_trip() -> None:
    args = PackingRulesArgs(climate="tropical", trip_length_days=5)
    result = packing_rules(args)
    assert result.items == [
        "light clothing",
        "sunscreen",
        "insect repellent",
        "rain jacket",
        "extra outfit set",
    ]


def test_short_medium_boundary_differs() -> None:
    short_result = packing_rules(PackingRulesArgs(climate="temperate", trip_length_days=3))
    medium_result = packing_rules(PackingRulesArgs(climate="temperate", trip_length_days=4))
    assert short_result.items != medium_result.items
    assert short_result.items[-1] == "day bag"
    assert medium_result.items[-1] == "extra outfit set"


def test_medium_long_boundary_differs() -> None:
    medium_result = packing_rules(PackingRulesArgs(climate="cold", trip_length_days=7))
    long_result = packing_rules(PackingRulesArgs(climate="cold", trip_length_days=8))
    assert medium_result.items != long_result.items
    assert medium_result.items[-1] == "extra outfit set"
    assert long_result.items[-2:] == ["extra outfit set", "laundry bag"]


def test_invalid_climate_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        PackingRulesArgs(climate="volcanic", trip_length_days=5)


def test_non_positive_trip_length_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        PackingRulesArgs(climate="alpine", trip_length_days=0)


def test_deterministic_repeated_calls_match() -> None:
    args = PackingRulesArgs(climate="alpine", trip_length_days=10)
    result_a = packing_rules(args)
    result_b = packing_rules(args)
    assert result_a == result_b
    assert result_a.items == result_b.items
