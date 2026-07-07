"""Deterministic sandbox implementation of the packing_rules tool."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from toolsmith.tools.schemas import ToolSpec, registry

# Base packing items per climate. Pure lookup table, no computation.
_CLIMATE_BASE_ITEMS: dict[str, list[str]] = {
    "tropical": ["light clothing", "sunscreen", "insect repellent", "rain jacket"],
    "desert": ["sun hat", "sunscreen", "water bottle", "light layers"],
    "temperate": ["light jacket", "umbrella", "comfortable shoes"],
    "cold": ["thermal layers", "winter coat", "gloves", "wool socks"],
    "alpine": ["hiking boots", "insulated jacket", "altitude medication", "gloves"],
}

# Length-based extras. Boundaries (inclusive):
#   trip_length_days <= 3            -> short trip: "day bag" only
#   4 <= trip_length_days <= 7        -> medium trip: adds "extra outfit set"
#   trip_length_days > 7 (i.e. >= 8)  -> long trip: adds "extra outfit set" and
#                                        "laundry bag"
_SHORT_TRIP_MAX_DAYS = 3
_MEDIUM_TRIP_MAX_DAYS = 7


class PackingRulesArgs(BaseModel):
    """Arguments for a packing_rules call: destination climate and trip length."""

    climate: Literal["tropical", "desert", "temperate", "cold", "alpine"]
    trip_length_days: int = Field(gt=0)


class PackingRulesResult(BaseModel):
    """Result of a packing_rules call."""

    items: list[str]


def _length_based_extras(trip_length_days: int) -> list[str]:
    """Return the extra packing items dictated by trip length alone."""
    if trip_length_days <= _SHORT_TRIP_MAX_DAYS:
        return ["day bag"]
    if trip_length_days <= _MEDIUM_TRIP_MAX_DAYS:
        return ["extra outfit set"]
    return ["extra outfit set", "laundry bag"]


def packing_rules(args: PackingRulesArgs) -> PackingRulesResult:
    """Combine climate base items with trip-length extras into a packing list."""
    base_items = _CLIMATE_BASE_ITEMS[args.climate]
    extras = _length_based_extras(args.trip_length_days)
    return PackingRulesResult(items=[*base_items, *extras])


registry.register(
    ToolSpec(
        name="packing_rules",
        description=(
            "Look up recommended packing items for a destination climate and "
            "trip length."
        ),
        args_model=PackingRulesArgs,
        returns_model=PackingRulesResult,
        sandbox_fn=packing_rules,
    )
)
