"""Shared constants, world-data loading, and TaskSpec-building helpers for local task generation."""

from __future__ import annotations

import json
import re
import uuid
from datetime import date
from pathlib import Path

from toolsmith.data.taskspec import TaskSpec, Tier

_IATA_CODE_PATTERN = re.compile(r"^[A-Z]{3}$")

SEED = 20260901
SANDBOX_TODAY = date(2026, 9, 1)
_WORLDDATA_DIR = Path("src/toolsmith/tools/sandbox/worlddata")

# The country_info sandbox tool only has fixture data for these 8 countries.
COUNTRY_FIXTURE_KEYS = [
    "France",
    "Japan",
    "United States",
    "United Kingdom",
    "Australia",
    "Egypt",
    "Brazil",
    "Iceland",
]
COUNTRY_CURRENCY = {
    "France": "EUR",
    "Japan": "JPY",
    "United States": "USD",
    "United Kingdom": "GBP",
    "Australia": "AUD",
    "Egypt": "EGP",
    "Brazil": "BRL",
    "Iceland": "ISK",
}
COUNTRY_CLIMATE = {
    "France": "temperate",
    "Japan": "temperate",
    "United States": "temperate",
    "United Kingdom": "temperate",
    "Australia": "desert",
    "Egypt": "desert",
    "Brazil": "tropical",
    "Iceland": "cold",
}
COUNTRY_TIMEZONE = {
    "France": "Europe/Paris",
    "Japan": "Asia/Tokyo",
    "United States": "America/New_York",
    "United Kingdom": "Europe/London",
    "Australia": "Australia/Sydney",
    "Egypt": "Africa/Cairo",
    "Brazil": "America/Sao_Paulo",
    "Iceland": "Atlantic/Reykjavik",
}
POI_CATEGORIES = ["landmark", "market", "museum", "park", "restaurant", "temple"]
PACKING_CLIMATES = ["tropical", "desert", "temperate", "cold", "alpine"]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def load_world() -> dict[str, object]:
    """Load the sandbox's fixed world-data fixtures used to ground generated tasks.

    Filters out any flight whose origin/dest isn't a 3-uppercase-letter code: the world
    generator's IATA-code collision fallback can produce a code like "DU1" (Dublin, digit
    suffix), which FlightSearchArgs' `^[A-Z]{3}$` pattern always rejects -- such a flight can
    never be booked, so a task built around it would be permanently unsolvable.
    """
    flights = json.loads((_WORLDDATA_DIR / "flights.json").read_text())
    return {
        "cities": json.loads((_WORLDDATA_DIR / "cities.json").read_text()),
        "flights": [
            f
            for f in flights
            if _IATA_CODE_PATTERN.match(f["origin"]) and _IATA_CODE_PATTERN.match(f["dest"])
        ],
        "fx_rates": json.loads((_WORLDDATA_DIR / "fx_rates.json").read_text()),
        "pois": json.loads((_WORLDDATA_DIR / "pois.json").read_text()),
    }


def tool_cond(tool_name: str, args: dict) -> dict:
    """Build a `tool_was_called_with` goal-condition dict."""
    return {"type": "tool_was_called_with", "tool_name": tool_name, "args": args}


def build_spec(tier: Tier, user_prompt: str, goal_spec: list[dict]) -> TaskSpec:
    """Build a validated TaskSpec with a fresh id and provisional min_steps/split.

    Mirrors scripts/generate_tasks.py's parse_batch_response: min_steps=0 and split="train" are
    placeholders here — scripts/validate_tasks.py recomputes both.
    """
    return TaskSpec.model_validate(
        {
            "id": f"{tier.lower()}-{uuid.uuid4().hex[:12]}",
            "tier": tier,
            "user_prompt": user_prompt,
            "goal_spec": goal_spec,
            "min_steps": 0,
            "split": "train",
        }
    )


def dedupe(specs: list[TaskSpec]) -> list[TaskSpec]:
    """Drop tasks whose user_prompt text (normalized) duplicates an earlier one."""
    seen: set[str] = set()
    deduped = []
    for spec in specs:
        key = spec.user_prompt.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(spec)
    return deduped
