"""T1 dispatcher: single-tool-call tasks across all 12 sandbox tools (see t1a.py/t1b.py)."""

from __future__ import annotations

import random

from scripts.task_generation import t1a, t1b
from toolsmith.data.taskspec import TaskSpec


def generate(rng: random.Random, world: dict, per_archetype: int) -> list[TaskSpec]:
    """Generate T1 tasks across all 12 tools, ~per_archetype instances of each."""
    cities, flights = world["cities"], world["flights"]
    currencies = [*world["fx_rates"].keys(), "USD"]
    return [
        *t1a.geocode(rng, cities, per_archetype),
        *t1a.weather(rng, cities, per_archetype),
        *t1a.flight(rng, flights, per_archetype),
        *t1a.currency(rng, currencies, per_archetype),
        *t1a.timezone(rng, cities, per_archetype),
        *t1a.calendar(rng, cities, per_archetype),
        *t1b.country(rng, per_archetype),
        *t1b.poi(rng, cities, per_archetype),
        *t1b.distance(rng, cities, per_archetype),
        *t1b.packing(rng, per_archetype),
        *t1b.unit(rng, per_archetype),
        *t1b.datetime_math(rng, per_archetype),
    ]
