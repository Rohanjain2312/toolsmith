"""T4 archetypes: traps -- requests the agent CANNOT satisfy by calling tools correctly
(per data/prompts/t4.txt's tier spec). The correct behavior is to decline or ask for
clarification, not to hallucinate a tool call with invented arguments, so every goal_spec here
is a single `answer_contains_fact` condition (a short substring a correct declining/clarifying
final answer must contain) -- never a tool_was_called_with condition.
"""

from __future__ import annotations

import random

from scripts.task_generation.common import ABSENT_CITIES, ABSENT_COUNTRIES, build_spec, fact_cond
from toolsmith.data.taskspec import TaskSpec


def _trap(user_prompt: str, fact: str) -> TaskSpec:
    return build_spec("T4", user_prompt, [fact_cond(fact)])


def unknown_city(rng: random.Random, n: int) -> list[TaskSpec]:
    """Ask about a real city absent from the sandbox's 50-city fixture."""
    templates = [
        "What's the weather like in {city} this week?",
        "Can you find {city}'s coordinates for me?",
        "What's the best way to get around {city}?",
    ]
    picks = rng.sample(ABSENT_CITIES, min(n, len(ABSENT_CITIES)))
    return [_trap(rng.choice(templates).format(city=city), "cannot") for city in picks]


def unknown_country(rng: random.Random, n: int) -> list[TaskSpec]:
    """Ask about a real country absent from the country_info sandbox fixture."""
    templates = [
        "What currency does {country} use, and what plug type do I need?",
        "Tell me about visiting {country} — currency and language.",
    ]
    picks = rng.sample(ABSENT_COUNTRIES, min(n, len(ABSENT_COUNTRIES)))
    return [_trap(rng.choice(templates).format(country=c), "don't have") for c in picks]


def missing_argument(rng: random.Random, cities: list[dict], n: int) -> list[TaskSpec]:
    """Ask for something missing a required detail (amount, date, unit, title) -- city name
    alone varies the phrasing without resolving the actual missing piece."""
    templates = [
        "Can you convert some money to euros for me?",
        "Schedule a meeting for me sometime next week in {city}.",
        "What's the weather going to be like on my trip to {city}?",
        "Book me a flight to {city}.",
        "Convert this measurement for me.",
        "What should I pack for my trip to {city}?",
    ]
    specs = []
    for _ in range(n):
        c = rng.choice(cities)
        specs.append(_trap(rng.choice(templates).format(city=c["name"]), "clarify"))
    return specs


def unroutable_flight(
    rng: random.Random, cities: list[dict], flights: list[dict], n: int
) -> list[TaskSpec]:
    """Ask for a flight between two real cities with no direct route in the sandbox data."""
    routed = {(f["origin"], f["dest"]) for f in flights}
    codes = [c["code"] for c in cities]
    candidates: list[tuple[str, str]] = []
    for _ in range(n * 20):
        origin, dest = rng.sample(codes, 2)
        if (origin, dest) not in routed:
            candidates.append((origin, dest))
        if len(candidates) >= n:
            break
    return [
        _trap(f"Find me a direct flight from {origin} to {dest} next week.", "cannot")
        for origin, dest in candidates
    ]


def no_data_capability(rng: random.Random, cities: list[dict], n: int) -> list[TaskSpec]:
    """Ask for information no sandbox tool provides (ratings, reviews, historical data)."""
    templates = [
        "What's the best-rated restaurant in {city}?",
        "What was the temperature in {city} last month?",
        "Which hotel in {city} has the most 5-star reviews?",
        "How crowded is the museum near {city} usually?",
        "What's the exchange rate trend for USD to EUR over the past year?",
    ]
    specs = []
    for _ in range(n):
        c = rng.choice(cities)
        specs.append(_trap(rng.choice(templates).format(city=c["name"]), "don't have"))
    return specs


def generate(rng: random.Random, world: dict, per_archetype: int) -> list[TaskSpec]:
    """Generate T4 trap tasks across 5 archetypes, ~per_archetype instances each."""
    cities, flights = world["cities"], world["flights"]
    return [
        *unknown_city(rng, per_archetype),
        *unknown_country(rng, per_archetype),
        *missing_argument(rng, cities, per_archetype),
        *unroutable_flight(rng, cities, flights, per_archetype),
        *no_data_capability(rng, cities, per_archetype),
    ]
