"""T1 archetypes, part 2: country_info, poi_search, distance_calc, packing_rules, unit_convert,
datetime_math."""

from __future__ import annotations

import random
from datetime import timedelta

from scripts.task_generation.common import (
    COUNTRY_FIXTURE_KEYS,
    PACKING_CLIMATES,
    POI_CATEGORIES,
    SANDBOX_TODAY,
    WEEKDAYS,
    build_spec,
    tool_cond,
)
from toolsmith.data.taskspec import TaskSpec


def country(rng: random.Random, n: int) -> list[TaskSpec]:
    templates = [
        "Tell me about {country} — currency, language, and plug type.",
        "What do I need to know about {country} before I visit?",
        "I'm planning a trip to {country}, what's the local currency and plug type?",
    ]
    specs = []
    for i in range(n):
        c = COUNTRY_FIXTURE_KEYS[i % len(COUNTRY_FIXTURE_KEYS)]
        specs.append(
            build_spec(
                "T1",
                rng.choice(templates).format(country=c),
                [tool_cond("country_info", {"country": c})],
            )
        )
    return specs


def poi(rng: random.Random, cities: list[dict], n: int) -> list[TaskSpec]:
    specs = []
    for _ in range(n):
        c = rng.choice(cities)
        category = rng.choice(POI_CATEGORIES)
        radius = rng.choice([1.0, 2.0, 5.0, 10.0])
        specs.append(
            build_spec(
                "T1",
                f"Find {category}s near {c['name']} within {radius} km.",
                [
                    tool_cond(
                        "poi_search",
                        {
                            "lat": c["lat"],
                            "lon": c["lon"],
                            "category": category,
                            "radius_km": radius,
                        },
                    )
                ],
            )
        )
    return specs


def distance(rng: random.Random, cities: list[dict], n: int) -> list[TaskSpec]:
    specs = []
    for _ in range(n):
        c1, c2 = rng.sample(cities, 2)
        specs.append(
            build_spec(
                "T1",
                f"How far is it from {c1['name']} to {c2['name']}?",
                [
                    tool_cond(
                        "distance_calc",
                        {
                            "lat1": c1["lat"],
                            "lon1": c1["lon"],
                            "lat2": c2["lat"],
                            "lon2": c2["lon"],
                        },
                    )
                ],
            )
        )
    return specs


def packing(rng: random.Random, n: int) -> list[TaskSpec]:
    specs = []
    for _ in range(n):
        clim = rng.choice(PACKING_CLIMATES)
        days = rng.randint(2, 21)
        specs.append(
            build_spec(
                "T1",
                f"What should I pack for a {days}-day trip somewhere with a {clim} climate?",
                [tool_cond("packing_rules", {"climate": clim, "trip_length_days": days})],
            )
        )
    return specs


def unit(rng: random.Random, n: int) -> list[TaskSpec]:
    options = [
        ("temperature", "C", "F"),
        ("temperature", "F", "C"),
        ("temperature", "C", "K"),
        ("distance", "km", "mi"),
        ("distance", "mi", "km"),
        ("distance", "m", "km"),
        ("weight", "kg", "lb"),
        ("weight", "lb", "kg"),
    ]
    specs = []
    for _ in range(n):
        cat, from_u, to_u = rng.choice(options)
        value = round(rng.uniform(1, 300), 1)
        specs.append(
            build_spec(
                "T1",
                f"Convert {value} {from_u} to {to_u}.",
                [
                    tool_cond(
                        "unit_convert",
                        {"value": value, "category": cat, "from_unit": from_u, "to_unit": to_u},
                    )
                ],
            )
        )
    return specs


def datetime_math(rng: random.Random, n: int) -> list[TaskSpec]:
    specs = []
    for i in range(n):
        op = ["add_days", "next_weekday", "weekday_of"][i % 3]
        base = (SANDBOX_TODAY + timedelta(days=rng.randint(0, 30))).isoformat()
        if op == "add_days":
            days = rng.randint(1, 60)
            specs.append(
                build_spec(
                    "T1",
                    f"What date is {days} days after {base}?",
                    [
                        tool_cond(
                            "datetime_math", {"operation": op, "base_date": base, "days": days}
                        )
                    ],
                )
            )
        elif op == "next_weekday":
            weekday = rng.choice(WEEKDAYS)
            specs.append(
                build_spec(
                    "T1",
                    f"What's the next {weekday} after {base}?",
                    [
                        tool_cond(
                            "datetime_math",
                            {"operation": op, "base_date": base, "weekday": weekday},
                        )
                    ],
                )
            )
        else:
            specs.append(
                build_spec(
                    "T1",
                    f"What day of the week is {base}?",
                    [tool_cond("datetime_math", {"operation": op, "base_date": base})],
                )
            )
    return specs
