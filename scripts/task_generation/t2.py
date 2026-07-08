"""T2 archetypes: two-step chains where the second call depends on the first's real output."""

from __future__ import annotations

import random
from datetime import timedelta

from scripts.task_generation.common import (
    COUNTRY_CURRENCY,
    COUNTRY_FIXTURE_KEYS,
    POI_CATEGORIES,
    SANDBOX_TODAY,
    build_spec,
    tool_cond,
)
from toolsmith.data.taskspec import TaskSpec
from toolsmith.tools.sandbox.geocode_city import GeocodeCityArgs, geocode_city


def geocode_weather(rng: random.Random, cities: list[dict], n: int) -> list[TaskSpec]:
    specs = []
    for _ in range(n):
        c = rng.choice(cities)
        d = (SANDBOX_TODAY + timedelta(days=rng.randint(0, 13))).isoformat()
        real = geocode_city(GeocodeCityArgs(city=c["name"]))
        specs.append(
            build_spec(
                "T2",
                f"I'm traveling to {c['name']} on {d} — first find its location, then tell "
                f"me the weather forecast.",
                [
                    tool_cond("geocode_city", {"city": c["name"]}),
                    tool_cond("weather_lookup", {"lat": real.lat, "lon": real.lon, "date": d}),
                ],
            )
        )
    return specs


def geocode_poi(rng: random.Random, cities: list[dict], n: int) -> list[TaskSpec]:
    specs = []
    for _ in range(n):
        c = rng.choice(cities)
        category = rng.choice(POI_CATEGORIES)
        radius = rng.choice([1.0, 2.0, 5.0])
        real = geocode_city(GeocodeCityArgs(city=c["name"]))
        specs.append(
            build_spec(
                "T2",
                f"Locate {c['name']}, then find {category}s within {radius} km of it.",
                [
                    tool_cond("geocode_city", {"city": c["name"]}),
                    tool_cond(
                        "poi_search",
                        {
                            "lat": real.lat,
                            "lon": real.lon,
                            "category": category,
                            "radius_km": radius,
                        },
                    ),
                ],
            )
        )
    return specs


def geocode_timezone(rng: random.Random, cities: list[dict], n: int) -> list[TaskSpec]:
    specs = []
    for _ in range(n):
        c = rng.choice(cities)
        real = geocode_city(GeocodeCityArgs(city=c["name"]))
        specs.append(
            build_spec(
                "T2",
                f"Find {c['name']}'s coordinates, then tell me its UTC offset.",
                [
                    tool_cond("geocode_city", {"city": c["name"]}),
                    tool_cond("timezone_info", {"lat": real.lat, "lon": real.lon}),
                ],
            )
        )
    return specs


def flight_currency(
    rng: random.Random, flights: list[dict], currencies: list[str], n: int
) -> list[TaskSpec]:
    specs = []
    for f in rng.sample(flights, min(n, len(flights))):
        d = f["depart"].split("T")[0]
        target = rng.choice([c for c in currencies if c != f["currency"]])
        specs.append(
            build_spec(
                "T2",
                f"Find a flight from {f['origin']} to {f['dest']} on {d}, then convert its "
                f"price to {target}.",
                [
                    tool_cond(
                        "flight_search", {"origin": f["origin"], "dest": f["dest"], "date": d}
                    ),
                    tool_cond(
                        "currency_convert",
                        {
                            "amount": f["price"],
                            "from_currency": f["currency"],
                            "to_currency": target,
                        },
                    ),
                ],
            )
        )
    return specs


def country_currency(rng: random.Random, currencies: list[str], n: int) -> list[TaskSpec]:
    specs = []
    for i in range(n):
        country = COUNTRY_FIXTURE_KEYS[i % len(COUNTRY_FIXTURE_KEYS)]
        my_currency = rng.choice([c for c in currencies if c != COUNTRY_CURRENCY[country]])
        amount = round(rng.uniform(50, 2000), 2)
        specs.append(
            build_spec(
                "T2",
                f"I'm visiting {country} with {amount} {my_currency} — look up its currency, "
                f"then convert my money to it.",
                [
                    tool_cond("country_info", {"country": country}),
                    tool_cond(
                        "currency_convert",
                        {
                            "amount": amount,
                            "from_currency": my_currency,
                            "to_currency": COUNTRY_CURRENCY[country],
                        },
                    ),
                ],
            )
        )
    return specs


def datetime_weather(rng: random.Random, cities: list[dict], n: int) -> list[TaskSpec]:
    specs = []
    for _ in range(n):
        c = rng.choice(cities)
        days = rng.randint(1, 13)
        target = (SANDBOX_TODAY + timedelta(days=days)).isoformat()
        specs.append(
            build_spec(
                "T2",
                f"What date is {days} days from now, and what's the weather in {c['name']} then?",
                [
                    tool_cond(
                        "datetime_math",
                        {
                            "operation": "add_days",
                            "base_date": SANDBOX_TODAY.isoformat(),
                            "days": days,
                        },
                    ),
                    tool_cond("weather_lookup", {"lat": c["lat"], "lon": c["lon"], "date": target}),
                ],
            )
        )
    return specs


def generate(rng: random.Random, world: dict, per_archetype: int) -> list[TaskSpec]:
    """Generate T2 tasks across 6 two-step chain archetypes, ~per_archetype instances each."""
    cities, flights = world["cities"], world["flights"]
    currencies = [*world["fx_rates"].keys(), "USD"]
    return [
        *geocode_weather(rng, cities, per_archetype),
        *geocode_poi(rng, cities, per_archetype),
        *geocode_timezone(rng, cities, per_archetype),
        *flight_currency(rng, flights, currencies, per_archetype),
        *country_currency(rng, currencies, per_archetype),
        *datetime_weather(rng, cities, per_archetype),
    ]
