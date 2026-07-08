"""T4 archetypes: four-step chains, grounded via real sandbox tool execution."""

from __future__ import annotations

import random
from datetime import timedelta

from scripts.task_generation.common import (
    COUNTRY_CLIMATE,
    COUNTRY_CURRENCY,
    COUNTRY_FIXTURE_KEYS,
    COUNTRY_TIMEZONE,
    POI_CATEGORIES,
    SANDBOX_TODAY,
    build_spec,
    tool_cond,
)
from toolsmith.data.taskspec import TaskSpec
from toolsmith.tools.sandbox.geocode_city import GeocodeCityArgs, geocode_city


def full_trip_prep(rng: random.Random, cities: list[dict], n: int) -> list[TaskSpec]:
    specs = []
    for _ in range(n):
        c = rng.choice(cities)
        d = (SANDBOX_TODAY + timedelta(days=rng.randint(0, 13))).isoformat()
        trip_days = rng.randint(3, 14)
        category = rng.choice(POI_CATEGORIES)
        real = geocode_city(GeocodeCityArgs(city=c["name"]))
        specs.append(
            build_spec(
                "T4",
                f"I'm planning a {trip_days}-day trip to {c['name']} starting {d}. Find its "
                f"location, check the weather, find nearby {category}s, and tell me what to "
                f"pack for its {c['climate']} climate.",
                [
                    tool_cond("geocode_city", {"city": c["name"]}),
                    tool_cond("weather_lookup", {"lat": real.lat, "lon": real.lon, "date": d}),
                    tool_cond(
                        "poi_search",
                        {"lat": real.lat, "lon": real.lon, "category": category, "radius_km": 5.0},
                    ),
                    tool_cond(
                        "packing_rules", {"climate": c["climate"], "trip_length_days": trip_days}
                    ),
                ],
            )
        )
    return specs


def flight_currency_geocode_weather(
    rng: random.Random, flights: list[dict], cities: list[dict], n: int
) -> list[TaskSpec]:
    city_by_code = {c["code"]: c for c in cities}
    candidates = [f for f in flights if f["dest"] in city_by_code]
    specs = []
    for f in rng.sample(candidates, min(n, len(candidates))):
        dest_city = city_by_code[f["dest"]]
        d = f["depart"].split("T")[0]
        real = geocode_city(GeocodeCityArgs(city=dest_city["name"]))
        specs.append(
            build_spec(
                "T4",
                f"Find a flight from {f['origin']} to {f['dest']} on {d}, convert the price to "
                f"USD, then look up {dest_city['name']}'s location and weather that day.",
                [
                    tool_cond(
                        "flight_search", {"origin": f["origin"], "dest": f["dest"], "date": d}
                    ),
                    tool_cond(
                        "currency_convert",
                        {
                            "amount": f["price"],
                            "from_currency": f["currency"],
                            "to_currency": "USD",
                        },
                    ),
                    tool_cond("geocode_city", {"city": dest_city["name"]}),
                    tool_cond("weather_lookup", {"lat": real.lat, "lon": real.lon, "date": d}),
                ],
            )
        )
    return specs


def country_currency_packing_calendar(
    rng: random.Random, currencies: list[str], n: int
) -> list[TaskSpec]:
    specs = []
    for i in range(n):
        country = COUNTRY_FIXTURE_KEYS[i % len(COUNTRY_FIXTURE_KEYS)]
        my_currency = rng.choice([c for c in currencies if c != COUNTRY_CURRENCY[country]])
        amount = round(rng.uniform(100, 3000), 2)
        trip_days = rng.randint(3, 14)
        d = SANDBOX_TODAY + timedelta(days=rng.randint(0, 13))
        start_hour = rng.randint(8, 18)
        start = f"{d.isoformat()}T{start_hour:02d}:00:00"
        end = f"{d.isoformat()}T{start_hour + 1:02d}:00:00"
        specs.append(
            build_spec(
                "T4",
                f"I'm visiting {country} for {trip_days} days with {amount} {my_currency}. "
                f"Look up the country, convert my money, tell me what to pack, and schedule "
                f"an arrival check-in from {start_hour}:00 to {start_hour + 1}:00 on "
                f"{d.isoformat()} local time.",
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
                    tool_cond(
                        "packing_rules",
                        {"climate": COUNTRY_CLIMATE[country], "trip_length_days": trip_days},
                    ),
                    tool_cond(
                        "calendar_create_event",
                        {
                            "title": "Arrival check-in",
                            "start": start,
                            "end": end,
                            "timezone": COUNTRY_TIMEZONE[country],
                        },
                    ),
                ],
            )
        )
    return specs


def generate(rng: random.Random, world: dict, per_archetype: int) -> list[TaskSpec]:
    """Generate T4 tasks across 3 four-step chain archetypes, ~per_archetype instances each."""
    cities, flights = world["cities"], world["flights"]
    currencies = [*world["fx_rates"].keys(), "USD"]
    return [
        *full_trip_prep(rng, cities, per_archetype),
        *flight_currency_geocode_weather(rng, flights, cities, per_archetype),
        *country_currency_packing_calendar(rng, currencies, per_archetype),
    ]
