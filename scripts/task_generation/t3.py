"""T3 archetypes: 4-6 tool chains with dependencies (per data/prompts/t3.txt's tier spec).

Grounded via real sandbox tool execution, same as t2.py.
"""

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
    WEEKDAYS,
    build_spec,
    tool_cond,
)
from toolsmith.data.taskspec import TaskSpec
from toolsmith.tools.sandbox.geocode_city import GeocodeCityArgs, geocode_city

_FLIGHT_DEST_COUNTRY = {
    "PAR": "France",
    "TOK": "Japan",
    "NEW": "United States",
    "LON": "United Kingdom",
}


def country_currency_packing_calendar(
    rng: random.Random, currencies: list[str], n: int
) -> list[TaskSpec]:
    """4 steps: country_info + currency_convert + packing_rules + calendar_create_event."""
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
                "T3",
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


def flight_itinerary(rng: random.Random, flights: list[dict], n: int) -> list[TaskSpec]:
    """5 steps: flight_search + currency_convert + geocode_city + weather_lookup + country_info."""
    candidates = [f for f in flights if f["dest"] in _FLIGHT_DEST_COUNTRY]
    city_names = {"PAR": "Paris", "TOK": "Tokyo", "NEW": "New York", "LON": "London"}
    specs = []
    for f in rng.sample(candidates, min(n, len(candidates))):
        country = _FLIGHT_DEST_COUNTRY[f["dest"]]
        dest_name = city_names[f["dest"]]
        d = f["depart"].split("T")[0]
        real = geocode_city(GeocodeCityArgs(city=dest_name))
        specs.append(
            build_spec(
                "T3",
                f"Find a flight from {f['origin']} to {f['dest']} on {d}, convert the price "
                f"to USD, then look up {dest_name}'s location, its weather that day, and tell "
                f"me about {country}.",
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
                    tool_cond("geocode_city", {"city": dest_name}),
                    tool_cond("weather_lookup", {"lat": real.lat, "lon": real.lon, "date": d}),
                    tool_cond("country_info", {"country": country}),
                ],
            )
        )
    return specs


def full_trip_with_budget(
    rng: random.Random, cities: list[dict], currencies: list[str], n: int
) -> list[TaskSpec]:
    """5 steps: geocode + weather + poi_search + packing_rules + currency_convert (budget)."""
    specs = []
    for _ in range(n):
        c = rng.choice(cities)
        d = (SANDBOX_TODAY + timedelta(days=rng.randint(0, 13))).isoformat()
        trip_days = rng.randint(3, 14)
        category = rng.choice(POI_CATEGORIES)
        budget = round(rng.uniform(200, 4000), 2)
        my_currency = rng.choice(currencies)
        real = geocode_city(GeocodeCityArgs(city=c["name"]))
        specs.append(
            build_spec(
                "T3",
                f"I'm planning a {trip_days}-day trip to {c['name']} starting {d} with a "
                f"budget of {budget} {my_currency}. Find its location, check the weather, "
                f"find nearby {category}s, tell me what to pack for its {c['climate']} "
                f"climate, and convert my budget to USD.",
                [
                    tool_cond("geocode_city", {"city": c["name"]}),
                    tool_cond(
                        "weather_lookup", {"lat": real.lat, "lon": real.lon, "date": d}
                    ),
                    tool_cond(
                        "poi_search",
                        {
                            "lat": real.lat,
                            "lon": real.lon,
                            "category": category,
                            "radius_km": 5.0,
                        },
                    ),
                    tool_cond(
                        "packing_rules",
                        {"climate": c["climate"], "trip_length_days": trip_days},
                    ),
                    tool_cond(
                        "currency_convert",
                        {"amount": budget, "from_currency": my_currency, "to_currency": "USD"},
                    ),
                ],
            )
        )
    return specs


def distance_timezone_weather_chain(
    rng: random.Random, cities: list[dict], n: int
) -> list[TaskSpec]:
    """6 steps: geocode x2 + distance_calc + timezone_info + weather_lookup + datetime_math."""
    specs = []
    for _ in range(n):
        c1, c2 = rng.sample(cities, 2)
        weekday = rng.choice(WEEKDAYS)
        r1 = geocode_city(GeocodeCityArgs(city=c1["name"]))
        r2 = geocode_city(GeocodeCityArgs(city=c2["name"]))
        specs.append(
            build_spec(
                "T3",
                f"I'm traveling from {c1['name']} to {c2['name']}. Look up both cities, tell "
                f"me the distance between them, {c2['name']}'s UTC offset, its weather on "
                f"{SANDBOX_TODAY.isoformat()}, and the next {weekday}.",
                [
                    tool_cond("geocode_city", {"city": c1["name"]}),
                    tool_cond("geocode_city", {"city": c2["name"]}),
                    tool_cond(
                        "distance_calc",
                        {"lat1": r1.lat, "lon1": r1.lon, "lat2": r2.lat, "lon2": r2.lon},
                    ),
                    tool_cond("timezone_info", {"lat": r2.lat, "lon": r2.lon}),
                    tool_cond(
                        "weather_lookup",
                        {"lat": r2.lat, "lon": r2.lon, "date": SANDBOX_TODAY.isoformat()},
                    ),
                    tool_cond(
                        "datetime_math",
                        {
                            "operation": "next_weekday",
                            "base_date": SANDBOX_TODAY.isoformat(),
                            "weekday": weekday,
                        },
                    ),
                ],
            )
        )
    return specs


def generate(rng: random.Random, world: dict, per_archetype: int) -> list[TaskSpec]:
    """Generate T3 tasks across 4 archetypes (4-6 steps each), ~per_archetype instances each."""
    cities, flights = world["cities"], world["flights"]
    currencies = [*world["fx_rates"].keys(), "USD"]
    return [
        *country_currency_packing_calendar(rng, currencies, per_archetype),
        *flight_itinerary(rng, flights, per_archetype),
        *full_trip_with_budget(rng, cities, currencies, per_archetype),
        *distance_timezone_weather_chain(rng, cities, per_archetype),
    ]
