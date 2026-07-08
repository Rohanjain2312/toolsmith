"""T3 archetypes: three-step chains, grounded via real sandbox tool execution."""

from __future__ import annotations

import random
from datetime import timedelta

from scripts.task_generation.common import SANDBOX_TODAY, WEEKDAYS, build_spec, tool_cond
from toolsmith.data.taskspec import TaskSpec
from toolsmith.tools.sandbox.geocode_city import GeocodeCityArgs, geocode_city

_FLIGHT_DEST_COUNTRY = {
    "PAR": "France",
    "TOK": "Japan",
    "NEW": "United States",
    "LON": "United Kingdom",
}


def distance_between_cities(rng: random.Random, cities: list[dict], n: int) -> list[TaskSpec]:
    specs = []
    for _ in range(n):
        c1, c2 = rng.sample(cities, 2)
        r1 = geocode_city(GeocodeCityArgs(city=c1["name"]))
        r2 = geocode_city(GeocodeCityArgs(city=c2["name"]))
        specs.append(
            build_spec(
                "T3",
                f"Look up {c1['name']} and {c2['name']}, then tell me the distance between them.",
                [
                    tool_cond("geocode_city", {"city": c1["name"]}),
                    tool_cond("geocode_city", {"city": c2["name"]}),
                    tool_cond(
                        "distance_calc",
                        {"lat1": r1.lat, "lon1": r1.lon, "lat2": r2.lat, "lon2": r2.lon},
                    ),
                ],
            )
        )
    return specs


def weather_and_packing(rng: random.Random, cities: list[dict], n: int) -> list[TaskSpec]:
    specs = []
    for _ in range(n):
        c = rng.choice(cities)
        d = (SANDBOX_TODAY + timedelta(days=rng.randint(0, 13))).isoformat()
        trip_days = rng.randint(3, 14)
        real = geocode_city(GeocodeCityArgs(city=c["name"]))
        specs.append(
            build_spec(
                "T3",
                f"I'm going to {c['name']} for {trip_days} days starting {d}. Find its "
                f"location, check the weather, and tell me what to pack for its "
                f"{c['climate']} climate.",
                [
                    tool_cond("geocode_city", {"city": c["name"]}),
                    tool_cond("weather_lookup", {"lat": real.lat, "lon": real.lon, "date": d}),
                    tool_cond(
                        "packing_rules", {"climate": c["climate"], "trip_length_days": trip_days}
                    ),
                ],
            )
        )
    return specs


def flight_currency_country(rng: random.Random, flights: list[dict], n: int) -> list[TaskSpec]:
    candidates = [f for f in flights if f["dest"] in _FLIGHT_DEST_COUNTRY]
    specs = []
    for f in rng.sample(candidates, min(n, len(candidates))):
        country = _FLIGHT_DEST_COUNTRY[f["dest"]]
        d = f["depart"].split("T")[0]
        specs.append(
            build_spec(
                "T3",
                f"Find a flight from {f['origin']} to {f['dest']} on {d}, convert the price "
                f"to USD, and tell me about {country}.",
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
                    tool_cond("country_info", {"country": country}),
                ],
            )
        )
    return specs


def timezone_and_next_weekday(rng: random.Random, cities: list[dict], n: int) -> list[TaskSpec]:
    specs = []
    for _ in range(n):
        c = rng.choice(cities)
        weekday = rng.choice(WEEKDAYS)
        real = geocode_city(GeocodeCityArgs(city=c["name"]))
        specs.append(
            build_spec(
                "T3",
                f"Find {c['name']}'s location and UTC offset, then tell me the next {weekday}.",
                [
                    tool_cond("geocode_city", {"city": c["name"]}),
                    tool_cond("timezone_info", {"lat": real.lat, "lon": real.lon}),
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
    """Generate T3 tasks across 4 three-step chain archetypes, ~per_archetype instances each."""
    cities, flights = world["cities"], world["flights"]
    return [
        *distance_between_cities(rng, cities, per_archetype),
        *weather_and_packing(rng, cities, per_archetype),
        *flight_currency_country(rng, flights, per_archetype),
        *timezone_and_next_weekday(rng, cities, per_archetype),
    ]
