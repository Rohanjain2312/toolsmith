"""T1 archetypes, part 1: geocode_city, weather_lookup, flight_search, currency_convert,
timezone_info, calendar_create_event."""

from __future__ import annotations

import random
from datetime import timedelta

from scripts.task_generation.common import SANDBOX_TODAY, build_spec, tool_cond
from toolsmith.data.taskspec import TaskSpec


def geocode(rng: random.Random, cities: list[dict], n: int) -> list[TaskSpec]:
    templates = [
        "What are the coordinates of {city}?",
        "Can you look up where {city} is located?",
        "I need the exact lat/lon for {city} for a mapping tool.",
    ]
    picks = rng.sample(cities, min(n, len(cities)))
    return [
        build_spec(
            "T1",
            rng.choice(templates).format(city=c["name"]),
            [tool_cond("geocode_city", {"city": c["name"]})],
        )
        for c in picks
    ]


def weather(rng: random.Random, cities: list[dict], n: int) -> list[TaskSpec]:
    templates = [
        "What's the weather forecast for {city} on {date}?",
        "Will it be sunny in {city} on {date}?",
        "Give me the {date} forecast for {city}.",
    ]
    specs = []
    for _ in range(n):
        c = rng.choice(cities)
        d = (SANDBOX_TODAY + timedelta(days=rng.randint(0, 13))).isoformat()
        specs.append(
            build_spec(
                "T1",
                rng.choice(templates).format(city=c["name"], date=d),
                [tool_cond("weather_lookup", {"lat": c["lat"], "lon": c["lon"], "date": d})],
            )
        )
    return specs


def flight(rng: random.Random, flights: list[dict], n: int) -> list[TaskSpec]:
    templates = [
        "Find flights from {origin} to {dest} on {date}.",
        "What flights are available from {origin} to {dest} on {date}?",
        "I need to fly from {origin} to {dest} on {date} — what's out there?",
    ]
    specs = []
    for f in rng.sample(flights, min(n, len(flights))):
        d = f["depart"].split("T")[0]
        specs.append(
            build_spec(
                "T1",
                rng.choice(templates).format(origin=f["origin"], dest=f["dest"], date=d),
                [tool_cond("flight_search", {"origin": f["origin"], "dest": f["dest"], "date": d})],
            )
        )
    return specs


def currency(rng: random.Random, currencies: list[str], n: int) -> list[TaskSpec]:
    templates = [
        "Convert {amount} {from_cur} to {to_cur}.",
        "How much is {amount} {from_cur} worth in {to_cur}?",
        "I have {amount} {from_cur}, what's that in {to_cur}?",
    ]
    specs = []
    for _ in range(n):
        from_cur, to_cur = rng.sample(currencies, 2)
        amount = round(rng.uniform(10, 5000), 2)
        specs.append(
            build_spec(
                "T1",
                rng.choice(templates).format(amount=amount, from_cur=from_cur, to_cur=to_cur),
                [
                    tool_cond(
                        "currency_convert",
                        {"amount": amount, "from_currency": from_cur, "to_currency": to_cur},
                    )
                ],
            )
        )
    return specs


def timezone(rng: random.Random, cities: list[dict], n: int) -> list[TaskSpec]:
    specs = []
    for i in range(n):
        c = rng.choice(cities)
        if i % 2 == 0:
            specs.append(
                build_spec(
                    "T1",
                    f"What timezone is {c['timezone']} in relative to UTC?",
                    [tool_cond("timezone_info", {"timezone": c["timezone"]})],
                )
            )
        else:
            specs.append(
                build_spec(
                    "T1",
                    f"What's the UTC offset at coordinates ({c['lat']}, {c['lon']})?",
                    [tool_cond("timezone_info", {"lat": c["lat"], "lon": c["lon"]})],
                )
            )
    return specs


def calendar(rng: random.Random, cities: list[dict], n: int) -> list[TaskSpec]:
    titles = [
        "Team sync",
        "Client dinner",
        "Museum visit",
        "Airport transfer",
        "Conference session",
        "Sightseeing tour",
    ]
    specs = []
    for _ in range(n):
        c = rng.choice(cities)
        d = SANDBOX_TODAY + timedelta(days=rng.randint(0, 13))
        start_hour = rng.randint(8, 18)
        start = f"{d.isoformat()}T{start_hour:02d}:00:00"
        end = f"{d.isoformat()}T{start_hour + 1:02d}:00:00"
        title = rng.choice(titles)
        specs.append(
            build_spec(
                "T1",
                f"Schedule '{title}' from {start_hour}:00 to {start_hour + 1}:00 on "
                f"{d.isoformat()} in {c['name']} time.",
                [
                    tool_cond(
                        "calendar_create_event",
                        {"title": title, "start": start, "end": end, "timezone": c["timezone"]},
                    )
                ],
            )
        )
    return specs
