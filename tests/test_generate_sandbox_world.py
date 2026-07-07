"""Tests for the seeded sandbox world-data generator."""

import json
from pathlib import Path

import scripts.generate_sandbox_world as gen_world


def test_regeneration_with_same_seed_is_byte_identical(tmp_path: Path) -> None:
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    gen_world.generate(out_a)
    gen_world.generate(out_b)

    for filename in ("cities.json", "weather.json", "flights.json", "pois.json", "fx_rates.json"):
        assert (out_a / filename).read_bytes() == (out_b / filename).read_bytes()


def test_generates_fifty_cities(tmp_path: Path) -> None:
    out = tmp_path / "world"
    gen_world.generate(out)
    cities = json.loads((out / "cities.json").read_text())
    assert len(cities) == 50
    assert {"name", "lat", "lon", "country", "timezone", "climate", "code"} <= cities[0].keys()


def test_generates_two_hundred_flights(tmp_path: Path) -> None:
    out = tmp_path / "world"
    gen_world.generate(out)
    flights = json.loads((out / "flights.json").read_text())
    assert len(flights) == 200
    assert all(f["origin"] != f["dest"] for f in flights)


def test_city_codes_are_unique(tmp_path: Path) -> None:
    out = tmp_path / "world"
    gen_world.generate(out)
    cities = json.loads((out / "cities.json").read_text())
    codes = [c["code"] for c in cities]
    assert len(codes) == len(set(codes))
