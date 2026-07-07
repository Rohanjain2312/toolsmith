"""Seeded generator for the sandbox's static world data: cities, weather, flights, FX, POIs."""

from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

SEED = 20260901
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "src" / "toolsmith" / "tools" / "sandbox" / "worlddata"

# name, lat, lon, country, IANA timezone
CITIES: list[tuple[str, float, float, str, str]] = [
    ("Paris", 48.8566, 2.3522, "France", "Europe/Paris"),
    ("Tokyo", 35.6762, 139.6503, "Japan", "Asia/Tokyo"),
    ("New York", 40.7128, -74.0060, "United States", "America/New_York"),
    ("London", 51.5074, -0.1278, "United Kingdom", "Europe/London"),
    ("Sydney", -33.8688, 151.2093, "Australia", "Australia/Sydney"),
    ("Cairo", 30.0444, 31.2357, "Egypt", "Africa/Cairo"),
    ("Rio de Janeiro", -22.9068, -43.1729, "Brazil", "America/Sao_Paulo"),
    ("Reykjavik", 64.1466, -21.9426, "Iceland", "Atlantic/Reykjavik"),
    ("Berlin", 52.5200, 13.4050, "Germany", "Europe/Berlin"),
    ("Rome", 41.9028, 12.4964, "Italy", "Europe/Rome"),
    ("Madrid", 40.4168, -3.7038, "Spain", "Europe/Madrid"),
    ("Amsterdam", 52.3676, 4.9041, "Netherlands", "Europe/Amsterdam"),
    ("Toronto", 43.6532, -79.3832, "Canada", "America/Toronto"),
    ("Los Angeles", 34.0522, -118.2437, "United States", "America/Los_Angeles"),
    ("Chicago", 41.8781, -87.6298, "United States", "America/Chicago"),
    ("Mexico City", 19.4326, -99.1332, "Mexico", "America/Mexico_City"),
    ("Sao Paulo", -23.5505, -46.6333, "Brazil", "America/Sao_Paulo"),
    ("Buenos Aires", -34.6037, -58.3816, "Argentina", "America/Argentina/Buenos_Aires"),
    ("Lagos", 6.5244, 3.3792, "Nigeria", "Africa/Lagos"),
    ("Nairobi", -1.2921, 36.8219, "Kenya", "Africa/Nairobi"),
    ("Johannesburg", -26.2041, 28.0473, "South Africa", "Africa/Johannesburg"),
    ("Dubai", 25.2048, 55.2708, "United Arab Emirates", "Asia/Dubai"),
    ("Istanbul", 41.0082, 28.9784, "Turkey", "Europe/Istanbul"),
    ("Moscow", 55.7558, 37.6173, "Russia", "Europe/Moscow"),
    ("Delhi", 28.7041, 77.1025, "India", "Asia/Kolkata"),
    ("Mumbai", 19.0760, 72.8777, "India", "Asia/Kolkata"),
    ("Bangkok", 13.7563, 100.5018, "Thailand", "Asia/Bangkok"),
    ("Singapore", 1.3521, 103.8198, "Singapore", "Asia/Singapore"),
    ("Hong Kong", 22.3193, 114.1694, "Hong Kong", "Asia/Hong_Kong"),
    ("Seoul", 37.5665, 126.9780, "South Korea", "Asia/Seoul"),
    ("Beijing", 39.9042, 116.4074, "China", "Asia/Shanghai"),
    ("Shanghai", 31.2304, 121.4737, "China", "Asia/Shanghai"),
    ("Jakarta", -6.2088, 106.8456, "Indonesia", "Asia/Jakarta"),
    ("Manila", 14.5995, 120.9842, "Philippines", "Asia/Manila"),
    ("Auckland", -36.8485, 174.7633, "New Zealand", "Pacific/Auckland"),
    ("Melbourne", -37.8136, 144.9631, "Australia", "Australia/Melbourne"),
    ("Vienna", 48.2082, 16.3738, "Austria", "Europe/Vienna"),
    ("Zurich", 47.3769, 8.5417, "Switzerland", "Europe/Zurich"),
    ("Stockholm", 59.3293, 18.0686, "Sweden", "Europe/Stockholm"),
    ("Oslo", 59.9139, 10.7522, "Norway", "Europe/Oslo"),
    ("Copenhagen", 55.6761, 12.5683, "Denmark", "Europe/Copenhagen"),
    ("Warsaw", 52.2297, 21.0122, "Poland", "Europe/Warsaw"),
    ("Athens", 37.9838, 23.7275, "Greece", "Europe/Athens"),
    ("Lisbon", 38.7223, -9.1393, "Portugal", "Europe/Lisbon"),
    ("Dublin", 53.3498, -6.2603, "Ireland", "Europe/Dublin"),
    ("Vancouver", 49.2827, -123.1207, "Canada", "America/Vancouver"),
    ("Santiago", -33.4489, -70.6693, "Chile", "America/Santiago"),
    ("Lima", -12.0464, -77.0428, "Peru", "America/Lima"),
    ("Bogota", 4.7110, -74.0721, "Colombia", "America/Bogota"),
    ("Casablanca", 33.5731, -7.5898, "Morocco", "Africa/Casablanca"),
]
assert len(CITIES) == 50

COUNTRY_CURRENCY: dict[str, str] = {
    "France": "EUR", "Germany": "EUR", "Italy": "EUR", "Spain": "EUR", "Netherlands": "EUR",
    "Austria": "EUR", "Portugal": "EUR", "Ireland": "EUR", "Greece": "EUR",
    "Japan": "JPY", "United States": "USD", "United Kingdom": "GBP", "Australia": "AUD",
    "Egypt": "EGP", "Brazil": "BRL", "Iceland": "ISK", "Canada": "CAD", "Mexico": "MXN",
    "Argentina": "ARS", "Nigeria": "NGN", "Kenya": "KES", "South Africa": "ZAR",
    "United Arab Emirates": "AED", "Turkey": "TRY", "Russia": "RUB", "India": "INR",
    "Thailand": "THB", "Singapore": "SGD", "Hong Kong": "HKD", "South Korea": "KRW",
    "China": "CNY", "Indonesia": "IDR", "Philippines": "PHP", "New Zealand": "NZD",
    "Switzerland": "CHF", "Sweden": "SEK", "Norway": "NOK", "Denmark": "DKK",
    "Poland": "PLN", "Chile": "CLP", "Peru": "PEN", "Colombia": "COP", "Morocco": "MAD",
}

FX_RATES_USD: dict[str, float] = {
    "USD": 1.0, "EUR": 0.92, "GBP": 0.79, "JPY": 149.5, "AUD": 1.52, "EGP": 48.5,
    "BRL": 5.4, "ISK": 137.0, "CAD": 1.36, "MXN": 17.0, "ARS": 900.0, "NGN": 1550.0,
    "KES": 129.0, "ZAR": 18.5, "AED": 3.67, "TRY": 32.5, "RUB": 92.0, "INR": 83.1,
    "THB": 35.0, "SGD": 1.34, "HKD": 7.82, "KRW": 1330.0, "CNY": 7.24, "IDR": 15600.0,
    "PHP": 56.5, "NZD": 1.64, "CHF": 0.88, "SEK": 10.4, "NOK": 10.6, "DKK": 6.88,
    "PLN": 4.0, "CLP": 930.0, "PEN": 3.75, "COP": 3900.0, "MAD": 9.9,
}

CLIMATE_PATTERNS: list[tuple[str, float]] = [
    ("Sunny", 28.0), ("Partly cloudy", 22.5), ("Rain showers", 18.0),
    ("Overcast", 20.0), ("Clear skies", 25.0), ("Windy", 19.5), ("Light snow", -2.0),
]
POI_CATEGORIES = ["museum", "park", "restaurant", "landmark", "temple", "market"]
FORECAST_WINDOW_DAYS = 14


def _climate_for_lat(lat: float) -> str:
    """Deterministic latitude-band approximation of climate (no real climate database offline)."""
    abs_lat = abs(lat)
    if abs_lat <= 15:
        return "tropical"
    if abs_lat <= 35:
        return "desert"
    if abs_lat <= 55:
        return "temperate"
    if abs_lat <= 66:
        return "cold"
    return "alpine"


def _assign_codes(cities: list[tuple[str, float, float, str, str]]) -> dict[str, str]:
    used: set[str] = set()
    codes: dict[str, str] = {}
    for name, *_rest in cities:
        letters = [c for c in name.upper() if c.isalpha()]
        base = "".join(letters[:3]).ljust(3, "X")
        candidate = base
        suffix = 1
        while candidate in used:
            candidate = (base[:2] + str(suffix))[:3]
            suffix += 1
        used.add(candidate)
        codes[name] = candidate
    return codes


def _generate_weather(rng: random.Random) -> dict[str, list[dict[str, object]]]:
    weather: dict[str, list[dict[str, object]]] = {}
    for name, *_rest in CITIES:
        picks = (rng.choice(CLIMATE_PATTERNS) for _ in range(FORECAST_WINDOW_DAYS))
        weather[name] = [
            {"summary": summary, "temp_c": round(base_temp + rng.uniform(-3.0, 3.0), 1)}
            for summary, base_temp in picks
        ]
    return weather


def _generate_flights(
    rng: random.Random, codes: dict[str, str], count: int = 200
) -> list[dict[str, object]]:
    names = [c[0] for c in CITIES]
    country_by_name = {c[0]: c[3] for c in CITIES}
    flights = []
    for i in range(count):
        origin, dest = rng.sample(names, 2)
        depart_hour = rng.randint(0, 23)
        duration_hours = rng.randint(1, 14)
        depart_dt = datetime(2026, 9, 1, depart_hour, 0, 0) + timedelta(days=rng.randint(0, 13))
        arrive_dt = depart_dt + timedelta(hours=duration_hours)
        flights.append({
            "id": f"FL{i + 1:04d}",
            "origin": codes[origin],
            "dest": codes[dest],
            "depart": depart_dt.isoformat(),
            "arrive": arrive_dt.isoformat(),
            "price": round(rng.uniform(80.0, 1200.0), 2),
            "currency": COUNTRY_CURRENCY[country_by_name[origin]],
        })
    return flights


def _generate_pois(rng: random.Random) -> list[dict[str, object]]:
    pois = []
    counter = 1
    for name, lat, lon, _country, _tz in CITIES:
        for _ in range(rng.randint(3, 5)):
            category = rng.choice(POI_CATEGORIES)
            pois.append({
                "name": f"{name} {category.capitalize()} {counter}",
                "lat": round(lat + rng.uniform(-0.05, 0.05), 4),
                "lon": round(lon + rng.uniform(-0.05, 0.05), 4),
                "category": category,
            })
            counter += 1
    return pois


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def generate(output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
    """Deterministically (re)generate all sandbox world-data JSON files under `output_dir`."""
    rng = random.Random(SEED)
    codes = _assign_codes(CITIES)
    cities_json = [
        {
            "name": name, "lat": lat, "lon": lon, "country": country, "timezone": tz,
            "climate": _climate_for_lat(lat), "code": codes[name],
        }
        for name, lat, lon, country, tz in CITIES
    ]
    weather_json = _generate_weather(rng)
    flights_json = _generate_flights(rng, codes)
    pois_json = _generate_pois(rng)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "cities.json", cities_json)
    _write_json(output_dir / "weather.json", weather_json)
    _write_json(output_dir / "flights.json", flights_json)
    _write_json(output_dir / "pois.json", pois_json)
    _write_json(output_dir / "fx_rates.json", FX_RATES_USD)


if __name__ == "__main__":
    generate()
