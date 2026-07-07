"""Proves importing toolsmith.tools.sandbox registers all 12 closed-registry tools."""

from toolsmith.tools import sandbox  # noqa: F401 (import triggers registration)
from toolsmith.tools.schemas import registry

_EXPECTED_TOOLS = {
    "geocode_city",
    "weather_lookup",
    "flight_search",
    "currency_convert",
    "timezone_info",
    "calendar_create_event",
    "country_info",
    "poi_search",
    "distance_calc",
    "packing_rules",
    "unit_convert",
    "datetime_math",
}


def test_all_twelve_tools_registered_on_package_import() -> None:
    names = {spec.name for spec in registry.list()}
    assert _EXPECTED_TOOLS <= names
