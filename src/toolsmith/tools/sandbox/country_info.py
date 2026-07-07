"""Sandbox implementation of the country_info tool: deterministic country facts lookup."""

from __future__ import annotations

from pydantic import BaseModel, Field

from toolsmith.tools.schemas import ToolSpec, registry


class CountryInfoArgs(BaseModel):
    """Input args for country_info: the country name to look up."""

    country: str = Field(min_length=1)


class CountryInfoResult(BaseModel):
    """Result of a country_info call: currency, languages, plug type, and a visa note."""

    country: str
    currency: str
    languages: list[str]
    plug_type: str
    visa_note: str


class CountryNotFoundError(ValueError):
    """Raised when the requested country is not present in the sandbox fixture."""


# Inline fixture. P1-T14's world generator covers cities/weather/flights/FX/POIs only
# (per the build spec) — country data is intentionally out of scope, kept as a fixture.
_COUNTRY_FIXTURE: dict[str, dict[str, str | list[str]]] = {
    "france": {
        "country": "France",
        "currency": "EUR",
        "languages": ["French"],
        "plug_type": "Type C/E",
        "visa_note": "Check destination visa requirements before travel.",
    },
    "japan": {
        "country": "Japan",
        "currency": "JPY",
        "languages": ["Japanese"],
        "plug_type": "Type A/B",
        "visa_note": "Check destination visa requirements before travel.",
    },
    "united states": {
        "country": "United States",
        "currency": "USD",
        "languages": ["English"],
        "plug_type": "Type A/B",
        "visa_note": "Check destination visa requirements before travel.",
    },
    "united kingdom": {
        "country": "United Kingdom",
        "currency": "GBP",
        "languages": ["English"],
        "plug_type": "Type G",
        "visa_note": "Check destination visa requirements before travel.",
    },
    "australia": {
        "country": "Australia",
        "currency": "AUD",
        "languages": ["English"],
        "plug_type": "Type I",
        "visa_note": "Check destination visa requirements before travel.",
    },
    "egypt": {
        "country": "Egypt",
        "currency": "EGP",
        "languages": ["Arabic"],
        "plug_type": "Type C/F",
        "visa_note": "Check destination visa requirements before travel.",
    },
    "brazil": {
        "country": "Brazil",
        "currency": "BRL",
        "languages": ["Portuguese"],
        "plug_type": "Type C/N",
        "visa_note": "Check destination visa requirements before travel.",
    },
    "iceland": {
        "country": "Iceland",
        "currency": "ISK",
        "languages": ["Icelandic"],
        "plug_type": "Type C/F",
        "visa_note": "Check destination visa requirements before travel.",
    },
}


def country_info(args: CountryInfoArgs) -> CountryInfoResult:
    """Look up a country's currency, languages, plug type, and visa note (case-insensitive)."""
    key = args.country.strip().lower()
    entry = _COUNTRY_FIXTURE.get(key)
    if entry is None:
        raise CountryNotFoundError(f"country not found in sandbox fixture: {args.country!r}")
    return CountryInfoResult(**entry)  # type: ignore[arg-type]


registry.register(
    ToolSpec(
        name="country_info",
        description="Look up a country's currency, official languages, plug type, and visa note.",
        args_model=CountryInfoArgs,
        returns_model=CountryInfoResult,
        sandbox_fn=country_info,
    )
)
