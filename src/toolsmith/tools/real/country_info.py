"""Real-mode country_info implementation backed by the free REST Countries API."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

from toolsmith.tools.sandbox.country_info import CountryInfoArgs, CountryInfoResult

BASE_URL = "https://restcountries.com/v3.1"


class CountryNotFoundError(ValueError):
    """Raised when the requested country is not found via the REST Countries API."""


class CountryInfoRequestError(RuntimeError):
    """Raised when the REST Countries API request fails or returns an unexpected payload."""


def _fetch_json(url: str) -> object:
    """Fetch and parse a JSON payload from `url`. Isolated so tests can monkeypatch it."""
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.loads(response.read())


def country_info_real(args: CountryInfoArgs) -> CountryInfoResult:
    """Look up real-world country currency/languages via the REST Countries API.

    NOTE: REST Countries does not expose electrical plug type or visa requirement
    data at all, so `plug_type` and `visa_note` are always fixed placeholder strings.
    """
    url = f"{BASE_URL}/name/{urllib.parse.quote(args.country)}?fields=name,currencies,languages"

    try:
        data = _fetch_json(url)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise CountryNotFoundError(
                f"country not found via REST Countries API: {args.country!r}"
            ) from exc
        raise CountryInfoRequestError(
            f"failed to fetch country data from REST Countries API: {exc}"
        ) from exc
    except (OSError, urllib.error.URLError) as exc:
        raise CountryInfoRequestError(
            f"failed to fetch country data from REST Countries API: {exc}"
        ) from exc

    if not data:
        raise CountryNotFoundError(
            f"country not found via REST Countries API: {args.country!r}"
        )

    try:
        entry = data[0]
        currency = next(iter(entry["currencies"].keys()))
        languages = list(entry["languages"].values())
    except (KeyError, IndexError, StopIteration, TypeError, AttributeError) as exc:
        raise CountryInfoRequestError(
            f"unexpected response shape from REST Countries API: missing {exc}"
        ) from exc

    return CountryInfoResult(
        country=args.country,
        currency=currency,
        languages=languages,
        plug_type="Not available from REST Countries API",
        visa_note="Check destination visa requirements before travel.",
    )
