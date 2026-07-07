"""Real-mode currency_convert client backed by the Frankfurter API."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from toolsmith.tools.sandbox.currency_convert import (
    CurrencyConvertArgs,
    CurrencyConvertResult,
)

# Frankfurter's latest-rates endpoint (ECB reference rates, no API key required).
# NOTE: reverify this domain against current Frankfurter docs before real-mode use —
# third-party API domains occasionally change; this is a best-effort implementation.
BASE_URL = "https://api.frankfurter.app"


class CurrencyConvertRequestError(RuntimeError):
    """Raised when the Frankfurter API request or response is invalid."""


def _fetch_json(url: str) -> dict:
    """Fetch a URL and parse its JSON body. Isolated for test monkeypatching."""
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.loads(response.read())


def currency_convert_real(args: CurrencyConvertArgs) -> CurrencyConvertResult:
    """Convert an amount between two currencies using live Frankfurter FX rates."""
    if args.from_currency == args.to_currency:
        return CurrencyConvertResult(
            amount=args.amount,
            from_currency=args.from_currency,
            to_currency=args.to_currency,
            converted=args.amount,
            rate=1.0,
        )

    url = f"{BASE_URL}/latest?base={args.from_currency}&symbols={args.to_currency}"

    try:
        data = _fetch_json(url)
        rate = data["rates"][args.to_currency]
    except (OSError, urllib.error.URLError) as exc:
        raise CurrencyConvertRequestError(
            f"failed to fetch FX rate from {url}: {exc}"
        ) from exc
    except KeyError as exc:
        raise CurrencyConvertRequestError(
            f"currency {args.to_currency!r} missing from Frankfurter response"
        ) from exc

    converted = args.amount * rate

    return CurrencyConvertResult(
        amount=args.amount,
        from_currency=args.from_currency,
        to_currency=args.to_currency,
        converted=converted,
        rate=rate,
    )
