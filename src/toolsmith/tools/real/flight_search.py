"""Real-mode flight_search client using Duffel TEST mode's offer-request -> offers flow."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime

from toolsmith.tools.sandbox.flight_search import (
    FlightOption,
    FlightSearchArgs,
    FlightSearchResult,
)

BASE_URL = "https://api.duffel.com"
DUFFEL_VERSION = "v2"


class MissingDuffelTokenError(RuntimeError):
    """Raised when DUFFEL_ACCESS_TOKEN is not set."""


class DuffelRequestError(RuntimeError):
    """Raised when a Duffel API call fails or returns an unexpected response shape."""


def _headers(include_content_type: bool) -> dict[str, str]:
    token = os.environ.get("DUFFEL_ACCESS_TOKEN")
    if not token:
        raise MissingDuffelTokenError("DUFFEL_ACCESS_TOKEN environment variable is not set")
    headers = {
        "Authorization": f"Bearer {token}",
        "Duffel-Version": DUFFEL_VERSION,
        "Accept": "application/json",
    }
    if include_content_type:
        headers["Content-Type"] = "application/json"
    return headers


def _request_json(
    url: str, method: str, headers: dict[str, str], body: dict | None = None
) -> dict:
    """Perform one HTTP request and parse the JSON response. Isolated for test monkeypatching."""
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=15) as response:
        return json.loads(response.read())


def _create_offer_request(args: FlightSearchArgs) -> str:
    """POST /air/offer_requests and return the new offer_request's id."""
    headers = _headers(include_content_type=True)
    body = {
        "data": {
            "slices": [
                {
                    "origin": args.origin,
                    "destination": args.dest,
                    "departure_date": args.date.isoformat(),
                }
            ],
            "passengers": [{"type": "adult"}],
            "cabin_class": "economy",
        }
    }
    try:
        response = _request_json(f"{BASE_URL}/air/offer_requests", "POST", headers, body)
        return response["data"]["id"]
    except (urllib.error.URLError, OSError, KeyError) as exc:
        raise DuffelRequestError(f"failed to create Duffel offer request: {exc}") from exc


def _fetch_offers(offer_request_id: str) -> list[dict]:
    """GET /air/offer_requests/{id}/offers and return the offers list."""
    headers = _headers(include_content_type=False)
    try:
        response = _request_json(
            f"{BASE_URL}/air/offer_requests/{offer_request_id}/offers", "GET", headers
        )
        return response["data"]
    except (urllib.error.URLError, OSError, KeyError) as exc:
        raise DuffelRequestError(f"failed to fetch Duffel offers: {exc}") from exc


def _offer_to_flight_option(offer: dict) -> FlightOption:
    """Map a Duffel offer object (first slice, first segment) to our FlightOption schema."""
    try:
        first_segment = offer["slices"][0]["segments"][0]
        return FlightOption(
            id=offer["id"],
            depart=datetime.fromisoformat(first_segment["departing_at"]),
            arrive=datetime.fromisoformat(first_segment["arriving_at"]),
            price=float(offer["total_amount"]),
            currency=offer["total_currency"],
        )
    except (KeyError, IndexError, ValueError) as exc:
        raise DuffelRequestError(f"unexpected Duffel offer shape: {exc}") from exc


def flight_search_real(args: FlightSearchArgs) -> FlightSearchResult:
    """Search flights via Duffel TEST mode's offer-request -> offers flow."""
    offer_request_id = _create_offer_request(args)
    offers = _fetch_offers(offer_request_id)
    flights = [_offer_to_flight_option(offer) for offer in offers]
    return FlightSearchResult(flights=flights)
