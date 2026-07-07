"""Tests for the real-mode flight_search Duffel client (network calls always mocked)."""

from __future__ import annotations

from datetime import date

import pytest

from toolsmith.tools.real import flight_search as flight_search_real_module
from toolsmith.tools.real.flight_search import (
    DuffelRequestError,
    MissingDuffelTokenError,
    flight_search_real,
)
from toolsmith.tools.sandbox.flight_search import FlightSearchArgs

_OFFER = {
    "id": "off_123",
    "total_amount": "245.50",
    "total_currency": "USD",
    "slices": [
        {
            "segments": [
                {
                    "operating_carrier": {"name": "Test Air"},
                    "departing_at": "2026-09-15T08:00:00",
                    "arriving_at": "2026-09-15T11:30:00",
                }
            ]
        }
    ],
}


def _args() -> FlightSearchArgs:
    return FlightSearchArgs(origin="BWI", dest="SLC", date=date(2026, 9, 15))


def _fake_request_json(captured_calls: list) -> callable:
    def _fake(url: str, method: str, headers: dict, body: dict | None = None) -> dict:
        captured_calls.append({"url": url, "method": method, "headers": headers, "body": body})
        if method == "POST":
            return {"data": {"id": "orq_abc"}}
        return {"data": [_OFFER]}

    return _fake


def test_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DUFFEL_ACCESS_TOKEN", "fake-test-token")
    monkeypatch.setattr(flight_search_real_module, "_request_json", _fake_request_json([]))

    result = flight_search_real(_args())

    assert len(result.flights) == 1
    flight = result.flights[0]
    assert flight.id == "off_123"
    assert flight.price == pytest.approx(245.50)
    assert flight.currency == "USD"
    assert flight.arrive > flight.depart


def test_missing_token_raises_without_requesting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DUFFEL_ACCESS_TOKEN", raising=False)

    def _fail_if_called(*args: object, **kwargs: object) -> dict:
        raise AssertionError("_request_json should not be called without a token")

    monkeypatch.setattr(flight_search_real_module, "_request_json", _fail_if_called)

    with pytest.raises(MissingDuffelTokenError):
        flight_search_real(_args())


def test_malformed_offer_shape_raises_duffel_request_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DUFFEL_ACCESS_TOKEN", "fake-test-token")

    def _fake(url: str, method: str, headers: dict, body: dict | None = None) -> dict:
        if method == "POST":
            return {"data": {"id": "orq_abc"}}
        return {"data": [{"id": "off_bad", "slices": []}]}

    monkeypatch.setattr(flight_search_real_module, "_request_json", _fake)

    with pytest.raises(DuffelRequestError):
        flight_search_real(_args())


def test_headers_are_correct(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DUFFEL_ACCESS_TOKEN", "fake-test-token")
    captured: list = []
    monkeypatch.setattr(flight_search_real_module, "_request_json", _fake_request_json(captured))

    flight_search_real(_args())

    post_call = next(c for c in captured if c["method"] == "POST")
    get_call = next(c for c in captured if c["method"] == "GET")

    assert post_call["headers"]["Authorization"] == "Bearer fake-test-token"
    assert post_call["headers"]["Duffel-Version"] == "v2"
    assert post_call["headers"]["Accept"] == "application/json"
    assert post_call["headers"]["Content-Type"] == "application/json"
    assert "Content-Type" not in get_call["headers"]
    assert post_call["body"]["data"]["slices"][0]["origin"] == "BWI"
    assert post_call["body"]["data"]["slices"][0]["destination"] == "SLC"


def test_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DUFFEL_ACCESS_TOKEN", "fake-test-token")
    monkeypatch.setattr(flight_search_real_module, "_request_json", _fake_request_json([]))

    first = flight_search_real(_args())
    second = flight_search_real(_args())

    assert first == second
