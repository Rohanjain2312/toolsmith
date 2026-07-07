"""Tests for the abstract Model interface and its StubModel test double."""

from __future__ import annotations

import pytest

from toolsmith.env.model import Model, StubModel, StubModelExhaustedError


def test_stub_model_returns_scripted_responses_in_order() -> None:
    model = StubModel(["a", "b", "c"])

    assert model.generate([], []) == "a"
    assert model.generate([], []) == "b"
    assert model.generate([], []) == "c"


def test_stub_model_raises_on_exhaustion() -> None:
    model = StubModel(["a", "b", "c"])

    for _ in range(3):
        model.generate([], [])

    with pytest.raises(StubModelExhaustedError):
        model.generate([], [])


def test_stub_model_call_count_tracks_calls() -> None:
    model = StubModel(["a", "b", "c"])

    assert model.call_count == 0
    model.generate([], [])
    assert model.call_count == 1
    model.generate([], [])
    assert model.call_count == 2
    model.generate([], [])
    assert model.call_count == 3


def test_model_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        Model()  # type: ignore[abstract]


def test_stub_model_is_a_model() -> None:
    assert isinstance(StubModel([]), Model)


def test_stub_model_ignores_messages_and_tools_arguments() -> None:
    model = StubModel(["x", "y"])

    first = model.generate(
        [{"role": "user", "content": "hello"}],
        [{"name": "geocode_city"}],
    )
    second = model.generate(
        [{"role": "user", "content": "totally different"}, {"role": "assistant", "content": "hi"}],
        [{"name": "flight_search"}, {"name": "weather_lookup"}],
    )

    assert first == "x"
    assert second == "y"
