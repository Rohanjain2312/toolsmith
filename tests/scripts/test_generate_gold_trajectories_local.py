"""Tests for local (no live LLM API) gold-trajectory construction."""

from __future__ import annotations

from scripts.generate_gold_trajectories_local import (
    _final_answer,
    build_gold_trajectories,
    build_scripted_responses,
)

import toolsmith.tools.sandbox  # noqa: F401  (registers all 12 sandbox tools)
from toolsmith.data.taskspec import TaskSpec, ToolWasCalledWithCondition
from toolsmith.env.state import EpisodeStatus
from toolsmith.tools.sandbox.geocode_city import GeocodeCityArgs
from toolsmith.tools.sandbox.geocode_city import geocode_city as real_geocode_city

_FINAL_ANSWER_SAMPLES = {
    "geocode_city": {
        "city": "Paris",
        "lat": 48.8,
        "lon": 2.3,
        "country": "France",
        "timezone": "Europe/Paris",
    },
    "weather_lookup": {
        "lat": 1,
        "lon": 1,
        "date": "2026-09-01",
        "summary": "Sunny",
        "temp_c": 20.0,
    },
    "flight_search": {"flights": []},
    "currency_convert": {
        "amount": 10,
        "from_currency": "USD",
        "to_currency": "EUR",
        "converted": 9.2,
    },
    "timezone_info": {
        "timezone": "UTC",
        "utc_offset_minutes": 0,
        "local_time": "2026-09-01T00:00:00",
    },
    "calendar_create_event": {"event_id": "abc", "status": "confirmed"},
    "country_info": {
        "country": "France",
        "currency": "EUR",
        "languages": ["French"],
        "plug_type": "Type C/E",
        "visa_note": "Check requirements.",
    },
    "poi_search": {"pois": []},
    "distance_calc": {"distance_km": 344.0},
    "packing_rules": {"items": ["shirt", "shoes"]},
    "unit_convert": {"value": 10, "converted": 22.0, "from_unit": "kg", "to_unit": "lb"},
    "datetime_math": {"result_date": "2026-09-05", "weekday_name": "Sat"},
}


def _spec(task_id: str, goal_spec: list) -> TaskSpec:
    return TaskSpec(
        id=task_id,
        tier="T2",
        user_prompt="x",
        goal_spec=goal_spec,
        min_steps=len(goal_spec),
        split="train",
    )


def test_final_answer_covers_all_twelve_tools() -> None:
    for tool_name, result in _FINAL_ANSWER_SAMPLES.items():
        answer = _final_answer(tool_name, result)
        assert isinstance(answer, str) and answer


def test_build_scripted_responses_executes_real_tools() -> None:
    spec = _spec(
        "t1",
        [ToolWasCalledWithCondition(tool_name="geocode_city", args={"city": "Paris"})],
    )

    responses = build_scripted_responses(spec)

    assert len(responses) == 2  # one tool-call turn + one final-answer turn
    assert '"tool": "geocode_city"' in responses[0]
    assert "Paris" in responses[1]


def test_build_scripted_responses_multi_step_chain() -> None:
    r = real_geocode_city(GeocodeCityArgs(city="Paris"))
    spec = _spec(
        "t2",
        [
            ToolWasCalledWithCondition(tool_name="geocode_city", args={"city": "Paris"}),
            ToolWasCalledWithCondition(
                tool_name="weather_lookup",
                args={"lat": r.lat, "lon": r.lon, "date": "2026-09-03"},
            ),
        ],
    )

    responses = build_scripted_responses(spec)

    assert len(responses) == 3


def test_build_gold_trajectories_all_pass_goal_check(tmp_path) -> None:
    distance_args = {"lat1": 0, "lon1": 0, "lat2": 1, "lon2": 1}
    specs = [
        _spec("t1", [ToolWasCalledWithCondition(tool_name="geocode_city", args={"city": "Tokyo"})]),
        _spec("t2", [ToolWasCalledWithCondition(tool_name="distance_calc", args=distance_args)]),
    ]

    states = build_gold_trajectories(specs, tmp_path)

    assert len(states) == 2
    assert all(state.status == EpisodeStatus.FINAL_ANSWER for state in states)
