"""End-to-end smoke test: a multi-tool episode from first tool call to final answer.

Scripts a StubModel through 4 sandbox tool calls (geocode, weather, currency, distance)
before giving a final answer, and asserts on both the episode state and the trajectory
JSONL file the runner writes to disk.
"""

import json
from pathlib import Path

from toolsmith.env.model import StubModel
from toolsmith.env.runner import run_episode
from toolsmith.env.state import EpisodeStatus

_GEOCODE_CALL = '{"tool": "geocode_city", "args": {"city": "Paris"}}'
_WEATHER_CALL = (
    '{"tool": "weather_lookup", '
    '"args": {"lat": 48.8566, "lon": 2.3522, "date": "2026-09-01"}}'
)
_CURRENCY_CALL = (
    '{"tool": "currency_convert", '
    '"args": {"amount": 100, "from_currency": "USD", "to_currency": "EUR"}}'
)
_DISTANCE_CALL = (
    '{"tool": "distance_calc", '
    '"args": {"lat1": 48.8566, "lon1": 2.3522, "lat2": 51.5074, "lon2": -0.1278}}'
)
_FINAL_ANSWER = (
    "Paris is sunny, 100 USD converts to roughly 92 EUR, and it's about 344 km to London."
)


def test_multi_tool_episode_end_to_end(tmp_path: Path) -> None:
    model = StubModel([_GEOCODE_CALL, _WEATHER_CALL, _CURRENCY_CALL, _DISTANCE_CALL, _FINAL_ANSWER])

    state = run_episode(
        task_id="e2e-smoke",
        user_message="Tell me about a trip to Paris, including weather, currency, and distance.",
        model=model,
        trajectory_dir=tmp_path,
    )

    assert state.status == EpisodeStatus.FINAL_ANSWER
    assert state.final_answer == _FINAL_ANSWER
    assert state.turn == 4
    assert len(state.tool_calls) == 4

    tool_names = [entry.tool_name for entry in state.tool_calls]
    assert tool_names == ["geocode_city", "weather_lookup", "currency_convert", "distance_calc"]
    assert all(entry.ok for entry in state.tool_calls)

    geocode_result, weather_result, currency_result, distance_result = (
        entry.result for entry in state.tool_calls
    )
    assert geocode_result["city"] == "Paris"
    assert geocode_result["country"] == "France"
    assert weather_result["lat"] == 48.8566
    assert currency_result["converted"] > 0
    assert distance_result["distance_km"] > 300

    roles = [m["role"] for m in state.messages]
    assert roles == [
        "system", "user",
        "assistant", "tool",
        "assistant", "tool",
        "assistant", "tool",
        "assistant", "tool",
        "assistant",
    ]

    trajectory_path = tmp_path / "e2e-smoke.jsonl"
    assert trajectory_path.exists()
    lines = [json.loads(line) for line in trajectory_path.read_text().strip().splitlines()]
    assert len(lines) == 5
    assert [line["event"] for line in lines] == [
        "tool_call", "tool_call", "tool_call", "tool_call", "final_answer",
    ]
    assert [line["tool_name"] for line in lines[:4]] == tool_names
    assert all(line["ok"] for line in lines[:4])
    assert lines[4]["text"] == _FINAL_ANSWER
