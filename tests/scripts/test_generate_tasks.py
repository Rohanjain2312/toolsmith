"""Tests for generate_tasks.py's response-parsing logic (no live API calls)."""

from __future__ import annotations

import json

import pytest
from scripts.generate_tasks import TaskGenerationRequestError, _dedupe, parse_batch_response


def test_parses_clean_json_array() -> None:
    raw = json.dumps(
        [
            {
                "user_prompt": "What's the weather in Paris on 2026-09-03?",
                "goal_spec": [
                    {
                        "type": "tool_was_called_with",
                        "tool_name": "weather_lookup",
                        "args": {"lat": 48.8566, "lon": 2.3522, "date": "2026-09-03"},
                    }
                ],
            }
        ]
    )

    specs = parse_batch_response(raw, "T1")

    assert len(specs) == 1
    assert specs[0].tier == "T1"
    assert specs[0].min_steps == 0
    assert specs[0].split == "train"
    assert specs[0].id.startswith("t1-")


def test_parses_json_wrapped_in_markdown_fence() -> None:
    raw = (
        "```json\n"
        + json.dumps(
            [
                {
                    "user_prompt": "Geocode Paris.",
                    "goal_spec": [
                        {
                            "type": "tool_was_called_with",
                            "tool_name": "geocode_city",
                            "args": {"city": "Paris"},
                        }
                    ],
                }
            ]
        )
        + "\n```"
    )

    specs = parse_batch_response(raw, "T1")

    assert len(specs) == 1
    assert specs[0].user_prompt == "Geocode Paris."


def test_malformed_json_raises() -> None:
    with pytest.raises(TaskGenerationRequestError):
        parse_batch_response("not json at all {", "T1")


def test_non_list_json_raises() -> None:
    with pytest.raises(TaskGenerationRequestError):
        parse_batch_response(json.dumps({"user_prompt": "x"}), "T1")


def test_item_missing_required_field_is_skipped() -> None:
    raw = json.dumps([{"user_prompt": "missing goal_spec"}])

    specs = parse_batch_response(raw, "T1")

    assert specs == []


def test_item_with_unknown_condition_type_is_skipped() -> None:
    raw = json.dumps(
        [
            {
                "user_prompt": "x",
                "goal_spec": [{"type": "not_a_real_condition"}],
            }
        ]
    )

    specs = parse_batch_response(raw, "T1")

    assert specs == []


def test_non_dict_items_are_skipped() -> None:
    raw = json.dumps(["not a dict", 42])

    specs = parse_batch_response(raw, "T1")

    assert specs == []


def test_dedupe_drops_case_and_whitespace_duplicates() -> None:
    raw = json.dumps(
        [
            {
                "user_prompt": "Geocode Paris.",
                "goal_spec": [
                    {
                        "type": "tool_was_called_with",
                        "tool_name": "geocode_city",
                        "args": {"city": "Paris"},
                    }
                ],
            },
            {
                "user_prompt": "  geocode paris.  ",
                "goal_spec": [
                    {
                        "type": "tool_was_called_with",
                        "tool_name": "geocode_city",
                        "args": {"city": "Paris"},
                    }
                ],
            },
        ]
    )

    specs = _dedupe(parse_batch_response(raw, "T1"))

    assert len(specs) == 1
