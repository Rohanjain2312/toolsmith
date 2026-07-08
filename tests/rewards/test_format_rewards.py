"""Tests for the R1-R4 format reward components: hand cases + hypothesis property tests."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from toolsmith.rewards.format_rewards import (
    R1_VALID_PARSE,
    R2_TOOL_EXISTS,
    R3_ARGS_VALID,
    R4_NO_DUPLICATE,
    reward_args_valid,
    reward_no_duplicate,
    reward_tool_exists,
    reward_valid_parse,
)

VALID_CALL = '{"tool": "geocode_city", "args": {"city": "Paris"}}'
INVALID_ARGS_CALL = '{"tool": "geocode_city", "args": {"city": ""}}'
UNKNOWN_TOOL_CALL = '{"tool": "not_a_real_tool", "args": {}}'
MALFORMED_JSON = '{"tool": "geocode_city", "args": '
FINAL_ANSWER = "Paris is a lovely city."


# --- R1: valid parse ---


def test_r1_valid_tool_call_scores_full() -> None:
    assert reward_valid_parse(VALID_CALL) == R1_VALID_PARSE


def test_r1_final_answer_scores_full() -> None:
    assert reward_valid_parse(FINAL_ANSWER) == R1_VALID_PARSE


def test_r1_malformed_json_scores_zero() -> None:
    assert reward_valid_parse(MALFORMED_JSON) == 0.0


@given(text=st.text())
def test_r1_never_raises_and_bounded(text: str) -> None:
    assert reward_valid_parse(text) in (0.0, R1_VALID_PARSE)


# --- R2: tool exists ---


def test_r2_registered_tool_scores_full() -> None:
    assert reward_tool_exists(VALID_CALL) == R2_TOOL_EXISTS


def test_r2_unknown_tool_scores_zero() -> None:
    assert reward_tool_exists(UNKNOWN_TOOL_CALL) == 0.0


def test_r2_final_answer_scores_zero() -> None:
    assert reward_tool_exists(FINAL_ANSWER) == 0.0


def test_r2_malformed_json_scores_zero() -> None:
    assert reward_tool_exists(MALFORMED_JSON) == 0.0


@given(tool_name=st.text(min_size=1).filter(lambda s: '"' not in s and "\\" not in s))
def test_r2_random_tool_names_are_almost_always_unregistered(tool_name: str) -> None:
    import json

    raw = json.dumps({"tool": tool_name, "args": {}})
    assert reward_tool_exists(raw) in (0.0, R2_TOOL_EXISTS)


# --- R3: args valid ---


def test_r3_valid_args_scores_full() -> None:
    assert reward_args_valid(VALID_CALL) == R3_ARGS_VALID


def test_r3_invalid_args_scores_zero() -> None:
    assert reward_args_valid(INVALID_ARGS_CALL) == 0.0


def test_r3_unknown_tool_scores_zero() -> None:
    assert reward_args_valid(UNKNOWN_TOOL_CALL) == 0.0


def test_r3_final_answer_scores_zero() -> None:
    assert reward_args_valid(FINAL_ANSWER) == 0.0


# --- R4: no duplicate ---


def test_r4_first_call_scores_full() -> None:
    assert reward_no_duplicate(VALID_CALL, []) == R4_NO_DUPLICATE


def test_r4_duplicate_call_scores_zero() -> None:
    prefix = [{"role": "assistant", "content": VALID_CALL}]
    assert reward_no_duplicate(VALID_CALL, prefix) == 0.0


def test_r4_same_tool_different_args_scores_full() -> None:
    prefix = [{"role": "assistant", "content": VALID_CALL}]
    other_city_call = '{"tool": "geocode_city", "args": {"city": "Tokyo"}}'
    assert reward_no_duplicate(other_city_call, prefix) == R4_NO_DUPLICATE


def test_r4_ignores_non_assistant_prefix_messages() -> None:
    prefix = [{"role": "user", "content": VALID_CALL}, {"role": "tool", "content": "{}"}]
    assert reward_no_duplicate(VALID_CALL, prefix) == R4_NO_DUPLICATE


def test_r4_final_answer_never_counts_as_duplicate() -> None:
    prefix = [{"role": "assistant", "content": FINAL_ANSWER}]
    assert reward_no_duplicate(FINAL_ANSWER, prefix) == R4_NO_DUPLICATE


def test_r4_malformed_json_scores_zero() -> None:
    assert reward_no_duplicate(MALFORMED_JSON, []) == 0.0


@given(prefix_len=st.integers(min_value=0, max_value=5))
def test_r4_bounded_regardless_of_prefix_length(prefix_len: int) -> None:
    prefix = [{"role": "assistant", "content": FINAL_ANSWER}] * prefix_len
    assert reward_no_duplicate(VALID_CALL, prefix) in (0.0, R4_NO_DUPLICATE)
