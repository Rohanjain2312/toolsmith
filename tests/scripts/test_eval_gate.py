"""Tests for the eval-gate CI script: score-drift detection + tool schema export validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.eval_gate import (
    EvalGateFailure,
    check_case,
    load_cases,
    validate_tool_schema_exports,
)

_CASE = {
    "case_id": "gate-paris",
    "prefix": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "geocode this city"},
    ],
    "completion": '{"tool": "geocode_city", "args": {"city": "Paris"}}',
    "frozen_responses": ["Paris located."],
    "goal_spec": [
        {"type": "tool_was_called_with", "tool_name": "geocode_city", "args": {"city": "Paris"}}
    ],
    "min_steps": 1,
    "expected_components": {
        "r1_valid_parse": 1.0,
        "r2_tool_exists": 0.5,
        "r3_args_valid": 1.0,
        "r4_no_duplicate": 0.5,
        "r5_goal_satisfied": 3.0,
        "r6_efficiency": 0.5,
        "penalty_hallucinated_tool": 0.0,
        "penalty_max_turns": 0.0,
        "total": 6.5,
    },
}


def test_validate_tool_schema_exports_passes() -> None:
    validate_tool_schema_exports()  # must not raise


def test_check_case_passes_when_scores_match() -> None:
    check_case(_CASE)  # must not raise


def test_check_case_fails_on_score_drift() -> None:
    tampered = {**_CASE, "expected_components": {**_CASE["expected_components"], "total": 99.0}}

    with pytest.raises(EvalGateFailure, match="score drift"):
        check_case(tampered)


def test_load_cases_reads_jsonl(tmp_path: Path) -> None:
    import json

    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps(_CASE) + "\n" + json.dumps(_CASE) + "\n")

    cases = load_cases(path)

    assert len(cases) == 2
    assert cases[0]["case_id"] == "gate-paris"


def test_generated_fixture_file_has_fifty_cases() -> None:
    cases = load_cases(Path("tests/fixtures/eval_gate_cases.jsonl"))

    assert len(cases) == 50


def test_generated_fixture_cases_all_pass_the_gate() -> None:
    for case in load_cases(Path("tests/fixtures/eval_gate_cases.jsonl")):
        check_case(case)  # must not raise for any of the checked-in baseline cases
