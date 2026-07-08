"""CI eval gate: replay cached reward-pipeline cases; fail on score drift or broken schemas."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from toolsmith.data.taskspec import TaskSpec
from toolsmith.env.anthropic_export import export_all_anthropic_tools
from toolsmith.env.model import StubModel
from toolsmith.env.openai_export import export_all_openai_tools
from toolsmith.rewards.composite import score_completion
from toolsmith.rewards.outcome_reward import ContinuationCache

DEFAULT_CASES_PATH = Path("tests/fixtures/eval_gate_cases.jsonl")
SCORE_TOLERANCE = 1e-6
EXPECTED_TOOL_COUNT = 12


class EvalGateFailure(RuntimeError):
    """Raised when a cached case's recomputed score drifts, or a schema export is broken."""


def load_cases(path: Path = DEFAULT_CASES_PATH) -> list[dict]:
    """Load cached (prefix, action, expected reward breakdown) cases from a JSONL file."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_tool_schema_exports() -> None:
    """Fail if the OpenAI/Anthropic tool exports don't cover exactly the 12 registered tools."""
    openai_tools = export_all_openai_tools()
    anthropic_tools = export_all_anthropic_tools()
    if len(openai_tools) != EXPECTED_TOOL_COUNT or len(anthropic_tools) != EXPECTED_TOOL_COUNT:
        raise EvalGateFailure(
            f"expected {EXPECTED_TOOL_COUNT} tools in each export, got "
            f"{len(openai_tools)} openai / {len(anthropic_tools)} anthropic"
        )


def check_case(case: dict) -> None:
    """Recompute one cached case's reward breakdown; raise if it drifts from the baseline."""
    goal_spec = TaskSpec.model_validate(
        {
            "id": case["case_id"],
            "tier": "T1",
            "user_prompt": "x",
            "goal_spec": case["goal_spec"],
            "min_steps": case["min_steps"],
            "split": "test",
        }
    ).goal_spec
    model = StubModel(case["frozen_responses"])
    cache = ContinuationCache(path=Path("/tmp") / f"eval_gate_{case['case_id']}.json")

    actual = score_completion(
        case["case_id"],
        case["prefix"],
        case["completion"],
        goal_spec,
        case["min_steps"],
        model,
        cache,
    )
    for key, expected_value in case["expected_components"].items():
        actual_value = actual[key]
        if abs(actual_value - expected_value) > SCORE_TOLERANCE:
            raise EvalGateFailure(
                f"score drift in {case['case_id']}.{key}: "
                f"expected {expected_value}, got {actual_value}"
            )


def main() -> int:
    cases_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CASES_PATH
    try:
        validate_tool_schema_exports()
        cases = load_cases(cases_path)
        for case in cases:
            check_case(case)
    except EvalGateFailure as exc:
        print(f"EVAL GATE FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"eval gate passed: {len(cases)} cases, schema exports OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
