"""Generate the eval-gate's 50 cached (prefix, action) reward-pipeline baseline cases.

One case per sandbox city (50 cities in worlddata/cities.json): a geocode_city call whose
frozen continuation always succeeds, so every case's expected reward breakdown is identical
in shape and easy to eyeball. Deterministic and fully offline — rerun any time reward logic
changes intentionally, to refresh the checked-in baseline in tests/fixtures/eval_gate_cases.jsonl.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from toolsmith.data.taskspec import TaskSpec
from toolsmith.env.model import StubModel
from toolsmith.rewards.composite import score_completion
from toolsmith.rewards.outcome_reward import ContinuationCache

CITIES_PATH = Path("src/toolsmith/tools/sandbox/worlddata/cities.json")
DEFAULT_OUTPUT_PATH = Path("tests/fixtures/eval_gate_cases.jsonl")
PREFIX = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "geocode this city"},
]


def build_case(city_name: str) -> dict:
    """Build one eval-gate case for `city_name`, including its baseline reward breakdown."""
    case_id = f"gate-{city_name.lower().replace(' ', '-')}"
    completion = json.dumps({"tool": "geocode_city", "args": {"city": city_name}})
    frozen_responses = [f"{city_name} located."]
    goal_spec_raw = [
        {"type": "tool_was_called_with", "tool_name": "geocode_city", "args": {"city": city_name}}
    ]

    spec = TaskSpec(
        id=case_id,
        tier="T1",
        user_prompt="x",
        goal_spec=goal_spec_raw,
        min_steps=1,
        split="test",
    )
    model = StubModel(frozen_responses)
    cache = ContinuationCache(path=Path("/tmp") / f"eval_gate_fixture_{case_id}.json")
    scored = score_completion(
        spec.id, PREFIX, completion, spec.goal_spec, spec.min_steps, model, cache
    )

    return {
        "case_id": case_id,
        "prefix": PREFIX,
        "completion": completion,
        "frozen_responses": frozen_responses,
        "goal_spec": goal_spec_raw,
        "min_steps": 1,
        "expected_components": scored,
    }


def main() -> int:
    output_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUTPUT_PATH
    cities = json.loads(CITIES_PATH.read_text())

    cases = [build_case(city["name"]) for city in cities]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for case in cases:
            f.write(json.dumps(case) + "\n")
    print(f"wrote {len(cases)} eval-gate cases to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
