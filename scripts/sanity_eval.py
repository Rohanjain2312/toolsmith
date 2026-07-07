"""Run val-split tasks through the episode runner; report JSON-validity % and correct-tool %."""

from __future__ import annotations

import sys
from pathlib import Path

from toolsmith.data.taskspec import TaskSpec, ToolWasCalledWithCondition
from toolsmith.env.model import Model, StubModel
from toolsmith.env.runner import DEFAULT_TRAJECTORY_DIR, run_episode
from toolsmith.env.state import EpisodeState, EpisodeStatus

DEFAULT_TASK_COUNT = 50
SANITY_EVAL_TRAJECTORY_DIR = Path("results/trajectories/sanity_eval")


def load_val_tasks(path: Path, count: int = DEFAULT_TASK_COUNT) -> list[TaskSpec]:
    """Load up to `count` val-split TaskSpecs from a validated tasks JSONL file."""
    specs = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        spec = TaskSpec.model_validate_json(line)
        if spec.split == "val":
            specs.append(spec)
        if len(specs) >= count:
            break
    return specs


def _is_json_valid(state: EpisodeState) -> bool:
    """A trajectory counts as JSON-valid iff the model never hit a parse failure."""
    return state.status is not EpisodeStatus.PARSE_FAILURE


def _required_tool_names(spec: TaskSpec) -> set[str]:
    """Tool names the task's goal spec requires to have been called."""
    return {
        condition.tool_name
        for condition in spec.goal_spec
        if isinstance(condition, ToolWasCalledWithCondition)
    }


def _called_correct_tool(spec: TaskSpec, state: EpisodeState) -> bool:
    """True iff every tool the goal spec requires was successfully called at least once."""
    required = _required_tool_names(spec)
    if not required:
        return True
    called = {entry.tool_name for entry in state.tool_calls if entry.ok}
    return required <= called


def run_sanity_eval(
    specs: list[TaskSpec], model: Model, trajectory_dir: Path = DEFAULT_TRAJECTORY_DIR
) -> dict[str, float]:
    """Run each task through the episode runner (greedy decoding is the caller's concern via
    `model`) and report aggregate JSON-validity / correct-tool rates."""
    if not specs:
        return {"task_count": 0, "json_valid_pct": 0.0, "correct_tool_pct": 0.0}

    json_valid = 0
    correct_tool = 0
    for spec in specs:
        state = run_episode(spec.id, spec.user_prompt, model, trajectory_dir=trajectory_dir)
        if _is_json_valid(state):
            json_valid += 1
        if _called_correct_tool(spec, state):
            correct_tool += 1

    n = len(specs)
    return {
        "task_count": n,
        "json_valid_pct": 100.0 * json_valid / n,
        "correct_tool_pct": 100.0 * correct_tool / n,
    }


def main() -> int:
    tasks_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/tasks.jsonl")
    specs = load_val_tasks(tasks_path)

    # Placeholder adapter: swap in a real greedy-decoding adapter (e.g. the llama.cpp adapter
    # from Phase 7) once one exists. StubModel here just keeps this script runnable end-to-end.
    model = StubModel(["no matching tool call for this placeholder run"] * len(specs))

    report = run_sanity_eval(specs, model, trajectory_dir=SANITY_EVAL_TRAJECTORY_DIR)
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
