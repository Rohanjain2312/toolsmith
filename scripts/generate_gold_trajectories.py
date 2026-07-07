"""Generate gold trajectories: run train-split tasks through the Anthropic adapter, keep passers."""

# No automated test for the live-API path: this calls the real Anthropic Messages API and
# requires ANTHROPIC_API_KEY. `load_train_tasks` and `generate_gold_trajectory` (pure logic)
# are unit tested with StubModel; the Anthropic adapter itself is tested separately with
# mocked HTTP responses in tests/env/test_anthropic_model.py.
# Run manually: `uv run python scripts/generate_gold_trajectories.py`.

from __future__ import annotations

import sys
from pathlib import Path

from toolsmith.data.taskspec import TaskSpec
from toolsmith.env.anthropic_model import AnthropicModel
from toolsmith.env.model import Model
from toolsmith.env.runner import DEFAULT_TRAJECTORY_DIR, run_episode
from toolsmith.rewards.goalcheck import check_goal

TASK_COUNT = 200
GOLD_TRAJECTORY_DIR = Path("results/gold_trajectories")


def load_train_tasks(path: Path, count: int = TASK_COUNT) -> list[TaskSpec]:
    """Load up to `count` train-split TaskSpecs from a validated tasks JSONL file."""
    specs = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        spec = TaskSpec.model_validate_json(line)
        if spec.split == "train":
            specs.append(spec)
        if len(specs) >= count:
            break
    return specs


def generate_gold_trajectory(
    spec: TaskSpec, model: Model, trajectory_dir: Path = DEFAULT_TRAJECTORY_DIR
) -> bool:
    """Run one task through the episode runner; return True iff its goal spec was satisfied."""
    state = run_episode(spec.id, spec.user_prompt, model, trajectory_dir=trajectory_dir)
    return check_goal(spec.goal_spec, state)


def main() -> int:
    tasks_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/tasks.jsonl")
    specs = load_train_tasks(tasks_path)
    model = AnthropicModel()

    kept = 0
    for spec in specs:
        if generate_gold_trajectory(spec, model, trajectory_dir=GOLD_TRAJECTORY_DIR):
            kept += 1
    print(f"{kept}/{len(specs)} gold trajectories passed the goal check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
