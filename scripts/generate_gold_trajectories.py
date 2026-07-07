"""Generate gold trajectories: run train-split tasks through the Anthropic adapter, keep passers."""

# No automated test for the live-API path: this calls the real Anthropic Messages API and
# requires ANTHROPIC_API_KEY. `load_train_tasks` and `generate_gold_trajectory` (pure logic)
# are unit tested with StubModel; the Anthropic adapter itself is tested separately with
# mocked HTTP responses in tests/env/test_anthropic_model.py.
# Run manually: `uv run python scripts/generate_gold_trajectories.py`.

from __future__ import annotations

import json
import sys
from pathlib import Path

from toolsmith.data.taskspec import TaskSpec
from toolsmith.env.anthropic_model import AnthropicModel
from toolsmith.env.model import Model
from toolsmith.env.runner import DEFAULT_TRAJECTORY_DIR, run_episode
from toolsmith.env.state import EpisodeState
from toolsmith.rewards.goalcheck import check_goal

TASK_COUNT = 200
GOLD_TRAJECTORY_DIR = Path("results/gold_trajectories")
GOLD_SFT_OUTPUT_PATH = Path("results/gold_trajectories.jsonl")


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
) -> EpisodeState | None:
    """Run one task through the episode runner; return its state iff the goal spec passed."""
    state = run_episode(spec.id, spec.user_prompt, model, trajectory_dir=trajectory_dir)
    return state if check_goal(spec.goal_spec, state) else None


def write_gold_sft_rows(states: list[EpisodeState], path: Path) -> None:
    """Write passing episode states as SFT rows, in prep_sft_data.py's {messages, source} shape."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for state in states:
            f.write(json.dumps({"messages": state.messages, "source": "gold"}) + "\n")


def main() -> int:
    tasks_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/tasks.jsonl")
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else GOLD_SFT_OUTPUT_PATH
    specs = load_train_tasks(tasks_path)
    model = AnthropicModel()

    passing_states = [
        state
        for spec in specs
        if (state := generate_gold_trajectory(spec, model, trajectory_dir=GOLD_TRAJECTORY_DIR))
        is not None
    ]

    write_gold_sft_rows(passing_states, output_path)
    print(f"{len(passing_states)}/{len(specs)} gold trajectories passed the goal check")
    print(f"wrote {len(passing_states)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
