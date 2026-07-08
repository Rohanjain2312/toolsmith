"""Extract per-turn decision points (conversation prefixes) from logged SFT-model trajectories."""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from toolsmith.data.taskspec import TaskSpec
from toolsmith.env.state import EpisodeState

DEFAULT_TASK_COUNT = 700
DEFAULT_SEED = 20260901
# T1/T2-weighted: cheap single/2-3-tool tasks dominate the decision-point pool.
_TIER_WEIGHTS: dict[str, float] = {"T1": 0.40, "T2": 0.35, "T3": 0.20, "T4": 0.05}


@dataclass(frozen=True)
class DecisionPoint:
    """One training sample: a task reference plus the prefix seen before an assistant turn."""

    task_id: str
    prefix: list[dict[str, Any]]


def load_trajectories(path: Path) -> list[EpisodeState]:
    """Load logged episode states (one EpisodeState.to_json() per line) from a JSONL file."""
    return [
        EpisodeState.from_json(line) for line in path.read_text().splitlines() if line.strip()
    ]


def select_task_subset(
    specs: list[TaskSpec],
    count: int = DEFAULT_TASK_COUNT,
    full_set: bool = False,
    seed: int = DEFAULT_SEED,
) -> list[TaskSpec]:
    """Select a T1/T2-weighted subset of tasks, or all tasks when `full_set` is True."""
    if full_set or len(specs) <= count:
        return list(specs)

    rng = random.Random(seed)
    by_tier: dict[str, list[TaskSpec]] = {}
    for spec in specs:
        by_tier.setdefault(spec.tier, []).append(spec)

    selected: list[TaskSpec] = []
    for tier, weight in _TIER_WEIGHTS.items():
        tier_specs = list(by_tier.get(tier, []))
        rng.shuffle(tier_specs)
        selected.extend(tier_specs[: round(count * weight)])
    return selected[:count]


def extract_decision_points(states: list[EpisodeState]) -> list[DecisionPoint]:
    """Emit one DecisionPoint per assistant turn: the exact prefix the model saw before acting."""
    points = []
    for state in states:
        for index, message in enumerate(state.messages):
            if message["role"] == "assistant":
                points.append(DecisionPoint(task_id=state.task_id, prefix=state.messages[:index]))
    return points


def write_decision_points(points: list[DecisionPoint], path: Path) -> None:
    """Write decision points to a JSONL file, one per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for point in points:
            f.write(json.dumps({"task_id": point.task_id, "prefix": point.prefix}) + "\n")


def main() -> int:
    tasks_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/tasks.jsonl")
    default_traj = Path("results/sft_trajectories.jsonl")
    trajectories_path = Path(sys.argv[2]) if len(sys.argv) > 2 else default_traj
    output_path = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("results/decision_points.jsonl")
    full_set = "--full-set" in sys.argv

    train_specs = [
        spec
        for line in tasks_path.read_text().splitlines()
        if line.strip()
        for spec in [TaskSpec.model_validate_json(line)]
        if spec.split == "train"
    ]
    subset = select_task_subset(train_specs, full_set=full_set)
    subset_ids = {spec.id for spec in subset}

    states = [s for s in load_trajectories(trajectories_path) if s.task_id in subset_ids]
    points = extract_decision_points(states)
    write_decision_points(points, output_path)
    print(f"wrote {len(points)} decision points from {len(states)} trajectories to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
