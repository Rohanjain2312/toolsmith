"""Validate generated tasks against the sandbox solver, reject unsolvable, split by tier."""

from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path

from pydantic import ValidationError

from toolsmith.data.solver import UnsolvableTaskError, solve
from toolsmith.data.taskspec import TaskSpec

SPLIT_SEED = 20260901
SPLIT_COUNTS: dict[str, int] = {"train": 1400, "val": 300, "test": 300}


class TaskValidationError(ValueError):
    """Raised when a task fails validation and should be rejected."""


def load_tasks(path: Path) -> list[TaskSpec]:
    """Load TaskSpecs from a JSONL file, skipping structurally invalid lines with a warning."""
    specs = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            specs.append(TaskSpec.model_validate_json(line))
        except ValidationError as exc:
            print(f"skipping malformed task on line {line_no}: {exc}", file=sys.stderr)
    return specs


def validate_task(spec: TaskSpec) -> TaskSpec:
    """Recompute min_steps via the solver, rejecting unsolvable tasks; return an updated spec."""
    try:
        min_steps = solve(spec.goal_spec)
    except UnsolvableTaskError as exc:
        raise TaskValidationError(f"task {spec.id} is unsolvable: {exc}") from exc
    return spec.model_copy(update={"min_steps": min_steps})


def validate_all(specs: list[TaskSpec]) -> tuple[list[TaskSpec], list[str]]:
    """Validate every task, returning (accepted specs, rejection messages for the rest)."""
    accepted = []
    rejections = []
    for spec in specs:
        try:
            accepted.append(validate_task(spec))
        except TaskValidationError as exc:
            rejections.append(str(exc))
    return accepted, rejections


def stratified_split(specs: list[TaskSpec], seed: int = SPLIT_SEED) -> list[TaskSpec]:
    """Assign train/val/test splits per tier, proportional to SPLIT_COUNTS' ratios."""
    rng = random.Random(seed)
    by_tier: dict[str, list[TaskSpec]] = defaultdict(list)
    for spec in specs:
        by_tier[spec.tier].append(spec)

    total_target = sum(SPLIT_COUNTS.values())
    result = []
    for tier_specs in by_tier.values():
        rng.shuffle(tier_specs)
        n = len(tier_specs)
        cursor = 0
        for split_name, target in SPLIT_COUNTS.items():
            take = round(n * target / total_target)
            for spec in tier_specs[cursor : cursor + take]:
                result.append(spec.model_copy(update={"split": split_name}))
            cursor += take
        # rounding remainder (at most a couple of items) defaults to train
        for spec in tier_specs[cursor:]:
            result.append(spec.model_copy(update={"split": "train"}))
    return result


def write_jsonl(specs: list[TaskSpec], path: Path) -> None:
    """Write TaskSpecs to a JSONL file, one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for spec in specs:
            f.write(spec.model_dump_json() + "\n")


def main() -> int:
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/generated_tasks.jsonl")
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results/tasks.jsonl")

    specs = load_tasks(input_path)
    accepted, rejections = validate_all(specs)
    for message in rejections:
        print(message, file=sys.stderr)
    print(f"{len(accepted)}/{len(specs)} tasks solvable")

    split_specs = stratified_split(accepted)
    write_jsonl(split_specs, output_path)
    print(f"wrote {len(split_specs)} tasks to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
