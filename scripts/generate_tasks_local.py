"""Locally synthesize travel-ops tasks, grounded in real sandbox world data (no live LLM API).

Alternative to scripts/generate_tasks.py's Anthropic API path, for environments without a live
API key: template-based generation (scripts/task_generation/) over the sandbox's fixed world
data, producing schema-valid TaskSpecs across all four tiers. Deterministic given SEED.

Run manually: `PYTHONPATH=src uv run python scripts/generate_tasks_local.py`.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

from scripts.task_generation import common, t1, t2, t3, t4
from toolsmith.data.taskspec import TaskSpec


def generate_all(seed: int = common.SEED, per_tier_target: int = 150) -> list[TaskSpec]:
    """Generate ~per_tier_target tasks for each of T1-T4, deduped by user_prompt text."""
    rng = random.Random(seed)
    world = common.load_world()
    return [
        *common.dedupe(t1.generate(rng, world, per_tier_target // 12 + 1)),
        *common.dedupe(t2.generate(rng, world, per_tier_target // 6 + 1)),
        *common.dedupe(t3.generate(rng, world, per_tier_target // 4 + 1)),
        *common.dedupe(t4.generate(rng, world, per_tier_target // 3 + 1)),
    ]


def main() -> int:
    output_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/generated_tasks.jsonl")
    per_tier_target = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    output_path.parent.mkdir(parents=True, exist_ok=True)

    specs = generate_all(per_tier_target=per_tier_target)
    by_tier: dict[str, int] = {}
    for spec in specs:
        by_tier[spec.tier] = by_tier.get(spec.tier, 0) + 1
    for tier, count in sorted(by_tier.items()):
        print(f"{tier}: generated {count} tasks")

    with output_path.open("w") as f:
        for spec in specs:
            f.write(spec.model_dump_json() + "\n")
    print(f"wrote {len(specs)} tasks to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
