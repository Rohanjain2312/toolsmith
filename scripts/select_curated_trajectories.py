"""Select curated SFT-vs-GRPO trajectory pairs from full runs, for the Space's replay tab."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from toolsmith.env.state import EpisodeState

DEFAULT_PAIR_COUNT = 20
DEFAULT_SEED = 20260901


def load_trajectories(path: Path) -> dict[str, EpisodeState]:
    """Load a JSONL file of EpisodeState.to_json() lines, keyed by task_id."""
    states = [
        EpisodeState.from_json(line) for line in path.read_text().splitlines() if line.strip()
    ]
    return {state.task_id: state for state in states}


def summarize(state: EpisodeState) -> dict:
    """Render one EpisodeState as the replay tab's compact {status, tool_calls, final_answer}."""
    return {
        "status": state.status.value,
        "tool_calls": [
            {"tool_name": entry.tool_name, "args": entry.args, "ok": entry.ok}
            for entry in state.tool_calls
        ],
        "final_answer": state.final_answer,
    }


def select_curated_pairs(
    sft_states: dict[str, EpisodeState],
    grpo_states: dict[str, EpisodeState],
    user_prompts: dict[str, str],
    count: int = DEFAULT_PAIR_COUNT,
    seed: int = DEFAULT_SEED,
) -> list[dict]:
    """Pick up to `count` task ids present in both runs, pairing their SFT/GRPO summaries."""
    common_ids = sorted(set(sft_states) & set(grpo_states))
    rng = random.Random(seed)
    rng.shuffle(common_ids)
    selected = common_ids[:count]
    return [
        {
            "task_id": task_id,
            "user_prompt": user_prompts.get(task_id, ""),
            "sft": summarize(sft_states[task_id]),
            "grpo": summarize(grpo_states[task_id]),
        }
        for task_id in selected
    ]


def write_curated_pairs(pairs: list[dict], output_dir: Path) -> None:
    """Write each curated pair to its own JSON file under `output_dir`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for pair in pairs:
        (output_dir / f"{pair['task_id']}.json").write_text(json.dumps(pair, indent=2))


def main() -> int:
    sft_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/sft_trajectories.jsonl")
    default_grpo = Path("results/grpo_trajectories.jsonl")
    grpo_path = Path(sys.argv[2]) if len(sys.argv) > 2 else default_grpo
    tasks_path = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("results/tasks.jsonl")
    output_dir = Path(sys.argv[4]) if len(sys.argv) > 4 else Path("space/replays")

    sft_states = load_trajectories(sft_path)
    grpo_states = load_trajectories(grpo_path)
    user_prompts = {
        spec["id"]: spec["user_prompt"]
        for line in tasks_path.read_text().splitlines()
        if line.strip()
        for spec in [json.loads(line)]
    }

    pairs = select_curated_pairs(sft_states, grpo_states, user_prompts)
    write_curated_pairs(pairs, output_dir)
    print(f"wrote {len(pairs)} curated trajectory pairs to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
