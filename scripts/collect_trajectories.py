"""Run a trained checkpoint through the episode runner and persist full EpisodeState trajectories.

Producer for the EpisodeState.to_json() JSONL files that data/decision_points.py,
scripts/audit_trajectories.py, and scripts/select_curated_trajectories.py already default to
reading (results/sft_trajectories.jsonl, results/grpo_trajectories.jsonl) -- until this script
existed, nothing in the repo wrote that format, so the documented pipeline (SFT trajectories ->
decision points -> GRPO training -> curated replay pairs) could not run end-to-end from a clean
checkout. See git history for "BUGFIX-T02" for the full incident writeup.

Human-run: point TOOLSMITH_GGUF_PATH at a locally-exported checkpoint (see
notebooks/src/03_export_gguf.py) and run once per checkpoint (SFT, then GRPO), passing the
matching output filename. Pure logic (load_tasks/collect_trajectories/write_trajectories) is
unit tested with StubModel; no live model call is exercised in tests.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from toolsmith.data.taskspec import TaskSpec
from toolsmith.env.model import Model
from toolsmith.env.runner import DEFAULT_TRAJECTORY_DIR, run_episode
from toolsmith.env.state import EpisodeState

DEFAULT_OUTPUT_PATH = Path("results/sft_trajectories.jsonl")


def load_tasks(path: Path) -> list[TaskSpec]:
    """Load every TaskSpec from a validated tasks JSONL file (no split filtering)."""
    return [
        TaskSpec.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()
    ]


def collect_trajectories(
    specs: list[TaskSpec], model: Model, trajectory_dir: Path = DEFAULT_TRAJECTORY_DIR
) -> list[EpisodeState]:
    """Run every task through the episode runner, returning each task's full final state."""
    return [
        run_episode(spec.id, spec.user_prompt, model, trajectory_dir=trajectory_dir)
        for spec in specs
    ]


def write_trajectories(states: list[EpisodeState], path: Path) -> None:
    """Write one EpisodeState.to_json() dump per line -- the shape load_trajectories() expects."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for state in states:
            f.write(state.to_json() + "\n")


def _default_model() -> Model:
    """Build the live model adapter: llama.cpp over a local GGUF named by TOOLSMITH_GGUF_PATH."""
    from toolsmith.env.llamacpp_model import LlamaCppModel

    model_path = os.environ.get("TOOLSMITH_GGUF_PATH")
    if not model_path:
        raise RuntimeError("TOOLSMITH_GGUF_PATH is not set — point it at a local GGUF file")
    return LlamaCppModel(model_path=model_path)


def main() -> int:
    tasks_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/tasks.jsonl")
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT_PATH

    specs = load_tasks(tasks_path)
    states = collect_trajectories(specs, _default_model())
    write_trajectories(states, output_path)
    print(f"wrote {len(states)} trajectories to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
