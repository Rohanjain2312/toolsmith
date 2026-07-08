"""Sample and pretty-print trajectories for human audit, flagging reward-hacking patterns."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from toolsmith.data.taskspec import TaskSpec, ToolWasCalledWithCondition
from toolsmith.env.state import EpisodeState, EpisodeStatus

DEFAULT_SAMPLE_SIZE = 20
DEFAULT_SEED = 20260901


def load_trajectories(path: Path) -> list[EpisodeState]:
    """Load logged episode states (one EpisodeState.to_json() per line) from a JSONL file."""
    return [
        EpisodeState.from_json(line) for line in path.read_text().splitlines() if line.strip()
    ]


def sample_trajectories(
    states: list[EpisodeState], n: int = DEFAULT_SAMPLE_SIZE, seed: int = DEFAULT_SEED
) -> list[EpisodeState]:
    """Randomly sample up to `n` trajectories (deterministic given `seed`)."""
    rng = random.Random(seed)
    return rng.sample(states, k=min(n, len(states)))


def _required_tool_names(spec: TaskSpec) -> set[str]:
    return {c.tool_name for c in spec.goal_spec if isinstance(c, ToolWasCalledWithCondition)}


def flag_answer_without_required_tools(state: EpisodeState, spec: TaskSpec) -> bool:
    """True iff the episode reached a final answer without calling all goal-required tools."""
    if state.status is not EpisodeStatus.FINAL_ANSWER:
        return False
    required = _required_tool_names(spec)
    if not required:
        return False
    called = {entry.tool_name for entry in state.tool_calls if entry.ok}
    return not required <= called


def flag_trivial_json_gaming(state: EpisodeState) -> bool:
    """True iff the same (tool, args) pair was called more than once (JSON-farming proxy)."""
    seen = set()
    for entry in state.tool_calls:
        key = (entry.tool_name, json.dumps(entry.args, sort_keys=True))
        if key in seen:
            return True
        seen.add(key)
    return False


def flag_repeated_stalling(state: EpisodeState) -> bool:
    """True iff the episode was cut off by the max-turns limit without concluding."""
    return state.status is EpisodeStatus.MAX_TURNS


def audit_trajectory(state: EpisodeState, spec: TaskSpec | None) -> dict[str, object]:
    """Build a human-readable audit record for one trajectory, with suspicious-pattern flags."""
    flags: dict[str, bool] = {
        "trivial_json_gaming": flag_trivial_json_gaming(state),
        "repeated_stalling": flag_repeated_stalling(state),
    }
    if spec is not None:
        flags["answer_without_required_tools"] = flag_answer_without_required_tools(state, spec)
    return {
        "task_id": state.task_id,
        "status": state.status.value,
        "turns": state.turn,
        "tool_calls": [(entry.tool_name, entry.ok) for entry in state.tool_calls],
        "final_answer": state.final_answer,
        "flags": flags,
    }


def format_audit_record(record: dict[str, object]) -> str:
    """Render one audit record as a human-readable multi-line block."""
    lines = [
        f"=== {record['task_id']} ({record['status']}, {record['turns']} turns) ===",
        f"tool calls: {record['tool_calls']}",
        f"final answer: {record['final_answer']!r}",
    ]
    flags: dict[str, bool] = record["flags"]  # type: ignore[assignment]
    active_flags = [name for name, hit in flags.items() if hit]
    if active_flags:
        lines.append(f"FLAGGED: {', '.join(active_flags)}")
    return "\n".join(lines)


def main() -> int:
    trajectories_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "results/sft_trajectories.jsonl"
    )
    tasks_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results/tasks.jsonl")

    states = load_trajectories(trajectories_path)
    specs_by_id = {
        spec.id: spec
        for line in tasks_path.read_text().splitlines()
        if line.strip()
        for spec in [TaskSpec.model_validate_json(line)]
    }

    for state in sample_trajectories(states):
        record = audit_trajectory(state, specs_by_id.get(state.task_id))
        print(format_audit_record(record))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
