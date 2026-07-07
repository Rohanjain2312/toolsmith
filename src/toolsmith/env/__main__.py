"""CLI entry point: `python -m toolsmith.env --task <path> --mode sandbox` runs one stub episode."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from toolsmith.env.model import StubModel
from toolsmith.env.runner import run_episode


def main(argv: list[str] | None = None) -> int:
    """Run one episode from a task JSON file using a StubModel, printing a trajectory summary."""
    parser = argparse.ArgumentParser(description="Run one ToolSmith episode with a stub model.")
    parser.add_argument("--task", required=True, type=Path, help="Path to a task JSON file.")
    parser.add_argument("--mode", default="sandbox", choices=["sandbox", "real"])
    args = parser.parse_args(argv)

    os.environ["TOOLSMITH_MODE"] = args.mode
    task = json.loads(args.task.read_text())

    model = StubModel(task["stub_responses"])
    state = run_episode(task["task_id"], task["user_message"], model)

    print(f"task_id: {state.task_id}")
    print(f"status: {state.status.value}")
    print(f"turns: {state.turn}")
    print(f"tool_calls: {len(state.tool_calls)}")
    print(f"final_answer: {state.final_answer}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
