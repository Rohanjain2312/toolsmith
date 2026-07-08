"""Replay tab: loads curated SFT-vs-GRPO trajectory pairs and renders them step by step."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gradio as gr

DEFAULT_REPLAYS_DIR = Path(__file__).parent / "replays"


def load_curated_trajectories(replays_dir: Path = DEFAULT_REPLAYS_DIR) -> list[dict[str, Any]]:
    """Load every curated trajectory-pair JSON file from `replays_dir` (see
    scripts/select_curated_trajectories.py for how they're produced)."""
    if not replays_dir.exists():
        return []
    return [json.loads(path.read_text()) for path in sorted(replays_dir.glob("*.json"))]


def format_trajectory(summary: dict[str, Any]) -> str:
    """Render one model's trajectory summary (status/tool_calls/final_answer) as markdown."""
    lines = [f"**status:** {summary['status']}"]
    for call in summary["tool_calls"]:
        marker = "OK" if call["ok"] else "FAIL"
        lines.append(f"- [{marker}] `{call['tool_name']}({call['args']})`")
    lines.append(f"\n**final_answer:** {summary['final_answer']}")
    return "\n".join(lines)


def build_replay_tab(replays_dir: Path = DEFAULT_REPLAYS_DIR) -> None:
    """Populate the active gr.Tab context with a trajectory selector and SFT-vs-GRPO panes."""
    trajectories = load_curated_trajectories(replays_dir)
    if not trajectories:
        gr.Markdown(
            "No curated trajectories yet — run `scripts/select_curated_trajectories.py` "
            "after training, then re-deploy the Space."
        )
        return

    by_task_id = {t["task_id"]: t for t in trajectories}
    labels = list(by_task_id.keys())
    first = by_task_id[labels[0]]

    selector = gr.Dropdown(choices=labels, value=labels[0], label="Curated task")
    prompt_display = gr.Markdown(f"**Prompt:** {first['user_prompt']}")
    with gr.Row():
        with gr.Column():
            gr.Markdown("### SFT")
            sft_display = gr.Markdown(format_trajectory(first["sft"]))
        with gr.Column():
            gr.Markdown("### GRPO")
            grpo_display = gr.Markdown(format_trajectory(first["grpo"]))

    def _on_select(task_id: str) -> tuple[str, str, str]:
        record = by_task_id[task_id]
        return (
            f"**Prompt:** {record['user_prompt']}",
            format_trajectory(record["sft"]),
            format_trajectory(record["grpo"]),
        )

    selector.change(
        _on_select, inputs=selector, outputs=[prompt_display, sft_display, grpo_display]
    )
