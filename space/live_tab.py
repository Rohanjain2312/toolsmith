"""Live tab: wires the llama.cpp adapter + episode runner to the UI, streaming steps as they run."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import gradio as gr

from toolsmith.env.model import Model
from toolsmith.env.runner import run_episode

LIVE_TRAJECTORY_DIR = Path("results/trajectories/space_live")
DEFAULT_GGUF_PATH = "toolsmith.Q4_K_M.gguf"  # pinned in the Space repo via Git LFS (P7-T07)


def _build_model() -> Model:
    """Build the live model adapter: llama.cpp over the GGUF pinned in the Space repo."""
    from toolsmith.env.llamacpp_model import LlamaCppModel

    model_path = os.environ.get("TOOLSMITH_GGUF_PATH", DEFAULT_GGUF_PATH)
    return LlamaCppModel(model_path=model_path)


def run_live_task(
    prompt: str, real_mode: bool, trajectory_dir: Path = LIVE_TRAJECTORY_DIR
) -> Iterator[dict[str, Any]]:
    """Run one task through the episode runner, yielding the trajectory summary progressively
    (one more tool call revealed per yield) so the UI updates step by step as it runs.

    `real_mode` sets TOOLSMITH_MODE to "real" for the keyless real APIs (Open-Meteo, Nominatim,
    Frankfurter, RestCountries, zoneinfo); sandbox (the default) is always deterministic —
    flight_search/poi_search/calendar_create_event still need API keys this public Space
    doesn't have configured, so they stay limited even in real mode.
    """
    os.environ["TOOLSMITH_MODE"] = "real" if real_mode else "sandbox"
    state = run_episode("space-live", prompt, _build_model(), trajectory_dir=trajectory_dir)

    revealed: list[dict[str, Any]] = []
    for entry in state.tool_calls:
        revealed.append({"tool_name": entry.tool_name, "args": entry.args, "ok": entry.ok})
        yield {"status": "in_progress", "tool_calls": list(revealed), "final_answer": None}
    yield {
        "status": state.status.value,
        "tool_calls": list(revealed),
        "final_answer": state.final_answer,
    }


def build_live_tab() -> None:
    """Populate the active gr.Tab context with the live task input, run button, and output."""
    gr.Markdown(
        "Sandbox mode by default (deterministic, always works). The keyless real-API toggle "
        "tries live public APIs for the tools that need no key — flight search, POI search, "
        "and calendar still require keys this public Space doesn't have configured."
    )
    task_input = gr.Textbox(label="Task", placeholder="Find flights BWI to SLC next Monday")
    real_mode_toggle = gr.Checkbox(label="Try keyless real APIs", value=False)
    run_btn = gr.Button("Run")
    output = gr.JSON(label="Trajectory")

    run_btn.click(fn=run_live_task, inputs=[task_input, real_mode_toggle], outputs=output)
