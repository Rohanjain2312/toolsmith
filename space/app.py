"""ToolSmith Gradio Space: two-tab demo shell (Replay default, Live second).

No inference wiring yet — that lands in replay_tab.py / live_tab.py. This file just builds the
layout: theme, expectation banner, and tab order (Replay first = default landing tab, so a
recruiter sees curated results in seconds; Live second, since the free-CPU Space is slow).
"""

from __future__ import annotations

import gradio as gr

COLAB_LINK = (
    "https://colab.research.google.com/github/Rohanjain2312/toolsmith/blob/main/"
    "notebooks/04_live_demo.ipynb"
)


def build_app() -> gr.Blocks:
    """Build (but do not launch) the ToolSmith demo Blocks app."""
    with gr.Blocks(title="ToolSmith") as demo:
        gr.Markdown("# ToolSmith — Tool-Calling Agent Demo")
        gr.Markdown(
            "⚠️ CPU demo — expect ~2-4 tok/s in Live mode. "
            f"For full speed, use the [Colab notebook]({COLAB_LINK})."
        )

        with gr.Tabs():
            with gr.Tab("Replay (instant)"):  # first tab = default landing tab
                gr.Markdown("Curated SFT vs GRPO trajectory comparisons — coming in P7-T05.")
            with gr.Tab("Live"):
                gr.Markdown("Run a live task against the model (sandbox mode) — coming in P7-T06.")

    return demo


if __name__ == "__main__":
    app = build_app()
    # mcp_server=True lands in P7-T07 alongside the rest of Space packaging.
    app.launch(theme=gr.themes.Soft())
