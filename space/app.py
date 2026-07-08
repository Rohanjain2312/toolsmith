"""ToolSmith Gradio Space: two-tab demo (Replay default, Live second).

Layout: theme, expectation banner, and tab order (Replay first = default landing tab, so a
recruiter sees curated results in seconds; Live second, since the free-CPU Space is slow).
Tab bodies are wired in from replay_tab.py / live_tab.py.
"""

from __future__ import annotations

import gradio as gr

try:
    # Flat layout: how this file actually runs once pushed as an HF Space repo root
    # (app.py, replay_tab.py, live_tab.py all sit side by side there, no "space" package).
    from live_tab import build_live_tab
    from replay_tab import build_replay_tab
except ImportError:
    # Package layout: how it runs inside THIS repo (pytest, `python -m space.app`).
    from space.live_tab import build_live_tab
    from space.replay_tab import build_replay_tab

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
                build_replay_tab()
            with gr.Tab("Live"):
                build_live_tab()

    return demo


def main() -> None:
    """Build and launch the Space app.

    mcp_server=True auto-converts every function wired to a component (via .click(), etc.)
    into an MCP tool, using its docstring + type hints for the schema — so live_tab.py's
    run_live_task doubles as both the UI handler and the Space's public MCP tool, with no
    separate registration needed (same mechanism as serve/mcp_server.py's @mcp.tool).
    """
    app = build_app()
    app.launch(theme=gr.themes.Soft(), mcp_server=True)


if __name__ == "__main__":
    main()
