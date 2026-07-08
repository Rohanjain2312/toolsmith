"""Local-launch smoke test for the Gradio Space shell (no server actually started/bound)."""

from __future__ import annotations

import gradio as gr
import pytest
from space.app import build_app

from space import app as app_module


def test_build_app_returns_blocks() -> None:
    demo = build_app()

    assert isinstance(demo, gr.Blocks)


def test_replay_tab_is_first_and_default() -> None:
    demo = build_app()

    tabs = [b for b in demo.blocks.values() if type(b).__name__ == "Tab"]
    labels = [tab.label for tab in tabs]

    assert labels[0] == "Replay (instant)"
    assert "Live" in labels


def test_app_shows_expectation_banner() -> None:
    demo = build_app()

    markdown_text = " ".join(
        b.value for b in demo.blocks.values() if type(b).__name__ == "Markdown"
    )

    assert "CPU demo" in markdown_text


def test_main_launches_with_mcp_server_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    class _FakeApp:
        def launch(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(app_module, "build_app", lambda: _FakeApp())

    app_module.main()

    assert captured["mcp_server"] is True
