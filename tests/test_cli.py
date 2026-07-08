"""Tests for the `toolsmith` console-script entry point (serve/run dispatch)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from toolsmith import cli


def test_run_dispatches_to_env_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    task = {"task_id": "cli-x", "user_message": "hi", "stub_responses": ["done"]}
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(task))

    captured: dict = {}

    def fake_run_episode_cli(argv: list[str]) -> int:
        captured["argv"] = argv
        return 0

    monkeypatch.setattr(cli, "_run_episode_cli", fake_run_episode_cli)

    exit_code = cli.main(["run", "--task", str(task_path)])

    assert exit_code == 0
    assert captured["argv"] == ["--task", str(task_path), "--mode", "sandbox"]


def test_run_passes_through_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_run_episode_cli(argv: list[str]) -> int:
        captured["argv"] = argv
        return 0

    monkeypatch.setattr(cli, "_run_episode_cli", fake_run_episode_cli)

    cli.main(["run", "--task", "x.json", "--mode", "real"])

    assert captured["argv"] == ["--task", "x.json", "--mode", "real"]


def test_serve_requires_mcp_flag() -> None:
    with pytest.raises(SystemExit):
        cli.main(["serve"])


def test_serve_dispatches_to_mcp_run(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"ran": False}
    monkeypatch.setattr(cli, "_run_mcp_server", lambda: called.__setitem__("ran", True))

    exit_code = cli.main(["serve", "--mcp"])

    assert exit_code == 0
    assert called["ran"] is True


def test_missing_command_raises_system_exit() -> None:
    with pytest.raises(SystemExit):
        cli.main([])
