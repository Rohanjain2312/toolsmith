"""Smoke test for the `python -m toolsmith.env` CLI entry point."""

import json
from pathlib import Path

from toolsmith.env.__main__ import main


def _write_task(tmp_path: Path, **overrides: object) -> Path:
    task = {
        "task_id": "cli-smoke",
        "user_message": "Say hello.",
        "stub_responses": ["Hello from the sandbox."],
    }
    task.update(overrides)
    path = tmp_path / "task.json"
    path.write_text(json.dumps(task))
    return path


def test_cli_runs_one_episode_and_prints_summary(tmp_path, capsys, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    task_path = _write_task(tmp_path)

    exit_code = main(["--task", str(task_path), "--mode", "sandbox"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "task_id: cli-smoke" in captured.out
    assert "status: final_answer" in captured.out
    assert "Hello from the sandbox." in captured.out


def test_cli_defaults_to_sandbox_mode(tmp_path, capsys, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    task_path = _write_task(tmp_path)

    exit_code = main(["--task", str(task_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "status: final_answer" in captured.out


def test_cli_reports_tool_call_count(tmp_path, capsys, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    task_path = _write_task(
        tmp_path,
        user_message="How far apart are these points?",
        stub_responses=[
            '{"tool": "distance_calc", "args": {"lat1": 0, "lon1": 0, "lat2": 0, "lon2": 0}}',
            "The distance is 0 km.",
        ],
    )

    exit_code = main(["--task", str(task_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "tool_calls: 1" in captured.out
