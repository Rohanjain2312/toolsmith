"""Tests for the model-agnostic eval runner, driven with StubModel."""

from __future__ import annotations

from pathlib import Path

from toolsmith.data.taskspec import TaskSpec, ToolWasCalledWithCondition
from toolsmith.env.model import StubModel
from toolsmith.eval.runner import run_eval


def _spec(task_id: str) -> TaskSpec:
    return TaskSpec(
        id=task_id,
        tier="T1",
        user_prompt="Geocode Paris.",
        goal_spec=[ToolWasCalledWithCondition(tool_name="geocode_city", args={"city": "Paris"})],
        min_steps=1,
        split="test",
    )


def test_run_eval_all_metrics_perfect(tmp_path: Path) -> None:
    specs = [_spec("t1-a"), _spec("t1-b")]
    responses = [
        '{"tool": "geocode_city", "args": {"city": "Paris"}}',
        "Paris is at 48.8N.",
        '{"tool": "geocode_city", "args": {"city": "Paris"}}',
        "Paris is at 48.8N.",
    ]
    model = StubModel(responses)

    report = run_eval(specs, model, trajectory_dir=tmp_path)

    assert report["task_count"] == 2
    assert report["json_valid_pct"] == 100.0
    assert report["correct_tool_pct"] == 100.0
    assert report["arg_accuracy_pct"] == 100.0
    assert report["task_completion_pct"] == 100.0


def test_run_eval_wrong_args_lowers_arg_accuracy_but_not_correct_tool(tmp_path: Path) -> None:
    specs = [_spec("t1-a")]
    responses = ['{"tool": "geocode_city", "args": {"city": "Tokyo"}}', "Tokyo is nice."]
    model = StubModel(responses)

    report = run_eval(specs, model, trajectory_dir=tmp_path)

    assert report["correct_tool_pct"] == 100.0  # right tool, called successfully
    assert report["arg_accuracy_pct"] == 0.0  # but wrong args vs the goal spec
    assert report["task_completion_pct"] == 0.0  # goal never actually satisfied


def test_run_eval_wrong_tool_lowers_correct_tool_and_arg_accuracy(tmp_path: Path) -> None:
    specs = [_spec("t1-a")]
    weather_call = '{"tool": "weather_lookup", "args": {"lat": 0, "lon": 0, "date": "2026-09-03"}}'
    model = StubModel([weather_call, "ok"])

    report = run_eval(specs, model, trajectory_dir=tmp_path)

    assert report["correct_tool_pct"] == 0.0
    assert report["arg_accuracy_pct"] == 0.0


def test_run_eval_parse_failure_lowers_json_valid_pct(tmp_path: Path) -> None:
    specs = [_spec("t1-a")]
    model = StubModel(['{"tool": "geocode_city", "args": '])

    report = run_eval(specs, model, trajectory_dir=tmp_path)

    assert report["json_valid_pct"] == 0.0
    assert report["task_completion_pct"] == 0.0


def test_run_eval_empty_task_list(tmp_path: Path) -> None:
    report = run_eval([], StubModel([]), trajectory_dir=tmp_path)

    assert report == {
        "task_count": 0,
        "json_valid_pct": 0.0,
        "correct_tool_pct": 0.0,
        "arg_accuracy_pct": 0.0,
        "task_completion_pct": 0.0,
    }


def test_run_eval_arg_accuracy_averages_across_multiple_required_calls(tmp_path: Path) -> None:
    spec = TaskSpec(
        id="t2-a",
        tier="T2",
        user_prompt="Geocode Paris then check weather.",
        goal_spec=[
            ToolWasCalledWithCondition(tool_name="geocode_city", args={"city": "Paris"}),
            ToolWasCalledWithCondition(
                tool_name="weather_lookup", args={"lat": 48.8566, "lon": 2.3522}
            ),
        ],
        min_steps=2,
        split="test",
    )
    responses = [
        '{"tool": "geocode_city", "args": {"city": "Paris"}}',
        '{"tool": "weather_lookup", "args": {"lat": 0, "lon": 0, "date": "2026-09-03"}}',
        "done",
    ]
    model = StubModel(responses)

    report = run_eval([spec], model, trajectory_dir=tmp_path)

    assert report["arg_accuracy_pct"] == 50.0  # 1 of 2 required calls had matching args
