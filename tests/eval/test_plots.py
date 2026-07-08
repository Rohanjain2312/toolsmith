"""Smoke tests for eval plots: confirm files get created with real content, on fixture data."""

from __future__ import annotations

from pathlib import Path

from toolsmith.eval.plots import plot_four_way_comparison, plot_reward_curves


def test_plot_four_way_comparison_creates_nonempty_file(tmp_path: Path) -> None:
    results = {
        "base": {"task_completion_pct": 20.0},
        "sft": {"task_completion_pct": 45.0},
        "grpo": {"task_completion_pct": 60.0},
        "gpt-4o-mini": {"task_completion_pct": 65.0},
    }
    output_path = tmp_path / "comparison.png"

    returned = plot_four_way_comparison(results, "task_completion_pct", output_path)

    assert returned == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_reward_curves_creates_nonempty_file(tmp_path: Path) -> None:
    csv_path = tmp_path / "wandb_export.csv"
    csv_path.write_text(
        "step,reward/r1_valid_parse,reward/r5_goal_satisfied,other_column\n"
        "1,0.5,0.0,x\n"
        "2,0.8,1.5,y\n"
        "3,1.0,3.0,z\n"
    )
    output_path = tmp_path / "curves.png"

    returned = plot_reward_curves(csv_path, output_path)

    assert returned == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_reward_curves_creates_parent_dirs(tmp_path: Path) -> None:
    csv_path = tmp_path / "wandb_export.csv"
    csv_path.write_text("step,reward/r1_valid_parse\n1,0.5\n")
    output_path = tmp_path / "nested" / "dir" / "curves.png"

    plot_reward_curves(csv_path, output_path)

    assert output_path.exists()
