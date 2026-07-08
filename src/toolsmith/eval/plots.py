"""Eval result plots: 4-way model comparison bars, reward-component curves from a W&B export."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display backend needed for scripted/CI runs

import matplotlib.pyplot as plt  # noqa: E402 (must follow matplotlib.use)


def plot_four_way_comparison(
    results: dict[str, dict[str, float]], metric: str, output_path: Path
) -> Path:
    """Bar chart comparing one metric across the model names in `results`; saves a PNG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    names = list(results.keys())
    values = [results[name][metric] for name in names]

    fig, ax = plt.subplots()
    ax.bar(names, values)
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} — model comparison")
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_reward_curves(wandb_csv_path: Path, output_path: Path) -> Path:
    """Plot per-component reward curves (columns named "reward/...") from a W&B export CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(wandb_csv_path.read_text().splitlines()))
    reward_columns = [col for col in (rows[0].keys() if rows else []) if col.startswith("reward/")]
    steps = [float(row["step"]) for row in rows]

    fig, ax = plt.subplots()
    for column in reward_columns:
        values = [float(row[column]) for row in rows]
        ax.plot(steps, values, label=column.removeprefix("reward/"))
    ax.set_xlabel("step")
    ax.set_ylabel("reward")
    ax.set_title("Reward components over training")
    ax.legend()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
