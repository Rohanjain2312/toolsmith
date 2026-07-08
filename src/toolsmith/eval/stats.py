"""Bootstrap confidence intervals and results-table rendering (CSV + markdown) for eval metrics."""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

DEFAULT_BOOTSTRAP_SAMPLES = 1000
DEFAULT_CONFIDENCE = 0.95
DEFAULT_SEED = 20260901


@dataclass(frozen=True)
class ConfidenceInterval:
    """A point estimate (the sample mean) with its bootstrap confidence interval bounds."""

    point: float
    lower: float
    upper: float


def bootstrap_ci(
    values: list[float],
    n_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> ConfidenceInterval:
    """Compute a percentile bootstrap CI for the mean of `values`."""
    if not values:
        return ConfidenceInterval(point=0.0, lower=0.0, upper=0.0)

    rng = random.Random(seed)
    n = len(values)
    point = sum(values) / n

    means = []
    for _ in range(n_samples):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()

    alpha = (1.0 - confidence) / 2.0
    lower_idx = max(0, int(alpha * n_samples))
    upper_idx = min(n_samples - 1, int((1.0 - alpha) * n_samples))
    return ConfidenceInterval(point=point, lower=means[lower_idx], upper=means[upper_idx])


def write_results_csv(rows: list[dict[str, object]], path: Path) -> None:
    """Write a list of {column: value} rows to a CSV file (empty file if `rows` is empty)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_results_markdown(rows: list[dict[str, object]], path: Path) -> None:
    """Write a list of {column: value} rows as a markdown table (empty file if empty)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    columns = list(rows[0].keys())
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
        *("| " + " | ".join(str(row[c]) for c in columns) + " |" for row in rows),
    ]
    path.write_text("\n".join(lines) + "\n")
