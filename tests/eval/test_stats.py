"""Tests for bootstrap CIs and results-table rendering, on fixture numbers."""

from __future__ import annotations

from pathlib import Path

from toolsmith.eval.stats import bootstrap_ci, write_results_csv, write_results_markdown

# --- bootstrap_ci ---


def test_bootstrap_ci_constant_values_collapses_to_that_value() -> None:
    result = bootstrap_ci([5.0, 5.0, 5.0, 5.0])

    assert result.point == 5.0
    assert result.lower == 5.0
    assert result.upper == 5.0


def test_bootstrap_ci_point_within_bounds() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 0.0, 2.5]

    result = bootstrap_ci(values)

    assert result.lower <= result.point <= result.upper


def test_bootstrap_ci_is_deterministic_given_seed() -> None:
    values = [1.0, 4.0, 9.0, 2.0, 7.0]

    first = bootstrap_ci(values, seed=42)
    second = bootstrap_ci(values, seed=42)

    assert first == second


def test_bootstrap_ci_different_seeds_can_differ() -> None:
    values = [1.0, 4.0, 9.0, 2.0, 7.0, 15.0, 0.5]

    first = bootstrap_ci(values, seed=1, n_samples=200)
    second = bootstrap_ci(values, seed=2, n_samples=200)

    assert first.point == second.point  # point estimate is seed-independent
    assert (first.lower, first.upper) != (second.lower, second.upper)


def test_bootstrap_ci_empty_values() -> None:
    result = bootstrap_ci([])

    assert result.point == 0.0
    assert result.lower == 0.0
    assert result.upper == 0.0


def test_bootstrap_ci_wider_interval_for_higher_confidence() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 0.0, 8.0, 3.5]

    narrow = bootstrap_ci(values, confidence=0.80, seed=7)
    wide = bootstrap_ci(values, confidence=0.99, seed=7)

    assert (wide.upper - wide.lower) >= (narrow.upper - narrow.lower)


# --- write_results_csv / write_results_markdown ---


def test_write_results_csv_round_trip(tmp_path: Path) -> None:
    rows = [
        {"model": "base", "task_completion_pct": 40.0},
        {"model": "sft", "task_completion_pct": 62.5},
    ]
    path = tmp_path / "results.csv"

    write_results_csv(rows, path)
    lines = path.read_text().splitlines()

    assert lines[0] == "model,task_completion_pct"
    assert lines[1] == "base,40.0"
    assert lines[2] == "sft,62.5"


def test_write_results_csv_empty_rows(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"

    write_results_csv([], path)

    assert path.read_text() == ""


def test_write_results_markdown_has_header_and_separator(tmp_path: Path) -> None:
    rows = [{"model": "base", "pct": 40.0}, {"model": "sft", "pct": 62.5}]
    path = tmp_path / "results.md"

    write_results_markdown(rows, path)
    text = path.read_text()
    lines = text.splitlines()

    assert lines[0] == "| model | pct |"
    assert lines[1] == "| --- | --- |"
    assert "base" in lines[2]
    assert "sft" in lines[3]


def test_write_results_markdown_empty_rows(tmp_path: Path) -> None:
    path = tmp_path / "empty.md"

    write_results_markdown([], path)

    assert path.read_text() == ""
