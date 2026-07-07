"""Tests for the tier prompt-template loader."""

from __future__ import annotations

import pytest

from toolsmith.data.prompts import UnknownTierError, load_prompt_template, render_prompt


@pytest.mark.parametrize("tier", ["T1", "T2", "T3", "T4"])
def test_load_prompt_template_reads_file_for_every_tier(tier: str) -> None:
    text = load_prompt_template(tier)

    assert "{world_context}" in text
    assert "goal_spec" in text


def test_load_prompt_template_unknown_tier_raises() -> None:
    with pytest.raises(UnknownTierError):
        load_prompt_template("T5")  # type: ignore[arg-type]


def test_render_prompt_injects_world_context() -> None:
    rendered = render_prompt("T1", "Paris (PAR, France, Europe/Paris)")

    assert "Paris (PAR, France, Europe/Paris)" in rendered
    assert "{world_context}" not in rendered


def test_t1_template_scopes_to_single_tool_call() -> None:
    text = load_prompt_template("T1")

    assert "T1" in text
    assert "tool_was_called_with" in text


def test_t2_template_mentions_multi_tool_chain() -> None:
    text = load_prompt_template("T2")

    assert "T2" in text


def test_t4_template_mentions_traps() -> None:
    text = load_prompt_template("T4")

    assert "answer_contains_fact" in text
