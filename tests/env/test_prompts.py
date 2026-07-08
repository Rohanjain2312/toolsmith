"""Tests for the versioned system-prompt loader."""

from __future__ import annotations

import pytest

from toolsmith.env.prompts import (
    SYSTEM_PROMPT_VERSION,
    UnknownPromptVersionError,
    load_system_prompt_template,
)


def test_default_version_loads() -> None:
    text = load_system_prompt_template()

    assert "tool" in text.lower()
    assert len(text) > 0


def test_pinned_version_constant_matches_a_real_file() -> None:
    # Loading with the explicit pinned version must succeed identically to the default.
    assert load_system_prompt_template(SYSTEM_PROMPT_VERSION) == load_system_prompt_template()


def test_unknown_version_raises() -> None:
    with pytest.raises(UnknownPromptVersionError):
        load_system_prompt_template("v999")
