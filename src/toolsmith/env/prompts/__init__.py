"""Versioned system-prompt loader: pins which prompt file build_system_prompt uses.

Changing the episode protocol's instructions means adding a new system_v{N}.txt file and
bumping SYSTEM_PROMPT_VERSION here, not editing an existing file in place — the eval-gate CI
workflow (P8-T01) runs on every push, so any prompt-file change gets covered automatically,
and a versioned file (vs. mutating v1) keeps old cached trajectories/rewards reproducible.
"""

from __future__ import annotations

from pathlib import Path

SYSTEM_PROMPT_VERSION = "v1"
_PROMPTS_DIR = Path(__file__).parent


class UnknownPromptVersionError(ValueError):
    """Raised when no system_{version}.txt file exists for the requested version."""


def load_system_prompt_template(version: str = SYSTEM_PROMPT_VERSION) -> str:
    """Load the pinned system-prompt template text for `version`."""
    path = _PROMPTS_DIR / f"system_{version}.txt"
    if not path.exists():
        raise UnknownPromptVersionError(f"no prompt file for version: {version!r}")
    return path.read_text()
