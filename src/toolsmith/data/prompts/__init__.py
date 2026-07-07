"""Prompt templates for synthetic task generation, one per difficulty tier (T1-T4)."""

from __future__ import annotations

from pathlib import Path

from toolsmith.data.taskspec import Tier

_TEMPLATES_DIR = Path(__file__).parent
_TEMPLATE_FILENAMES: dict[str, str] = {
    "T1": "t1.txt", "T2": "t2.txt", "T3": "t3.txt", "T4": "t4.txt",
}


class UnknownTierError(ValueError):
    """Raised when a tier has no associated prompt template."""


def load_prompt_template(tier: Tier) -> str:
    """Load the raw prompt template text for a given task tier."""
    filename = _TEMPLATE_FILENAMES.get(tier)
    if filename is None:
        raise UnknownTierError(f"no prompt template for tier: {tier!r}")
    return (_TEMPLATES_DIR / filename).read_text()


def render_prompt(tier: Tier, world_context: str) -> str:
    """Render a tier's prompt template with the given sandbox world-data context injected."""
    template = load_prompt_template(tier)
    return template.format(world_context=world_context)
