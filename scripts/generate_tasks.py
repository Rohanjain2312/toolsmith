"""Generate synthetic travel-ops tasks via the Anthropic API, validate, dedupe. Human runs this."""

# No automated test for the network path: this hits the real Anthropic Messages API and
# requires ANTHROPIC_API_KEY. Only `parse_batch_response` / `_dedupe` (pure parsing/dedupe
# logic) are unit tested, against canned fixtures.
# Run manually: `uv run python scripts/generate_tasks.py`.

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
import uuid
from datetime import date
from pathlib import Path

from pydantic import ValidationError

from toolsmith.data.prompts import render_prompt
from toolsmith.data.taskspec import TaskSpec, Tier

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MODEL = "claude-sonnet-5"
SANDBOX_TODAY = date(2026, 9, 1)
_WORLDDATA_DIR = Path("src/toolsmith/tools/sandbox/worlddata")
_TIERS: tuple[Tier, ...] = ("T1", "T2", "T3", "T4")


class TaskGenerationRequestError(RuntimeError):
    """Raised when the Anthropic API request fails or its response can't be used."""


def _build_world_context() -> str:
    """Summarize the sandbox world data (cities, IATA codes, routes) for prompt grounding."""
    cities = json.loads((_WORLDDATA_DIR / "cities.json").read_text())
    flights = json.loads((_WORLDDATA_DIR / "flights.json").read_text())
    city_lines = [f"{c['name']} ({c['code']}, {c['country']}, {c['timezone']})" for c in cities]
    route_lines = sorted({f"{f['origin']}->{f['dest']}" for f in flights})[:30]
    return (
        "Cities:\n" + "\n".join(city_lines)
        + "\n\nSample flight routes (IATA codes):\n" + ", ".join(route_lines)
        + f"\n\nSandbox 'today': {SANDBOX_TODAY.isoformat()}"
    )


def _call_anthropic(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """POST one prompt to the Anthropic Messages API and return the raw text response."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise TaskGenerationRequestError("ANTHROPIC_API_KEY is not set")

    body = json.dumps(
        {"model": model, "max_tokens": 4096, "messages": [{"role": "user", "content": prompt}]}
    ).encode("utf-8")
    request = urllib.request.Request(
        ANTHROPIC_API_URL,
        data=body,
        method="POST",
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read())
    except (OSError, urllib.error.URLError) as exc:
        raise TaskGenerationRequestError(f"Anthropic API request failed: {exc}") from exc

    try:
        return "".join(block["text"] for block in payload["content"] if block["type"] == "text")
    except (KeyError, TypeError) as exc:
        raise TaskGenerationRequestError(f"unexpected Anthropic response shape: {payload}") from exc


def _strip_code_fence(text: str) -> str:
    """Strip a leading/trailing ``` fence (optionally with a language tag) if present."""
    if not text.startswith("```"):
        return text
    body = text.strip("`")
    newline = body.find("\n")
    return body[newline + 1 :] if newline != -1 else body


def parse_batch_response(raw_text: str, tier: Tier) -> list[TaskSpec]:
    """Parse one model response into validated TaskSpecs with fresh ids and provisional fields.

    `min_steps` and `split` are placeholders here — scripts/validate_tasks.py recomputes
    min_steps via the solver and assigns the real stratified split.
    """
    stripped = _strip_code_fence(raw_text.strip()).strip()
    try:
        raw_items = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise TaskGenerationRequestError(f"model response was not valid JSON: {exc}") from exc
    if not isinstance(raw_items, list):
        raise TaskGenerationRequestError("model response JSON was not a list")

    specs = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        try:
            spec = TaskSpec.model_validate(
                {
                    "id": f"{tier.lower()}-{uuid.uuid4().hex[:12]}",
                    "tier": tier,
                    "user_prompt": item["user_prompt"],
                    "goal_spec": item["goal_spec"],
                    "min_steps": 0,
                    "split": "train",
                }
            )
        except (ValidationError, KeyError):
            continue
        specs.append(spec)
    return specs


def _dedupe(specs: list[TaskSpec]) -> list[TaskSpec]:
    """Drop tasks whose user_prompt text (normalized) duplicates an earlier one."""
    seen: set[str] = set()
    deduped = []
    for spec in specs:
        key = spec.user_prompt.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(spec)
    return deduped


def generate_tasks_for_tier(tier: Tier, world_context: str, batches: int = 1) -> list[TaskSpec]:
    """Call the Anthropic API `batches` times for one tier and return validated, deduped tasks."""
    prompt = render_prompt(tier, world_context)
    all_specs: list[TaskSpec] = []
    for _ in range(batches):
        all_specs.extend(parse_batch_response(_call_anthropic(prompt), tier))
    return _dedupe(all_specs)


def main() -> int:
    world_context = _build_world_context()
    output_path = Path("results/generated_tasks.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_specs: list[TaskSpec] = []
    for tier in _TIERS:
        specs = generate_tasks_for_tier(tier, world_context)
        print(f"{tier}: generated {len(specs)} tasks")
        all_specs.extend(specs)

    with output_path.open("w") as f:
        for spec in all_specs:
            f.write(spec.model_dump_json() + "\n")
    print(f"wrote {len(all_specs)} tasks to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
