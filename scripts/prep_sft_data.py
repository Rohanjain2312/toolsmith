"""Prepare public function-calling SFT data (xLAM-60k + Glaive v2) as Qwen3 chat-template JSONL."""

# Best-effort schema mapping for Salesforce/xlam-function-calling-60k and
# glaiveai/glaive-function-calling-v2 — reverify both dataset schemas on HF before a real run;
# public dataset column layouts occasionally change. Downloading requires the `datasets`
# library and network access, so `_load_xlam_dataset`/`_load_glaive_dataset` are never called
# in tests; only the pure row-conversion functions are tested, against canned fixture rows.
# Run manually: `uv run python scripts/prep_sft_data.py`.

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

MAX_ROWS = 10_000


def _tools_system_prompt(tools: list[dict[str, Any]]) -> str:
    """Render a tool-list system prompt matching the episode runner's protocol style."""
    lines = [
        "Respond with a single JSON tool call or a plain-text final answer.",
        "Available tools:",
    ]
    for tool in tools:
        lines.append(f"- {tool.get('name', '?')}: {tool.get('description', '')}")
    return "\n".join(lines)


def convert_xlam_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """Convert one xLAM-60k row (query/answers/tools JSON strings) to Qwen3 chat messages."""
    try:
        answers = json.loads(row["answers"])
        tools = json.loads(row["tools"])
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    if not answers or not isinstance(answers, list):
        return None

    messages: list[dict[str, str]] = [
        {"role": "system", "content": _tools_system_prompt(tools)},
        {"role": "user", "content": row["query"]},
    ]
    for call in answers:
        if not isinstance(call, dict) or "name" not in call:
            return None
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps({"tool": call["name"], "args": call.get("arguments", {})}),
            }
        )
    return {"messages": messages, "source": "xlam"}


def convert_glaive_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """Convert one Glaive-v2 row (system + chat transcript) to Qwen3 chat messages."""
    chat = row.get("chat", "")
    if "<functioncall>" not in chat:
        return None  # not a function-calling example

    messages: list[dict[str, str]] = [{"role": "system", "content": row.get("system", "")}]
    for turn in chat.split("USER:")[1:]:
        user_part, _, rest = turn.partition("ASSISTANT:")
        assistant_text = rest.split("FUNCTION RESPONSE:")[0].replace("<|endoftext|>", "").strip()
        if not assistant_text:
            return None
        messages.append({"role": "user", "content": user_part.strip()})
        messages.append({"role": "assistant", "content": assistant_text})
    return {"messages": messages, "source": "glaive"}


def filter_and_cap(
    rows: list[dict[str, Any] | None], cap: int = MAX_ROWS
) -> list[dict[str, Any]]:
    """Drop unconvertible (None) rows and cap the remainder at `cap`."""
    return [row for row in rows if row is not None][:cap]


def _load_xlam_dataset() -> list[dict[str, Any]]:
    """Download the xLAM-60k dataset from the HF Hub. Not covered by tests (network + HF auth)."""
    from datasets import load_dataset

    return list(load_dataset("Salesforce/xlam-function-calling-60k", split="train"))


def _load_glaive_dataset() -> list[dict[str, Any]]:
    """Download the Glaive-v2 dataset from the HF Hub. Not covered by tests (network + HF auth)."""
    from datasets import load_dataset

    return list(load_dataset("glaiveai/glaive-function-calling-v2", split="train"))


def main() -> int:
    output_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/sft_public_data.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    xlam_rows = [convert_xlam_row(r) for r in _load_xlam_dataset()]
    glaive_rows = [convert_glaive_row(r) for r in _load_glaive_dataset()]
    rows = filter_and_cap(xlam_rows + glaive_rows)

    with output_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} SFT rows to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
