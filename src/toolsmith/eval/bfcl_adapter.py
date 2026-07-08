"""Convert our tool-call protocol into BFCL's AST-subset harness expectations.

BFCL (`bfcl-eval` on PyPI) runs as a separate CLI in its own workspace (`bfcl generate` /
`bfcl evaluate`, driven by `BFCL_PROJECT_ROOT`) — it is not a dependency of this repo. Every
BFCL model handler implements two conversion methods, `decode_ast` and `decode_execute`; BFCL
already ships a `qwen_fc.py` handler for Qwen's native tool-call format, but our SFT/GRPO model
is trained to emit OUR project's protocol (`{"tool": name, "args": {...}}` JSON text, per
env/parser.py), not Qwen's native format. This module implements that pair of conversions
against our protocol; a human wires them into a small custom handler class inside the separate
BFCL workspace (subclassing `qwen_fc.py` and delegating `decode_ast`/`decode_execute` here).

Scope: only the `simple`, `multiple`, and `parallel` AST-checked categories, per plan scope
(exec categories and live-API categories are explicitly out of scope for a free-tier build).
"""

from __future__ import annotations

import json

from toolsmith.env.parser import find_balanced_brace_spans

SUPPORTED_CATEGORIES: tuple[str, ...] = ("simple", "multiple", "parallel")


class UnsupportedBFCLCategoryError(ValueError):
    """Raised when a BFCL test category outside our AST-subset scope is requested."""


def validate_category(category: str) -> None:
    """Raise UnsupportedBFCLCategoryError unless `category` is in SUPPORTED_CATEGORIES."""
    if category not in SUPPORTED_CATEGORIES:
        raise UnsupportedBFCLCategoryError(
            f"unsupported BFCL category {category!r}; scoped to {SUPPORTED_CATEGORIES}"
        )


def _decode_calls(response: str) -> list[tuple[str, dict]]:
    """Extract every valid (tool_name, args) pair from a raw model response.

    Unlike env/parser.py's parse_model_output (which takes only the first candidate, since the
    episode loop generates one action per turn), this takes ALL valid spans found — required
    for BFCL's "parallel" category, where a single response may contain multiple calls.
    """
    calls = []
    for candidate in find_balanced_brace_spans(response.strip()):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "tool" in parsed and "args" in parsed:
            name, args = parsed["tool"], parsed["args"]
            if isinstance(name, str) and isinstance(args, dict):
                calls.append((name, args))
    return calls


def decode_ast(response: str) -> list[dict[str, dict]]:
    """Convert a raw model response into BFCL's AST format: [{func_name: {arg: value}}, ...].

    Argument values keep their native JSON types (str/int/float/bool/list/dict) since BFCL's
    AST checker compares them structurally against ground truth, not as strings. Returns an
    empty list (never raises) for unparseable or tool-call-free responses, so a wrong or
    malformed response simply fails the AST match rather than crashing the harness.
    """
    return [{name: args} for name, args in _decode_calls(response)]


def _render_arg(value: object) -> str:
    """Render one argument value as a Python literal, for the decode_execute call-string form."""
    return repr(value)


def decode_execute(response: str) -> list[str]:
    """Convert a raw model response into BFCL's callable-string form: ["func(arg=value)", ...]."""
    calls = _decode_calls(response)
    return [
        f"{name}({', '.join(f'{key}={_render_arg(value)}' for key, value in args.items())})"
        for name, args in calls
    ]
