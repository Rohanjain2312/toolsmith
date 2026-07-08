"""Outcome reward R5: roll a candidate action to completion via a frozen policy, cached on disk."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from toolsmith.data.taskspec import GoalCondition
from toolsmith.env.executor import execute_tool_call
from toolsmith.env.model import Model
from toolsmith.env.openai_export import export_all_openai_tools
from toolsmith.env.parser import (
    ParsedFinalAnswer,
    ParsedToolCall,
    ToolCallParseError,
    parse_model_output,
)
from toolsmith.env.runner import MAX_TURNS, run_loop
from toolsmith.env.state import EpisodeState, EpisodeStatus, ToolCallLogEntry
from toolsmith.env.truncation import truncate_tool_result
from toolsmith.rewards.goalcheck import check_goal

R5_GOAL_SATISFIED = 3.0
DEFAULT_CACHE_PATH = Path("results/r5_cache.json")


def _hash_json(value: Any) -> str:
    """Stable sha256 hash of a JSON-serializable value (canonical key ordering)."""
    canonical = json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class ContinuationCache:
    """Disk-persisted cache of (state, action) -> completed EpisodeState, keyed by content hash.

    Sandbox determinism guarantees a given (prefix, candidate action) pair always rolls out to
    the same completion, so caching it is a correctness-safe, MANDATORY speedup for R5's
    rollout-to-completion cost. Persists to `path` on every write so it survives process
    restarts (e.g. a fresh Colab session); tracks hits/misses for W&B hit-rate logging.
    """

    path: Path
    hits: int = field(default=0, init=False)
    misses: int = field(default=0, init=False)
    _store: dict[str, str] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.path.exists():
            self._store = json.loads(self.path.read_text())

    def _key(self, prefix: list[dict[str, Any]], raw_text: str) -> str:
        return f"{_hash_json(prefix)}:{_hash_json(raw_text)}"

    def get(self, prefix: list[dict[str, Any]], raw_text: str) -> EpisodeState | None:
        """Return the cached completed EpisodeState for (prefix, raw_text), or None on a miss."""
        cached = self._store.get(self._key(prefix, raw_text))
        if cached is None:
            self.misses += 1
            return None
        self.hits += 1
        return EpisodeState.from_json(cached)

    def put(self, prefix: list[dict[str, Any]], raw_text: str, state: EpisodeState) -> None:
        """Store a completed EpisodeState for (prefix, raw_text) and persist the cache to disk."""
        self._store[self._key(prefix, raw_text)] = state.to_json()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._store))

    @property
    def hit_rate(self) -> float:
        """Fraction of get() calls that were cache hits (0.0 if none made yet)."""
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


def _reconstruct_prior_tool_calls(prefix: list[dict[str, Any]]) -> list[ToolCallLogEntry]:
    """Re-derive ToolCallLogEntry objects for every tool call already made earlier in `prefix`.

    A decision point's prefix carries only raw messages (system/user/assistant/tool), not the
    original episode's ToolCallLogEntry list, so a candidate rolled out from a prefix beyond
    turn 0 would otherwise start from an empty tool_calls history. Sandbox determinism (same
    input -> identical output, always) guarantees re-executing the same (tool, args) pair
    reproduces the original ok/result/error exactly, so this recovers real values instead of
    guessing them from the logged message text.

    Every assistant message inside `prefix` is guaranteed to be a tool-call turn: a prefix is
    always `state.messages[:index]` for some assistant-turn index (decision_points.py), and an
    episode terminates immediately on its first final answer or parse failure, so no assistant
    turn before the last one can be anything but a successful parse of a tool call.
    """
    entries: list[ToolCallLogEntry] = []
    for turn, message in enumerate(m for m in prefix if m["role"] == "assistant"):
        parsed = parse_model_output(message["content"])
        assert isinstance(parsed, ParsedToolCall)
        exec_result = execute_tool_call(parsed.name, parsed.args)
        entries.append(
            ToolCallLogEntry(
                turn=turn,
                tool_name=parsed.name,
                args=parsed.args,
                ok=exec_result.ok,
                result=exec_result.result,
                error=exec_result.error,
            )
        )
    return entries


def _build_continuation_state(
    task_id: str, prefix: list[dict[str, Any]], raw_text: str, frozen_model: Model
) -> EpisodeState:
    """Execute the candidate action, then roll the episode to completion with a frozen policy."""
    import toolsmith.tools.sandbox  # noqa: F401  (registers all 12 sandbox tools)

    prior_tool_calls = _reconstruct_prior_tool_calls(prefix)
    elapsed_turns = sum(1 for m in prefix if m["role"] == "assistant")
    messages = [*prefix, {"role": "assistant", "content": raw_text}]

    try:
        parsed = parse_model_output(raw_text)
    except ToolCallParseError:
        return EpisodeState(
            task_id=task_id,
            messages=messages,
            tool_calls=prior_tool_calls,
            turn=elapsed_turns,
            status=EpisodeStatus.PARSE_FAILURE,
        )

    if isinstance(parsed, ParsedFinalAnswer):
        return EpisodeState(
            task_id=task_id,
            messages=messages,
            tool_calls=prior_tool_calls,
            turn=elapsed_turns,
            status=EpisodeStatus.FINAL_ANSWER,
            final_answer=parsed.text,
        )

    assert isinstance(parsed, ParsedToolCall)
    exec_result = execute_tool_call(parsed.name, parsed.args)
    tool_calls = [
        *prior_tool_calls,
        ToolCallLogEntry(
            turn=elapsed_turns,
            tool_name=parsed.name,
            args=parsed.args,
            ok=exec_result.ok,
            result=exec_result.result,
            error=exec_result.error,
        ),
    ]
    tool_content = (
        truncate_tool_result(exec_result.result)
        if exec_result.ok and exec_result.result is not None
        else {"error": exec_result.error}
    )
    messages.append({"role": "tool", "tool_name": parsed.name, "content": json.dumps(tool_content)})
    state = EpisodeState(
        task_id=task_id, messages=messages, tool_calls=tool_calls, turn=elapsed_turns + 1
    )

    run_loop(state, frozen_model, export_all_openai_tools(), MAX_TURNS)
    return state


def get_or_build_final_state(
    task_id: str,
    prefix: list[dict[str, Any]],
    raw_text: str,
    frozen_model: Model,
    cache: ContinuationCache,
) -> EpisodeState:
    """Return the cached completed EpisodeState for (prefix, raw_text), building it on a miss."""
    cached = cache.get(prefix, raw_text)
    if cached is not None:
        return cached
    state = _build_continuation_state(task_id, prefix, raw_text, frozen_model)
    cache.put(prefix, raw_text, state)
    return state


def compute_outcome_reward(
    task_id: str,
    prefix: list[dict[str, Any]],
    raw_text: str,
    goal_spec: list[GoalCondition],
    frozen_model: Model,
    cache: ContinuationCache,
) -> float:
    """R5: +3.0 iff the completed episode (cached on (prefix, raw_text)) satisfies the goal spec."""
    state = get_or_build_final_state(task_id, prefix, raw_text, frozen_model, cache)
    return R5_GOAL_SATISFIED if check_goal(goal_spec, state) else 0.0
