"""Episode state container tracking messages, tool calls, and termination status."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class EpisodeStatus(str, Enum):  # noqa: UP042 (str mixin required for JSON serialization)
    """The running/termination state of an episode."""

    IN_PROGRESS = "in_progress"
    FINAL_ANSWER = "final_answer"
    MAX_TURNS = "max_turns"
    PARSE_FAILURE = "parse_failure"


@dataclass(frozen=True)
class ToolCallLogEntry:
    """A single logged tool-execution outcome for one turn of an episode."""

    turn: int
    tool_name: str
    args: dict[str, Any]
    ok: bool
    result: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class EpisodeState:
    """Mutable state of an in-progress or completed episode."""

    task_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[ToolCallLogEntry] = field(default_factory=list)
    turn: int = 0
    status: EpisodeStatus = EpisodeStatus.IN_PROGRESS
    final_answer: str | None = None

    def to_json(self) -> str:
        """Serialize the episode state to a JSON string."""
        data: dict[str, Any] = asdict(self)
        data["status"] = self.status.value
        return json.dumps(data)

    @classmethod
    def from_json(cls, s: str) -> EpisodeState:
        """Deserialize a JSON string back into an EpisodeState."""
        data = json.loads(s)
        tool_calls = [ToolCallLogEntry(**entry) for entry in data["tool_calls"]]
        return cls(
            task_id=data["task_id"],
            messages=data["messages"],
            tool_calls=tool_calls,
            turn=data["turn"],
            status=EpisodeStatus(data["status"]),
            final_answer=data["final_answer"],
        )
