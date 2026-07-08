"""R6 efficiency bonus and terminal penalties (hallucinated-tool, max-turns), post-episode."""

from __future__ import annotations

from toolsmith.env.state import EpisodeState, EpisodeStatus

R6_EFFICIENCY_BONUS = 0.5
PENALTY_HALLUCINATED_TOOL = -1.0
PENALTY_MAX_TURNS = -1.0

_UNKNOWN_TOOL_ERROR_PREFIX = "unknown tool:"


def reward_efficiency(state: EpisodeState, min_steps: int) -> float:
    """R6: +0.5 if a final answer was reached in at most min_steps + 1 successful tool calls."""
    if state.status is not EpisodeStatus.FINAL_ANSWER:
        return 0.0
    successful_calls = sum(1 for entry in state.tool_calls if entry.ok)
    return R6_EFFICIENCY_BONUS if successful_calls <= min_steps + 1 else 0.0


def penalty_hallucinated_tool(state: EpisodeState) -> float:
    """-1.0 if any tool call in the episode named an unregistered ("hallucinated") tool."""
    for entry in state.tool_calls:
        if (
            not entry.ok
            and entry.error is not None
            and entry.error.startswith(_UNKNOWN_TOOL_ERROR_PREFIX)
        ):
            return PENALTY_HALLUCINATED_TOOL
    return 0.0


def penalty_max_turns(state: EpisodeState) -> float:
    """-1.0 if the episode was cut off by the max-turns limit without reaching a final answer."""
    return PENALTY_MAX_TURNS if state.status is EpisodeStatus.MAX_TURNS else 0.0
