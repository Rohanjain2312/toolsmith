"""Episode loop: system prompt assembly, generate/parse/execute loop, trajectory logging."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from toolsmith.env.executor import execute_tool_call
from toolsmith.env.model import Model
from toolsmith.env.openai_export import export_all_openai_tools
from toolsmith.env.parser import (
    ParsedFinalAnswer,
    ParsedToolCall,
    ToolCallParseError,
    parse_model_output,
)
from toolsmith.env.prompts import load_system_prompt_template
from toolsmith.env.state import EpisodeState, EpisodeStatus, ToolCallLogEntry
from toolsmith.env.truncation import truncate_tool_result

MAX_TURNS = 8
DEFAULT_TRAJECTORY_DIR = Path("results/trajectories")


def build_system_prompt(tools: list[dict]) -> str:
    """Assemble a system prompt describing the tool-call protocol and available tools."""
    lines = [load_system_prompt_template()]
    for tool in tools:
        fn = tool["function"]
        lines.append(f"- {fn['name']}: {fn['description']}")
    return "\n".join(lines)


def _log(trajectory_file, event: dict) -> None:
    trajectory_file.write(json.dumps(event) + "\n")


def run_loop(
    state: EpisodeState,
    model: Model,
    tools: list[dict],
    max_turns: int,
    log: Callable[[dict], None] | None = None,
) -> None:
    """Drive an episode's generate/parse/execute loop in place until termination.

    Shared by `run_episode` (fresh episodes, with trajectory-file logging) and the GRPO R5
    outcome reward's frozen-policy continuation (resuming from an existing prefix, unlogged).
    """
    while state.turn < max_turns:
        raw = model.generate(state.messages, tools)
        state.messages.append({"role": "assistant", "content": raw})

        try:
            parsed = parse_model_output(raw)
        except ToolCallParseError as exc:
            state.status = EpisodeStatus.PARSE_FAILURE
            if log is not None:
                log({"turn": state.turn, "event": "parse_failure", "error": str(exc)})
            return

        if isinstance(parsed, ParsedFinalAnswer):
            state.final_answer = parsed.text
            state.status = EpisodeStatus.FINAL_ANSWER
            if log is not None:
                log({"turn": state.turn, "event": "final_answer", "text": parsed.text})
            return

        assert isinstance(parsed, ParsedToolCall)
        exec_result = execute_tool_call(parsed.name, parsed.args)
        state.tool_calls.append(
            ToolCallLogEntry(
                turn=state.turn,
                tool_name=parsed.name,
                args=parsed.args,
                ok=exec_result.ok,
                result=exec_result.result,
                error=exec_result.error,
            )
        )
        tool_content = (
            truncate_tool_result(exec_result.result)
            if exec_result.ok and exec_result.result is not None
            else {"error": exec_result.error}
        )
        state.messages.append(
            {"role": "tool", "tool_name": parsed.name, "content": json.dumps(tool_content)}
        )
        if log is not None:
            log(
                {
                    "turn": state.turn,
                    "event": "tool_call",
                    "tool_name": parsed.name,
                    "args": parsed.args,
                    "ok": exec_result.ok,
                }
            )
        state.turn += 1
    else:
        state.status = EpisodeStatus.MAX_TURNS
        if log is not None:
            log({"turn": state.turn, "event": "max_turns"})


def run_episode(
    task_id: str,
    user_message: str,
    model: Model,
    max_turns: int = MAX_TURNS,
    trajectory_dir: Path = DEFAULT_TRAJECTORY_DIR,
) -> EpisodeState:
    """Run one episode to completion: final answer, max-turns, or parse-failure."""
    tools = export_all_openai_tools()
    state = EpisodeState(
        task_id=task_id,
        messages=[
            {"role": "system", "content": build_system_prompt(tools)},
            {"role": "user", "content": user_message},
        ],
    )
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = trajectory_dir / f"{task_id}.jsonl"

    with trajectory_path.open("w") as trajectory_file:
        run_loop(state, model, tools, max_turns, log=lambda event: _log(trajectory_file, event))

    return state
