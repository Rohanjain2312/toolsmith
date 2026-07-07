"""Episode loop: system prompt assembly, generate/parse/execute loop, trajectory logging."""

from __future__ import annotations

import json
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
from toolsmith.env.state import EpisodeState, EpisodeStatus, ToolCallLogEntry
from toolsmith.env.truncation import truncate_tool_result

MAX_TURNS = 8
DEFAULT_TRAJECTORY_DIR = Path("results/trajectories")


def build_system_prompt(tools: list[dict]) -> str:
    """Assemble a system prompt describing the tool-call protocol and available tools."""
    lines = [
        "You are a travel-ops assistant. Respond with EITHER a single JSON tool call "
        '(e.g. {"tool": "<name>", "args": {...}}) OR a plain-text final answer.',
        "Available tools:",
    ]
    for tool in tools:
        fn = tool["function"]
        lines.append(f"- {fn['name']}: {fn['description']}")
    return "\n".join(lines)


def _log(trajectory_file, event: dict) -> None:
    trajectory_file.write(json.dumps(event) + "\n")


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
        while state.turn < max_turns:
            raw = model.generate(state.messages, tools)
            state.messages.append({"role": "assistant", "content": raw})

            try:
                parsed = parse_model_output(raw)
            except ToolCallParseError as exc:
                state.status = EpisodeStatus.PARSE_FAILURE
                event = {"turn": state.turn, "event": "parse_failure", "error": str(exc)}
                _log(trajectory_file, event)
                break

            if isinstance(parsed, ParsedFinalAnswer):
                state.final_answer = parsed.text
                state.status = EpisodeStatus.FINAL_ANSWER
                event = {"turn": state.turn, "event": "final_answer", "text": parsed.text}
                _log(trajectory_file, event)
                break

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
            _log(
                trajectory_file,
                {
                    "turn": state.turn,
                    "event": "tool_call",
                    "tool_name": parsed.name,
                    "args": parsed.args,
                    "ok": exec_result.ok,
                },
            )
            state.turn += 1
        else:
            state.status = EpisodeStatus.MAX_TURNS
            _log(trajectory_file, {"turn": state.turn, "event": "max_turns"})

    return state
