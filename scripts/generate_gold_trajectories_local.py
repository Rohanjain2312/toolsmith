"""Construct gold SFT trajectories locally (no live LLM API), for train-split tasks.

Alternative to scripts/generate_gold_trajectories.py's Anthropic API path. Every task this
repo's local generator produces (scripts/generate_tasks_local.py) has a goal_spec made entirely
of `tool_was_called_with` conditions -- i.e. the goal_spec IS the exact ordered tool-call
sequence a correct agent must produce. So instead of asking a live model to improvise that
sequence, this scripts it directly: each condition's (tool_name, args) becomes one scripted
assistant turn, executed for real against the sandbox (deterministic, so the trajectory's tool
results are genuine, not fabricated), then a final natural-language answer is templated from the
last tool's real result. The scripted StubModel is fed through the exact same
generate_gold_trajectory()/write_gold_sft_rows() pipeline the API-based script uses, so the
check_goal() pass/fail gate and output format are unchanged.

Run manually: `PYTHONPATH=".:src" uv run python scripts/generate_gold_trajectories_local.py`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts.generate_gold_trajectories import (
    GOLD_SFT_OUTPUT_PATH,
    GOLD_TRAJECTORY_DIR,
    generate_gold_trajectory,
    load_train_tasks,
    write_gold_sft_rows,
)
from toolsmith.data.taskspec import (
    AnswerContainsFactCondition,
    TaskSpec,
    ToolWasCalledWithCondition,
)
from toolsmith.env.executor import execute_tool_call
from toolsmith.env.model import StubModel
from toolsmith.env.state import EpisodeState


def _final_answer(tool_name: str, result: dict) -> str:
    """Template a natural-language final answer from one tool's real (executed) result."""
    if tool_name == "geocode_city":
        return (
            f"{result['city']} is located at ({result['lat']}, {result['lon']}) in "
            f"{result['country']} (timezone {result['timezone']})."
        )
    if tool_name == "weather_lookup":
        return (
            f"The forecast for {result['date']} is {result['summary']} with a temperature "
            f"around {result['temp_c']}°C."
        )
    if tool_name == "flight_search":
        flights = result["flights"]
        if not flights:
            return "I didn't find any matching flights for that route and date."
        f = flights[0]
        return (
            f"I found flight {f['id']} for {f['price']} {f['currency']}, "
            f"departing {f['depart']}."
        )
    if tool_name == "currency_convert":
        return (
            f"{result['amount']} {result['from_currency']} converts to about "
            f"{result['converted']:.2f} {result['to_currency']}."
        )
    if tool_name == "timezone_info":
        return (
            f"That's timezone {result['timezone']}, UTC offset "
            f"{result['utc_offset_minutes']} minutes, local time {result['local_time']}."
        )
    if tool_name == "calendar_create_event":
        return f"Done — event {result['event_id']} is {result['status']}."
    if tool_name == "country_info":
        langs = ", ".join(result["languages"])
        return (
            f"{result['country']} uses {result['currency']}, speaks {langs}, plug type "
            f"{result['plug_type']}. {result['visa_note']}"
        )
    if tool_name == "poi_search":
        pois = result["pois"]
        if not pois:
            return "I didn't find any matching points of interest nearby."
        p = pois[0]
        return f"The closest match is {p['name']}, about {p['distance_km']:.2f} km away."
    if tool_name == "distance_calc":
        return f"They're about {result['distance_km']:.1f} km apart."
    if tool_name == "packing_rules":
        return f"For that trip, pack: {', '.join(result['items'])}."
    if tool_name == "unit_convert":
        return (
            f"{result['value']} {result['from_unit']} is about "
            f"{result['converted']:.2f} {result['to_unit']}."
        )
    if tool_name == "datetime_math":
        return f"That date is {result['result_date']} ({result['weekday_name']})."
    raise ValueError(f"no final-answer template for tool {tool_name!r}")


_DECLINE_TEMPLATES = {
    "cannot": (
        "I'm sorry, but I cannot find that in my available data — "
        "could you double-check the details?"
    ),
    "don't have": (
        "I don't have that information in my system — is there anything else I can help with?"
    ),
    "clarify": (
        "I'd need a bit more detail to help with that — could you clarify what "
        "you're looking for?"
    ),
}


def _decline_answer(fact: str) -> str:
    """A short declining/clarifying final answer containing the goal's required fact substring.

    For T4 "trap" tasks (goal_spec is a single answer_contains_fact condition): the correct
    agent behavior is exactly this -- no tool call, a plain-text refusal or clarifying question.
    """
    template = _DECLINE_TEMPLATES.get(fact)
    if template is None:
        raise ValueError(f"no decline-answer template for fact {fact!r}")
    return template


def build_scripted_responses(spec: TaskSpec) -> list[str]:
    """Build the exact scripted turns a StubModel needs to satisfy spec's goal_spec.

    Two shapes, matching this repo's local task generator:
    - tool_was_called_with conditions (T1-T3): one JSON tool-call turn per condition (executed
      for real to confirm it succeeds), then one final-answer turn templated from the last
      tool's real result.
    - a single answer_contains_fact condition (T4 traps): no tool calls at all -- one
      declining/clarifying final-answer turn containing the required fact substring.
    """
    import toolsmith.tools.sandbox  # noqa: F401  (registers all 12 sandbox tools)

    if len(spec.goal_spec) == 1 and isinstance(spec.goal_spec[0], AnswerContainsFactCondition):
        return [_decline_answer(spec.goal_spec[0].fact)]

    responses = []
    last_tool: str | None = None
    last_result: dict | None = None
    for condition in spec.goal_spec:
        if not isinstance(condition, ToolWasCalledWithCondition):
            raise ValueError(f"{spec.id}: gold-trajectory scripting only supports tool conditions")
        responses.append(json.dumps({"tool": condition.tool_name, "args": condition.args}))
        exec_result = execute_tool_call(condition.tool_name, condition.args)
        if not exec_result.ok:
            raise ValueError(f"{spec.id}: {condition.tool_name} failed: {exec_result.error}")
        last_tool, last_result = condition.tool_name, exec_result.result
    responses.append(_final_answer(last_tool, last_result))
    return responses


def build_gold_trajectories(specs: list[TaskSpec], trajectory_dir: Path) -> list[EpisodeState]:
    """Script and run every task, keeping only those whose goal check passes."""
    states = []
    for spec in specs:
        model = StubModel(build_scripted_responses(spec))
        state = generate_gold_trajectory(spec, model, trajectory_dir=trajectory_dir)
        if state is not None:
            states.append(state)
    return states


def main() -> int:
    tasks_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/tasks.jsonl")
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else GOLD_SFT_OUTPUT_PATH
    specs = load_train_tasks(tasks_path)

    states = build_gold_trajectories(specs, GOLD_TRAJECTORY_DIR)
    write_gold_sft_rows(states, output_path)
    print(f"{len(states)}/{len(specs)} gold trajectories passed the goal check")
    print(f"wrote {len(states)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
