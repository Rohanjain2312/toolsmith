---
license: mit
task_categories:
  - other
tags:
  - tool-calling
  - agents
  - synthetic
  - travel
pretty_name: ToolSmith Tasks
---

# ToolSmith Tasks

Synthetic travel-ops tool-calling tasks for training and evaluating {MODEL_NAME}, generated
against the ToolSmith deterministic 12-tool sandbox.

## Dataset Summary

- **Tasks:** {TOTAL_TASK_COUNT} across 4 difficulty tiers (T1 single-tool, T2 2-3 tool chains,
  T3 4-6 tool chains with dependencies, T4 traps/impossible requests)
- **Splits:** train / val / test, stratified by tier ({TRAIN_COUNT} / {VAL_COUNT} / {TEST_COUNT})
- **Generation:** synthetic, produced by prompting Claude against the sandbox world data
  (`src/toolsmith/tools/sandbox/worlddata/`), validated for 100% solvability by a bounded BFS
  solver (`src/toolsmith/data/solver.py`)
- **Goal specs:** every task carries a machine-checkable goal spec (not a gold trajectory) —
  rewards verify sandbox outcomes, not paths

## Fields

| Field | Type | Description |
|---|---|---|
| `id` | string | unique task id |
| `tier` | string | one of T1, T2, T3, T4 |
| `user_prompt` | string | the natural-language traveler request |
| `goal_spec` | list | machine-checkable conditions (see below) |
| `min_steps` | int | solver-computed minimum sandbox tool calls to satisfy the goal |
| `split` | string | one of train, val, test |

### Goal condition types

- `answer_contains_fact` — final answer text must contain a given substring
- `tool_was_called_with` — a successful tool call must match a tool name + arg subset
- `calendar_event_exists` — a `calendar_create_event` call with exact fields must have succeeded
- `numeric_within_tolerance` — a number from the final answer or a tool result must be near
  an expected value

## How This Dataset Is Used

- **SFT** (`notebooks/src/01_sft_warmstart.py`): `train`-split tasks are replayed through the
  Anthropic API inside the episode runner; trajectories whose goal spec passes become gold SFT
  rows (`scripts/generate_gold_trajectories.py`).
- **GRPO** (`notebooks/src/02_grpo_training.py`): `goal_spec` feeds the R5 outcome reward
  (`src/toolsmith/rewards/outcome_reward.py`) directly — every candidate action is scored by
  executing it in the sandbox and checking these same conditions. `min_steps` feeds the R6
  efficiency bonus.
- **Eval** (`src/toolsmith/eval/runner.py`): the `test` split is the held-out 4-way comparison
  suite ({TEST_TASK_COUNT} tasks); see the model card's evaluation table for results.

## Contamination Controls

SFT gold trajectories are drawn only from the `train` split; `test` never touches training.
This dataset was generated independently of, and shares no rows with, the xLAM/Glaive public
function-calling corpora used for SFT warm-start.

## License

MIT.
