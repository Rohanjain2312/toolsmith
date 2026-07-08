---
license: mit
base_model: unsloth/Qwen3-4B-Instruct-2507
tags:
  - tool-calling
  - agents
  - grpo
  - lora
  - travel
  - unsloth
pipeline_tag: text-generation
---

# ToolSmith Qwen3-4B

Qwen3-4B-Instruct-2507, post-trained for reliable multi-step tool calling via LoRA SFT
warm-start followed by step-level GRPO with verifiable, sandbox-computed rewards. No labels
or gold trajectories were used at the RL stage — every reward is computed by executing the
candidate action in a deterministic 12-tool travel-ops sandbox and checking a machine-checkable
goal spec.

## Training Details

**Base model:** `unsloth/Qwen3-4B-Instruct-2507` (non-thinking variant — avoids `<think>` block
token blowup during RL rollouts).

**Stage 1 — SFT warm-start** (`notebooks/src/01_sft_warmstart.py`):
- LoRA r={LORA_R}, alpha={LORA_ALPHA}, dropout 0, target modules: q/k/v/o/gate/up/down_proj
- Data: {SFT_PUBLIC_ROW_COUNT} rows from xLAM-60k + Glaive v2 (public function-calling corpora)
  + {SFT_GOLD_ROW_COUNT} gold trajectories (Anthropic-generated, sandbox-verified, train split
  only)
- Sequence length {MAX_SEQ_LENGTH}, batch {BATCH_SIZE} x grad-accum {GRAD_ACCUM_STEPS}, cosine
  LR {SFT_LEARNING_RATE}, {NUM_EPOCHS} epoch(s), `train_on_responses_only` (loss on assistant
  tokens only)

**Stage 2 — step-level GRPO** (`notebooks/src/02_grpo_training.py`):
- Decision-point dataset: {DECISION_POINT_COUNT} points from a T1/T2-weighted subset of
  {DECISION_POINT_TASK_COUNT} train tasks
- G={NUM_GENERATIONS} generations/prompt, max completion {MAX_COMPLETION_LENGTH} tokens
  (single action, not a full episode), KL β={KL_BETA}, LR={GRPO_LEARNING_RATE}
- Curriculum: {CURRICULUM_TIERS} core; conditional expansion documented but not required
- Ran for {GRPO_STEP_COUNT} steps; halted by the early-stopping exit gate (SFT + 10 pts task
  completion held for 2 consecutive 50-step checkpoints) / reached the step ceiling — {HALT_REASON}

## Reward Spec

| Component | Description | Weight |
|---|---|---|
| R1 | Output parses as valid tool-call JSON or a valid final answer | +1.0 |
| R2 | Called tool exists in the closed 12-tool registry | +0.5 |
| R3 | Args pass that tool's Pydantic validation | +1.0 |
| R4 | No redundant call (same tool + args already in the prefix) | +0.5 |
| R5 | Sandbox goal spec satisfied (execute action → complete episode with a frozen greedy
  policy → goal-check) | +3.0 |
| R6 | Efficiency bonus: episode finished within min_steps + 1 tool calls | +0.5 |
| Penalty | Hallucinated (unregistered) tool name | −1.0 |
| Penalty | Max-turns limit hit without a final answer | −1.0 |

R5's rollout-to-completion cost is amortized by a mandatory, disk-persisted cache keyed on
(prefix, action) content hash — sandbox determinism guarantees a cached entry is always valid.

## Evaluation

4-way comparison on {TEST_TASK_COUNT} held-out test tasks, greedy decoding, identical system
prompt and tool schemas across all four models, bootstrap 95% CIs:

| Model | JSON valid % | Correct tool % | Arg accuracy % | Task completion % | Cost / 1k calls |
|---|---|---|---|---|---|
| Qwen3-4B base | {BASE_JSON_VALID} | {BASE_CORRECT_TOOL} | {BASE_ARG_ACC} | {BASE_COMPLETION} | ~$0 (self-host) |
| + SFT | {SFT_JSON_VALID} | {SFT_CORRECT_TOOL} | {SFT_ARG_ACC} | {SFT_COMPLETION} | ~$0 (self-host) |
| + SFT + GRPO | {GRPO_JSON_VALID} | {GRPO_CORRECT_TOOL} | {GRPO_ARG_ACC} | {GRPO_COMPLETION} | ~$0 (self-host) |
| GPT-4o-mini | {GPT4OMINI_JSON_VALID} | {GPT4OMINI_CORRECT_TOOL} | {GPT4OMINI_ARG_ACC} | {GPT4OMINI_COMPLETION} | {GPT4OMINI_COST} |

**BFCL** (AST subsets: simple / multiple / parallel only — live-API categories out of scope
for a free-tier build): base {BFCL_BASE}, SFT {BFCL_SFT}, GRPO {BFCL_GRPO}. Reported to check
that domain RL didn't wreck general tool-calling ability.

**LLM judge** (blinded 1-5 rubric, {JUDGE_TASK_COUNT} tasks, calibrated against
{JUDGE_CALIBRATION_COUNT} hand-scored samples): {JUDGE_MAE} mean absolute error vs. human
scores.

## Honesty Caveats

- A domain-tuned 4B model beating a generalist model on its own sandbox is the expected
  outcome, not evidence of general capability — this is why BFCL (out-of-domain) is reported
  alongside the in-domain comparison.
- Step-level GRPO (single-action completions scored via frozen-policy rollout-to-completion)
  is not equivalent to trajectory-level RL; credit assignment across a full multi-step episode
  is noisier than it would be with true multi-turn rollouts. Documented as a scoped tradeoff,
  not a limitation hidden from the reader.
- Report both wins and reward-hacking incidents observed during training; see
  `results/notes.md` for the trajectory-audit log.

## Contamination Disclosure

- SFT gold trajectories are drawn only from the `train` split of `rohanjain2312/toolsmith-tasks`;
  the held-out `test` split is never touched by training or hyperparameter selection.
- The public SFT corpus (xLAM-60k, Glaive v2) shares distribution with BFCL's own training
  data — this model is **not** claimed to be BFCL-training-clean, and the BFCL score above
  should be read with that in mind.
- GRPO's decision-point dataset is generated by replaying the SFT checkpoint over train-split
  tasks only.

## Intended Use

Research and portfolio demonstration of label-free, sandbox-verified RL for tool-calling.
Domain-scoped to the 12-tool travel-ops sandbox described in `src/toolsmith/tools/`; not
intended as a general-purpose function-calling model without further fine-tuning.

## License

MIT.
