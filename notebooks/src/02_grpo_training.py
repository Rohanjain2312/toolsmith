# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ToolSmith — Step-Level GRPO Training
#
# Colab free-T4 notebook (human-run only; never executed by Claude Code). Continues from the
# SFT-warm-started checkpoint (`01_sft_warmstart.py`) with step-level GRPO over decision points
# (`results/decision_points.jsonl`, from `src/toolsmith/data/decision_points.py`): each training
# sample is one conversation prefix, the policy generates a single next action (G candidates),
# and the composite R1-R6 reward (`src/toolsmith/rewards/composite.py`) scores each candidate.
#
# TRL's `GRPOTrainer` is single-turn — it can't run a multi-step episode natively. R5 (outcome
# reward) works around this by executing the candidate action then rolling the rest of the
# episode to completion with a FROZEN policy snapshot (greedy decoding), never the live-training
# weights. This is not optional: the R5 continuation cache is only correct because the frozen
# policy never changes mid-run, so the same (prefix, action) pair always rolls out identically.
#
# Before running: mount Drive, upload `results/decision_points.jsonl` and `results/tasks.jsonl`
# (or point the paths below at their Drive copies), and set `HF_TOKEN` / `WANDB_API_KEY` in the
# Colab secrets panel.

# %%
# Colab-only setup cell: clone the toolsmith repo and install it editable (matches README's
# Quickstart install method) so later cells can `import toolsmith`, and so relative paths like
# `results/tasks.jsonl` resolve against the cloned repo root. Skip if already cloned.
# %cd /content
# !git clone https://github.com/Rohanjain2312/toolsmith.git
# %cd /content/toolsmith
# %pip install -q -e .

# %%
# Colab-only install cell. Skip if the environment already has these.
# %pip install -q "unsloth @ git+https://github.com/unslothai/unsloth.git" unsloth_zoo
# %pip install -q trl vllm wandb datasets bitsandbytes

# %%
# `pip install -e .`'s editable-install finder isn't reliable for direct (non-pytest) imports in
# this repo -- see CLAUDE.md's "Known environment quirks" for the macOS Gatekeeper .pth case.
# Insert the clone's src/ directly so `toolsmith.*` resolves regardless of that, and to guard
# against a same-named-but-unrelated PyPI package ("toolsmith", not this project) ever shadowing
# it if one ends up pulled in transitively.
import sys  # noqa: E402 (Colab-only cell, top-of-cell by design)

sys.path.insert(0, "/content/toolsmith/src")

# %%
# unsloth must be imported before trl -- at import time it monkeypatches trl's trainer configs
# (SFTConfig, GRPOConfig, ...), and importing trl first leaves that patch half-applied, corrupting
# defaults like eos_token/pad_token (confirmed root cause + fix from an unsloth maintainer:
# https://github.com/unslothai/unsloth/issues/2797 -- filed against SFTTrainer, but the patch is
# applied generically across every trl {X}Trainer/{X}Config pair, GRPOTrainer included).
from unsloth import FastLanguageModel

# isort: split
import json
import os
from pathlib import Path

import wandb
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams

from toolsmith.data.decision_points import extract_decision_points, select_task_subset
from toolsmith.data.taskspec import TaskSpec
from toolsmith.env.model import Model
from toolsmith.env.runner import run_episode
from toolsmith.rewards.composite import make_reward_func
from toolsmith.rewards.goalcheck import check_goal
from toolsmith.rewards.outcome_reward import ContinuationCache

# %%
# --- Config ---
MAX_SEQ_LENGTH = 2048
MAX_PROMPT_LENGTH = 1536
MAX_COMPLETION_LENGTH = 256  # single action, not a full episode; 1536 + 256 <= MAX_SEQ_LENGTH
LORA_R = 16
LORA_ALPHA = 32
NUM_GENERATIONS = 4  # G=4 default per plan; drop to 3 if OOM or hours run long (documented dial)
KL_BETA = 0.04
LEARNING_RATE = 5e-6
CHECKPOINT_STEPS = 50  # also the val-gate evaluation cadence
MAX_STEPS_CEILING = 600
EXIT_GATE_MARGIN_PTS = 10  # halt once val task-completion holds >= SFT baseline + this, twice
RESUME_FROM_CHECKPOINT = False  # flip to True on every session after the first
DECISION_POINT_REFRESH = False  # re-run the current policy to regenerate decision points
# mid-run if reward plateaus below the exit gate; default OFF (a full generation pass, not free)

# Curriculum: T1/T2 decision points are the core run. Expand to T3/T4 only if the gate is
# cleared with budget remaining (conditional, per plan) — flip this list to add tiers.
CURRICULUM_TIERS = ["T1", "T2"]

MODEL_NAME = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"
SFT_LORA_REPO = "rohanjain2312/toolsmith-qwen3-4b"  # swap point: the P3-T03 SFT-tuned adapter
HF_MODEL_REPO = "rohanjain2312/toolsmith-qwen3-4b"

DRIVE_CHECKPOINT_DIR = "/content/drive/MyDrive/toolsmith-checkpoints/grpo"
R5_CACHE_PATH = Path("/content/drive/MyDrive/toolsmith-checkpoints/r5_cache.json")
# Matches where 01_sft_warmstart.py's final cell writes it
# (f"{DRIVE_CHECKPOINT_DIR}/../decision_points.jsonl" there, with that notebook's own
# sft_warmstart-specific DRIVE_CHECKPOINT_DIR) -- override to "results/decision_points.jsonl"
# if you'd rather upload the file directly instead of reading it from Drive.
DECISION_POINTS_PATH = Path("/content/drive/MyDrive/toolsmith-checkpoints/decision_points.jsonl")
TASKS_PATH = Path("results/tasks.jsonl")

# %%
# Bridge Colab Secrets (key icon, left sidebar) into env vars the rest of this notebook reads
# directly. Add HF_TOKEN and WANDB_API_KEY there first, then grant this notebook access when
# prompted -- adding a Colab secret does NOT auto-populate os.environ on its own.
from google.colab import userdata  # noqa: E402 (Colab-only import, top-of-cell by design)

os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY")

# %%
# Mount Drive for checkpoint + R5-cache persistence across Colab sessions.
from google.colab import drive  # noqa: E402 (Colab-only import, top-of-cell by design)

drive.mount("/content/drive")
os.makedirs(DRIVE_CHECKPOINT_DIR, exist_ok=True)

# %%
# --- Load model with vLLM fast inference enabled ---
# Swap point: once the SFT LoRA is pushed (01_sft_warmstart.py -> SFT_LORA_REPO), load that
# adapter here as the GRPO starting policy instead of a fresh get_peft_model() init below.
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    fast_inference=True,  # vLLM-backed generation for rollouts
    gpu_memory_utilization=0.6,  # lower if OOM; leaves room for training + the frozen-policy copy
)

model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=LORA_ALPHA,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)


# %%
class FastGenerateFrozenModel(Model):
    """Adapts Unsloth's fast_generate (vLLM) as a greedy Model, for two distinct uses:

    1. R5 rollouts: constructed once with a FIXED `lora_request` snapshot (the SFT
       checkpoint) and never updated — the R5 continuation cache is only correct as long as
       this policy stays fixed for the whole run, or the same cached (prefix, action) entry
       would go stale as training progressed.
    2. Decision-point refresh (below): constructed with `lora_request=None` to sample from
       the model's CURRENT (live, in-training) adapter weights instead of a frozen snapshot.
    """

    def __init__(self, fast_model, lora_request, sampling_params: SamplingParams) -> None:
        self._model = fast_model
        self._lora_request = lora_request
        self._sampling_params = sampling_params

    def generate(self, messages: list[dict], tools: list[dict]) -> str:
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        output = self._model.fast_generate(
            [prompt_text], sampling_params=self._sampling_params, lora_request=self._lora_request
        )
        return output[0].outputs[0].text


# The frozen snapshot is the SFT checkpoint's LoRA, loaded once, independent of the live
# GRPO-training `model` object above.
frozen_lora_request = model.load_lora(SFT_LORA_REPO)
frozen_model = FastGenerateFrozenModel(
    model, frozen_lora_request, SamplingParams(temperature=0.0, max_tokens=MAX_COMPLETION_LENGTH)
)

# %%
# --- Decision-point dataset (from src/toolsmith/data/decision_points.py), curriculum-filtered ---
train_specs = {
    spec.id: spec
    for line in TASKS_PATH.read_text().splitlines()
    if line.strip()
    for spec in [TaskSpec.model_validate_json(line)]
}
task_lookup = {
    task_id: (spec.goal_spec, spec.min_steps) for task_id, spec in train_specs.items()
}

decision_point_rows = []
for line in DECISION_POINTS_PATH.read_text().splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    spec = train_specs.get(row["task_id"])
    if spec is not None and spec.tier in CURRICULUM_TIERS:
        decision_point_rows.append({"prompt": row["prefix"], "task_ids": row["task_id"]})

decision_point_dataset = Dataset.from_list(decision_point_rows)

# %%
# --- Composite R1-R6 reward, wired to the frozen policy + mandatory persistent R5 cache ---
r5_cache = ContinuationCache(path=R5_CACHE_PATH)
composite_reward_func = make_reward_func(task_lookup, frozen_model, r5_cache)

wandb.login(key=os.environ["WANDB_API_KEY"])

# %%
training_args = GRPOConfig(
    learning_rate=LEARNING_RATE,
    beta=KL_BETA,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    logging_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=NUM_GENERATIONS,
    max_prompt_length=MAX_PROMPT_LENGTH,
    max_completion_length=MAX_COMPLETION_LENGTH,
    max_steps=CHECKPOINT_STEPS,  # raised in CHECKPOINT_STEPS increments by the loop below
    save_steps=CHECKPOINT_STEPS,
    max_grad_norm=0.1,
    report_to="wandb",
    output_dir=DRIVE_CHECKPOINT_DIR,
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[composite_reward_func],
    args=training_args,
    train_dataset=decision_point_dataset,
)


# %%
def run_val_gate(val_specs: list[TaskSpec], policy: Model) -> float:
    """Greedy task-completion % over val_specs, used for the early-stopping exit gate."""
    passed = sum(
        check_goal(
            spec.goal_spec,
            run_episode(spec.id, spec.user_prompt, policy, trajectory_dir=Path("/tmp/val_gate")),
        )
        for spec in val_specs
    )
    return 100.0 * passed / len(val_specs) if val_specs else 0.0


val_specs = [s for s in train_specs.values() if s.split == "val"]
SFT_BASELINE_COMPLETION_PCT = 0.0  # human fills in: measured via scripts/sanity_eval.py post-SFT


def refresh_decision_points() -> Dataset:
    """Re-run the CURRENT (live, in-training) policy over a fresh T1/T2-weighted task subset
    and rebuild the decision-point dataset from its trajectories.

    Costs a full generation pass over the subset — not free — which is why
    DECISION_POINT_REFRESH defaults to False (see plan §7): only called when the reward has
    plateaued below the exit gate, so the training loop below gates this behind that flag.
    """
    live_policy = FastGenerateFrozenModel(
        model, None, SamplingParams(temperature=0.0, max_tokens=MAX_COMPLETION_LENGTH)
    )
    curriculum_pool = [
        spec
        for spec in train_specs.values()
        if spec.split == "train" and spec.tier in CURRICULUM_TIERS
    ]
    subset = select_task_subset(curriculum_pool)
    states = [
        run_episode(spec.id, spec.user_prompt, live_policy, trajectory_dir=Path("/tmp/refresh"))
        for spec in subset
    ]
    points = extract_decision_points(states)
    rows = [{"prompt": point.prefix, "task_ids": point.task_id} for point in points]
    return Dataset.from_list(rows)

# %%
# --- Train in CHECKPOINT_STEPS-sized chunks; evaluate + early-stop between chunks ---
# A chunked outer loop (rather than an internal TRL callback) keeps the exit-gate logic in
# plain, auditable Python instead of depending on GRPOTrainer-internal control-flow hooks.
step = 0
consecutive_gate_passes = 0
resume = RESUME_FROM_CHECKPOINT

while step < MAX_STEPS_CEILING and consecutive_gate_passes < 2:
    training_args.max_steps = min(step + CHECKPOINT_STEPS, MAX_STEPS_CEILING)
    trainer.train(resume_from_checkpoint=resume)
    resume = True
    step = training_args.max_steps

    completion_pct = run_val_gate(val_specs, frozen_model)  # frozen policy: cheap, cache-safe
    wandb.log(
        {
            "step": step,
            "val_task_completion_pct": completion_pct,
            "r5_cache_hit_rate": r5_cache.hit_rate,
        }
    )
    if composite_reward_func.last_component_log:
        for key in composite_reward_func.last_component_log[0]:
            values = [row[key] for row in composite_reward_func.last_component_log]
            wandb.log({f"reward/{key}": sum(values) / len(values), "step": step})

    if completion_pct >= SFT_BASELINE_COMPLETION_PCT + EXIT_GATE_MARGIN_PTS:
        consecutive_gate_passes += 1
    else:
        consecutive_gate_passes = 0
        if DECISION_POINT_REFRESH:
            trainer.train_dataset = refresh_decision_points()

# %%
# --- Save LoRA adapter + push to HF Hub ---
model.save_lora(f"{DRIVE_CHECKPOINT_DIR}/grpo_lora")
model.push_to_hub(HF_MODEL_REPO, token=os.environ["HF_TOKEN"])
tokenizer.push_to_hub(HF_MODEL_REPO, token=os.environ["HF_TOKEN"])
