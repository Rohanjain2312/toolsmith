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
# trl pinned to 0.24.0 (top of unsloth's own declared compatible range) to match what the
# mergekit/llm_blender import-chain stub fix below was actually verified against -- an unpinned
# install could silently drift to a trl version where that stub's assumptions don't hold.
# %pip install -q "unsloth @ git+https://github.com/unslothai/unsloth.git" unsloth_zoo
# %pip install -q "trl==0.24.0" wandb datasets bitsandbytes
# Plain `pip install vllm` resolves to the PyPI default wheel, compiled against CUDA 13 (needs
# libcudart.so.13). Colab's T4 tier reports CUDA 12.8 system-wide -- that .so doesn't exist
# there, so vllm's compiled extension fails to load, and unsloth permanently disables all vllm
# imports for the rest of the kernel session the first time that happens (so reinstalling
# afterward, in the same session, does nothing -- both packages have to be right on the FIRST
# install). No published vllm release has a cu128-tagged asset (checked the GitHub releases API
# directly across several recent versions), but cu129 does -- and since only the .so's MAJOR
# version has to match (CUDA's own minor-version compatibility guarantee), a cu129 build's
# libcudart.so.12 is satisfied by a CUDA 12.8 system, unlike the default build's libcudart.so.13.
#
# T4-specific: unlike a G4 (RTX PRO 6000 Blackwell, sm_120), T4 (sm_75) doesn't need CUDA 13 at
# all -- FlashInfer's ">=12.9 to recognize Blackwell" requirement doesn't apply to this
# architecture, so there's no reason to chase CUDA 13 here the way the G4 path had to.
#
# `--torch-backend` only steers which index TORCH resolves against -- it does NOT redirect which
# CUDA-variant of vllm itself gets installed. `vllm` (bare, no wheel pin) always resolves to its
# plain PyPI default, which targets CUDA 13 regardless of --torch-backend. The G4/Blackwell fix
# (a previous version of this cell) appeared to work with `--torch-backend=cu130` alone only
# because vllm's plain default ALREADY targets CUDA 13 -- torch was the only piece that needed
# steering there. On T4, that coincidence doesn't hold: vllm needs to actually be cu129, and the
# only way to get that is a specific wheel, not a --torch-backend flag. Confirmed on a live T4
# run: `--torch-backend=cu129` alone still produced "vLLM was built for CUDA 13" (the exact same
# warning as before any of this was fixed) -- vllm's own wheel was never actually redirected.
#
# `--reinstall` is still needed (uv won't replace an already-installed package that nominally
# satisfies a constraint), and vllm + its torch dependency are still resolved together in one
# command (vllm's compiled kernels are ABI-coupled to a specific torch build) -- just installing
# the actual cu129 wheel this time instead of trusting --torch-backend to pick it.
# %pip install -q uv
# !uv pip install --system --reinstall https://github.com/vllm-project/vllm/releases/download/v0.24.0/vllm-0.24.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu129  # noqa: E501

# %%
# Colab's kernel has numpy already imported (and its C extension loaded into the process) before
# any of our cells run -- some Colab-internal machinery pulls it in at kernel startup. The vllm
# reinstall above then upgrades numpy ON DISK as a transitive dependency, but the already-loaded
# copy in this process can't be swapped (numpy is a C extension; Python can't hot-reload one), so
# the next `import numpy` anywhere just returns the stale cached module -- except unsloth_zoo's
# own consistency check (unsloth_zoo/temporary_patches/utils.py, hit via `from unsloth import
# FastLanguageModel`) compares the loaded version against the on-disk one and raises rather than
# silently running with a mismatch. Re-pinning numpy back to exactly what's already loaded (not a
# hardcoded version -- Colab's pre-loaded numpy version isn't guaranteed stable across kernel
# snapshots) satisfies that check without needing a restart: the loaded copy keeps working
# exactly as it already was, this just makes the on-disk metadata agree with it.
import subprocess  # noqa: E402
import sys  # noqa: E402

if "numpy" in sys.modules:
    _loaded_numpy_version = sys.modules["numpy"].__version__
    _numpy_pin = f"numpy=={_loaded_numpy_version}"
    subprocess.run(
        ["pip", "install", "-q", _numpy_pin, "--force-reinstall", "--no-deps"],
        check=True,
    )

# %%
# `pip install -e .`'s editable-install finder isn't reliable for direct (non-pytest) imports in
# this repo -- see CLAUDE.md's "Known environment quirks" for the macOS Gatekeeper .pth case.
# Insert the clone's src/ directly so `toolsmith.*` resolves regardless of that, and to guard
# against a same-named-but-unrelated PyPI package ("toolsmith", not this project) ever shadowing
# it if one ends up pulled in transitively. Also drop any toolsmith.* entries already cached in
# sys.modules from an earlier broken import attempt in this kernel -- re-running this cell after
# a failure (e.g. while iterating on a Colab error) leaves the parent `toolsmith` module cached
# with the wrong __path__, and inserting into sys.path alone can't fix an already-cached parent.
import sys  # noqa: E402 (Colab-only cell, top-of-cell by design)

for _name in list(sys.modules):
    if _name == "toolsmith" or _name.startswith("toolsmith."):
        del sys.modules[_name]
sys.path.insert(0, "/content/toolsmith/src")

# %%
# unsloth must be imported before trl -- at import time it monkeypatches trl's trainer configs
# (SFTConfig, GRPOConfig, ...), and importing trl first leaves that patch half-applied, corrupting
# defaults like eos_token/pad_token (confirmed root cause + fix from an unsloth maintainer:
# https://github.com/unslothai/unsloth/issues/2797 -- filed against SFTTrainer, but the patch is
# applied generically across every trl {X}Trainer/{X}Config pair, GRPOTrainer included).
from unsloth import FastLanguageModel  # noqa: E402

# %%
# trl==0.24.0's GRPOTrainer import chain unconditionally imports two optional, heavy deps this
# notebook never uses: trl/trainer/callbacks.py has bare top-level `from ..mergekit_utils import
# ...` (needs the `mergekit` package) and `from .judges import BasePairwiseJudge` (needs
# `llm_blender`) -- neither import is gated behind an availability check in this trl version, so
# pip-installing them would be the only way to satisfy them for real. That's not viable here:
# `llm_blender` itself is broken against current transformers (imports the since-removed
# TRANSFORMERS_CACHE symbol). Neither MergeConfig/merge_models nor BasePairwiseJudge are actually
# used by anything below -- both are referenced only inside MergeModelCallback/WinRateCallback,
# optional callback classes this notebook never instantiates -- so stub both packages instead of
# installing them; the stubs are never touched at runtime, only satisfy the module-level import.
# Also coerce trl.import_utils's `_vllm_ascend_available` flag: transformers>=4.48's
# `_is_package_available()` returns a (bool, version) tuple, and trl assigns that tuple directly
# to the flag without unpacking it, so a non-empty tuple is truthy even when the package is
# absent. This is the same bug class unsloth's own import_fixes.py documents (fix_trl_vllm_ascend)
# for this exact flag, but that fix doesn't reach mergekit/judges since those two imports aren't
# flag-gated at all in this trl version -- hence the stubs above, not just a flag coercion.
# isort: split
import importlib.machinery  # noqa: E402
import sys  # noqa: E402
import types  # noqa: E402


def _stub_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    sys.modules[name] = module
    return module


_mergekit = _stub_module("mergekit")
_mergekit_config = _stub_module("mergekit.config")
_mergekit_config.MergeConfiguration = type("MergeConfiguration", (), {})
_mergekit.config = _mergekit_config

_llm_blender = _stub_module("llm_blender")
_llm_blender.Blender = type("Blender", (), {})

import trl.import_utils as _trl_import_utils  # noqa: E402

_trl_import_utils._vllm_ascend_available = False

# isort: split
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

import wandb  # noqa: E402
from datasets import Dataset  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402
from vllm import SamplingParams  # noqa: E402

from toolsmith.data.decision_points import extract_decision_points, select_task_subset  # noqa: E402
from toolsmith.data.taskspec import TaskSpec  # noqa: E402
from toolsmith.env.model import Model  # noqa: E402
from toolsmith.env.runner import run_episode  # noqa: E402
from toolsmith.rewards.composite import make_reward_func  # noqa: E402
from toolsmith.rewards.goalcheck import check_goal  # noqa: E402
from toolsmith.rewards.outcome_reward import ContinuationCache  # noqa: E402

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
