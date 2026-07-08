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
# # ToolSmith — Merge + GGUF Export
#
# Colab free-T4 notebook (human-run only; never executed by Claude Code). Loads the final GRPO
# checkpoint, pushes a merged 16-bit model to the HF Hub, then exports a Q4_K_M GGUF quant for
# the free-tier llama.cpp Space demo (Phase 7).
#
# Budget note: the GGUF export step needs llama.cpp build tools, which Unsloth auto-installs on
# its first call (~3 min) — a fresh Colab session's first run will be slower than later ones;
# that's expected, not a failure.
#
# Before running: mount Drive, set the real GRPO checkpoint step in `GRPO_CHECKPOINT_STEP`
# below, and set `HF_TOKEN` in the Colab secrets panel.

# %%
# Colab-only install cell. Skip if the environment already has these.
# %pip install -q "unsloth @ git+https://github.com/unslothai/unsloth.git"

# %%
import os
import shutil
from pathlib import Path

from unsloth import FastLanguageModel

# %%
# --- Config ---
MAX_SEQ_LENGTH = 2048
QUANTIZATION_METHOD = "q4_k_m"  # matches the plan's Q4_K_M target
HF_MODEL_REPO = "rohanjain2312/toolsmith-qwen3-4b"

GRPO_CHECKPOINT_STEP = "<final_step>"  # human fills in the actual step number after training
GRPO_CHECKPOINT_DIR = Path(
    f"/content/drive/MyDrive/toolsmith-checkpoints/grpo/checkpoint-{GRPO_CHECKPOINT_STEP}"
)
GGUF_OUTPUT_DIR = Path("toolsmith_gguf")  # local scratch disk, not Drive (freed once pushed)

# %%
# Bridge Colab Secrets (key icon, left sidebar) into env vars the rest of this notebook reads
# directly. Add HF_TOKEN there first, then grant this notebook access when prompted -- adding a
# Colab secret does NOT auto-populate os.environ on its own.
from google.colab import userdata  # noqa: E402 (Colab-only import, top-of-cell by design)

os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")

# %%
# Mount Drive to read the GRPO checkpoint.
from google.colab import drive  # noqa: E402 (Colab-only import, top-of-cell by design)

drive.mount("/content/drive")

# %%
# --- Load the LoRA checkpoint (adapter, not merged) that GRPO training produced ---
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=str(GRPO_CHECKPOINT_DIR),
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

# %%
# --- Merged 16-bit save + push (separate artifact from GGUF, per plan §9) ---
model.push_to_hub_merged(
    HF_MODEL_REPO, tokenizer, save_method="merged_16bit", token=os.environ["HF_TOKEN"]
)

# %%
# --- GGUF export: local save first, verify, THEN push ---
# Deliberately two steps (not save+push combined): several linked Unsloth GitHub issues report
# "quantization failed mid-push, lost everything" — saving locally and verifying the file
# exists before pushing is cheap insurance against wasting a GPU session on a failed push.
# NOTE: this call triggers Unsloth's llama.cpp build on a fresh Colab session (first call
# only) — budget ~3 min beyond normal export time; not a hang or a failure.
model.save_pretrained_gguf(
    str(GGUF_OUTPUT_DIR), tokenizer, quantization_method=QUANTIZATION_METHOD
)

gguf_files = list(GGUF_OUTPUT_DIR.glob(f"*{QUANTIZATION_METHOD.upper()}*.gguf"))
if not gguf_files:
    raise FileNotFoundError(
        f"no {QUANTIZATION_METHOD} GGUF file found under {GGUF_OUTPUT_DIR} — export failed, "
        "do not proceed to push"
    )
gguf_path = gguf_files[0]

# %%
# --- Free disk before pushing: merge -> HF-16bit -> llama.cpp bf16 GGUF -> quantize keeps
# intermediates around, and free Colab disk (~78GB) gets tight after model + checkpoints +
# llama.cpp build. Remove everything under GGUF_OUTPUT_DIR except the final quantized file.
for path in GGUF_OUTPUT_DIR.iterdir():
    if path != gguf_path:
        shutil.rmtree(path) if path.is_dir() else path.unlink()

# %%
# --- Push the verified GGUF file to the HF Hub ---
model.push_to_hub_gguf(
    HF_MODEL_REPO, tokenizer, quantization_method=QUANTIZATION_METHOD, token=os.environ["HF_TOKEN"]
)
print(f"pushed {gguf_path.name} to https://huggingface.co/{HF_MODEL_REPO}")
