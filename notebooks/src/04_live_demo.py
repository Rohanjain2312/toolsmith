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
# # ToolSmith — Colab Live Demo (full-speed T4 via vLLM)
#
# Colab free-T4 notebook (human-run only; never executed by Claude Code). Loads the merged
# 16-bit ToolSmith checkpoint (pushed by `03_export_gguf.py`'s `push_to_hub_merged` step) via
# vLLM for full-speed inference, then runs one interactive task through the episode runner.
#
# This is the "fast" counterpart to the HF Space demo — the Space runs the quantized GGUF on
# free CPU (~2-4 tok/s by design), while this notebook runs the merged 16-bit weights on a
# free T4 GPU for a real-speed feel. Badge this notebook in the README next to the Space link.
#
# Before running: mount Drive if you want trajectory logs persisted, and set `TASK_PROMPT`
# below to whatever travel-ops request you want to try.

# %%
# Colab-only install cell. Skip if the environment already has these.
# %pip install -q vllm

# %%
from pathlib import Path
from typing import Any

from vllm import LLM, SamplingParams

from toolsmith.env.model import Model
from toolsmith.env.runner import run_episode

# %%
# --- Config ---
HF_MODEL_REPO = "rohanjain2312/toolsmith-qwen3-4b"  # merged 16-bit weights from P6-T01
MAX_TOKENS = 512
TEMPERATURE = 0.0  # greedy by default; raise for more varied demo responses
TASK_PROMPT = "What's the weather in Paris this week, and what should I pack for a 3-day trip?"

# %%
llm = LLM(model=HF_MODEL_REPO, dtype="bfloat16")
tokenizer = llm.get_tokenizer()
sampling_params = SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_TOKENS)


# %%
class VLLMModel(Model):
    """Model adapter over a loaded vLLM engine, for full-speed interactive demo generation."""

    def generate(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str:
        """Render `messages` through the chat template and generate one completion."""
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        output = llm.generate([prompt_text], sampling_params)
        return output[0].outputs[0].text


# %%
# --- Interactive task cell: edit TASK_PROMPT above and re-run this cell to try another task ---
state = run_episode(
    task_id="live-demo",
    user_message=TASK_PROMPT,
    model=VLLMModel(),
    trajectory_dir=Path("results/trajectories/live_demo"),
)

print(f"status: {state.status.value}")
for entry in state.tool_calls:
    print(f"  tool call: {entry.tool_name}({entry.args}) -> ok={entry.ok}")
print(f"final answer: {state.final_answer}")
