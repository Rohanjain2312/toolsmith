---
title: ToolSmith
emoji: 🧳
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
pinned: false
license: mit
tags:
  - tool-calling
  - agents
  - mcp-server-track
---

# ToolSmith Demo

Tool-calling agent demo for `rohanjain2312/toolsmith-qwen3-4b` (Qwen3-4B, LoRA SFT + step-level
GRPO over a deterministic 12-tool travel-ops sandbox).

- **Replay tab** (default): instant curated SFT-vs-GRPO trajectory comparisons.
- **Live tab**: run a real task against the quantized model on this Space's free CPU
  (~2-4 tok/s — for full speed, use the Colab notebook linked in the app).

This Space also runs as a public MCP server (`mcp_server=True`) — every Live-tab function is
auto-exposed as an MCP tool. View the generated schema at `/gradio_api/mcp/schema`, or via
"View API → MCP" in the footer.

See the [GitHub repo](https://github.com/Rohanjain2312/toolsmith) for training code, the
reward spec, and eval results.
