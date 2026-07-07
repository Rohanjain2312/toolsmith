# CLAUDE.md — ToolSmith

## What this project is
ToolSmith: post-trained small-LLM tool-calling agent. Qwen3-4B-Instruct-2507 + LoRA SFT + step-level GRPO (verifiable rewards). Deterministic 12-tool travel-ops sandbox. Eval harness, MCP server, Gradio demo. All free-tier infrastructure; GPU work happens on Google Colab (human-run), never here.

## Source of truth
- `TOOLSMITH_BUILD_INSTRUCTIONS.md` — every task, acceptance criteria, phase protocol. THE task source. Follow it exactly.
- `ToolSmith_Project_Plan.md` — architecture rationale. Read-only reference.
- This file — standing rules + environment.

## Execution model
- One PHASE per session. Human names the phase. Tasks in listed order.
- Commit per task: `P{phase}-T{nn}: short description`. Push only at phase end.
- Phase end: spawn checker subagent (fresh context) per the Phase Execution Protocol in build instructions. Fix all findings before closing.
- Never start another phase in the same session.

## Environment
- Repo: https://github.com/Rohanjain2312/toolsmith.git
- Local: `/Users/rohanjain/Desktop/UMD - MSML/toolsmith` (path contains spaces — quote it in shell commands)
- HF: user `rohanjain2312`; model `toolsmith-qwen3-4b`; dataset `toolsmith-tasks`; Space `toolsmith-demo`
- PyPI package: `toolsmith-agent`
- Sandbox "today": `2026-09-01` (single constant, used by datetime_math + task generation)
- Flights real-mode provider: Duffel TEST mode
- Env vars (names only; never read values): ANTHROPIC_API_KEY, OPENAI_API_KEY, HF_TOKEN, WANDB_API_KEY, DUFFEL_ACCESS_TOKEN, OPENTRIPMAP_API_KEY, GOOGLE_CREDENTIALS_PATH, TOOLSMITH_MODE

## Commands
- Tests: `uv run pytest -x`
- Lint: `uv run ruff check --fix`
- Single episode: `PYTHONPATH=src uv run python -m toolsmith.env --task <path> --mode sandbox` (see Known environment quirks)
- Notebook convert (human runs): `uvx jupytext --to ipynb notebooks/src/<file>.py`

## Hard rules
1. Python 3.11. Pydantic v2 only (`model_validate`, `field_validator`, `ConfigDict`). uv for everything.
2. Test-first. No implementation before its failing test exists (where testable).
3. Files ≤300 lines. Tasks ≤~150 LOC. Bigger → stop, ask human to split.
4. NEVER read/write/print `.env`, `credentials.json`, `token.json`, `~/.config/*`. Env var names only.
5. NEVER network calls in tests. pytest-socket enforces. Sandbox implementations + mocked responses only.
6. Tool registry closed after Phase 1. No new tools, schema fields, or reward components — ever.
7. Sandbox = deterministic. Same input → identical output. Seeded everything.
8. Notebooks: jupytext percent `.py` under `notebooks/src/` only. Never `.ipynb`. Never execute training code.
9. Uncertain external API (TRL, Unsloth, FastMCP, Gradio, BFCL, Duffel)? STOP and ask human for current docs snippet. Do not guess from memory.
10. One-line purpose docstring per module. Type hints on public functions.
11. No TODO/FIXME in committed code.

## Known environment quirks
- macOS (this machine) tags `uv`-generated `.venv` `.pth` files with the Gatekeeper "hidden" chflag on every editable-install rebuild. CPython's `site.py` silently skips hidden `.pth` files, so `import toolsmith` fails outside pytest even though the package is "installed." Fixed permanently for tests via `pythonpath = ["src"]` in `[tool.pytest.ini_options]` (pyproject.toml) — do not remove it. For ad-hoc CLI runs outside pytest, prefix with `PYTHONPATH=src`, e.g. `PYTHONPATH=src uv run python -m toolsmith.env --task <path> --mode sandbox`. If direct imports still fail, run `chflags nohidden .venv/lib/python3.11/site-packages/*.pth`.

## HF Hub operations
`hf` CLI installed + authenticated. Official HF CLI agent skill available. Prefer `hf` CLI (or huggingface_hub library) for: dataset uploads, model pushes, Space file pushes, repo creation checks. Never embed HF_TOKEN in code — CLI auth + env var handle it.

## Definition of done
Task: tests green, ruff clean, typed, committed.
Phase: all tasks committed, checker report clean, pushed, CI green.

## Domain quick-reference
- 12 tools: geocode_city, weather_lookup, flight_search, currency_convert, timezone_info, calendar_create_event, country_info, poi_search, distance_calc, packing_rules, unit_convert, datetime_math
- Two implementations per tool: `tools/sandbox/` (default) + `tools/real/`; switched by `TOOLSMITH_MODE`
- Episode: system prompt + tool defs → generate → parse → execute → loop; ends on final answer / 8 turns / parse failure; results truncated to 512 tokens
- Rewards: R1 valid parse +1.0, R2 tool exists +0.5, R3 args valid +1.0, R4 no duplicate +0.5, R5 goal satisfied +3.0, R6 efficiency +0.5; penalties: hallucinated tool −1.0, max-turns −1.0
- Task tiers T1–T4; specs carry machine-checkable goals + solver-computed `min_steps`
