# TOOLSMITH — BUILD INSTRUCTIONS FOR CLAUDE CODE

You (Claude Code) will build ToolSmith: a post-trained small-LLM tool-calling agent with a deterministic sandbox environment, SFT + step-level GRPO training pipeline, evaluation harness, MCP server, and Gradio demo. This file defines every task. Execute ONE PHASE per session, tasks strictly in order within the phase.

---

## PHASE EXECUTION PROTOCOL (every session)

1. Human names the phase (e.g., "Execute Phase 1"). Work ONLY that phase.
2. Execute the phase's tasks in listed order. Within a task: test-first, implement, `pytest -x`, `ruff check --fix`, git commit with message `P1-T05: currency_convert tool + tests`, then next task.
3. If any task exceeds ~150 LOC or its scope surprises you: pause, report, ask the human before proceeding.
4. **Phase-end verification (mandatory): spawn a checker subagent** with fresh context and this mandate:
   - Run full test suite from scratch
   - Diff-review every file changed this phase against the task acceptance criteria in this document
   - Hunt rule violations: network calls in tests, secrets touched, files >300 lines, missing type hints, Pydantic v1 syntax, invented schema fields/tools
   - Verify determinism claims (run sandbox tests twice, compare)
   - Produce a findings report: PASS items, FAIL items with file:line
5. Main agent fixes every FAIL, re-runs checker. Repeat until clean report.
6. Only then: final commit `Phase N complete + verified`, print phase summary (files created, test count, checker findings fixed), stop.
7. Never start the next phase in the same session.

---

## PROJECT CONTEXT (read once, internalize)

- Base model: Qwen3-4B-Instruct-2507 (non-thinking variant). Training happens on Google Colab (human runs it) — you author training code but NEVER execute it.
- Domain: travel-ops assistant with 12 tools. Every tool has TWO implementations: `sandbox` (deterministic, seeded, offline) and `real` (free public APIs).
- Training strategy: SFT warm-start (LoRA via Unsloth) → step-level GRPO (TRL GRPOTrainer, single-action completions, verifiable rewards computed by running the sandbox inside the reward function).
- Everything must run free-tier: no paid services, no GPU assumptions in repo code (GPU code lives only in notebook sources).
- Python 3.11. Pydantic v2. pytest. ruff. Package manager: uv.

## GLOBAL RULES (apply to every task)

1. ONE phase per session (protocol above). Never look into future phases.
2. Test-first where a test target exists: write the test, watch it fail, implement, watch it pass.
3. File size limit: 300 lines. Split before exceeding.
4. Task size limit: ~150 LOC of implementation. If a task turns out bigger, stop and tell the human to split it.
5. NEVER read, write, or print: `.env`, `credentials.json`, `token.json`, any file under `~/.config`. Reference env vars by name only.
6. NEVER make network calls in tests. All tests run offline against sandbox implementations or mocked responses. Use pytest-socket to enforce.
7. NEVER invent new tools, schema fields, or reward components. The registry defined in Phase 1 is closed.
8. Pydantic v2 syntax only (`model_validate`, `field_validator`, `ConfigDict`).
9. Commit after every task (message format: `P{phase}-T{nn}: short description`). Push only at phase end, after checker passes.
10. Every module gets a one-line docstring stating its purpose.
11. Notebook code is authored as jupytext percent-format `.py` files under `notebooks/src/`. Never create `.ipynb` directly.
12. If an external library API is uncertain (TRL, Unsloth, FastMCP, Gradio), say so and ask the human for the current documentation snippet instead of guessing from memory.
13. This file = single task source. No separate task-tracking file. Progress lives in git history (task-ID commit messages + phase-complete commits).

## DEFINITION OF DONE (every task)

- All tests pass (`pytest -x`)
- `ruff check` clean
- No TODO/FIXME left in changed files
- Type hints on all public functions
- Committed with correct message format

## DEFINITION OF DONE (every phase)

- All phase tasks committed
- Checker subagent report fully clean
- Pushed; CI green

## ENVIRONMENT FACTS (confirmed)

- GitHub repo: https://github.com/Rohanjain2312/toolsmith.git
- Local path: `/Users/rohanjain/Desktop/UMD - MSML/toolsmith`
- HF username: `rohanjain2312`
- HF model repo: `rohanjain2312/toolsmith-qwen3-4b`
- HF dataset repo: `rohanjain2312/toolsmith-tasks`
- HF Space: `rohanjain2312/toolsmith-demo`
- Drive checkpoint folder: `toolsmith-checkpoints`
- Env var names in `.env` (values NEVER accessible to you): ANTHROPIC_API_KEY, OPENAI_API_KEY, HF_TOKEN, WANDB_API_KEY, DUFFEL_ACCESS_TOKEN, OPENTRIPMAP_API_KEY, GOOGLE_CREDENTIALS_PATH, TOOLSMITH_MODE
- Flight real-API provider: Duffel TEST mode (Amadeus decommissioned July 2026)
- Sandbox "today" constant: `2026-09-01`
- PyPI package name: `toolsmith-agent`
- HF CLI (`hf`) installed + authenticated on this machine; official HF CLI agent skill available — prefer it for Hub uploads, repo file ops, Space pushes

---

# PHASE 0 — REPO FOUNDATION

**P0-T01 — Package skeleton.**
Create `pyproject.toml` (uv-compatible, project name `toolsmith-agent`, src layout), `src/toolsmith/__init__.py`, empty subpackages per the repo map below, `tests/__init__.py`, pytest + ruff config, MIT LICENSE, `.gitignore` (Python, `.env`, `credentials.json`, `token.json`, `.ipynb_checkpoints`, worlddata regen artifacts), and `.env.example` listing exactly the env var names from ENVIRONMENT FACTS with empty values + one-line comments. Human fills real `.env` after this phase.
Repo map:
```
src/toolsmith/{tools,tools/real,tools/sandbox,env,rewards,data,eval,serve,utils}
notebooks/src/  space/  tests/  results/  scripts/
```
Accept: `uv run pytest` collects zero tests without error; `ruff check` clean.

**P0-T02 — CI workflow.**
Create `.github/workflows/ci.yml`: on push/PR → install via uv, run ruff, run pytest. CPU only. No secrets used.
Accept: YAML valid; steps match local commands.

**P0-T03 — Test network guard.**
Add pytest-socket; global conftest disabling network for all tests; a marker `@pytest.mark.allow_network` reserved but unused.
Accept: a demo test proving socket calls raise.

---

# PHASE 1 — TOOL REGISTRY + SANDBOX + EPISODE RUNNER

**P1-T01 — ToolSpec base + registry.**
`tools/schemas.py`: abstract tool definition (name, description, args model, returns model), a registry object (register/get/list, rejects duplicates and unknown names).
Accept: registry unit tests (register, duplicate rejection, unknown lookup).

**P1-T02 through P1-T13 — One task per tool.**
For EACH of the 12 tools below, one task: define Pydantic args/result schemas + sandbox implementation + minimum 5 unit tests (happy path, boundary, invalid args, determinism check — same input twice → identical output, and one domain-specific edge).
Sandbox implementations read from static seeded datasets (created in P1-T14); until then, use small inline fixture data and note the swap point.
Order:
- P1-T02 `geocode_city` (city name → lat/lon/country/timezone name)
- P1-T03 `weather_lookup` (lat/lon + date → forecast summary)
- P1-T04 `flight_search` (origin/dest IATA + date → list of flights: id, depart/arrive, price, currency)
- P1-T05 `currency_convert` (amount, from, to → converted amount, rate)
- P1-T06 `timezone_info` (timezone name or lat/lon → UTC offset, current local time; stdlib zoneinfo — NO network even in real mode)
- P1-T07 `calendar_create_event` (title, start, end, timezone → event id, status)
- P1-T08 `country_info` (country → currency, languages, plug type, visa-note field)
- P1-T09 `poi_search` (lat/lon, category, radius → list of POIs)
- P1-T10 `distance_calc` (two lat/lon pairs → km, haversine; pure function, identical in both modes)
- P1-T11 `packing_rules` (destination climate + trip length → packing checklist; pure lookup table)
- P1-T12 `unit_convert` (value, from-unit, to-unit; temperature/distance/weight)
- P1-T13 `datetime_math` (date arithmetic, weekday resolution, "next Monday" style resolution relative to a FIXED sandbox 'today')
Accept per task: 5+ tests green; tool registered; schema exports without error.

**P1-T14 — Sandbox world generator.**
`scripts/generate_sandbox_world.py`: seeded generation of static world data — 50 cities (coords, timezone, climate), fixed weather table per city/date-offset, 200 synthetic flights over city pairs, fixed FX table, POI sets per city. Writes JSON files under `src/toolsmith/tools/sandbox/worlddata/`. Sandbox tools from earlier tasks now load this data (refactor inline fixtures out).
Accept: regeneration with same seed → byte-identical files; all Phase 1 tests still green.

**P1-T15 — OpenAI function-call exporter.**
Registry → OpenAI tools JSON format. Tests validate structure for all 12 tools.

**P1-T16 — Anthropic tool exporter.**
Same for Anthropic tool-use format.

**P1-T17 — Tool-call parser.**
`env/parser.py`: extract tool call (name + args JSON) or final answer from raw model text. Handle: clean JSON, JSON in code fences, leading prose, trailing junk, malformed JSON, hallucinated tool name (parse succeeds, validation flags), multiple JSON objects (take first). 10+ edge-case tests.

**P1-T18 — Episode state.**
`env/state.py`: dataclass — task spec, message history, tool-call log, turn counter, terminal status enum. JSON serialization round-trip test.

**P1-T19 — Tool executor.**
`env/executor.py`: dispatch parsed call → registry → sandbox/real implementation via `TOOLSMITH_MODE` env var (default sandbox). Unknown tool → structured error result (NOT exception). Arg validation failure → structured error result. Tests for both paths.

**P1-T20 — Result truncation.**
Truncate any tool result to 512 tokens (approximate by chars/4; document approximation). Test with oversized flight-search result.

**P1-T21 — Model interface.**
`env/model.py`: abstract base — `generate(messages, tools) -> str`. Implement StubModel (returns scripted sequence, for tests). Later adapters (llama.cpp, vLLM, OpenAI, Anthropic) implement same ABC.

**P1-T22 — Episode loop.**
`env/runner.py`: system prompt assembly (tool defs injected), loop (generate → parse → execute → append), termination (final answer / max 8 turns / parse failure), trajectory JSONL logging to `results/trajectories/`. Tests with StubModel covering all three terminations.

**P1-T23 — CLI entry.**
`python -m toolsmith.env --task path.json --mode sandbox` runs one episode with StubModel, prints trajectory summary. Smoke test.

**P1-T24 through P1-T29 — Real API clients (one task each).**
Thin clients matching sandbox schemas exactly. Each with mocked-response tests (responses library or monkeypatch — never live calls).
- P1-T24 Open-Meteo (weather; no key)
- P1-T25 Nominatim (geocode; no key; MUST send custom User-Agent header; 1 req/s rate-limit guard)
- P1-T26 Frankfurter (currency; no key)
- P1-T27 RestCountries (country info; no key)
- P1-T28 Duffel TEST mode (flights; Bearer token `DUFFEL_ACCESS_TOKEN` from env; offer-request → offers flow; map Duffel offer fields to our flight schema; test mode returns sandbox airlines — acceptable, real mode = decorative)
- P1-T29 OpenTripMap (POI; key from env var)

**P1-T30 — Google Calendar real client.**
google-api-python-client wrapper; reads cached `token.json` path from env var; raises clear instruction message when token missing ("run scripts/auth_google.py"). Mocked tests only.

**P1-T31 — Google auth helper.**
`scripts/auth_google.py`: Desktop-app OAuth flow → caches token. You write it; human runs it. No test (document why).

**P1-T32 — End-to-end smoke.**
Full episode: multi-tool task, StubModel scripted through 4 tool calls to final answer, sandbox mode, assertions on trajectory contents.

---

# PHASE 2 — TASK DATA

**P2-T01 — Task spec model.**
`data/taskspec.py`: Pydantic model — id, tier (T1–T4), user prompt, goal spec (machine-checkable conditions over sandbox facts / final answer), `min_steps` int, split field. Round-trip tests.

**P2-T02 — Goal-spec checker.**
`rewards/goalcheck.py`: evaluate goal spec against a finished episode (final answer text + sandbox interaction log). Condition types: answer-contains-fact, tool-was-called-with, calendar-event-exists, numeric-within-tolerance. Tests per condition type.

**P2-T03 — Solver.**
`data/solver.py`: brute-force/BFS over tool-call sequences (bounded depth 6) proving a task solvable in sandbox and returning `min_steps`. Tests on hand-built T1 and T2 tasks.

**P2-T04 — Generation prompt files.**
`data/prompts/`: four prompt templates (T1–T4) instructing Claude API to emit task specs as JSON matching P2-T01 schema, grounded in the sandbox world data (city list, flight routes injected). No API calls in this task — just templates + a loader.

**P2-T05 — Generation script.**
`scripts/generate_tasks.py`: calls Anthropic API (key from env), batches, validates each returned spec against schema, dedupes. Human runs it. Tests: response-parsing logic only, with canned fixtures.

**P2-T06 — Validation + split script.**
`scripts/validate_tasks.py`: run solver over every task; reject unsolvable; verify `min_steps`; stratified split 1400/300/300 by tier; write JSONL. Tests on fixture tasks.

**P2-T07 — HF dataset upload script.**
`scripts/upload_dataset.py` + dataset card template. Human runs it.

---

# PHASE 3 — SFT PIPELINE

**P3-T01 — Public data prep.**
`scripts/prep_sft_data.py`: download xLAM-60k + Glaive v2 from HF, filter to function-calling samples, cap ~10k rows, convert to Qwen3 chat-template message format, write JSONL. Tests: conversion logic on fixture rows (no downloads in tests).

**P3-T02 — Gold trajectory generator.**
`scripts/generate_gold_trajectories.py`: for 200 train-split tasks, call Anthropic API as the agent inside the episode runner (Anthropic adapter implements the P1-T21 ABC), keep only trajectories passing goal check. Tests: adapter logic with mocked responses.

**P3-T03 — SFT notebook source.**
`notebooks/src/01_sft_warmstart.py` (jupytext percent format): Unsloth load Qwen3-4B-Instruct-2507 4-bit, LoRA r=16/alpha=32, train_on_responses_only, seq 2048, batch 2 × grad-accum 8, cosine LR 2e-4, 1–2 epochs, W&B logging, checkpoint to Drive + HF Hub push, resume support. ASK HUMAN for current Unsloth notebook snippet before writing — do not rely on memory.
Accept: file imports cleanly under `python -m py_compile`; cell structure logical; no execution.

**P3-T04 — Sanity eval script.**
`scripts/sanity_eval.py`: run 50 val tasks through episode runner with a provided model adapter, greedy decoding, report JSON-validity % and correct-tool %. Tests with StubModel.

---

# PHASE 4 — GRPO PIPELINE

**P4-T01 — Decision-point extractor.**
`data/decision_points.py`: replay logged SFT-model trajectories, emit every intermediate conversation prefix as a training sample (prefix + task ref). Configurable task-subset parameter, default ~700 train tasks (T1/T2-weighted) → ~2–2.5k points; full-set expansion behind a flag. Tests on fixture trajectories.

**P4-T02 — Format rewards R1–R4.**
`rewards/format_rewards.py`: R1 valid parse (+1.0), R2 tool exists (+0.5), R3 args validate (+1.0), R4 no duplicate call vs prefix history (+0.5). Property-based tests (hypothesis) + hand cases.

**P4-T03 — Outcome reward R5 + continuation cache.**
`rewards/outcome_reward.py`: execute candidate action in sandbox → complete episode with frozen policy adapter (greedy) → goal check (+3.0). Cache keyed on (state-hash, action-hash) — MANDATORY, not optional: sandbox determinism guarantees validity. Cache must PERSIST to disk (configurable path; human points it at Drive on Colab) and survive process restarts; include hit-rate counter logged to W&B. Tests: cache behavior, persistence round-trip, goal-check integration with StubModel as frozen policy.

**P4-T04 — R6 + penalties.**
Efficiency bonus vs `min_steps` (+0.5); hallucinated-tool penalty (−1.0); max-turns penalty (−1.0). Tests.

**P4-T05 — Composite reward function.**
`rewards/composite.py`: TRL-compatible reward callable — takes prompts + completions, returns float list; wires R1–R6; logs per-component values for W&B. Tests: full pipeline on fixtures, StubModel frozen policy.

**P4-T06 — GRPO notebook source.**
`notebooks/src/02_grpo_training.py`: Unsloth fast_inference, TRL GRPOTrainer, G=4 (config var; G=3 documented fallback for hours/VRAM), max_completion 256, KL β=0.04, LR 5e-6, curriculum flag (tier filter; T1/T2 core, T3/T4 conditional stage), persistent R5 cache path pointed at Drive, checkpoint every 50 steps, resume, W&B per-component reward logging + cache hit-rate, EARLY-STOPPING logic: val task-completion every 50 steps, halt when exit gate (SFT + 10 pts) holds 2 consecutive checkpoints, decision-point refresh behind a flag DEFAULT OFF. ASK HUMAN for current Unsloth GRPO snippet first.
Accept: compiles; no execution.

**P4-T07 — Trajectory audit script.**
`scripts/audit_trajectories.py`: sample N trajectories from a run, pretty-print for human reading, flag suspicious patterns (answer without required tools, trivial-JSON gaming, repeated stalling). Tests on fixtures.

---

# PHASE 5 — EVALUATION

**P5-T01 — Eval runner.**
`eval/runner.py`: model-agnostic — any P1-T21 adapter × task list → metrics (JSON-valid %, correct-tool %, arg accuracy %, task completion %), greedy decoding enforced. Tests with StubModel.

**P5-T02 — OpenAI adapter.**
GPT-4o-mini adapter implementing the model ABC with native function calling mapped to our schema exports. Mocked tests.

**P5-T03 — BFCL adapter.**
Wrapper conforming our model outputs to BFCL AST-subset harness expectations (simple/multiple/parallel categories only). ASK HUMAN for current BFCL repo interface before writing. Tests on fixture cases.

**P5-T04 — LLM judge.**
`eval/judge.py`: Anthropic-API rubric scoring (1–5) of final answers, model identity blinded, calibration mode comparing judge scores to a human-scored CSV. Mocked tests.

**P5-T05 — Stats + tables.**
`eval/stats.py`: bootstrap CIs per metric; results tables → CSV + markdown. Tests on fixture numbers.

**P5-T06 — Plots.**
`eval/plots.py`: 4-way comparison bar charts, reward-component curves from W&B export CSV. Smoke tests (files created).

---

# PHASE 6 — PACKAGING

**P6-T01 — Export notebook source.** `notebooks/src/03_export_gguf.py`: merge LoRA, GGUF Q4_K_M export, HF Hub push. Compiles only.
**P6-T02 — Model card.** Template with training details, reward spec, eval table placeholders, contamination disclosure, license.
**P6-T03 — Dataset card.** Same treatment for toolsmith-tasks.

---

# PHASE 7 — SERVING + DEMO

**P7-T01 — llama.cpp adapter.**
Model ABC implementation over llama-cpp-python loading local GGUF. Mocked tests (no model download in tests).

**P7-T02 — FastMCP server.**
`serve/mcp_server.py`: exposes one MCP tool — "run_toolsmith_task(prompt)" → executes episode, returns trajectory summary. ASK HUMAN for current FastMCP snippet. Tests: handler logic with StubModel.

**P7-T03 — CLI.**
`toolsmith serve --mcp` and `toolsmith run --task` entry points in pyproject scripts. Tests.

**P7-T04 — Gradio shell.**
`space/app.py`: two-tab layout, replay tab FIRST/default, expectation banner, theme. No inference wiring yet. Local-launch smoke test.

**P7-T05 — Replay tab.**
Loads curated trajectory JSONs from `space/replays/`, renders step-by-step with SFT-vs-GRPO side-by-side pairs. Include a script selecting 20 curated trajectories from results.

**P7-T06 — Live tab.**
Wires llama.cpp adapter + episode runner, streaming step display, sandbox default, keyless-real-API toggle.

**P7-T07 — MCP flag + Space packaging.**
`mcp_server=True` in launch; Space README/config files; GGUF via Git LFS pointer; requirements file.

**P7-T08 — Colab demo notebook source.**
`notebooks/src/04_live_demo.py`: T4 vLLM load, interactive task cell.

---

# PHASE 8 — CI GATES

**P8-T01 — Eval-gate workflow.**
`.github/workflows/eval-gate.yml`: replays 50 cached trajectories through reward pipeline; fails PR on unexpected score drift; validates tool schema exports. CPU only.
**P8-T02 — Prompt versioning.**
System prompt moved to `src/toolsmith/env/prompts/system_v1.txt`; loader pins version; eval-gate covers prompt file changes.
**P8-T03 — Intentional-failure verification.**
Doc + fixture PR branch demonstrating the gate blocks a reward-code regression. Human executes the PR dance.

---

# PHASE 9 — DOCS

**P9-T01 — README.** Hero structure: GIF placeholder, results table placeholders, architecture diagram (mermaid), quickstart, MCP config snippet, honest-limitations section.
**P9-T02 — Architecture diagram.** Mermaid source checked into repo, rendered in README.
**P9-T03 — Demo GIF script.** Shell script + instructions for recording Space interaction (human executes).

---

## TASK COUNT: 78 across 10 phases. One phase per session. Phases 0→1→2→3→4→5 sequential; 6–9 after 5. Checker subagent verifies every phase before it closes.
