# ToolSmith — Full Project Plan (Zero-Cost Build)

**Goal:** Post-train a small open model (GRPO) for reliable multi-step tool calling. Match GPT-4o-mini accuracy at ~3% inference cost. Ship as MCP server + public demo.

**Total cash budget:** $0 infrastructure. ~$10–30 of existing Claude/OpenAI API credits (data generation + LLM judge).

---

## 0. Constraint Checklist (your requirements → plan decisions)

| Your constraint | Plan decision |
|---|---|
| Zero cost except API credits | All infra on free tiers (verified below) |
| No local GPU / slow MacBook | All training on Colab free T4; MacBook only for code editing + CPU unit tests |
| Colab available | Unsloth GRPO notebooks run on free T4 (confirmed — Unsloth publishes free-tier GRPO + tool-calling notebooks) |
| Claude + OpenAI credits | Used only for: synthetic data generation, LLM-judge scoring, GPT-4o-mini baseline comparison |
| Claude Code, GitHub, VS Code | Repo-first workflow; Claude Code for scaffolding, tests, refactors |
| Hugging Face | Model hosting (Hub, free), demo (Spaces free CPU tier), datasets (free) |
| AWS free tier | NOT needed — avoided deliberately (HF Spaces simpler, zero risk of surprise bills) |
| Free public demo | HF Space (free CPU, GGUF quantized model) + replay mode + Colab live-demo badge |
| Free data sources | Public HF datasets + keyless APIs + Claude-generated synthetic tasks |

---

## 1. Project Summary

- **Name:** ToolSmith
- **Model:** Qwen3-4B-Instruct-2507 (LOCKED; non-thinking variant — avoids `<think>` block token blowup during RL rollouts; fits free T4 in 4-bit via Unsloth)
- **Training:** SFT warm-start (LoRA) → GRPO with verifiable rewards
- **Domain:** Travel-ops assistant. 12 tools: flight search, weather, geocoding, currency, timezone, calendar, country info, POI search, distance calc, packing rules, unit conversion, datetime math
- **Key trick:** Deterministic sandbox mirrors of every tool → rewards fully verifiable, no labels needed at RL stage
- **Deliverables:** Trained LoRA + merged GGUF on HF Hub, MCP server, Gradio demo Space, eval report (4-way comparison), W&B training report, Medium article

**Success criteria (define before building):**
1. GRPO model beats SFT model by ≥10 pts on held-out task-completion rate
2. GRPO model within 5 pts of GPT-4o-mini on custom multi-turn suite
3. Reported cost-per-1k-calls comparison (self-hosted vs API)
4. BFCL score reported honestly (even if mid — the comparison story matters)

**Honesty caveat (state in article):** domain-tuned 4B beating a generalist on its own sandbox = expected, not magic. BFCL (out-of-domain) keeps the claim honest. Report both.

---

## 2. Day 0 — Accounts & Setup (all free)

1. **GitHub repo:** `toolsmith` — public, MIT license, README stub
2. **Hugging Face:** already have (rohanjain2312). Create:
   - Model repo: `rohanjain2312/toolsmith-qwen3-4b`
   - Dataset repo: `rohanjain2312/toolsmith-tasks`
   - Space: `rohanjain2312/toolsmith-demo` (Gradio, CPU basic — free)
3. **Weights & Biases:** free personal account. Project: `toolsmith`
4. **Google Colab:** free tier. Mount Google Drive for checkpoints
5. **Kaggle account (backup GPU):** free 30 GPU-hrs/week (T4×2 / P100). Fallback when Colab disconnects
6. **API keys (all free tiers, no card where possible):**
   - Open-Meteo — weather, NO key needed
   - Nominatim (OpenStreetMap) — geocoding, NO key needed (1 req/sec limit; REQUIRES custom User-Agent header or requests get blocked)
   - Frankfurter.dev — currency rates, NO key needed
   - RestCountries — country info, NO key needed
   - Duffel — flight search, free TEST mode token (Amadeus Self-Service decommissioned July 17, 2026; registrations paused — replaced)
   - OpenTripMap — POI search, free key
   - Timezone — NO API. Python stdlib `zoneinfo` (WorldTimeAPI dropped: flaky, unnecessary network dependency)
7. **Google Calendar (real OAuth — LOCKED decision):**
   - Create Google Cloud project (free, no card)
   - Enable Google Calendar API
   - Configure OAuth consent screen: External, Testing mode (no verification needed; add own Gmail as test user)
   - Create OAuth 2.0 credentials → Desktop app type → download `credentials.json`
   - Scope: `https://www.googleapis.com/auth/calendar.events` (minimal)
   - First local run triggers browser consent → caches `token.json`
   - Demo Space uses sandbox calendar; real OAuth demoed via Colab notebook + README GIF (Space cannot run interactive OAuth cleanly)
8. **Secrets handling:** `.env` locally, GitHub Actions secrets, HF Space secrets. Never commit keys
9. **Local dev env (MacBook, CPU only):** Python 3.11, uv or poetry, pytest, ruff, pre-commit hooks

---

## 3. Repository Structure

```
toolsmith/
├── README.md                  # hero readme: demo GIF, results table, architecture diagram
├── pyproject.toml
├── .github/workflows/
│   ├── ci.yml                 # lint + unit tests + schema tests (CPU, free minutes)
│   └── eval-gate.yml          # regression eval on cached trajectories
├── src/toolsmith/
│   ├── tools/                 # 12 tool implementations
│   │   ├── schemas.py         # Pydantic schemas, JSON-schema export
│   │   ├── real/              # real API clients (demo mode)
│   │   └── sandbox/           # deterministic mocks (training + eval mode)
│   ├── env/                   # episode runner: multi-turn loop, state, termination
│   ├── rewards/               # reward functions (staged, see §7)
│   ├── data/                  # synthetic task generation + dataset builders
│   ├── eval/                  # custom suite + BFCL adapter + judge
│   ├── serve/                 # MCP server + llama.cpp CPU inference
│   └── utils/
├── notebooks/
│   ├── 01_sft_warmstart.ipynb     # Colab
│   ├── 02_grpo_training.ipynb     # Colab
│   └── 03_live_demo.ipynb         # Colab badge in README
├── space/                     # Gradio app (deployed to HF Space)
├── tests/                     # pytest: every tool, every reward, env loop
└── results/                   # eval tables, plots, trajectory samples
```

---

## 4. Phase 1 — Sandbox Environment + Tools

**Why sandbox first:** GRPO needs thousands of rollouts. Real APIs = rate limits + nondeterminism + cost. Sandbox = instant, deterministic, free.

Steps:
1. Define all 12 tool Pydantic schemas. Export to OpenAI/Anthropic function-call JSON format
2. Build sandbox versions: each tool backed by a small static dataset (e.g., 200 synthetic flights, 50 cities with fixed weather, fixed FX table). Seeded RNG → same input, same output, always
3. Build real versions: thin clients over the free APIs above. Same schemas. Toggle via `TOOLSMITH_MODE=sandbox|real`
4. Build episode runner:
   - System prompt with tool definitions
   - Loop: model output → parse tool call → execute → append result → repeat
   - Termination: final answer emitted, max 8 turns, or parse failure
   - Tool-result truncation: cap each result at 512 tokens (flight search returns can explode context past 2048 budget)
   - Full trajectory logging (JSON lines)
   - Model interface = abstract class; stub model, llama.cpp, vLLM, OpenAI all swap behind it
5. Unit tests: every tool, every schema, parser edge cases (malformed JSON, hallucinated tool names, wrong arg types)
6. CI: GitHub Actions runs pytest + ruff on every push

Claude Code usage: scaffold schemas, generate sandbox datasets, write tests.

**Exit criteria:** `python -m toolsmith.env --task sample.json` runs a full episode against sandbox with a stub model.

---

## 5. Phase 2 — Data

Two data streams:

**A. Public SFT data (free, HF):**
- `Salesforce/xlam-function-calling-60k` — accept license on HF
- Glaive function-calling v2
- Filter to single + multi-step samples; convert to Qwen chat template; keep ~8–12k rows

**B. Synthetic domain tasks (Claude credits, ~$5–10):**
1. Prompt Claude to generate 2,000 travel-ops tasks across 4 difficulty tiers:
   - T1: single tool ("weather in Lisbon Friday?")
   - T2: 2–3 tool chain ("flight BWI→SLC next Monday, budget in EUR")
   - T3: 4–6 tool chain with dependencies ("plan day trip: geocode → POI → distance → weather → calendar entry")
   - T4: traps — impossible requests, ambiguous args, tools missing needed data (model must ask or decline)
2. Each task ships with a machine-checkable goal spec (which sandbox facts constitute a correct answer) — NOT a gold trajectory. Rewards verify outcomes, not paths. Spec ALSO stores `min_steps` (computed at generation time by a solver script) — required by reward R6
3. Generate + validate: run each goal spec against sandbox to prove solvability; solver confirms `min_steps`
4. Splits: 1,400 train / 300 val / 300 held-out test (test never touches training)
5. **Contamination controls:** SFT gold trajectories drawn ONLY from train split; xLAM/Glaive share distribution with BFCL-style data — disclose in model card, never claim BFCL-clean training
6. Push dataset to HF Hub (public — portfolio value)

**Exit criteria:** dataset card on HF Hub, validation script proves 100% of tasks solvable in sandbox.

---

## 6. Phase 3 — SFT Warm-Start (Colab free T4)

Purpose: teach exact tool-call format first, so GRPO doesn't waste steps learning JSON syntax.

1. Colab notebook: Unsloth + `Qwen3-4B-Instruct-2507` in 4-bit, LoRA r=16, alpha=32
2. Train on: public function-calling data (from 5A) + 200 Claude-generated gold trajectories for domain format (train split only; verify each against sandbox before including)
3. Config: max_seq_len 2048, batch 2 + grad-accum 8, 1–2 epochs, cosine LR 2e-4; `train_on_responses_only` (loss on assistant tokens only — critical, tool results in loss = garbage gradients)
4. Checkpoint to Google Drive every 100 steps + push LoRA to HF Hub (private until release)
5. Log to W&B
6. Quick sanity eval: 50 val tasks, greedy decoding, measure JSON-validity rate + correct-tool rate. Target: >90% valid JSON before moving on

Colab survival tactics:
- Save/resume script (`resume_from_checkpoint=True`)
- Keep sessions <3.5h; split epochs across sessions
- Kaggle notebook mirror as backup (30 free GPU-hrs/week)

---

## 7. Phase 4 — GRPO Training (Colab free T4)

Core phase. Base on Unsloth's free-tier GRPO notebook pattern.

**CRITICAL ARCHITECTURE DECISION (previous plan glossed over this):**
TRL `GRPOTrainer` = single-turn. It generates ONE completion per prompt. Multi-step tool episodes need environment interaction mid-rollout — NOT natively supported. Plan v1 hand-waved this. Fixed approach:

**Primary: step-level GRPO (works inside stock TRL/Unsloth, T4-proven):**
1. Build decision-point dataset: run SFT model over a ~700-task train subset (T1/T2-weighted), log every intermediate conversation state (prefix) → ~2–2.5k decision points. Expand toward full 1,400 tasks ONLY if reward plateaus below the exit gate
2. Each GRPO sample = one prefix → model generates next action only (G=4, max_completion 256 — cheaper than 768, more steps per hour)
3. Reward per completion (runs inside custom `reward_func` — arbitrary Python allowed):
   - Format rewards R1–R4 scored directly on the action
   - Outcome reward R5: execute action in sandbox → roll episode to completion with frozen SFT policy (greedy) → check goal spec. Verifiable, label-free, per-step credit
   - R6 from `min_steps` in task spec
4. Mid-training decision-point refresh: CONDITIONAL, default OFF. Trigger only if reward plateaus below gate (refresh costs a full GPU generation pass — not free)

**Stretch (only if primary lands early):** trajectory-level RL via ART or `verifiers` library (vLLM + LoRA multi-turn GRPO). Document as future work otherwise. Honest scoping beats fake ambition.

**Reward function (staged, verifiable — no labels, no judge in the loop):**

| Stage | Reward | Weight |
|---|---|---|
| R1 | Output parses as valid tool-call JSON / valid final answer | +1.0 |
| R2 | Called tool exists in registry | +0.5 |
| R3 | Args pass Pydantic validation | +1.0 |
| R4 | No redundant calls (same tool+args twice) | +0.5 |
| R5 | Task goal spec satisfied (sandbox state / answer check) | +3.0 |
| R6 | Efficiency bonus: ≤ optimal steps + 1 | +0.5 |
| Penalty | Hallucinated tool name | −1.0 |
| Penalty | Max turns exceeded without answer | −1.0 |

**Anti-reward-hacking checks (interview gold — document every incident):**
- Watch for: model answering without calling required tools; gaming R1 by emitting trivial valid JSON; verbose stalling
- Mitigation: R5 dominates; length penalty; log 20 random trajectories per eval step and read them

**Training config:**
- Unsloth `fast_inference=True` (vLLM rollouts on T4)
- num_generations G=4 default, max_completion 256 (single action, not full episode). Dial: drop to G=3 (−25% compute) if projected hours exceed budget or OOM
- R5 continuation cache: MANDATORY from day one, persisted to Drive (survives Colab sessions). Sandbox deterministic → (state, action) result never recomputed. Single biggest hour-saver
- Curriculum: T1/T2 decision points = core run. T3/T4 stage = conditional, budget permitting after gate cleared (skipping them = honest, disclosed tradeoff)
- Early stopping: evaluate val task-completion every 50 steps; STOP once exit gate (SFT + 10 pts) holds for 2 consecutive checkpoints. Never run to step ceiling on autopilot
- KL β=0.04, LR 5e-6, 300–600 step ceiling
- Checkpoint every 50 steps → Drive + HF Hub
- W&B: reward curves per component (R1…R6 separately — this plot goes in the article)

**GPU-hour efficiency dials (apply in this order):** persistent R5 cache → subset decision points → early stopping → conditional refresh OFF → G=3 → defer T3/T4. Parallel wall-clock trick: run evals, BFCL, decision-point work, and GGUF export on Kaggle (30 GPU-hrs/wk) WHILE GRPO occupies Colab — two free GPU pools, concurrent. (Note: one training run can't split across platforms; only surrounding jobs parallelize.)

**Budget reality check:** with dials applied expect ~12–18 Colab T4 hours; 20–35h = worst case with no dials. Free tier gives ~3–4h/session, usage caps vary. Kaggle covers gaps + parallel jobs.

**Exit criteria:** val task-completion ≥ SFT + 10 pts.

---

## 8. Phase 5 — Evaluation

**Four-way comparison (the headline table):**
| Model | JSON valid % | Correct tool % | Arg accuracy % | Task completion % | Cost/1k tasks |
|---|---|---|---|---|---|
| Qwen3-4B base | | | | | ~$0 self-host |
| + SFT | | | | | ~$0 |
| + SFT + GRPO | | | | | ~$0 |
| GPT-4o-mini | | | | | $ (from OpenAI credits) |

1. Custom suite: 300 held-out tasks, run all 4 models in sandbox (open models via Colab vLLM; GPT-4o-mini via API — few $ of credits). Protocol: identical system prompt + schemas, greedy decoding (temp 0), 3 seeds for sandbox tie-breaks, per-metric bootstrap CIs
2. BFCL: scope to AST subsets only (simple / multiple / parallel — skips live-API categories, fits Colab). Run base vs SFT vs GRPO. Report deltas; checks domain RL didn't wreck general tool calling
3. LLM-judge (Claude credits): score final-answer quality on 100 tasks, 1–5 rubric; calibrate judge on 20 hand-scored samples first; judge NEVER sees model identity (blind)
4. Latency + cost table: measured T4 throughput → project cost per 1k calls vs GPT-4o-mini list price
5. Error taxonomy: classify 50 failures (wrong tool / wrong args / bad plan / gave up) — goes in article
6. All results → `results/` as CSV + plots; W&B report public link

---

## 9. Phase 6 — Packaging

1. Merge LoRA → export GGUF Q4_K_M (~2.5 GB) via Unsloth on Colab
2. Push to HF Hub: merged 16-bit + GGUF + LoRA adapters
3. Model card: training details, reward spec, eval table, limitations, license
4. Dataset card for toolsmith-tasks

---

## 10. Phase 7 — Free Demo

Three layers, all free:

**A. HF Space (primary, free CPU basic — 2 vCPU / 16GB):**
- Gradio app, GGUF Q4 via llama-cpp-python. Realistic: ~2–4 tok/s, NOT fast. Design around it
- **Replay tab = DEFAULT landing tab** (instant, recruiter sees results in 5 seconds): playback of 20 curated trajectories incl. side-by-side SFT-vs-GRPO failure/success pairs
- Live tab second: type task → tool calls stream (sandbox mode; keyless real-API toggle). Progress indicators + expectation-setting banner
- Free Spaces sleep after ~48h inactivity; cold start re-downloads 2.5GB GGUF → pin model file in Space repo via Git LFS (free ≤ limits) so restart = load-from-disk only
- Enable `mcp_server=True` in Gradio launch → Space doubles as a public MCP server. Claude users connect your model as a tool. Strong differentiator
- Banner: "CPU demo — full-speed Colab link below"

**B. Colab live-demo notebook (full speed):**
- Badge in README → loads GGUF/LoRA on free T4, vLLM inference, interactive cell UI

**C. MCP server (local install):**
- Publish to PyPI (free). NOTE: name `toolsmith` likely taken — check first; fallback `toolsmith-agent`. `pip install toolsmith-agent` → `toolsmith serve --mcp`
- Build on FastMCP (official Python SDK) — minimal boilerplate
- README shows Claude Desktop config snippet

---

## 11. Phase 8 — CI/CD + Observability (runs alongside build)

GitHub Actions free tier (CPU only — plan around it):
1. `ci.yml`: ruff + pytest on push (tools, rewards, parser)
2. `eval-gate.yml`: replays 50 cached trajectories through reward pipeline; fails PR if reward code changes scores unexpectedly. Also validates all tool schemas against BFCL format
3. Prompt versioning: system prompt lives in versioned file; changing it requires eval-gate pass
4. W&B: public workspace link in README

(Full GPU eval stays manual on Colab — Actions has no free GPU. Documented honestly in README; that candor itself signals production maturity.)

---

## 12. Phase 9 — Writeup + Distribution

1. **README:** demo GIF (record Space), architecture diagram, results table, quickstart, MCP config
2. **Medium article** (your FinCompress playbook): lead with counterintuitive finding — likely candidates: a reward-hacking story, or "4B + $0 GPU budget vs GPT-4o-mini". Include W&B reward curves
3. **LinkedIn post:** ~120 words, hook-first
4. **Resume bullets (fill real numbers):**
   - "Post-trained Qwen3-4B via GRPO (verifiable staged rewards) on $0 GPU budget (Colab T4); lifted multi-step tool-call task completion XX→XX%, within X pts of GPT-4o-mini at ~3% cost"
   - "Built deterministic 12-tool sandbox enabling label-free RL; designed 6-component reward with documented reward-hacking mitigations"
   - "Shipped as public MCP server + HF Space demo; BFCL-benchmarked; CI eval gate blocks prompt/reward regressions"
5. HF Space + Hub links on resume projects line

---

## 13. Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Colab free caps block training | High | Kaggle 30h/wk backup; aggressive checkpointing; curriculum keeps runs short |
| GRPO unstable / reward hacks | Medium | SFT warm-start; staged rewards; read trajectories; lower LR; it becomes article content either way |
| Step-level GRPO ≠ trajectory RL (credit assignment noise from frozen-policy continuations) | Medium | Accept + disclose; refresh decision points mid-training; trajectory-level = documented future work |
| R5 rollout-to-completion too slow on T4 | Medium | Persistent (state, action) cache MANDATORY day one; subset decision points; T1/T2-heavy curriculum |
| T4 VRAM OOM at G=4 | Medium | Drop to G=3, max_completion 512, seq 1536 |
| CPU Space demo too slow | Medium | Replay mode carries the demo; Colab badge for speed |
| 4B model plateaus below target | Low-Med | Story still works: report honest gap + cost analysis; optionally rerun with 8B on Kaggle |
| Duffel test-mode limitations (sandbox airlines only) | Low | Sandbox default; real APIs only decorative |
| OAuth token expiry / consent friction | Medium | Testing-mode refresh tokens expire ~7 days — re-auth script provided; sandbox calendar always works as fallback |
| Scope creep | High | Phases 1–5 = minimum shippable; 6–9 polish |

---

## 14. Cost Table (final)

| Item | Cost |
|---|---|
| GPU (Colab free T4 + Kaggle backup) | $0 |
| Hosting (HF Hub, Space CPU, GitHub, PyPI) | $0 |
| Data APIs (keyless/free tiers) | $0 |
| W&B, GitHub Actions | $0 |
| Synthetic data gen (Claude credits) | ~$5–10 |
| LLM judge + GPT-4o-mini baseline (credits) | ~$5–15 |
| **Cash out of pocket** | **$0** |

---

## 15. Execution Playbook — Exact Order of Operations

**A. Project start (before any code):**
1. Complete Day 0 setup (§2): all accounts, API keys, Google Cloud OAuth credentials
2. Create GitHub repo. Clone locally
3. Write `CLAUDE.md` at repo root (contents spec: §18)
4. Add `TOOLSMITH_BUILD_INSTRUCTIONS.md` (companion file) to repo root — Claude Code's single task source (no separate TASKS.md; git history tracks progress)
5. First commit: README stub + CLAUDE.md + build instructions + license

**B. Build loop (one Claude Code session per PHASE, Phases 0–2):**
1. Open Claude Code in repo
2. Prompt: "Execute Phase N per TOOLSMITH_BUILD_INSTRUCTIONS.md"
3. Claude Code executes all phase tasks in order, committing per task
4. Phase end: Claude Code spawns checker subagent — full test run + diff review vs acceptance criteria + rule-violation hunt; fixes findings until clean
5. You spot-review final diff summary. Push. Verify CI green on GitHub
6. Next phase = next session

**C. Data generation gate (end of Phase 2):**
1. Run Claude synthetic-task generation script yourself (uses your API key)
2. Run validation script — 100% solvability required
3. Push dataset to HF Hub
4. Do NOT start Phase 3 until validation passes

**D. Training loop (Phases 3–4, manual Colab work):**
1. Claude Code authors notebook `.py` sources locally
2. You convert via jupytext, upload to Colab
3. Run SFT session(s); checkpoints → Drive + HF Hub
4. Run sanity eval script; >90% JSON validity gate
5. Run decision-point extraction on Colab
6. Run GRPO sessions (multiple); monitor W&B between sessions
7. After each session: download trajectory samples, read 20, log reward-hacking observations in `results/notes.md`

**E. After training (Phase 5):**
1. Freeze final checkpoint. Tag repo `v1.0-model`
2. Run full eval suite on Colab (all 4 models)
3. Run BFCL AST subsets
4. Run judge script locally (API-based, CPU fine)
5. Commit all results CSVs + plots to `results/`

**F. Ship (Phases 6–8):**
1. GGUF export on Colab → push all artifacts to HF Hub
2. Write model + dataset cards
3. Build Gradio app locally with Claude Code; test on MacBook CPU
4. Push Space; verify replay tab + live tab + MCP endpoint
5. Publish PyPI package
6. Wire eval-gate CI; verify one intentional failing PR gets blocked
7. Record demo GIF

**G. Distribute (Phase 9):**
1. Finalize README (results tables now real)
2. Medium article draft → edit → publish
3. LinkedIn post
4. Resume bullets with real numbers; update resume PDF
5. Pin repo on GitHub profile; add links to HF profile

**Standing rules throughout:**
- Never merge red CI
- Never train past a failed gate
- Every Colab session starts by pulling latest repo + checkpoint
- Log every anomaly immediately (article fuel)

---

## 16. Locked Decisions

1. Base model: **Qwen3-4B-Instruct-2507** (non-thinking). VRAM fallback if OOM: G=3, seq 1536; last resort Llama-3.2-3B
2. Calendar: **real Google OAuth** (Desktop-app flow, Testing mode). Training + Space demo still use sandbox calendar; OAuth showcased via Colab notebook + README GIF
3. GRPO mechanism: **step-level** (stock TRL/Unsloth compatible). Trajectory-level = stretch/future work

---

## 17. Critique Log (issues found in plan v1 → fixes applied)

| # | Severity | Issue | Fix |
|---|---|---|---|
| 1 | CRITICAL | Plan implied episode-level GRPO; TRL GRPOTrainer = single-turn. Whole training phase unbuildable as written | §7 rewritten: step-level GRPO with decision-point dataset + rollout-to-completion R5 |
| 2 | HIGH | "Qwen3-4B-Instruct" ambiguous; hybrid variant emits `<think>` blocks → rollout token blowup, reward parsing chaos | Locked to Instruct-2507 non-thinking variant |
| 3 | HIGH | R6 needed "optimal steps" but plan stored no gold paths — unverifiable reward | Task spec now stores solver-computed `min_steps` |
| 4 | HIGH | SFT loss over full sequence (incl. tool results) → garbage gradients | `train_on_responses_only` added |
| 5 | MED | max_completion 768 for whole episodes: wrong unit + slow + VRAM risk | 256 (single action) under step-level design |
| 6 | MED | GRPO hour estimate ignored R5 rollout cost | 15–25h → 20–35h; continuation caching added |
| 7 | MED | No eval decoding protocol → numbers not comparable across 4 models | Greedy, identical prompts, blind judge, bootstrap CIs |
| 8 | MED | "Run BFCL" unscoped — full harness includes live-API categories, infeasible free | Scoped to AST subsets (simple/multiple/parallel) |
| 9 | MED | Space speed claim (3–8 tok/s) optimistic; ignored 48h sleep + cold-start re-download | 2–4 tok/s honest; LFS-pinned model; replay tab default |
| 10 | MED | Data contamination unaddressed (SFT gold vs test split; xLAM vs BFCL distribution) | Contamination controls in §5; disclosure in model card |
| 11 | LOW | WorldTimeAPI flaky + pointless network dep | Python `zoneinfo`, zero network |
| 12 | LOW | Nominatim blocks default User-Agent | Custom UA header noted |
| 13 | LOW | Tool results can blow 2048 context | 512-token result truncation in runner |
| 14 | LOW | PyPI name `toolsmith` likely taken | Check + `toolsmith-agent` fallback |
| 15 | LOW | Domain-tuned-vs-generalist comparison risked overclaiming | Honesty caveat added to success criteria |

---

## 18. Claude Code Build Guide

**Core rule: one task = one file OR one function + its test. ≤150 LOC per task. Never "build phase 1" as a single instruction.**

Authoritative sources (this section defers to them — no duplication):
- `TOOLSMITH_BUILD_INSTRUCTIONS.md` — full 78-task breakdown, acceptance criteria, phase execution protocol, checker-subagent mandate
- `CLAUDE.md` — standing rules, environment facts, commands

### What Claude Code CANNOT do here (plan around it)
1. **No GPU, no Colab execution.** Author training code as `.py` in `notebooks/src/` (jupytext percent format) → convert to `.ipynb` → YOU run on Colab manually. Claude Code writes + reviews; never executes training
2. **No OAuth browser flow.** You run `python scripts/auth_google.py` yourself once
3. **No HF Space runtime debugging.** Test Gradio app locally first (`python space/app.py`), then push
4. **Long monolith generation degrades quality.** Hence micro-tasks in the build instructions file

### Anticipated Claude Code failure points (specific)
| Failure | Prevention |
|---|---|
| Invents tool schemas mid-task, drifts from registry | Schemas locked early in Phase 1; CLAUDE.md forbids new tools |
| Writes tests calling real APIs | CLAUDE.md rule + pytest-socket blocks network in tests |
| Regenerates whole file for small edit, loses code | Keep files <300 lines; commit after every task |
| Notebook cells as giant single blob | Author as jupytext `.py` percent cells; review as code |
| Pydantic v1/v2 syntax mixing | Pin v2 in CLAUDE.md + pyproject |
| Reward function edge cases silently wrong | Property-based tests (hypothesis) required per reward |
| Stale TRL/Unsloth API usage from training data | Paste current library snippet into session when asked |
