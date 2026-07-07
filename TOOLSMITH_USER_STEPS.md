# TOOLSMITH — USER STEPS (Your Actions, Start to Finish)

Everything YOU do. Claude Code handles code; Colab handles GPUs; this file handles you.
Work top to bottom. Check off as you go.

---

## STAGE 1 — ACCOUNTS + API KEYS (do all before anything else)

### 1.1 GitHub — DONE ✅
- [x] Repo: https://github.com/Rohanjain2312/toolsmith.git

### 1.2 Hugging Face (account: rohanjain2312)
- [x] Model repo: rohanjain2312/toolsmith-qwen3-4b
- [ ] Dataset repo: `toolsmith-tasks` — CREATE IF NOT DONE (New Dataset → private for now)
- [x] Space: rohanjain2312/toolsmith-demo
- [ ] Settings → Access Tokens → New token → type WRITE → name `toolsmith` (if not done)
- [ ] Accept xLAM dataset terms: open huggingface.co/datasets/Salesforce/xlam-function-calling-60k → click agree

### 1.3 Weights & Biases
- [ ] wandb.ai → sign up (free personal)
- [ ] wandb.ai/authorize → copy API key

### 1.4 Google Colab + Drive — DONE ✅
- [x] Colab signed in
- [x] Drive folder `toolsmith-checkpoints` created

### 1.5 Kaggle (backup GPU)
- [ ] kaggle.com → sign up
- [ ] Settings → phone verification (REQUIRED for GPU + internet access in notebooks)

### 1.6 Anthropic API (credits exist)
- [ ] console.anthropic.com → API Keys → create key `toolsmith`

### 1.7 OpenAI API (credits exist)
- [ ] platform.openai.com → API Keys → create key `toolsmith`

### 1.8 Duffel (flight search — REPLACES Amadeus)
Amadeus Self-Service shuts down July 17, 2026; new registrations already paused. Duffel replaces it: modern REST, self-serve signup, free test mode (realistic sandbox flight offers, no card).
- [ ] duffel.com → Sign up (free)
- [ ] Dashboard → stay in TEST mode
- [ ] Developers → Access tokens → create test token → copy

### 1.9 OpenTripMap (POI search)
- [ ] dev.opentripmap.org → sign up → copy free API key

### 1.10 Google Cloud (Calendar OAuth)
- [ ] console.cloud.google.com → New Project → `toolsmith`
- [ ] APIs & Services → Library → search "Google Calendar API" → Enable
- [ ] APIs & Services → OAuth consent screen → External → fill app name + your email → Save
- [ ] Audience/Testing → Add test user → your own Gmail
- [ ] Credentials → Create Credentials → OAuth client ID → Application type: Desktop app → Create
- [ ] Download JSON → rename `credentials.json` → keep OUTSIDE repo for now (repo `.gitignore` will cover it later)
- [ ] NOTE: Testing-mode refresh tokens expire ~7 days. Re-run auth script when calendar calls fail

### 1.11 PyPI (package publish — nothing more to do now)
There IS no "create project" button. PyPI projects appear automatically on first upload (`uv publish`, Stage 7). Done for now:
- [x] Registered + logged in
- [ ] Enable 2FA (Account settings)
- [ ] Stage 7 only: Account settings → API tokens → create token
- Package name LOCKED: `toolsmith-agent` (avoids collisions; repo stays `toolsmith`)

### 1.12 Keyless APIs — nothing to do
Open-Meteo, Nominatim, Frankfurter, RestCountries: no accounts, no keys.

---

## STAGE 2 — LOCAL MACHINE SETUP (MacBook)

- [ ] Install uv: docs.astral.sh/uv → standalone installer command
- [ ] Verify git installed (`git --version`)
- [ ] Verify Claude Code CLI (`claude --version`)
- [x] Repo cloned: `/Users/rohanjain/Desktop/UMD - MSML/toolsmith`
- [ ] Install HF CLI + agent skill (lets Claude Code manage HF Hub directly — uploads, repo files, Space pushes):
  - `pip install -U huggingface_hub` (or `uv tool install huggingface_hub`)
  - `hf auth login` → paste HF write token
  - Install official HF CLI Skill into Claude Code per huggingface.co/docs/hub/en/agents-cli
- [ ] Copy into repo root: `ToolSmith_Project_Plan.md`, `TOOLSMITH_BUILD_INSTRUCTIONS.md`, `CLAUDE.md`, this file
- [ ] `.env` handling: Claude Code creates `.env.example` + `.gitignore` in Phase 0 (task P0-T01). After Phase 0 finishes, YOU copy `.env.example` → `.env` and fill real values. Expected entries:
  ```
  ANTHROPIC_API_KEY=
  OPENAI_API_KEY=
  HF_TOKEN=
  WANDB_API_KEY=
  DUFFEL_ACCESS_TOKEN=
  OPENTRIPMAP_API_KEY=
  GOOGLE_CREDENTIALS_PATH=/absolute/path/to/credentials.json
  TOOLSMITH_MODE=sandbox
  ```
- [ ] First commit + push: the four docs above

(No TASKS.md — build instructions file itself = single task source; progress tracked via git commit messages + phase-complete commits.)

### ➜ REPORT-BACK CHECKPOINT — DONE ✅
Environment facts already written into TOOLSMITH_BUILD_INSTRUCTIONS.md. Remaining loose end: dataset repo `rohanjain2312/toolsmith-tasks` — create on HF if not done yet.

---

## STAGE 3 — CLAUDE CODE BUILD SESSIONS (one phase = one session)

For each phase 0 → 5 (then 6–9 after training):
- [ ] Open terminal in repo → `claude`
- [ ] Prompt: `Execute Phase N per TOOLSMITH_BUILD_INSTRUCTIONS.md. Follow the Phase Execution Protocol including the checker subagent.`
- [ ] Let it run tasks + per-task commits
- [ ] Read its phase summary + checker report
- [ ] Spot-check: open 2–3 changed files, skim tests
- [ ] `git push` → confirm GitHub Actions CI green
- [ ] Close session. Next phase = fresh session
- When Claude Code asks for current library docs (Unsloth/TRL/FastMCP/BFCL/Gradio): fetch the referenced page/notebook, paste the snippet

Sequencing notes:
- Phases 0, 1, 2 back to back
- STOP after Phase 2 → Stage 4 below
- Phase 3, 4, 5 sessions interleave with Colab work (Stages 5–6)
- Phases 6–9 after evaluation done

---

## STAGE 4 — DATA GENERATION (you run, after Phase 2 code exists)

- [ ] `uv run python scripts/generate_tasks.py` (uses your Anthropic key; ~$5–10 credits)
- [ ] `uv run python scripts/validate_tasks.py` → must report 100% solvable; rerun generation for rejects
- [ ] `uv run python scripts/upload_dataset.py` → check dataset appears on HF Hub
- [ ] Run Google auth once: `uv run python scripts/auth_google.py` → browser consent → `token.json` cached
- [ ] Gate: do NOT proceed to training until validation passes

---

## STAGE 5 — TRAINING ON COLAB (after Phase 3 + 4 code exists)

Setup per Colab session:
- [ ] Runtime → Change runtime type → T4 GPU
- [ ] Colab Secrets (key icon): add HF_TOKEN, WANDB_API_KEY
- [ ] Mount Drive; clone repo; pull latest

SFT:
- [ ] Convert notebook: `uvx jupytext --to ipynb notebooks/src/01_sft_warmstart.py` → upload/open in Colab
- [ ] Run; checkpoints land in Drive + HF Hub; keep sessions <3.5h; resume across sessions
- [ ] After training: run `scripts/sanity_eval.py` on Colab → gate: >90% JSON validity. Below gate → more SFT data/epochs

GRPO (target: ~12–18 GPU hours with dials; 20–35h worst case):
- [ ] Run decision-point extraction (Colab, uses SFT checkpoint) — subset mode (~700 tasks) first
- [ ] Convert + run `02_grpo_training.py`; point R5 cache path at Drive (cache persists across sessions — biggest hour-saver); multiple sessions; watch W&B reward curves + cache hit-rate between sessions
- [ ] Early stop: once val gate (SFT + 10 pts) holds 2 consecutive checkpoints, STOP. Never run to step ceiling on autopilot
- [ ] T3/T4 curriculum stage: only if gate cleared AND GPU budget remains
- [ ] Hours running over? Dial order: verify cache hits high → G=3 → skip T3/T4 → refresh stays OFF
- [ ] After EVERY session: run `scripts/audit_trajectories.py`, read 20 trajectories, write observations in `results/notes.md` (reward-hacking stories = article gold)
- [ ] Parallel trick: run evals / BFCL / GGUF export / extraction jobs on Kaggle (30 GPU-hrs/week) WHILE GRPO occupies Colab. One training run cannot split across platforms; surrounding jobs can
- [ ] Gate: val task-completion ≥ SFT + 10 pts

---

## STAGE 6 — EVALUATION (after Phase 5 code exists)

- [ ] Tag repo: `git tag v1.0-model && git push --tags`
- [ ] Colab: run eval runner for base / SFT / GRPO models (greedy)
- [ ] Colab: run GPT-4o-mini adapter eval (~$5 credits)
- [ ] Colab: run BFCL AST subsets (simple/multiple/parallel) for all three local variants
- [ ] MacBook: run judge script (`eval/judge.py`) — API-based, CPU fine; calibrate on 20 hand-scored samples first (you score them)
- [ ] Commit all CSVs + plots to `results/`; publish W&B report link

---

## STAGE 7 — SHIP (Phases 6–8 code)

- [ ] Colab: run `03_export_gguf.py` → merged + GGUF + LoRA pushed to HF Hub
- [ ] Fill real numbers into model card + dataset card; flip HF repos to PUBLIC
- [ ] Test Gradio app locally: `uv run python space/app.py` → both tabs work
- [ ] Push `space/` contents to the HF Space repo (git remote); add Space secret HF_TOKEN if needed; GGUF via LFS
- [ ] Verify Space: replay tab default, live tab responds, MCP endpoint listed
- [ ] PyPI: create API token → `uv build` → `uv publish`
- [ ] Test install elsewhere: `pip install <package>` in fresh venv → `toolsmith run --task`
- [ ] Verify eval-gate CI: open the intentional-failure PR branch → confirm blocked → close PR
- [ ] Record demo GIF (script in repo); add to README

---

## STAGE 8 — DISTRIBUTE

- [ ] README: real results tables, GIF, links — final review
- [ ] Medium article: draft (reward-hacking story or $0-GPU angle) → publish
- [ ] LinkedIn post ~120 words, hook-first
- [ ] Resume: add project with real metrics; regenerate PDF
- [ ] Pin repo on GitHub profile; add Space link to HF profile
- [ ] Send me final numbers → I draft bullets/article/post with you

---

## STANDING RULES (whole project)

- Never merge/push red CI
- Never train past a failed gate
- Every Colab session: pull latest repo first
- Log anomalies same day in `results/notes.md`
- Keys live in `.env` + Colab Secrets only. Never in chat, never in code, never in commits
