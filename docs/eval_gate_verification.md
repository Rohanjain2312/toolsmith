# Verifying the eval gate actually blocks a reward regression

This is a one-time sanity check that `.github/workflows/eval-gate.yml` really fails a PR when
reward-scoring code regresses — not just when tests are literally broken. **Human-run**: this
exercises GitHub PR checks, which Claude Code does not have the standing authorization to
create/push/open on your behalf.

## Why this check exists

`ci.yml` runs `pytest` + `ruff`, which catches broken *code*. It does **not** catch a reward
component silently changing VALUE while staying internally consistent — e.g. someone "fixing"
`R5_GOAL_SATISFIED` from `3.0` to `1.5` wouldn't fail any existing unit test (nothing asserts the
literal constant), but it would silently change the GRPO training signal. `eval-gate.yml`
(`scripts/eval_gate.py`) exists specifically to catch this class of regression, by replaying 50
cached (prefix, action) cases through the live reward pipeline and comparing against a
checked-in baseline (`tests/fixtures/eval_gate_cases.jsonl`).

## Steps

1. Create a throwaway branch:
   ```
   git checkout -b test/verify-eval-gate-blocks-regression
   ```

2. Apply the fixture regression (drops `R5_GOAL_SATISFIED` from `3.0` to `1.5` —
   a real, believable typo-class mistake, not a syntax error):
   ```
   git apply scripts/fixtures/intentional_reward_regression.patch
   git add -u
   git commit -m "test: intentional reward regression (verify eval-gate catches this)"
   ```

3. Push and open a PR against `main`:
   ```
   git push -u origin test/verify-eval-gate-blocks-regression
   gh pr create --title "DO NOT MERGE: eval-gate verification" --body "Verifying eval-gate.yml blocks a reward-value regression. Closing without merging."
   ```

4. In the PR's checks tab, confirm:
   - `CI` (`ci.yml`) — **passes**. Ruff and pytest have nothing to complain about; this is the
     point — it proves eval-gate is catching something CI structurally cannot.
   - `Eval Gate` (`eval-gate.yml`) — **fails**, with a log line like:
     ```
     EVAL GATE FAILED: score drift in gate-paris.r5_goal_satisfied: expected 3.0, got 1.5
     ```

5. Confirm locally too (optional, faster feedback loop than waiting on Actions):
   ```
   PYTHONPATH=src uv run python scripts/eval_gate.py
   # exit code 1, same "score drift" message
   ```

6. Clean up — close the PR without merging, then delete the branch:
   ```
   gh pr close --delete-branch
   ```
   (or, if you didn't push: `git checkout main && git branch -D test/verify-eval-gate-blocks-regression`)

## What "pass" looks like

- `ci.yml` green, `eval-gate.yml` red, on the same commit. That divergence is the whole proof:
  a reward-value regression that's invisible to lint/unit-tests is caught by the gate.
- The failure message names the exact case and component that drifted
  (`gate-paris.r5_goal_satisfied`), not just "something failed" — useful signal for whoever
  has to debug a real regression later.

## If you need to test a different kind of regression

`scripts/fixtures/intentional_reward_regression.patch` only exercises one reward constant. To
test schema-export breakage instead, temporarily comment out one `registry.register(...)` call
in `src/toolsmith/tools/sandbox/*.py` — `validate_tool_schema_exports()` in `scripts/eval_gate.py`
fails as soon as either tool export list drops below 12.
