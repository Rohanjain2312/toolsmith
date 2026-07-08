# Verifying the eval gate actually blocks a reward regression

This is a one-time sanity check that `.github/workflows/eval-gate.yml` really fails a PR when
reward-scoring code regresses. **Human-run**: this exercises GitHub PR checks, which Claude
Code does not have the standing authorization to create/push/open on your behalf.

## Why this check exists

`eval-gate.yml` (`scripts/eval_gate.py`) replays 50 cached (prefix, action) cases through the
live reward pipeline and compares against a checked-in baseline
(`tests/fixtures/eval_gate_cases.jsonl`), and separately validates that both tool-schema
exports still cover all 12 registered tools. It's a fast, single-purpose, independent-of-pytest
check of exactly the reward pipeline's output values against a real snapshot — useful
defense-in-depth even where its coverage overlaps with unit tests, and the part of the codebase
most consequential to get wrong silently (a reward-value regression changes the GRPO training
signal without necessarily looking "broken").

**Honest caveat, verified below, not assumed:** in this codebase's *current* state, the
fixture regression below (`R5_GOAL_SATISFIED` 3.0 → 1.5) is **also** caught by ordinary
`pytest` — `tests/rewards/test_composite.py` already hardcodes that constant's expected value,
and this phase's own `tests/scripts/test_eval_gate.py` calls the gate's `check_case()` function
directly. So for *this specific* regression, `ci.yml` and `eval-gate.yml` both go red — that's
redundant confirmation from two independent code paths, not proof that eval-gate catches
something CI structurally cannot. Don't claim otherwise without re-verifying; the value of a
dedicated gate is real (a fast, targeted check that doesn't require running the whole suite,
and a safety net for reward changes that a future PR's unit tests might not fully cover), but
it is not, today, a strict superset of what pytest already checks.

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
   - `Eval Gate` (`eval-gate.yml`) — **fails**, with a log line like:
     ```
     EVAL GATE FAILED: score drift in gate-paris.r5_goal_satisfied: expected 3.0, got 1.5
     ```
   - `CI` (`ci.yml`) — **also fails** (at the pytest step, on the two `test_composite.py`
     assertions and two `test_eval_gate.py` assertions that hardcode the same constant). This
     is expected for this fixture, not a bug in either workflow — see the caveat above.

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

- `eval-gate.yml` red on the regression commit, with a failure message naming the exact case
  and component that drifted (`gate-paris.r5_goal_satisfied`), not just "something failed" —
  useful signal for whoever has to debug a real regression later.
- If you want to see eval-gate catch something `pytest` genuinely misses, look for a regression
  that only manifests on inputs the hand-written unit tests don't happen to cover (e.g. one of
  the 49 non-Paris/Tokyo cities in the 50-case set) rather than a top-level reward constant —
  those are directly asserted in multiple unit test files and will always be caught by pytest
  too.

## If you need to test a different kind of regression

`scripts/fixtures/intentional_reward_regression.patch` only exercises one reward constant. To
test schema-export breakage instead, temporarily comment out one `registry.register(...)` call
in `src/toolsmith/tools/sandbox/*.py` — `validate_tool_schema_exports()` in `scripts/eval_gate.py`
fails as soon as either tool export list drops below 12 (note: `tests/tools/sandbox/test_all_registered.py`
already asserts this too, via `pytest`, for the same reason described above).
