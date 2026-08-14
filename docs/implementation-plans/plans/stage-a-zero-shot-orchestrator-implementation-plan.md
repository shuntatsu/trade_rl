# Stage A Zero-Shot Evaluation Orchestrator Implementation Plan

**Goal:** Build the A6a deterministic validation and sealed-test orchestrator over the existing Stage A v2 contracts.

**Architecture:** Add immutable evaluation-cell and schedule contracts, a protocol-driven orchestrator, and atomic per-phase artifact publication. Real checkpoint loading and production execution remain behind the evaluator protocol for A6b.

**Tech Stack:** Python 3.12, dataclasses, Protocol, pytest, Ruff, MyPy, Import Linter.

## Global Constraints

- Keep all existing Stage A v2 schemas unchanged.
- Do not import serving, Torch, Stable-Baselines3, PostgreSQL, or market adapters.
- Evaluate the exact declared Cartesian product.
- Evaluate one shared baseline per triplet/fold/seed cell.
- Recompute validation selection before ledger access.
- Authorize every test fold before the first test evaluation.
- Evaluate only the selected candidate on test.
- Publish complete immutable directories only.

## Task 1: Cell, schedule, and run contracts

**Files:**
- `trade_rl/workflows/stage_a_zero_shot_runner_contracts.py`
- `tests/workflows/test_stage_a_zero_shot_runner_contracts.py`

- Add RED tests for policy/baseline identity closure, finite results, unique schedule folds, exact plan closure, and run identity consistency.
- Implement immutable request, result, evaluator protocol, schedule, validation-run, and sealed-test-run contracts.
- Compute request and run content digests from every identity-bearing field.
- Run the contract test until GREEN.

## Task 2: Validation orchestration

**Files:**
- `trade_rl/workflows/stage_a_zero_shot_runner.py`
- `tests/workflows/test_stage_a_zero_shot_runner.py`

- Add a recording evaluator and assert exact baseline/policy call counts.
- Assert deterministic triplet, fold, seed, baseline, candidate ordering.
- Assert shared baseline evidence and growth across candidates in each cell.
- Assert request/result digest mismatch fails closed.
- Implement complete validation observation generation and call the maintained v2 selection gate.
- Run validation tests until GREEN.

## Task 3: Sealed-test orchestration

**Files:**
- `trade_rl/workflows/stage_a_zero_shot_runner.py`
- `tests/workflows/test_stage_a_zero_shot_runner.py`

- Test that failed or forged validation output causes zero ledger and test-evaluator calls.
- Test that all fold authorizations occur before the first test evaluation.
- Test selected-only policy requests and one shared baseline per test cell.
- Test repeated sealed-test execution is rejected by the ledger.
- Recompute validation selection, authorize every fold, build selected-only test evidence, and call the maintained v2 final gate.
- Run sealed-test tests until GREEN.

## Task 4: Atomic phase publication

**Files:**
- `trade_rl/workflows/stage_a_zero_shot_artifacts.py`
- `tests/workflows/test_stage_a_zero_shot_artifacts.py`

- Test exact validation and sealed-test package filenames.
- Inject a writer failure and prove the final package and staging directory are absent.
- Reject publication over an existing package.
- Implement unique staging directories, maintained atomic file writers, canonical access-record JSON, completed-directory rename, and recursive failure cleanup.
- Run publisher tests until GREEN.

## Task 5: Documentation and verification

**Files:**
- `docs/operations/stage-a-zero-shot-evaluation-plan.md`

- Mark A6a complete and preserve canonical loading, real source verification, CLI, and PostgreSQL wiring for A6b.
- Run all new workflow tests and existing Stage A contract/gate/hardening tests.
- Run Ruff, formatter check, MyPy, and Import Linter.
- Run full branch-aware pytest and the critical coverage ratchet.
- Record the exact verified head SHA and integrate only that head.
