# Walk-Forward E2E Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a concrete adapter-driven nested walk-forward workflow that trains, selects, evaluates, stitches, and emits gate-ready identity-bound results.

**Architecture:** Keep market data and model implementations behind typed adapters. The workflow owns chronology, fold-local request scoping, deterministic selection, baseline fallback, one-shot sealed OOS evaluation, stitching, and identity validation.

**Tech Stack:** Python 3.12, dataclasses, NumPy, pytest, existing `trade_rl` domain and workflow modules.

## Global Constraints

- No outer-test data may reach training, checkpoint validation, or configuration selection.
- Selection rules are deterministic and fixed before outer-OOS execution.
- Baseline fallback remains a first-class result.
- Existing production status remains NO-GO.

---

### Task 1: Define executable fold contracts

**Files:**
- Create: `trade_rl/workflows/fold_runner.py`
- Test: `tests/workflows/test_fold_runner.py`

- [x] Write failing tests for range-scoped adapter requests, deterministic candidate selection, baseline fallback, and exactly-once sealed OOS evaluation.
- [x] Run the focused tests and confirm RED.
- [x] Implement immutable request/result records and `ConcreteFoldRunner`.
- [x] Run focused tests and confirm GREEN.

### Task 2: Execute and stitch all folds

**Files:**
- Modify: `trade_rl/workflows/walk_forward.py`
- Test: `tests/workflows/test_walk_forward_execution.py`

- [x] Write failing tests for chronological execution, non-overlapping OOS stitching, and dataset identity enforcement.
- [x] Run focused tests and confirm RED.
- [x] Implement `execute_walk_forward` and typed run result.
- [x] Run focused tests and confirm GREEN.

### Task 3: Bind final evaluation identity into gates

**Files:**
- Modify: `trade_rl/domain/evaluation.py`
- Modify: `trade_rl/domain/releases.py`
- Test: `tests/domain/test_artifact_invariants.py`

- [x] Write failing tests showing a gate cannot authorize a different dataset or selected policy identity.
- [x] Run focused tests and confirm RED.
- [x] Add dataset, selected-policy, and final-evaluation identity to `GateDecision` and enforce them in release construction.
- [x] Run focused tests and confirm GREEN.

### Task 4: Document and verify

**Files:**
- Modify: `README.md`
- Modify: `docs/RESEARCH_STATUS.md`

- [x] Document the concrete workflow boundary and remaining adapter requirements.
- [x] Run Ruff, format, Mypy, Import Linter, full pytest with branch coverage, and CLI smoke.
- [x] Review the complete diff and update the existing pull request.
