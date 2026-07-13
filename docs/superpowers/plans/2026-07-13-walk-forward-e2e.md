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

- [ ] Write failing tests for range-scoped adapter requests, deterministic candidate selection, baseline fallback, and exactly-once sealed OOS evaluation.
- [ ] Run the focused tests and confirm RED.
- [ ] Implement immutable request/result records and `ConcreteFoldRunner`.
- [ ] Run focused tests and confirm GREEN.

### Task 2: Execute and stitch all folds

**Files:**
- Modify: `trade_rl/workflows/walk_forward.py`
- Test: `tests/workflows/test_walk_forward_execution.py`

- [ ] Write failing tests for chronological execution, non-overlapping OOS stitching, and dataset identity enforcement.
- [ ] Run focused tests and confirm RED.
- [ ] Implement `execute_walk_forward` and typed run result.
- [ ] Run focused tests and confirm GREEN.

### Task 3: Bind evaluation identity into gates

**Files:**
- Modify: `trade_rl/domain/evaluation.py`
- Modify: `trade_rl/domain/releases.py`
- Test: `tests/domain/test_artifact_invariants.py`

- [ ] Write failing tests showing a gate for evaluation A cannot authorize selection evaluation B.
- [ ] Run focused tests and confirm RED.
- [ ] Add evaluation digest to `GateDecision` and enforce it in release construction.
- [ ] Run focused tests and confirm GREEN.

### Task 4: Document and verify

**Files:**
- Modify: `README.md`
- Modify: `docs/RESEARCH_STATUS.md`

- [ ] Document the concrete workflow boundary and remaining adapter requirements.
- [ ] Run Ruff, format, Mypy, Import Linter, full pytest with branch coverage, and CLI smoke.
- [ ] Review the complete diff and open a stacked PR.
