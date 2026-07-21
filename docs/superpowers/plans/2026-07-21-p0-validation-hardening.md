# P0 Validation Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add persistent research-integrity gates and independent verification for sealed tests, serving parity, historical metadata, accounting, execution sensitivity, and multi-seed evidence.

**Architecture:** Introduce narrow trust-boundary interfaces in production code and keep mathematical oracles under tests. PostgreSQL enforces cross-process uniqueness, canonical observation reconstruction unifies training and serving, and phase-aware promotion rejects non-historical metadata. All evaluation evidence is bound to exact identities and commit provenance.

**Tech Stack:** Python 3.12, NumPy, pytest, psycopg 3, PostgreSQL 16, Stable-Baselines3, Docker Compose, GitHub Actions.

## Global Constraints

- Use TDD: every production change begins with a failing test.
- Do not reuse production accounting or execution functions in the independent oracle.
- Preserve research-only and no-direct-order-routing boundaries.
- Fail closed on identity mismatch, duplicate sealed-test access, and non-historical promotion.
- Bind verification output to exact commit SHA and image digest.

---

### Task 1: Establish exact baseline evidence

**Files:**
- Create: `docs/verification/2026-07-21-p0-validation-baseline.md`
- Temporary: `.github/workflows/p0-source-export.yml`

- [ ] Export the exact PR head as an Actions artifact.
- [ ] Run `ruff check .`, `ruff format --check .`, `mypy .`, and `pytest -q` locally from the exported tree.
- [ ] Confirm the existing CI, PostgreSQL workflow, Serving E2E, and training-image jobs run on the same head SHA.
- [ ] Record workflow IDs, head SHA, source-tree digest, lockfile digest, and Docker image digest.

### Task 2: Persistent sealed-test ledger

**Files:**
- Modify: `trade_rl/evaluation/walk_forward/sealed_test.py`
- Modify: `trade_rl/workflows/fold_runner.py`
- Modify: `trade_rl/catalog/contracts.py`
- Modify: `trade_rl/catalog/postgres.py`
- Modify: `trade_rl/catalog/sql/0001_artifact_catalog.sql` or add an ordered migration
- Test: `tests/evaluation/walk_forward/test_sealed_test.py`
- Test: `tests/catalog/test_postgres_integration.py`

- [ ] Write a failing test that two independent ledger instances reject the same `(plan, dataset, fold)` authorization when backed by one PostgreSQL catalog.
- [ ] Add a `SealedTestLedgerProtocol` and inject it into `ConcreteFoldRunner`.
- [ ] Add an immutable PostgreSQL record with a unique key on plan, dataset, and fold.
- [ ] Map uniqueness violations to the existing sealed-test error contract.
- [ ] Run focused unit and PostgreSQL integration tests.

### Task 3: Training-serving observation parity

**Files:**
- Modify: `trade_rl/rl/observations.py`
- Modify: `trade_rl/serving/runtime.py`
- Modify: `trade_rl/integrations/sb3_serving.py`
- Test: `tests/e2e/test_research_to_serving_v2.py`
- Test: `tests/serving/test_observation_parity.py`

- [ ] Write a failing test that steps a real environment several times and captures non-zero structured state.
- [ ] Compare feature order, availability mask, staleness, book state, pending target, previous action, raw flat observation, normalized observation, member action, and ensemble action.
- [ ] Extract one canonical reconstruction function used by both environment and serving.
- [ ] Require flat serving inputs to be produced by the canonical reconstruction boundary or a verified state snapshot.
- [ ] Run focused parity and E2E tests.

### Task 4: Historical metadata promotion gate

**Files:**
- Modify: `trade_rl/workflows/binance_metadata_modes.py`
- Modify: `trade_rl/workflows/training_run.py`
- Modify: `trade_rl/serving/package.py`
- Test: `tests/workflows/test_binance_metadata_modes.py`
- Test: `tests/serving/test_package.py`

- [ ] Write failing tests that selected-final and serving promotion reject `frozen_snapshot` and `conservative_static` metadata.
- [ ] Add a phase-aware promotion validator requiring `historical_signed` evidence and matching dataset/rule-history identity.
- [ ] Keep development and sensitivity runs permissive but explicitly marked non-promotable.
- [ ] Run focused workflow and serving-package tests.

### Task 5: Independent accounting oracle

**Files:**
- Create: `tests/oracles/manual_accounting.py`
- Create: `tests/simulation/test_manual_accounting_oracle.py`

- [ ] Implement independent equations for cash, quantities, fees, funding, split, delisting recovery, margin, PnL, and reward without importing production accounting/execution helpers.
- [ ] Build a two-symbol, five-bar deterministic market fixture.
- [ ] Cover no-cost, fees, partial fills, split, delisting, and margin-deficit scenarios.
- [ ] Compare every intermediate state and final reward to production results with explicit tolerances.

### Task 6: Execution sensitivity and multi-seed evidence

**Files:**
- Create: `trade_rl/evaluation/execution_sensitivity.py`
- Create: `tests/evaluation/test_execution_sensitivity.py`
- Modify: `trade_rl/evaluation/metrics.py`
- Create: `tests/evaluation/test_seed_aggregate.py`

- [ ] Write failing tests for the complete parameter matrix: fees 1/2/4, spread 1/2, slippage 1/2/4, capacity 100/50/25, signal delay 0/1/2, limit fill optimistic/neutral/conservative, tradability delay 0/1.
- [ ] Add deterministic sensitivity identities and reject incomplete matrices.
- [ ] Aggregate per-seed return, median, worst seed, max drawdown, turnover, baseline difference, and bootstrap confidence interval without best-seed selection.
- [ ] Verify an unused-period evaluation is identity-bound and not fed into selection.

### Task 7: Full verification and evidence publication

**Files:**
- Update: `docs/verification/2026-07-21-p0-validation-baseline.md`
- Delete: `.github/workflows/p0-source-export.yml`

- [ ] Run all focused tests, then `ruff check .`, `ruff format --check .`, `mypy .`, and `pytest -q`.
- [ ] Run PostgreSQL integration, Serving E2E, Docker Compose validation, and training-image build in GitHub Actions.
- [ ] Record exact head SHA, workflow IDs, job conclusions, source-tree digest, lockfile digest, and Docker image digest.
- [ ] Remove temporary source-export infrastructure before final integration.
