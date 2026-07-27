# Causal Scenario C3 Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract and verify the evaluation-only C3 core as an independent, mergeable change on current `main`.

**Architecture:** The implementation remains inside `trade_rl.evaluation`. Immutable contracts bind all identities and arrays; a pre-replay decision artifact enforces chronology; realized comparison consumes only a reloaded decision; fold and aggregate reports recompute paired evidence; a pure fail-closed gate determines Phase A eligibility. CLI, walk-forward orchestration, training, Serving, release, promotion, Studio, and direct execution remain out of scope.

**Tech Stack:** Python 3.12, NumPy, immutable dataclasses, existing artifact hashing/canonical JSON helpers, C1/C2 causal-scenario APIs, paired moving-block bootstrap, Pytest.

## Global Constraints

- Production remains `NO-GO`.
- C3 is evaluation-only and must not enter maintained training, Serving, promotion, release, Studio, or order-routing paths.
- A decision must be persisted and reloaded before realized replay.
- Compared policies must share exact replay identity and independent state clones.
- At least six folds and 180 selection days are required for a passing formal Phase A gate.
- Missing, non-finite, incompatible, or tampered evidence fails closed.

---

### Task 1: Add RED evaluation contracts

**Files:**
- Create: `tests/evaluation/test_causal_scenario_c3.py`
- Create: `tests/evaluation/test_causal_scenario_c3_identity_daily.py`
- Create: `tests/evaluation/test_causal_scenario_c3_perfect_information.py`
- Create: `tests/evaluation/test_causal_scenario_c3_plan_closure.py`

**Interfaces:**
- Consumes missing `trade_rl.evaluation.causal_scenario_c3_*` modules.
- Produces failing behavioral contracts for immutable identity, chronology, replay equivalence, daily aggregation, gate closure, and Perfect-Information compatibility.

- [x] Add the tests from the approved C3 evaluation design without production modules.
- [x] Run focused pytest and confirm failure is caused by missing C3 modules.
- [x] Commit the RED state.

### Task 2: Add immutable contracts and decision persistence

**Files:**
- Create: `trade_rl/evaluation/causal_scenario_c3_contracts.py`
- Create: `trade_rl/evaluation/causal_scenario_c3_prediction.py`
- Create: `trade_rl/evaluation/causal_scenario_c3_decision_artifact.py`
- Test: `tests/evaluation/test_causal_scenario_c3.py`
- Test: `tests/evaluation/test_causal_scenario_c3_identity_daily.py`

**Interfaces:**
- Produces `CausalScenarioC3Config`, `C3ReplayIdentity`, `PersistedScenarioDecision`, `RealizedPolicyOutcome`, `CausalScenarioQueryComparison`, `PerfectInformationComparison`, `C3PredictionEvidence`, `LoadedC3Decision`, writer, and loader.

- [x] Validate strict schema versions, positive/non-negative integer fields, finite values, SHA-256 identities, unique candidate identities, read-only C-contiguous arrays, ranking reconstruction, and decision digest closure.
- [x] Publish deterministic `decision.json` and `arrays.npz` with exact file closure, atomic replacement, idempotent identical writes, and conflict rejection.
- [x] Reload and recompute all identities before exposing a decision to replay.
- [x] Run focused tests and commit GREEN.

### Task 3: Add realized and adverse comparisons

**Files:**
- Create: `trade_rl/evaluation/causal_scenario_c3_runner.py`
- Create: `trade_rl/evaluation/causal_scenario_c3_perfect_information.py`
- Create: `trade_rl/evaluation/causal_scenario_c3_adverse_source.py`
- Create: `trade_rl/evaluation/causal_scenario_c3_adverse.py`
- Test: `tests/evaluation/test_causal_scenario_c3.py`
- Test: `tests/evaluation/test_causal_scenario_c3_perfect_information.py`
- Create: `tests/evaluation/test_causal_scenario_c3_adverse.py`
- Create: `tests/evaluation/test_causal_scenario_c3_adverse_source.py`

**Interfaces:**
- Produces `build_persisted_scenario_decision`, `run_c3_query_comparison`, compatibility evaluation for Perfect-Information evidence, verified adverse-source loading, and nominal/adverse comparison results.

- [x] Require exact replay-identity equality for all comparators.
- [x] Use fresh cloned replay state for Trend, Scenario Oracle, deterministic PPO, each seeded random comparator, and optional Perfect-Information comparison.
- [x] Apply the selected residual only at the first decision and zero residual afterward where declared.
- [x] Recompute realized ranking, top-one regret, and Spearman correlation from complete finite candidate outcomes.
- [x] Reject adverse evidence with source, interval, schema, or digest mismatch.
- [x] Run focused tests and commit GREEN.

### Task 4: Add fold/aggregate reports and artifacts

**Files:**
- Create: `trade_rl/evaluation/causal_scenario_c3_report.py`
- Create: `trade_rl/evaluation/causal_scenario_c3_artifact.py`
- Test: `tests/evaluation/test_causal_scenario_c3_identity_daily.py`
- Create: `tests/evaluation/test_causal_scenario_c3_adverse_report_binding.py`

**Interfaces:**
- Produces immutable calibration buckets, policy execution summaries, fold reports, aggregate reports, adverse evidence bindings, deterministic writer, and loader.

- [x] Compute daily paired log-growth differences using maintained moving-block inference with the configured block and resample count.
- [x] Preserve independent fold distributions; never concatenate fold equity curves.
- [x] Aggregate ranking, calibration, execution, economic cost, drawdown, termination, scenario coverage, and adverse evidence.
- [x] Persist exact JSON/NPZ closure and recompute aggregates on load.
- [x] Run focused tests and commit GREEN.

### Task 5: Add the pure Phase A gate and exports

**Files:**
- Create: `trade_rl/evaluation/causal_scenario_c3_gate.py`
- Modify: `trade_rl/evaluation/__init__.py`
- Test: `tests/evaluation/test_causal_scenario_c3_plan_closure.py`

**Interfaces:**
- Produces immutable gate thresholds/evidence and `evaluate_phase_a_entry_gate`.

- [x] Encode all nine approved conditions and one stable reason for every failed condition.
- [x] Bind thresholds and all input evidence in a deterministic gate digest.
- [x] Keep evaluation pure: no filesystem, dataset, model, environment, network, or mutable-state access.
- [x] Export only evaluation-layer public APIs.
- [x] Run focused tests and commit GREEN.

### Task 6: Verify exact head and open the independent PR

**Files:**
- Modify: `docs/superpowers/plans/2026-07-27-causal-scenario-c3-core.md` only if verification evidence requires correction.

**Interfaces:**
- Produces a mergeable evaluation-only PR based on current `main`.

- [x] Run focused Ruff and format checks.
- [x] Run focused Mypy.
- [x] Run focused C3 tests: `30 passed`.
- [ ] Run import-linter on the exact PR head.
- [ ] Run the complete pytest suite with branch coverage and critical-coverage validation.
- [ ] Run CLI smoke and repository architecture guards.
- [ ] Confirm no C3 imports exist in training, Serving, promotion, release, Studio, or order-routing packages.
- [x] Open draft PR #223.
- [ ] Wait for exact-head CI, repair failures, and mark ready only after all required jobs pass.

## Extraction verification

The isolated extractor verified the RED state before production modules existed, then passed the focused C3 suite (`30 passed`), Ruff, Ruff format, and Mypy. Two inconsistencies inherited from draft PR #196 were corrected without weakening contracts: legacy fold-report fixtures now provide the required adverse-evidence digest, and adverse-threshold construction uses explicit typed dataclass arguments instead of an untyped heterogeneous `**payload` expansion.
