# Causal Alpha V7 Causal Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a causal, artifact-bound V7 calibration and attribution path through fixed Signal, Selection, Admission, BC, and RL gates.

**Architecture:** Reuse V4 as the base shared forecaster, fit one purged train-only shared calibration layer per knowledge cutoff, and compare control, symmetric contrarian, and calibrated target paths under identical V6 simulator economics. Persist cutoff checkpoints and attribution evidence without exposing Admission data.

**Tech Stack:** Python 3.12, NumPy, pytest, Ruff, Mypy, Docker, PostgreSQL/Binance research data, existing Trade RL artifact and simulator contracts.

**Spec:** `docs/implementation-plans/specs/2026-08-26-causal-alpha-v7-causal-calibration-design.md`

## Global Constraints

- Reward is exactly `100 * net_log_return`; no auxiliary reward terms.
- No symbol ID, symbol-specific intercept, symbol exclusion, gate relaxation, or Admission tuning.
- Calibration labels end strictly before the Selection knowledge cutoff and are separated from the base-fit cutoff by at least the maximum label horizon.
- Candidate execution, costs, caps, risk, cadence, and episode identities are paired and identical.
- Working prediction/calibration blocks contain at most 4,096 rows.
- Admission, BC, and RL remain unreachable until all preceding gates pass.

---

### Task 1: Calibration contracts and causal ranges

**Files:**
- Create: `trade_rl/learning/causal_alpha_v7.py`
- Create: `tests/learning/test_causal_alpha_v7.py`

**Interfaces:**
- Produces: `CausalAlphaV7Candidate`, `CausalAlphaV7CalibrationConfig`, `CausalAlphaV7CalibrationRange`, and content-addressed payload methods.
- Consumes: existing hashing and SHA-256 validation contracts only.

- [ ] **Step 1: Write failing contract tests** for canonical candidate order, fixed tail/purge values, invalid or overlapping ranges, digest tampering, and a feature schema containing `symbol` or `symbol_id`.
- [ ] **Step 2: Run** `uv run pytest -q tests/learning/test_causal_alpha_v7.py` **and verify failures are caused by missing V7 contracts.**
- [ ] **Step 3: Implement minimal frozen dataclasses** with early validation, exact candidate values `v6_control`, `symmetric_contrarian`, `causal_calibrated`, and content digests.
- [ ] **Step 4: Run** `uv run pytest -q tests/learning/test_causal_alpha_v7.py`, `uv run ruff check trade_rl/learning/causal_alpha_v7.py tests/learning/test_causal_alpha_v7.py`, and focused Mypy; require pass.
- [ ] **Step 5: Commit** with `feat: define causal alpha v7 calibration contracts`.

### Task 2: Purged shared calibration fit

**Files:**
- Create: `trade_rl/learning/causal_alpha_v7_calibration.py`
- Create: `tests/learning/test_causal_alpha_v7_calibration.py`

**Interfaces:**
- Consumes: V7 calibration contracts and existing `CausalAlphaRidgeModel` fitting primitive.
- Produces: `fit_causal_alpha_v7_calibration(...) -> CausalAlphaV7CalibrationFit` and `CausalAlphaV7CalibrationFit.predict(...)`.

- [ ] **Step 1: Write failing tests** proving labels beyond the calibration end are rejected, symbol identity features are rejected, both direction supports are required, results are deterministic across symbol order, and prediction calls never exceed 4,096 rows.
- [ ] **Step 2: Run the new test file** and observe the missing fit API fail.
- [ ] **Step 3: Implement a shared regularized calibration fit** over prediction, direction, uncertainty, volatility, liquidity, stress, and slow agreement; store range/support/schema/model digests.
- [ ] **Step 4: Run unit tests, Ruff, and focused Mypy** until clean.
- [ ] **Step 5: Commit** with `feat: fit purged causal alpha v7 calibration`.

### Task 3: Fixed V7 candidate target paths

**Files:**
- Create: `trade_rl/learning/causal_alpha_v7_target.py`
- Create: `tests/learning/test_causal_alpha_v7_target.py`
- Modify: `trade_rl/learning/causal_alpha_v7.py`

**Interfaces:**
- Consumes: V4 forecast, V7 calibration fit, V6 target config, costs, caps, risk caps, and actionable mask.
- Produces: `causal_alpha_v7_target_paths(...) -> Mapping[CausalAlphaV7Candidate, CausalAlphaV7TargetPath]`.

- [ ] **Step 1: Write failing tests** proving control equals V6 fast-only, contrarian negates forecast/direction only, calibrated uses calibrated outputs, and all candidates preserve costs/caps/cadence/initial weight.
- [ ] **Step 2: Run the target tests** and verify missing implementation failures.
- [ ] **Step 3: Implement minimal adapters** around the existing V6 target compiler; do not duplicate target optimization logic.
- [ ] **Step 4: Run target, V6 regression, Ruff, and focused Mypy checks.**
- [ ] **Step 5: Commit** with `feat: compile fixed causal alpha v7 candidates`.

### Task 4: Reconciled replay attribution

**Files:**
- Create: `trade_rl/workflows/universal_causal_alpha_v7_attribution.py`
- Create: `tests/workflows/test_universal_causal_alpha_v7_attribution.py`

**Interfaces:**
- Consumes: target path, calibrated inputs, causal quartile boundaries, and `ActionPathEvaluation` step/performance evidence.
- Produces: `CausalAlphaV7AttributionEvidence` with fixed bins, support, gross/net contribution, cost, exposure-hours, and digest.

- [ ] **Step 1: Write failing tests** for long/short/flat, transition, confidence, volatility, liquidity, and slow-agreement bins; require gross/net/cost totals to reconcile and reject unsupported or nonfinite bins.
- [ ] **Step 2: Run attribution tests** and observe missing API failure.
- [ ] **Step 3: Implement fixed aggregate attribution** without raw row serialization or candidate-specific accounting.
- [ ] **Step 4: Run attribution tests, Ruff, and focused Mypy.**
- [ ] **Step 5: Commit** with `feat: attribute causal alpha v7 replay economics`.

### Task 5: Signal, Selection, and stage orchestration

**Files:**
- Create: `trade_rl/workflows/universal_causal_alpha_v7_signal.py`
- Create: `trade_rl/workflows/universal_causal_alpha_v7_selection.py`
- Create: `trade_rl/workflows/universal_causal_alpha_v7_stage_entry.py`
- Create: `tests/workflows/test_universal_causal_alpha_v7_signal.py`
- Create: `tests/workflows/test_universal_causal_alpha_v7_selection.py`
- Create: `tests/workflows/test_universal_causal_alpha_v7_stage_entry.py`

**Interfaces:**
- Consumes: V4 prepared stage data, V7 calibration and target APIs, V6 replay simulator and universal gate values.
- Produces: artifact-bound Signal/Selection evidence, paired replay/attribution digests, and stage JSON output.

- [ ] **Step 1: Write failing tests** for causal ranges, long/short support, three-way paired identity, unchanged reward/gates, and Selection rejection preventing Admission.
- [ ] **Step 2: Run new workflow tests** and verify missing modules fail.
- [ ] **Step 3: Implement cutoff-grouped fitting and replay** with explicit garbage collection and 4,096-row calibration blocks.
- [ ] **Step 4: Run V7 tests plus V4-V6 regression, Ruff, Mypy, and Import Linter.**
- [ ] **Step 5: Commit** with `feat: evaluate causal alpha v7 selection`.

### Task 6: Checkpoint, pipeline, and artifact store

**Files:**
- Create: `trade_rl/workflows/universal_causal_alpha_v7_checkpoint.py`
- Create: `trade_rl/workflows/universal_causal_alpha_v7_artifact_store.py`
- Create: `trade_rl/workflows/universal_causal_alpha_v7_pipeline.py`
- Create: `tests/workflows/test_universal_causal_alpha_v7_checkpoint.py`
- Create: `tests/workflows/test_universal_causal_alpha_v7_pipeline.py`

**Interfaces:**
- Consumes: V7 stage evidence and existing V6 Admission/BC/RL handoff contracts where compatible.
- Produces: append-only cutoff checkpoint, terminal result envelope, and fail-closed stage ordering.

- [ ] **Step 1: Write failing tests** for resume digest identity, duplicates/reordering/torn records, normal Selection rejection, and Admission/BC/RL absence.
- [ ] **Step 2: Run tests** and observe missing checkpoint/pipeline failures.
- [ ] **Step 3: Implement append-only checkpoint validation and pipeline ordering.**
- [ ] **Step 4: Run new and inherited pipeline tests, Ruff, Mypy, and Import Linter.**
- [ ] **Step 5: Commit** with `feat: checkpoint causal alpha v7 research`.

### Task 7: Authored config, runner, and Docker provenance

**Files:**
- Create: `examples/binance/universal-causal-alpha-v7-research.json`
- Create: `trade_rl/workflows/universal_causal_alpha_v7_runner.py`
- Create: `scripts/run_universal_causal_alpha_v7_research.py`
- Create: `tests/workflows/test_universal_causal_alpha_v7_runner.py`
- Modify: `pyproject.toml`
- Modify: `docs/ARCHITECTURE.md`

**Interfaces:**
- Consumes: fixed V7 config and V7 stage/pipeline APIs.
- Produces: `trade-rl-causal-alpha-v7` CLI and provenance-bound Docker command.

- [ ] **Step 1: Write failing runner tests** for exact authored config, source/runtime/V4 digests, non-root execution, fixed stage order, and terminal exit codes.
- [ ] **Step 2: Run tests** and verify missing runner/config failure.
- [ ] **Step 3: Implement runner and CLI**, update package/architecture contracts, and avoid environment-selected thresholds.
- [ ] **Step 4: Run all V7 tests, V4-V6 regression, Ruff, Linux Mypy, Import Linter, and `git diff --check`.**
- [ ] **Step 5: Commit** with `feat: expose causal alpha v7 research runner`.

### Task 8: Memory diagnostic and complete real-data run

**Files:**
- Create after run: `report/causal-alpha-v7-<terminal>-20260826.md`
- Produce in Docker volume: `var/runs/causal-alpha-v7-prod-20260826/*`

**Interfaces:**
- Consumes: immutable V7 image, PostgreSQL/Binance data, runtime/V4 manifests, and authored config.
- Produces: Signal, Selection, optional Admission/BC/RL evidence, checkpoint, result envelope, and report.

- [ ] **Step 1: Build the image** with frozen lock/source/runtime labels and verify torch compile, non-root user, and manifest digests.
- [ ] **Step 2: Run one worst-cutoff diagnostic** under the 8 GB limit and require exit 0 without OOM.
- [ ] **Step 3: Launch the complete real-data run** with a durable named container and volume-backed output.
- [ ] **Step 4: Monitor checkpoint, reward, gross/net wealth, per-symbol results, turnover, costs, RSS, and OOM state; fix implementation defects only through new failing tests and rebuild.**
- [ ] **Step 5: Respect terminal gates:** continue to Admission/BC/RL only if evidence passes; never relax thresholds or open holdout after rejection.
- [ ] **Step 6: Write and commit the Japanese GPT handoff report** with branch, commits, image/container IDs, artifact hashes, per-symbol wealth, attribution, reward equality, gate outcome, and explicit non-claims.

## Self-review

- Spec coverage: every causal boundary, candidate, attribution, checkpoint, gate, provenance, memory, and real-run requirement maps to Tasks 1-8.
- Placeholder scan: no deferred implementation placeholders or threshold choices remain.
- Type consistency: candidate, calibration fit, target path, attribution, stage, checkpoint, pipeline, and runner outputs are introduced before downstream consumption.
