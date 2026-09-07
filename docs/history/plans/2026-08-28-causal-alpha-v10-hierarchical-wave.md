# Causal Alpha V10 Hierarchical Wave Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate a restart-safe, symbol-free two-stage 72h/4h wave policy that maximizes after-cost wealth without opening holdout early.

**Architecture:** V10 adds one pooled nonlinear slow-regime fit beside the existing V9 fast fit. A pure state machine combines slow continuation, fast entry/reversal, and causal execution-regime eligibility, while the stage runner persists all 216 paired replay leaves and reuses unchanged numerical gates.

**Tech Stack:** Python 3.12, NumPy, pytest, Ruff, Mypy, Docker, PostgreSQL-backed frozen Binance artifacts.

**Spec:** `docs/implementation-plans/specs/2026-08-28-causal-alpha-v10-hierarchical-wave-design.md`

## Global Constraints

- Reward is exactly `100 * net_log_return`.
- No symbol ID, symbol exclusion, holdout inspection, gate relaxation, or BC/RL before Admission.
- Keep the 15-minute simulator and target magnitude `0.10`.
- Fast horizon is 4h/4 weeks; slow horizon is 72h/12 weeks.
- All runtime leaves and evidence are atomic, digest-bound, and restart-safe.

---

### Task 1: Immutable V10 contracts and causal dual-horizon fit

**Files:**
- Create: `trade_rl/learning/causal_alpha_v10.py`
- Create: `trade_rl/learning/causal_alpha_v10_fit.py`
- Test: `tests/learning/test_causal_alpha_v10_fit.py`

**Interfaces:**
- Consumes: pooled feature arrays and aligned 4h/72h labels with label-end indices.
- Produces: `CausalAlphaV10Config`, `CausalAlphaV10TrainingRows`, `CausalAlphaV10DualFit`, and `fit_causal_alpha_v10(...)`.

- [ ] Write failing tests proving deterministic fits, `max_label_end < cutoff`, non-overlapping horizon rows, unique fast/slow digests, and rejection of symbol identity features.
- [ ] Run `.venv\\Scripts\\python.exe -m pytest tests/learning/test_causal_alpha_v10_fit.py -q` and confirm failures are caused by missing V10 contracts.
- [ ] Implement frozen V10 config validation and a shared private ridge-head fitter that returns a raw-plus-128-hidden fast fit and a 32-hidden-only slow fit.
- [ ] Run the focused test and confirm all cases pass.
- [ ] Run Ruff and Mypy on both modules and the test, then commit `feat: add causal alpha v10 dual horizon fit`.

### Task 2: Hierarchical exposure compiler

**Files:**
- Create: `trade_rl/learning/causal_alpha_v10_hierarchy.py`
- Test: `tests/learning/test_causal_alpha_v10_hierarchy.py`

**Interfaces:**
- Consumes: fast/slow three-head predictions, causal volatility/liquidity arrays and frozen attribution boundaries, caps, actionability, and initial weight.
- Produces: `causal_alpha_v10_hierarchical_target_path(...) -> CausalAlphaV6TargetPath` plus immutable hierarchy diagnostics.

- [ ] Write failing tests for two coherent confirmations, execution-regime entry veto, slow-authorized neutral fast hold, fast opposite exit after two observations, slow opposite exit, 24h slow-neutral expiry, inherited-position validation, no direct flip, and runtime no-trade-band clearance.
- [ ] Run the focused hierarchy tests and verify the intended failures.
- [ ] Implement only the fixed state machine from the spec, with four-hour cadence and immediate cap deleveraging.
- [ ] Run hierarchy plus `tests/simulation/test_target_exposure_controller.py` and confirm all cases pass.
- [ ] Run Ruff and Mypy, then commit `feat: add v10 hierarchical wave compiler`.

### Task 3: V10 evidence mapping and restart-safe stages

**Files:**
- Create: `trade_rl/workflows/universal_causal_alpha_v10_gates.py`
- Create: `trade_rl/workflows/universal_causal_alpha_v10_stage_entry.py`
- Create: `scripts/run_universal_causal_alpha_v10_research.py`
- Test: `tests/workflows/test_universal_causal_alpha_v10_gates.py`
- Test: `tests/workflows/test_universal_causal_alpha_v10_stage_entry.py`

**Interfaces:**
- Consumes: V8 control, V9 control, V10 hierarchy targets, runtime/V4 manifests, frozen metadata, and the V7 attribution boundaries.
- Produces: V10 Signal/Selection evidence, 216 replay leaves, terminal result, and `run_causal_alpha_v10_selection(...)`.

- [ ] Write failing tests for candidate mapping, 72-scope dual-horizon liveness, unchanged gate results, leaf identity validation, resume reuse, and no Selection bypass.
- [ ] Run the focused workflow tests and verify failures.
- [ ] Implement V10-owned evidence wrappers and stage runner by composing immutable V8/V9 components; do not duplicate numerical gate constants.
- [ ] Add the CLI with exit codes 0 pass, 3 rejected, and 5 execution failure.
- [ ] Run focused V8-V10 tests, Ruff, and Mypy, then commit `feat: add restart safe v10 selection`.

### Task 4: Provenance build and real-data Selection

**Files:**
- Modify only if evidence requires correction: V10 files and their tests.
- Create at runtime: `/workspace/var/runs/causal-alpha-v10-prod-20260828-r1/**`

**Interfaces:**
- Consumes: clean committed source, runtime manifest digest `6726b3737df9fbacf6787f3d02894e846c512a840bec4dd037538a02af1480b0`, frozen V4 context, DB volume.
- Produces: immutable Docker image and complete Signal/Selection artifacts.

- [ ] Compute clean source/lock digests and build `docker/Dockerfile.training` with all provenance arguments.
- [ ] Run a dual-fit Docker smoke proving slow label ends precede cutoff and both head tensors have shape `(3, rows)`.
- [ ] Launch the DB-backed V10 runner in a new output root and verify the first hierarchical replay has meaningful execution.
- [ ] Monitor every 27 leaves; aggregate net/gross wealth, per-symbol wealth, positive scopes, turnover, costs, and execution counts.
- [ ] If a code defect appears, stop while preserving leaves, reproduce with a failing test, implement one fix, revalidate, rebuild, and resume in a new run root.
- [ ] Complete all 216 leaves and inspect canonical Signal, Selection, and terminal result digests.

### Task 5: Admission, BC/RL, and reporting gates

**Files:**
- Create only after Selection passes: `trade_rl/workflows/universal_causal_alpha_v10_admission.py`
- Create only after Selection passes: `scripts/run_universal_causal_alpha_v10_admission.py`
- Create: `report/causal-alpha-v10-final-report.md`

**Interfaces:**
- Consumes: passed Selection digest and untouched holdout.
- Produces: Admission evidence; only a passed Admission may produce BC/RL training artifacts and final policy evidence.

- [ ] If Selection rejects, do not read holdout; document exact candidate gates, attribution, and next design evidence.
- [ ] If Selection passes, write failing tests that bind Admission to the selected config and immutable holdout contracts, then implement the unchanged numerical Admission gate.
- [ ] Run Admission once. Start BC/RL only if its canonical evidence passes.
- [ ] If BC/RL opens, train to the configured terminal budget and verify learned-policy uplift rather than baseline fallback.
- [ ] Write the final report with branch, commit, image manifest digest, commands, artifact paths/digests, gate outcomes, and explicit distinction between aggregate wealth and universal admission.
- [ ] Run final tests, Ruff, Mypy, `git diff --check`, and artifact digest verification before any completion claim.
