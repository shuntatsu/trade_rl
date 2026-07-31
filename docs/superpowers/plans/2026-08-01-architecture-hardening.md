# Architecture Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the audited architecture boundary leaks while preserving the existing package layout and active Studio work.

**Architecture:** Keep the current 17-layer dependency order. Centralize GPU smoke execution and validation in maintained operations code, remove the `rl` to `integrations` lazy edge, add explicit serving action semantics, move release publication orchestration to workflows, and consolidate GPU workflows around one reusable implementation.

**Tech Stack:** Python 3.12, pytest, Ruff, MyPy, Import Linter, GitHub Actions, Stable-Baselines3 optional integration, immutable JSON artifacts.

## Global Constraints

- Base all implementation changes on `0f8b1c9b0218d1b9b995f2934e34483308a282d9`.
- Do not modify files changed by active PR #318.
- Preserve the enforced 17-layer order in `.importlinter`.
- Do not add direct exchange execution or production-secret handling.
- Do not accept legacy serving schemas silently.
- Use exact immutable action SHAs and read-only workflow permissions.
- Keep each task independently reviewable and tested.

---

### Task 1: Authoritative GPU evidence contract

**Files:**
- Create: `trade_rl/operations/__init__.py`
- Create: `trade_rl/operations/gpu_training_smoke.py`
- Modify: `examples/binance-multitimeframe/run_gpu_training_smoke.py`
- Create: `tests/operations/test_gpu_training_smoke.py`
- Modify: `tests/examples/test_gpu_training_performance_assets.py`

**Interfaces:**
- Produces: `GPU_TRAINING_SMOKE_SCHEMA: str`.
- Produces: `validate_gpu_training_smoke_evidence(payload: object, *, expected_commit: str, expected_runtime_profile: str, minimum_timesteps: int) -> dict[str, object]`.
- Produces: `run_gpu_training_smoke(...) -> dict[str, object]` from the maintained operations module.

- [ ] Add failing schema, validator, and ownership tests.
- [ ] Verify RED through PR CI.
- [ ] Move the smoke implementation into `trade_rl.operations`.
- [ ] Replace the example with a thin wrapper.
- [ ] Run focused pytest, Ruff, and MyPy.

### Task 2: Layer-safe lazy exports

**Files:**
- Modify: `trade_rl/rl/__init__.py`
- Modify: `tests/architecture/test_public_lazy_exports.py`
- Create: `tests/architecture/test_lazy_export_layer_boundaries.py`

**Interfaces:**
- `trade_rl.rl.__all__` no longer contains SB3 backends.
- SB3 backends remain available from `trade_rl.integrations.sb3_training`.

- [ ] Add a failing layer-index test for lazy export targets.
- [ ] Verify RED through PR CI.
- [ ] Remove the reverse export without a compatibility shim.
- [ ] Run focused pytest and Import Linter.

### Task 3: Explicit serving action semantics

**Files:**
- Modify: `trade_rl/serving/bundle.py`
- Modify: the release packaging implementation before Task 4 moves it
- Modify: `trade_rl/serving/runtime.py`
- Modify: serving test helpers and focused bundle/runtime tests
- Modify: maintained schema documentation

**Interfaces:**
- `SERVING_BUNDLE_SCHEMA = "serving_bundle_v6"`.
- `ServingBundleManifest.action_mode: ActionMode` is required.
- `RuntimeSnapshot.action_mode: ActionMode` is required.
- Bundle JSON includes `action_mode` in the digest payload.

- [ ] Add failing residual and target-weight manifest tests.
- [ ] Verify RED through PR CI.
- [ ] Add schema v6 action mode construction, serialization, parsing, and runtime propagation.
- [ ] Replace residual-only runtime error wording.
- [ ] Update maintained schema documentation.
- [ ] Run focused serving, domain, and architecture tests.

### Task 4: Release publication workflow boundary

**Files:**
- Create: `trade_rl/workflows/release_packaging.py`
- Delete: `trade_rl/serving/package.py`
- Modify: all maintained imports of `trade_rl.serving.package`
- Create: `tests/architecture/test_release_packaging_boundary.py`

**Interfaces:**
- `trade_rl.workflows.release_packaging.package_selected_training_run(...) -> ServingBundleManifest`.
- No maintained module imports `trade_rl.serving.package`.

- [ ] Add failing ownership and import tests.
- [ ] Verify RED through PR CI.
- [ ] Move the module without semantic changes.
- [ ] Run packaging, end-to-end, and Import Linter checks.

### Task 5: Consolidated reusable GPU workflow

**Files:**
- Create: `.github/workflows/reusable-gpu-training-verification.yml`
- Modify: `.github/workflows/gpu-nightly.yml`
- Create: `.github/workflows/main-gpu-verification.yml`
- Delete: `.github/workflows/finalize-pr227-gpu-verification.yml`
- Create: `tests/architecture/test_gpu_workflow_consolidation.py`

**Interfaces:**
- Reusable workflow accepts `timesteps`, `runtime_profile`, `use_docker`, and `artifact_name`.
- Thin callers contain no evidence schema literal.
- Evidence validation invokes the maintained Python validator.

- [ ] Add failing workflow structure tests.
- [ ] Verify RED through PR CI.
- [ ] Implement the reusable workflow and callers.
- [ ] Run workflow security and focused tests.

### Task 6: Full verification and self-review

- [ ] Run Ruff, Ruff format, MyPy, Import Linter, and Vulture.
- [ ] Run full pytest with branch coverage and critical coverage.
- [ ] Run Studio tests, typecheck, build, and layout checks.
- [ ] Build the complete training image.
- [ ] Review the entire diff for scope, boundaries, compatibility, and dead code.
- [ ] Verify required CI checks on one unchanged final head.
- [ ] Keep the PR Draft until all required checks succeed.
