# Full Training Functional Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove whether every maintained learning capability can be enabled without being ignored, can complete an end-to-end training path, and cannot bypass causality, fold isolation, evaluation, artifact, or serving contracts.

**Architecture:** Build a temporary audit harness on an isolated branch. The harness generates deterministic causal market and signal artifacts, executes a feature matrix of short real training runs, then executes one maximal end-to-end run from dataset build through walk-forward selection, final fixed-seed ensemble training, export/reload, and structured serving. Static configuration-consumption checks and negative/fault-injection cases complement runtime execution so a feature cannot pass merely because a branch was never exercised.

**Tech Stack:** Python 3.12, NumPy, Gymnasium, Stable-Baselines3 2.3.2, sb3-contrib 2.3.0, PyTorch 2.3.1, pytest, Ruff, MyPy, GitHub Actions, Docker training image.

## Global Constraints

- Profitability and direct exchange order routing are outside this audit.
- The audited source is the current `main` tree.
- Product code is not modified unless a failing audit proves a defect.
- Every runtime case uses small deterministic data and timesteps; completion, identity, causality, and feature consumption are the assertions.
- A feature is "usable" only when enabling it changes a bound configuration/observation/action/model/artifact path and the run completes.
- The maximal run must include three symbols, native 15m/1h/4h/1d sequences, PPO, three fixed seeds, behavior cloning, checkpointing, fold-local normalization, walk-forward selection, final ensemble publication, reload, and structured inference.
- Algorithm-specific smoke cases cover PPO, SAC, TD3, and TQC separately because they cannot share one optimizer configuration.
- Alpha, factor-basis, risk-tilt, target-weight, residual-tilt, asset-set encoder, sequence encoder, resume, ONNX, and TorchScript are audited as separate capability cases when mutually exclusive.
- Any unsupported combination must fail during configuration validation with an explicit reason, never silently downgrade.

---

### Task 1: Inventory configuration and runtime consumption

**Files:**
- Create: `tools/audit_training_capabilities.py`
- Create: `artifacts/audit/training-capability-matrix.json`
- Inspect: `trade_rl/rl/training.py`
- Inspect: `trade_rl/rl/algorithm_configs.py`
- Inspect: `trade_rl/rl/environment_config.py`
- Inspect: `trade_rl/rl/actions.py`
- Inspect: `trade_rl/workflows/training_run.py`
- Inspect: `trade_rl/workflows/market_walk_forward.py`

**Interfaces:**
- Produces: a machine-readable matrix mapping each public configuration field to parser, validation, runtime consumer, identity payload, and test/runtime case.

- [ ] Enumerate dataclass/configuration fields with Python reflection and JSON schema loaders.
- [ ] Search the AST for each field's reads outside constructors and digest-only code.
- [ ] Mark fields with no runtime consumer, fields used only in identity, and mutually exclusive combinations.
- [ ] Verify the maintained full configuration does not contain unknown or ignored keys.
- [ ] Emit a failing exit code for any unconsumed maintained field.

### Task 2: Build deterministic causal audit fixtures

**Files:**
- Create: `tests/audit/test_full_training_capabilities.py`
- Create: `tools/run_training_capability_audit.py`

**Interfaces:**
- Produces: deterministic three-symbol market dataset, native timeframe sequences, effective-dated execution metadata, alpha artifact, factor artifact, and oracle-teacher artifact.

- [ ] Generate strictly causal OHLCV/funding/availability arrays with known timestamp spacing.
- [ ] Build 15m/1h/4h/1d feature windows through the maintained dataset builder rather than handcrafted observation tensors.
- [ ] Create content-addressed alpha and factor artifacts bound to the dataset and symbol order.
- [ ] Create a teacher artifact using only the training capability.
- [ ] Mutate one future bar and assert all earlier observations, normalizers, signals, and teacher labels remain byte-identical.

### Task 3: Execute the learning capability matrix

**Files:**
- Modify: `tests/audit/test_full_training_capabilities.py`
- Modify: `tools/run_training_capability_audit.py`

**Interfaces:**
- Produces: one isolated artifact directory and completion record per capability case.

- [ ] Run PPO with flat observations and the asset-set encoder.
- [ ] Run PPO with structured multi-timeframe sequences and the sequence encoder.
- [ ] Run PPO with behavior-cloning initialization and prove actor parameters change before PPO rollout.
- [ ] Run target-weight, residual-tilt, alpha-enabled, factor-basis, and risk-tilt action contracts in valid configurations.
- [ ] Run SAC, TD3, and TQC with their off-policy buffer settings.
- [ ] Run checkpoint save, interrupted-run resume, and final immutable publication.
- [ ] Run requested ONNX and TorchScript export behavior and reload/probe supported outputs.
- [ ] Assert every case records the enabled capability in environment, policy-loader, ensemble, and run identity artifacts.

### Task 4: Execute maximal end-to-end research flow

**Files:**
- Modify: `tools/run_training_capability_audit.py`
- Modify: `tests/audit/test_full_training_capabilities.py`

**Interfaces:**
- Produces: dataset → nested walk-forward → selected deployable ensemble recipe → final training → serving bundle → deterministic structured prediction evidence.

- [ ] Run two compact folds with purge, checkpoint-validation, configuration-selection, and sealed outer-test ranges.
- [ ] Include flat asset-set PPO, structured sequence PPO, and structured BC+PPO candidates.
- [ ] Require identical fixed seed sets between fold evaluation and final ensemble training.
- [ ] Reload every selected ensemble member and reproduce deterministic mean action.
- [ ] Build a serving bundle, verify normalizers/schema/state snapshot, and execute prediction from the dataset.
- [ ] Validate that all staged runs either publish completely or move to `failed/` without changing `latest.json`.

### Task 5: Fault injection and loophole checks

**Files:**
- Modify: `tests/audit/test_full_training_capabilities.py`

**Interfaces:**
- Produces: explicit rejection evidence for known bypass classes.

- [ ] Reject feature availability that becomes true before source data exists.
- [ ] Reject normalization fit ranges that cross train boundaries.
- [ ] Reject signal artifacts with future timestamps, wrong dataset IDs, wrong symbol order, or changed digests.
- [ ] Reject selection or final training with a changed seed ensemble recipe.
- [ ] Reject outer-test access before configuration selection and any second sealed access.
- [ ] Reject action dimensions that are configured but not backed by alpha/factor providers.
- [ ] Reject unsupported algorithm/policy/sequence combinations rather than silently selecting another policy.
- [ ] Reject resume checkpoints whose dataset, environment, action, or training identity differs.
- [ ] Reject final artifacts missing a policy, normalizer, checkpoint evidence, or declared export.

### Task 6: Run repository-wide verification and publish audit report

**Files:**
- Create: `docs/audits/2026-07-16-full-training-functional-audit.md`
- Remove before merge: temporary workflow files and large audit artifacts

**Interfaces:**
- Produces: a pass/partial/fail result for every capability, exact commands, runtimes, artifact identities, and evidence-backed defects.

- [ ] Run Ruff, format, MyPy, import-lint, existing full pytest, critical coverage, CLI smoke, Windows/Ubuntu compatibility, and Docker training-image probe.
- [ ] Run the capability matrix and maximal end-to-end audit under Python 3.12.
- [ ] Review logs for warnings, NaN/Inf, zero gradients, unchanged BC parameters, unused action dimensions, empty replay/rollout buffers, and unexpected CPU fallback.
- [ ] For every failure, reproduce with a focused RED test before changing product code.
- [ ] Write the audit report with feature-by-feature status and remaining limitations.
- [ ] Merge only evidence-backed fixes and retained regression tests; discard temporary audit orchestration.
