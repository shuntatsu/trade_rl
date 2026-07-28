# Architecture Audit Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the architecture-audit findings around strict configuration parsing, typed encoder/runtime modes, structured export and serving wiring, CUDA reproducibility, and diagnostics cost control without weakening existing fail-closed identities.

**Architecture:** Preserve `TrainingRunConfig` JSON v2 as the public boundary, but parse it through field-closed section validators and typed `StrEnum` values. Produce structured TorchScript artifacts during hierarchical training, bind them into the immutable run and serving manifests, and select the serving loader through one canonical deployment contract. Keep SB3 and Torch dependencies isolated in integration/serving adapters.

**Tech Stack:** Python 3.12, dataclasses, StrEnum, Stable-Baselines3 2.3.2, Torch 2.3.1, Pytest, Ruff, MyPy, Import Linter.

## Global Constraints

- Do not accept `training_run_config_v1` or omit `schema_version`.
- Reject every unknown JSON field at the section where it appears.
- Keep `domain`, `telemetry`, `learning`, and workflow framework boundaries valid under `.importlinter`.
- Derive architecture identity from the assembled model, never only from configuration.
- Restore model device and training state after export on success and failure.
- Keep Production status `NO-GO`; this work does not claim profitability or enable exchange order submission.
- Use tests first and verify each new test fails for the intended reason before production changes.

---

### Task 1: Field-closed v2 configuration and typed modes

**Files:**
- Create: `trade_rl/rl/training_modes.py`
- Create: `trade_rl/workflows/config_fields.py`
- Modify: `trade_rl/rl/training.py`
- Modify: `trade_rl/workflows/training_run.py`
- Test: `tests/workflows/test_training_run_config.py`
- Test: `tests/rl/test_training_config_active_fields.py`

**Interfaces:**
- Produces: `ObservationEncoder(StrEnum)` with `FLAT_MLP`, `ASSET_SET`, `HIERARCHICAL_SEQUENCE_V2`.
- Produces: `CudaRuntimeMode(StrEnum)` with `DETERMINISTIC`, `PERFORMANCE`.
- Produces: `require_exact_fields(mapping, *, required, optional, field)`.
- `ResidualTrainingConfig.observation_encoder` becomes `ObservationEncoder` and adds `cuda_runtime_mode: CudaRuntimeMode`.

- [ ] Write failing tests proving omitted schema, unknown top-level fields, unknown `exports` fields, and misspelled training fields are rejected with the unknown names in the message.
- [ ] Write failing tests proving string JSON values resolve to the typed enums and invalid values fail closed.
- [ ] Run focused tests and confirm failures are caused by permissive parsing/string fields.
- [ ] Implement exact-field validation before dataclass construction and require explicit `training_run_config_v2`.
- [ ] Convert values to `ObservationEncoder` and `CudaRuntimeMode` in `ResidualTrainingConfig.__post_init__`; include both canonical string values in digests.
- [ ] Run focused tests, Ruff, and MyPy.
- [ ] Commit as `feat: close training config fields and type runtime modes`.

### Task 2: Explicit CUDA reproducibility contract

**Files:**
- Modify: `trade_rl/integrations/sb3_training.py`
- Modify: `docs/CONFIGURATION.md`
- Test: `tests/integrations/test_sb3_training_performance.py`
- Test: `tests/integrations/test_sb3_training.py`

**Interfaces:**
- `_configure_torch_cuda_runtime(torch, device, mode)` returns evidence containing `mode`, deterministic algorithms, cuDNN benchmark/deterministic, and TF32 state.

- [ ] Write failing tests for deterministic mode (`benchmark=False`, `deterministic=True`, TF32 disabled) and performance mode (current accelerated settings).
- [ ] Run focused tests and confirm deterministic mode is not implemented.
- [ ] Implement the two explicit modes and record the resolved mode in runtime evidence and training-config identity.
- [ ] Document the speed/reproducibility trade-off and default.
- [ ] Run focused tests, Ruff, and MyPy.
- [ ] Commit as `feat: make CUDA reproducibility mode explicit`.

### Task 3: Diagnostics activation and interval control

**Files:**
- Modify: `trade_rl/rl/checkpointing.py`
- Modify: `trade_rl/rl/sequence_diagnostics.py`
- Modify: `trade_rl/integrations/sb3_training.py`
- Test: `tests/rl/test_sequence_diagnostics.py`
- Test: `tests/rl/test_checkpointing.py`

**Interfaces:**
- `build_sequence_diagnostics_callback(*, enabled: bool, rollout_interval: int)` returns `None` when disabled and probes only every N completed rollouts.
- `build_checkpoint_callback(..., sequence_diagnostics_enabled, sequence_diagnostics_interval)` includes the callback only when enabled.

- [ ] Write failing tests proving disabled diagnostics do not construct/probe and interval 3 probes only on the third rollout.
- [ ] Run tests and verify current always-on behavior fails them.
- [ ] Implement gated callback creation and interval accounting.
- [ ] Wire `tensorboard_enabled` and `tensorboard_log_interval` through checkpoint callback assembly.
- [ ] Run focused tests, Ruff, and MyPy.
- [ ] Commit as `perf: gate sequence diagnostics by TensorBoard interval`.

### Task 4: Export state safety

**Files:**
- Modify: `trade_rl/rl/structured_export.py`
- Test: `tests/rl/test_structured_export.py`

**Interfaces:**
- `export_structured_policy_actor` preserves the original policy device and training mode in a `finally` block.

- [ ] Write failing tests for state restoration after successful export and after a parity/export exception.
- [ ] Run focused tests and confirm the policy remains on CPU/eval before the fix.
- [ ] Capture original device and training flag; always restore them after export cleanup.
- [ ] Run focused tests, Ruff, and MyPy.
- [ ] Commit as `fix: restore policy state after structured export`.

### Task 5: Structured export in the canonical training run

**Files:**
- Modify: `trade_rl/workflows/training_run.py`
- Modify: `trade_rl/integrations/sb3_training.py`
- Modify: `trade_rl/rl/training.py`
- Modify: `trade_rl/artifacts/run_manifest.py`
- Test: `tests/workflows/test_training_run_config.py`
- Test: `tests/workflows/test_training_run.py`
- Test: `tests/integrations/test_sb3_training_active_architecture.py`

**Interfaces:**
- Public v2 JSON adds `exports.structured_torchscript` boolean.
- `PolicyTrainingResult` carries optional structured export manifest/model paths and digests.
- Hierarchical runs with `structured_torchscript=true` export from a canonical environment observation after training.
- Run manifest file closure includes both structured files.

- [ ] Write failing config and workflow tests for valid hierarchical structured export, invalid flat-policy use, and manifest file closure.
- [ ] Run focused tests and confirm no canonical export path exists.
- [ ] Add the config flag and inactive-field validation.
- [ ] Export after final model save using the environment’s canonical structured observation; return artifact metadata.
- [ ] Copy/register export artifacts into each member directory and run manifest.
- [ ] Run focused tests, Ruff, MyPy, and an actual CPU hierarchical BC-to-PPO smoke with structured export.
- [ ] Commit as `feat: publish structured policy exports with training runs`.

### Task 6: Canonical serving loader and unified deployment identity

**Files:**
- Create: `trade_rl/serving/policy_loader.py`
- Modify: `trade_rl/serving/runtime.py`
- Modify: `trade_rl/serving/package.py`
- Modify: `trade_rl/serving/bundle.py`
- Modify: `trade_rl/cli/extended.py`
- Test: `tests/serving/test_package.py`
- Test: `tests/serving/test_runtime.py`
- Test: `tests/serving/test_structured_policy.py`
- Test: `tests/e2e/test_research_to_serving_v2.py`

**Interfaces:**
- `RuntimeIdentityContract` adds `architecture_digest: str | None`.
- `canonical_policy_loader(contract, manifest)` selects the only compatible loader from observation schema and bound artifacts.
- Structured bundle packaging requires exactly one structured manifest/model pair per selected member and binds their digests.
- Runtime may omit an injected loader only when the canonical factory can construct one.

- [ ] Write failing tests for automatic structured loader selection, architecture mismatch, missing structured files, and flat/structured ambiguity.
- [ ] Run tests and confirm manual loader injection is currently required.
- [ ] Implement the unified deployment identity and canonical loader factory without importing SB3 into serving.
- [ ] Bind structured artifact files and architecture digest into serving manifests during packaging.
- [ ] Make runtime construct the canonical loader when none is injected; preserve explicit injection only for tests/custom adapters with identity verification.
- [ ] Add a read-only CLI activation smoke command that loads and performs the loader smoke observation without placing orders.
- [ ] Run serving/e2e tests, Ruff, MyPy, Import Linter, and serving/recovery smoke.
- [ ] Commit as `feat: add canonical structured serving loader path`.

### Task 7: Early action-contract rejection and final verification

**Files:**
- Modify: `trade_rl/integrations/sb3_model_assembly.py`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/CONFIGURATION.md`
- Test: `tests/integrations/test_sb3_model_assembly.py`
- Test: `tests/integrations/test_sb3_policy_identity_v2.py`

**Interfaces:**
- Hierarchical sequence assembly rejects non-`target_weight:<symbol>` actions before model construction and verifies exact symbol order.

- [ ] Write failing tests for non-target actions and symbol-order mismatch at assembly time.
- [ ] Run tests and confirm failure currently occurs only during later identity binding.
- [ ] Add early validation in `_sequence_policy_assembly` and document the maintained action contract.
- [ ] Run all focused tests.
- [ ] Run `pytest` with branch coverage and critical coverage.
- [ ] Run Ruff check/format, MyPy, Import Linter, dead-code, CLI smoke, serving/recovery smoke, Windows/Linux compatibility, and production training image.
- [ ] Run CPU BC→PPO with checkpoint resume and structured export. Run Docker CUDA smoke when the required self-hosted runner is connected; otherwise report it as queued rather than passing.
- [ ] Remove this completed plan from the maintained docs tree while preserving it in Git history.
- [ ] Commit as `test: verify architecture audit remediation`.
