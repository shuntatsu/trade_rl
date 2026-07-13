# Training Artifact Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a maintained CLI-driven path from validated market dataset artifacts to trained ensembles, canonical manifests, walk-forward outputs, immutable publication, SB3 serving, and deterministic actor exports.

**Architecture:** Add small focused modules for dataset artifacts, run manifests, pipeline orchestration, walk-forward adapters, policy export, and SB3 serving. Keep domain manifests immutable, keep filesystem writes atomic, and pass range-scoped dataset views through training and evaluation boundaries so leakage is structurally impossible.

**Tech Stack:** Python 3.12, NumPy, Gymnasium, Stable-Baselines3 2.3.2, PyTorch 2.3.1, optional ONNX, argparse, pytest.

## Global Constraints

- Base commit is `4a1a3ee28a8d45d16986b39e594abe62d112e446` from PR #29.
- `policy.zip` remains the authoritative model format.
- ONNX export is required only when requested and dependency support exists; TorchScript is best-effort.
- Every published file is content-bound by SHA-256 and byte size.
- Fold preprocessing fits on train bars only.
- Sealed test bars are evaluated once and never influence selection.
- Production status remains `NO-GO` without an approved release identity.
- Failures do not change `latest.json` or the active serving pointer.

---

### Task 1: Dataset artifact and range-scoped views

**Files:**
- Create: `trade_rl/data/artifacts.py`
- Modify: `trade_rl/data/__init__.py`
- Test: `tests/data/test_market_dataset_artifact.py`

**Interfaces:**
- Produces: `write_market_dataset_artifact(root: Path, dataset: MarketDataset) -> Path`
- Produces: `load_market_dataset_artifact(root: Path) -> MarketDataset`
- Produces: `MarketDatasetView(dataset: MarketDataset, start: int, stop: int)` with `subview` and `materialize`.

- [ ] Write tests that round-trip every required array, reject NPZ digest tampering, reject manifest metadata mismatch, and reject subviews escaping the parent range.
- [ ] Run `pytest tests/data/test_market_dataset_artifact.py -v` and verify failures are caused by missing interfaces.
- [ ] Implement canonical manifest writing, digest verification, strict array allow-listing, `MarketDataset` reconstruction, and range identity.
- [ ] Re-run the focused tests and commit.

### Task 2: Canonical run manifests and publication validation

**Files:**
- Create: `trade_rl/artifacts/run_manifest.py`
- Modify: `trade_rl/artifacts/__init__.py`
- Test: `tests/artifacts/test_run_manifest.py`

**Interfaces:**
- Produces: `RunFile`, `TrainingRunManifest`, `write_training_run_manifest`, `load_training_run_manifest`, `validate_training_run_directory`.
- Consumes: `PolicyEnsembleManifest`, dataset/environment/action/normalizer identities.

- [ ] Write tests for canonical bytes, file size/digest validation, member count mismatch, path traversal rejection, and missing file rejection.
- [ ] Run focused tests and verify RED.
- [ ] Implement immutable manifest records and complete directory validation.
- [ ] Run focused tests and commit.

### Task 3: Training run orchestration and CLI

**Files:**
- Create: `trade_rl/workflows/training_run.py`
- Modify: `trade_rl/cli/app.py`
- Modify: `trade_rl/rl/training.py`
- Test: `tests/workflows/test_training_run.py`
- Test: `tests/cli/test_train_run.py`

**Interfaces:**
- Produces: `TrainingRunConfig.from_json`, `execute_training_run(config_path, dataset_path, store_root, run_id) -> TrainingRunResult`.
- Consumes: dataset artifact loader, `ArtifactStore`, `StableBaselines3Backend`, environment factory builder, run manifest validation.

- [ ] Write a tiny deterministic training integration test that produces one or more `policy.zip` files, `ensemble.json`, `run.json`, and atomic `latest.json`.
- [ ] Write failure-isolation test proving a backend failure moves staging to `failed` and preserves the prior latest pointer.
- [ ] Write CLI test for `trade-rl train run` structured JSON output.
- [ ] Run focused tests and verify RED.
- [ ] Implement JSON configuration parsing, environment construction, ensemble training, manifest serialization, optional export invocation, staged validation, publication, and structured error output.
- [ ] Run focused tests and commit.

### Task 4: Project walk-forward adapters

**Files:**
- Create: `trade_rl/workflows/market_walk_forward.py`
- Modify: `trade_rl/workflows/__init__.py`
- Modify: `trade_rl/cli/app.py`
- Test: `tests/workflows/test_market_walk_forward.py`
- Test: `tests/cli/test_walk_forward_run.py`

**Interfaces:**
- Produces: fold-local normalizer fitting, range-bound candidate trainer/evaluator, deterministic selector, stitched sealed-OOS result, and `execute_market_walk_forward`.
- Consumes: `WalkForwardWorkflowConfig`, `MarketDatasetView`, training-run primitives.

- [ ] Write tests proving normalizer fit sees only train rows, evaluators cannot access outside assigned ranges, selection does not read sealed test results, and each sealed test range is evaluated once.
- [ ] Run focused tests and verify RED.
- [ ] Implement fold adapters and canonical fold/selection/OOS outputs.
- [ ] Add `trade-rl walk-forward run` CLI handler and structured output.
- [ ] Run focused tests and commit.

### Task 5: Stable-Baselines3 serving loader

**Files:**
- Create: `trade_rl/serving/sb3_loader.py`
- Modify: `trade_rl/serving/runtime.py`
- Modify: `trade_rl/serving/__init__.py`
- Modify: `trade_rl/serving/bundle.py`
- Test: `tests/serving/test_sb3_loader.py`
- Test: `tests/serving/test_runtime.py`

**Interfaces:**
- Produces: `StableBaselines3PolicyLoader` and deterministic ensemble `LoadedPolicy`.
- Runtime consumes dynamic action dimension and bounds from bundle identity.

- [ ] Write tests for loading PPO/SAC-family checkpoint declarations, exact dynamic action size, arithmetic ensemble mean, rejection of incomplete or corrupted members, and fail-closed member prediction errors.
- [ ] Run focused tests and verify RED.
- [ ] Implement algorithm-aware model loading and dynamic runtime validation.
- [ ] Run focused tests and commit.

### Task 6: ONNX and TorchScript actor exports

**Files:**
- Create: `trade_rl/rl/export.py`
- Modify: `pyproject.toml`
- Modify: `trade_rl/workflows/training_run.py`
- Test: `tests/rl/test_policy_export.py`

**Interfaces:**
- Produces: `PolicyExportConfig`, `export_policy_actor`, `verify_export_parity`, `ExportManifest`.
- Consumes: checkpoint path, algorithm, observation corpus, action and normalizer identities.

- [ ] Write tests for export metadata, TorchScript parity for a tiny PPO model, explicit unsupported status, and ONNX parity guarded by `pytest.importorskip("onnx")` and `pytest.importorskip("onnxruntime")`.
- [ ] Run focused tests and verify RED.
- [ ] Implement deterministic actor wrapper, TorchScript export, optional ONNX export, parity verification, and canonical `export.json`.
- [ ] Run focused tests and commit.

### Task 7: Documentation and complete verification

**Files:**
- Modify: `README.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `.github/workflows/ci.yml` only if optional export extras need a dedicated test job.

- [ ] Document artifact layouts, CLI examples, authoritative model format, export semantics, and production `NO-GO` boundary.
- [ ] Run Ruff lint and format checks.
- [ ] Run mypy and import-linter.
- [ ] Run full pytest with branch coverage and CLI smoke tests.
- [ ] Inspect the final diff for unrelated changes and temporary workflows.
- [ ] Open a draft PR targeting `agent/action-environment-comprehensive-hardening` with exact verification evidence.
