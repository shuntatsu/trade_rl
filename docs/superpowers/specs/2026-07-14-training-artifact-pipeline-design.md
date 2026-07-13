# Training Artifact Pipeline Design

## Goal

Provide one maintained path from a validated real-market dataset artifact to trained Stable-Baselines3 ensemble checkpoints, canonical manifests, nested walk-forward evaluation outputs, immutable publication, serving-time loading, and deterministic actor exports.

## Base and integration boundary

The implementation starts from PR #29 head `4a1a3ee28a8d45d16986b39e594abe62d112e446` on branch `agent/training-artifact-pipeline`. It uses the dynamic `ActionSpec`, observation schema v3, fold-fitted normalizer, multi-algorithm Stable-Baselines3 backend, and strict serving identity contracts from PR #29. PR #24 is not yet in this history, so the pipeline defines a maintained `manifest.json` plus `arrays.npz` market-dataset artifact loader whose interface can later be backed by PR #24 without changing training or evaluation orchestration.

## Commands

`trade-rl train run --config CONFIG --dataset DATASET --output STORE [--run-id ID]` loads and validates a dataset artifact and run configuration, trains one checkpoint per seed, writes manifests and optional exports into a staged run, validates every declared file digest, then atomically publishes the run.

`trade-rl walk-forward run --config CONFIG --dataset DATASET --output STORE [--run-id ID]` builds fold boundaries, fits preprocessing only on each train range, trains candidates, evaluates only through range-scoped dataset views, selects using validation ranges, evaluates the selected candidate once on sealed test ranges, writes fold and stitched-OOS results, and publishes atomically.

Both commands print a single JSON object and retain `production_status: "NO-GO"` until an approved release identity exists.

## Dataset artifact

A dataset directory contains canonical `manifest.json` and `arrays.npz`. The manifest records schema, dataset identity, symbol and feature ordering, period/calendar metadata, array names, shapes, dtypes, and SHA-256 of the NPZ payload. Loading rejects missing files, unsafe or unknown array names, digest mismatch, metadata mismatch, non-finite data, and invalid `MarketDataset` invariants.

A `MarketDatasetView` exposes a half-open bar range and rejects any attempt to construct a subview outside its parent range. Materialization preserves dataset identity plus range identity so training and evaluation cannot silently read bars outside the permitted fold.

## Run artifacts

Each published run contains `run.json`, `training-config.json`, `dataset-reference.json`, `ensemble.json`, member directories with `policy.zip`, `metrics.json`, and `export.json`, plus walk-forward outputs when applicable. Canonical JSON is written atomically. Every file declared in `run.json` records its relative path, byte size, and SHA-256. Validation fails closed if files are missing, extra declared files are inconsistent, member counts differ, digests mismatch, or manifest identities disagree.

`ArtifactStore` remains responsible for staging, failed-run isolation, immutable `runs/<run-id>` publication, and atomic `latest.json` update. A failed run never changes `latest.json`.

## Training orchestration

A `TrainingRunConfig` combines `ResidualTrainingConfig`, environment configuration, trend/risk/execution/reward/action contracts, initial capital, export settings, and optional Git commit identity. The environment factory is built from the loaded dataset and fold-fitted normalizer. `train_residual_ensemble` continues to produce checkpoint files and a `PolicyEnsembleManifest`; the pipeline serializes the manifest and validates checkpoint digests before publication.

## Walk-forward orchestration

Each fold uses four disjoint half-open views: train, checkpoint validation, configuration selection, and sealed test. The normalizer is fit from train observations only and then frozen. Candidate training receives only the train view. Evaluators receive only their assigned view. Candidate selection is deterministic and lexicographic: safety pass, positive absolute growth, baseline non-inferiority, then drawdown, stability, and cost tie-breakers. The sealed test result is computed once after selection and is never fed back into selection.

## Serving loader

`StableBaselines3PolicyLoader` reads all ensemble members from a validated serving bundle, verifies model files against manifest digests and the runtime identity contract, loads the declared SB3 algorithm, and returns a deterministic ensemble policy. Each member must produce a finite vector with the exact dynamic action dimension. The ensemble action is the arithmetic mean; any failed member rejects the full prediction.

The runtime validates action dimension and bounds from the active action contract rather than assuming two actions.

## Exports

`policy.zip` is the authoritative retraining and recovery format. ONNX is the required deterministic actor export when the optional `onnx` dependency is installed. TorchScript is best-effort. Export metadata records source checkpoint digest, algorithm, input/output shape and dtype, normalizer and action-spec identities, exporter version, status, and error reason.

Exports use a deterministic actor wrapper and are compared against SB3 deterministic predictions on a fixed finite observation corpus. An export is accepted only when shape and finite checks pass and maximum absolute error is within the configured tolerance. Unsupported TorchScript is recorded without failing the run; requested ONNX export failure fails the run.

## Failure semantics

Dataset, range, normalizer, checkpoint, manifest, export, ensemble, or identity mismatch failures are fail-closed. The staged directory is moved to `failed/<run-id>`, published and serving pointers remain unchanged, and the CLI exits nonzero with structured JSON on stderr.

## Verification

Tests cover dataset round-trip and tamper detection, canonical manifests, atomic publication and failure isolation, CLI tiny training, fold-only preprocessing, range escape rejection, sealed-test separation, SB3 save/load/predict, dynamic action dimensions, incomplete ensembles, ONNX parity when available, TorchScript parity or explicit unsupported status, and unreleased production activation rejection. Standard CI must pass Ruff, format, mypy, import-linter, dead-code reporting, pytest with branch coverage, and CLI smoke tests.
