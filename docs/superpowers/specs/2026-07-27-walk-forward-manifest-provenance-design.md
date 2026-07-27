# Walk-Forward Manifest and Provenance Design

## Problem

`execute_market_walk_forward()` publishes walk-forward evidence through `TrainingRunManifest`, even though the repository already defines `WalkForwardRunManifest`. It also passes the workflow configuration digest as `provenance_digest`, so the manifest field does not identify the source tree, Git state, dependency lock, runtime, or hardware that produced the evidence.

This lets structurally valid walk-forward directories be classified as training runs and makes the provenance field semantically false.

## Decision

Walk-forward publication will use the dedicated `WalkForwardRunManifest` contract and real `RuntimeProvenance` evidence.

The workflow will:

1. capture runtime provenance before candidate training begins;
2. write `provenance.json` into the staged run directory;
3. construct `WalkForwardRunManifest` with distinct evaluation, workflow configuration, policy-set, environment, and provenance digests;
4. write and validate the dedicated walk-forward manifest before publication;
5. keep production status `NO-GO` and preserve existing fold, selection, and sealed-test behavior.

## Manifest mapping

- `evaluation_digest`: `WalkForwardExecutionResult.evaluation_digest`
- `workflow_config_digest`: digest of the complete normalized `MarketWalkForwardConfig`
- `policy_set_digest`: digest of the trained policy registry
- `environment_digest`: digest of all candidate environment contracts
- `provenance_digest`: digest from `capture_runtime_provenance()`
- `fold_count`: number of completed walk-forward folds

## Compatibility

No existing training-run schema changes. Existing legacy walk-forward artifacts that were written as `training_run_v3` remain readable as legacy training manifests. Newly produced walk-forward runs use `walk_forward_run_v1`, allowing Studio to identify them as `walk-forward` without heuristics.

## Failure handling

Provenance capture is fail-closed. A valid Git commit and determinable dirty state are required, matching full training. Manifest validation remains inside the staged directory; failures are isolated through the existing `ArtifactStore.mark_failed()` path.

## Testing

A focused integration test will run the existing minimal market walk-forward fixture and assert:

- `run.json` uses `walk_forward_run_v1`;
- `provenance.json` exists and its digest equals the manifest `provenance_digest`;
- `evaluation_digest`, `workflow_config_digest`, `policy_set_digest`, and `fold_count` are present and correctly bound;
- the directory passes `validate_walk_forward_run_directory()` and is rejected by `validate_training_run_directory()`.

An architecture regression test will prevent `execute_market_walk_forward()` from returning to `TrainingRunManifest` publication.