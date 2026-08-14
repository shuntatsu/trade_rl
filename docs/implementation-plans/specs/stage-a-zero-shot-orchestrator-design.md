# Stage A Zero-Shot Evaluation Orchestrator Design

## Scope

The orchestrator is the deterministic control plane for Stage A validation and selected-only sealed-test evaluation. It consumes a Stage A evaluation plan, an immutable `StageAEvaluationDatasetManifest`, a cell evaluator, and an optional sealed-test ledger. Dataset selection and fold slicing are no longer caller-owned inputs.

Checkpoint loading, SB3 environment assembly, durable PostgreSQL sealed-test ledger persistence, and CLI wiring remain separate A6b-2 lanes.

## Identity model

`StageAZeroShotEvaluationPlan` v3 binds the symbol-disjoint manifests, the evaluation-dataset manifest digest, feature identity, execution identity, evaluation identity, candidates, folds, seeds, triplet IDs, and statistical thresholds. It does not carry a synthetic global dataset identity.

`StageAEvaluationDatasetManifest` binds:

- source closure and metadata evidence;
- one real PostgreSQL-backed `MarketDataset.dataset_id` per validation/test triplet;
- the exact three symbols and slot order for each triplet;
- a common full timeline used for causal feature and sequence warm-up;
- each fold's `configuration_selection` and `test` half-open `IndexRange`.

`StageAEvaluationCellRequest` v2 contains the manifest digest, real triplet dataset ID, exact scored range, and the remaining plan/candidate/checkpoint identities. `validate_manifest()` rejects any relabeling of split, triplet, fold, seed, dataset, range, feature, execution, evaluation, candidate, or checkpoint identity.

## Validation phase

`evaluate_validation()` iterates in deterministic triplet, fold, and seed order. For each cell it evaluates one shared baseline, then every candidate. Baseline and policy requests are created from the same manifest lookup and therefore share the same dataset and `configuration_selection` range.

The orchestrator constructs complete Stage A evidence v3 and passes it to `select_stage_a_validation_candidate`. No sealed-test ledger method is called during validation.

## Sealed-test phase

`evaluate_sealed_test(validation_run)` recomputes validation selection before any test authorization. A failed or forged validation result stops immediately.

Authorization is performed once for every declared test triplet × fold cell, not merely once per fold. `StageASealedTestAccessRecord` binds the plan, manifest, triplet, dataset, test range, selected candidate, and underlying ledger-record digest. Test requests are then created only for the selected candidate and shared baseline, using the manifest's `test` range.

## Schedule and access records

`StageATestSchedule` v2 is derived from the evaluation-dataset manifest. Its plan digest, manifest digest, evaluation identity, fold set, and test ranges must exactly match the plan and manifest. Caller-defined test ranges are rejected.

The current generic in-memory ledger is wrapped by Stage A-specific access records. Durable PostgreSQL one-shot persistence remains a later lane, but it must persist the same triplet/fold/dataset/range closure.

## Artifact publication

`StageAZeroShotArtifactPublisher` publishes independent immutable validation and sealed-test packages through sibling staging directories and atomic rename. Sealed-test access artifacts include manifest, triplet, dataset, range, and ledger identities. Existing destination directories and incomplete packages are rejected.

## Error handling

The system fails closed on schema downgrade, plan/manifest mismatch, undeclared triplet or fold, dataset or range drift, evaluator-result request substitution, incomplete evidence, forged validation output, premature test access, repeated authorization, and artifact rebinding. There is no compatibility adapter for pre-v3 Stage A contracts.

## Testing strategy

Tests cover exact Cartesian closure, shared baselines, manifest-derived requests and schedules, validation-before-test ordering, triplet × fold authorization, selected-only test execution, legacy-schema rejection, request/result substitution, immutable publication, and cleanup on failure.
