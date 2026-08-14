# Stage A Evaluation Dataset Manifest Design

## Status

Approved design for A6b-2 PostgreSQL fold slicing. This specification replaces the single-dataset assumption in the current Stage A zero-shot contracts with an immutable manifest that binds each declared evaluation triplet to its real PostgreSQL-backed dataset and each fold to its exact evaluation range.

## Problem

The current `StageAZeroShotEvaluationPlan` carries one `dataset_identity` for every validation and sealed-test cell. That is incompatible with the maintained PostgreSQL builder because `build_postgres_market_dataset()` produces content identity from the selected symbols, source tables, metadata evidence, indicator artifacts, requested time range, slot order, and triplet provenance. Different symbol triplets therefore produce different dataset IDs even when they share the same clock.

The current request, observation, replay, execution-evidence, and sealed-test ledger paths inherit this single identity. Adding fold slicing without changing that contract would force one of two invalid behaviors:

1. record a synthetic common dataset ID that is not the ID of the bytes executed; or
2. inject per-triplet dataset IDs into a plan and evidence model that still declares one global dataset.

Both approaches weaken evidence closure and can allow a valid execution artifact to be relabeled as another triplet or fold.

## Goals

The implementation must:

- bind one Stage A evaluation plan to one immutable evaluation-dataset manifest;
- bind every validation and test `triplet_id` to exactly three declared symbols and one real PostgreSQL-backed `MarketDataset.dataset_id`;
- bind every declared fold to exact `configuration_selection` and `test` `IndexRange` values;
- construct requests from the manifest rather than accepting caller-defined symbols, dataset IDs, or time ranges;
- preserve full dataset history for feature and sequence warm-up while separately binding the scored evaluation range;
- ensure policy and baseline executions for the same split, triplet, and fold use the same dataset and range;
- fail closed on source, manifest, triplet, fold, symbol order, dataset identity, feature identity, or range drift;
- make the sealed-test schedule and ledger derive from the same manifest;
- reject all pre-v3 Stage A plan, request, observation, replay, and evidence schemas rather than maintaining an ambiguous compatibility adapter.

## Non-goals

This lane does not implement:

- SB3 environment construction or policy action execution;
- one-shot PostgreSQL sealed-test ledger persistence;
- validation or sealed-test CLI commands;
- training-fold dataset construction;
- changes to symbol-disjoint partitioning or triplet enumeration;
- arbitrary caller-selected date ranges.

Those remain separate A6b-2 lanes and consume the contracts defined here.

## Architecture

### 1. Immutable evaluation-dataset manifest

Add `trade_rl/workflows/stage_a_evaluation_dataset_manifest.py` with the following public contracts.

```python
StageAEvaluationDatasetSplit = Literal["validation", "test"]

@dataclass(frozen=True, slots=True)
class StageAEvaluationDatasetTriplet:
    split: StageAEvaluationDatasetSplit
    triplet_id: str
    symbols: tuple[str, str, str]
    dataset_id: str

@dataclass(frozen=True, slots=True)
class StageAEvaluationDatasetFold:
    fold: int
    configuration_selection: IndexRange
    test: IndexRange

@dataclass(frozen=True, slots=True)
class StageAEvaluationDatasetManifest:
    symbol_disjoint_manifest_digest: str
    symbol_disjoint_triplet_manifest_digest: str
    source_closure_digest: str
    source_metadata_evidence_digest: str
    indicator_cache_id: str
    feature_identity: str
    timeline_start_time: datetime
    timeline_end_time: datetime
    triplets: tuple[StageAEvaluationDatasetTriplet, ...]
    folds: tuple[StageAEvaluationDatasetFold, ...]
    schema_version: str = "stage_a_evaluation_dataset_manifest_v1"
    digest: str = ""
```

`source_closure_digest` is the immutable identity of the maintained PostgreSQL source closure, not a `MarketDataset.dataset_id`. It binds the table contracts, source range, symbol vocabulary, metadata evidence, execution-rule evidence, and indicator-cache identity used to build all triplet datasets.

The manifest validates:

- canonical UTC, 15-minute-aligned, half-open timeline bounds;
- exactly one entry for every validation and test triplet declared by the symbol-disjoint triplet manifest;
- no train triplets;
- exact triplet-to-symbol closure and canonical member order from `SymbolDisjointTripletManifest`;
- exactly three unique symbols per triplet;
- unique `(split, triplet_id)` keys and unique dataset IDs;
- exact fold closure and order from the maintained `WalkForwardFold` sequence;
- `configuration_selection` and `test` ranges equal the corresponding maintained fold ranges;
- all ranges fall inside the common timeline and contain at least three scored bars;
- no overlap between configuration-selection and test ranges within a fold;
- no overlap between test ranges across folds;
- strict SHA-256 validation for all source, dataset, feature, and manifest identities;
- canonical digest recomputation and strict JSON field closure.

The manifest provides only deterministic lookup methods:

```python
def triplet_for(
    self, split: StageAEvaluationDatasetSplit, triplet_id: str
) -> StageAEvaluationDatasetTriplet: ...

def range_for(
    self, split: StageAEvaluationDatasetSplit, fold: int
) -> IndexRange: ...
```

No method accepts replacement symbols, dataset IDs, or ranges.

### 2. PostgreSQL manifest builder

Add `trade_rl/integrations/postgres_stage_a_evaluation_datasets.py` as the only PostgreSQL construction boundary for this lane.

The builder accepts:

- one validated `SymbolDisjointManifest`;
- one validated `SymbolDisjointTripletManifest` bound to that source;
- one maintained sequence of `WalkForwardFold` values;
- the exact common timeline start and end;
- symbol metadata and metadata evidence digest;
- execution-rule histories and their evidence digest;
- one preloaded native indicator bundle covering the complete symbol vocabulary and common timeline;
- an `IndicatorArtifactConnection`.

It performs the following deterministic process:

1. validate the symbol-disjoint and triplet manifests against each other;
2. validate that the common timeline covers the largest declared fold stop;
3. compute `source_closure_digest` from canonical source-table names, common timeline, ordered symbol vocabulary, metadata evidence, execution-rule evidence, indicator-cache ID, feature identity, and schema version;
4. iterate validation and test triplets in the canonical triplet-manifest order;
5. derive an ordered triplet subset from the preloaded indicator bundle without re-reading or mutating its artifacts;
6. call `build_postgres_market_dataset()` once per triplet for the complete common timeline, with `symbols` and `slot_symbols` equal to the canonical triplet member order and with immutable triplet provenance;
7. verify `dataset.feature_config_digest` equals the single declared manifest feature identity;
8. record the returned real `dataset_id` in the triplet entry;
9. bind every maintained fold's `configuration_selection` and `test` ranges;
10. return the manifest and an immutable mapping from `(split, triplet_id)` to the built `MarketDataset`.

The dataset mapping is an in-process construction result, not serialized inside the manifest. Consumers must verify `dataset.dataset_id == manifest.triplet_for(...).dataset_id` before use.

The builder must load the full common timeline for each triplet. It must not call `MarketDatasetView.materialize()` to make the execution dataset, because physically truncating the dataset at the scoring boundary would remove pre-range feature and recurrent-policy history. The exact scored range remains a separate request identity.

### 3. Stage A contract v3

Replace the current Stage A plan schema with `stage_a_zero_shot_evaluation_plan_v3`.

`StageAZeroShotEvaluationPlan` removes the global `dataset_identity` field and adds:

```python
evaluation_dataset_manifest_digest: str
```

The plan still owns candidate, seed, fold, triplet, feature, execution, evaluation, and gate identities. It validates that its validation/test triplet IDs and folds exactly match the separately loaded dataset manifest through an explicit `validate_dataset_manifest()` method.

Replace the cell-request schema with `stage_a_evaluation_cell_request_v3`. `StageAEvaluationCellRequest` adds:

```python
evaluation_dataset_manifest_digest: str
dataset_identity: str
evaluation_range: IndexRange
```

The orchestrator constructs those fields only through the manifest:

- `dataset_identity = manifest.triplet_for(split, triplet_id).dataset_id`;
- `evaluation_range = manifest.range_for(split, fold)`;
- `evaluation_dataset_manifest_digest = manifest.digest`.

The request digest therefore binds the exact real dataset and exact scored range. Seed and candidate changes do not change those two fields for the same split, triplet, and fold.

Replace the observation schema with `stage_a_zero_shot_observation_v3`. `StageAEvaluationObservation` includes the manifest digest and evaluation range in addition to the per-triplet dataset identity. Replace the evidence schema with `stage_a_zero_shot_evidence_v3`; evidence closure validates each observation against manifest lookup rather than against a global plan dataset field.

Replace `stage_a_execution_cell_identity_v1` and `stage_a_execution_replay_v1` with v2 equivalents because both structures gain manifest and range fields. `StageAExecutionCellIdentity`, `StageAExecutionReplayArtifact`, policy-source validation, execution producer, promotion store, and production evaluator must compare:

- manifest digest;
- real triplet dataset ID;
- exact evaluation range;
- split, triplet, fold, seed, candidate, checkpoint, feature, execution, and evaluation identities.

Execution event artifacts and promotion evidence continue to carry the real `dataset_id`. The episode executor receives the full triplet dataset plus the exact evaluation range and must produce actions, observations, equity, and order events only for the authorized scored interval while allowing read-only historical context before `evaluation_range.start`.

### 4. Manifest-backed test schedule

Replace the standalone schedule schema with `stage_a_test_schedule_v2`.

`StageATestSchedule` contains:

```python
plan_digest: str
evaluation_dataset_manifest_digest: str
fold_ranges: tuple[StageATestFoldRange, ...]
```

The only production constructor is:

```python
@classmethod
def from_manifest(
    cls,
    *,
    plan: StageAZeroShotEvaluationPlan,
    manifest: StageAEvaluationDatasetManifest,
) -> StageATestSchedule: ...
```

It copies each manifest fold's `test` range. Direct construction remains validated but cannot create a schedule whose fold ranges differ from the manifest.

The orchestrator receives both the plan and manifest, validates their closure once in `__init__`, and builds every request from manifest lookups. Add `StageASealedTestAccessRecord` and `StageASealedTestLedgerProtocol` in the Stage A runner contracts. They bind `plan_digest`, `evaluation_dataset_manifest_digest`, `triplet_id`, `dataset_id`, `fold`, `test_range`, selected candidate, and selected policy digest. The generic walk-forward `SealedTestLedger` remains unchanged.

Sealed-test authorization creates one Stage A access record per `(plan, manifest, triplet dataset, fold)` rather than one global dataset record per fold. The ledger therefore records every test triplet actually opened. A fold-level global record is insufficient because each triplet has different dataset bytes.

### 5. Identity and provenance model

The identity hierarchy is:

```text
SymbolDisjointManifest
        │
        ▼
SymbolDisjointTripletManifest
        │
        ├───────────────┐
        ▼               ▼
PostgreSQL source   WalkForwardFold sequence
closure identity        │
        └───────┬───────┘
                ▼
StageAEvaluationDatasetManifest
        │
        ├─ validation triplet → symbols → real dataset_id
        ├─ test triplet       → symbols → real dataset_id
        └─ fold → configuration_selection / test range
                │
                ▼
StageAZeroShotEvaluationPlan v3
                │
                ▼
StageAEvaluationCellRequest v3
                │
                ▼
execution replay / evidence / observation
```

A downstream artifact is valid only when every ancestor digest and lookup value matches exactly.

## Data flow

### Validation

1. Load and validate the symbol-disjoint manifests.
2. Build or load the canonical evaluation-dataset manifest and triplet datasets.
3. Load the Stage A plan and require `plan.validate_dataset_manifest(manifest)`.
4. For each validation triplet, fold, and seed, resolve the real dataset and `configuration_selection` range from the manifest.
5. Produce one shared baseline request and one policy request per candidate.
6. Execute only the authorized scored range with read-only pre-range context.
7. Publish replay and evidence bound to the request digest.
8. Build observations that retain manifest, dataset, and range identity.
9. Select the validation candidate from complete v3 evidence.

### Sealed test

1. Recompute validation selection from v3 evidence.
2. Build `StageATestSchedule.from_manifest()` and verify exact closure.
3. Authorize each selected test triplet dataset and fold range once.
4. Evaluate only the selected candidate and shared baseline.
5. Publish sealed-test evidence and access records containing manifest digest, dataset ID, triplet ID, fold, and exact test range.
6. Reject any second access to the same `(plan_digest, manifest_digest, triplet_id, dataset_id, fold)` key.

## Warm-up and range semantics

`evaluation_range` is an absolute, half-open index range over the full triplet dataset.

The environment may read bars before `evaluation_range.start` only to construct causal observations and recurrent state. It must:

- reset portfolio, order, reward, and execution accounting at `evaluation_range.start`;
- emit no scored action, order, reward, or equity before that index;
- stop after processing the final authorized bar before `evaluation_range.stop`;
- produce an equity curve covering only the scored interval, including the initial scored portfolio value;
- reject any order-event timestamp outside the authorized range;
- reject observation digests or action counts inconsistent with the authorized range.

A later SB3 environment lane will implement these mechanics. This lane establishes the immutable range contract consumed by that executor.

## Serialization and migration

Add strict canonical JSON readers and writers for the dataset manifest. Unknown, missing, duplicated, unsorted, or non-canonical fields fail closed.

The following old schemas are explicitly unsupported after this migration:

- `stage_a_zero_shot_evaluation_plan_v2`;
- `stage_a_evaluation_cell_request_v1`;
- `stage_a_zero_shot_observation_v2`;
- `stage_a_zero_shot_evidence_v2`;
- `stage_a_execution_cell_identity_v1` and `stage_a_execution_replay_v1`;
- `stage_a_test_schedule_v1`;
- `stage_a_sealed_test_access_records_v1`.

No automatic conversion is provided because a v2 plan does not contain enough information to prove the real per-triplet dataset ID or fold range. Tests must verify that loading old schema payloads raises a clear unsupported-schema error.

Existing generic `MarketDataset`, `MarketDatasetView`, `WalkForwardFold`, execution-event, and promotion-evidence schemas remain unchanged. `StageAPolicySourceBinding` and the request-index schemas also remain structurally unchanged because their existing `request_digest` field transitively binds the new manifest, dataset, and range identity; their validation logic is updated to resolve the request through the manifest.

## Error handling

The implementation fails closed on:

- a triplet manifest not bound to the supplied symbol-disjoint manifest;
- validation or test triplet closure mismatch;
- triplet symbol substitution or member-order drift;
- duplicate or missing triplet entries;
- a dataset built from symbols or slot order different from the triplet entry;
- dataset ID substitution;
- feature identity drift across triplets;
- source table, metadata evidence, execution-rule evidence, or indicator-cache drift;
- non-UTC or non-15-minute-aligned timeline bounds;
- fold closure, order, purge, selection-range, or test-range drift;
- a range outside the common timeline or a range with fewer than three bars;
- overlapping sealed-test ranges;
- plan-to-manifest digest mismatch;
- request-to-manifest dataset or range mismatch;
- policy and baseline requests using different dataset or range identities for one cell;
- event timestamps outside the authorized evaluation range;
- loading any rejected legacy schema.

There is no fallback to caller-supplied symbols, dates, ranges, dataset identities, or compatibility defaults.

## Testing strategy

### Contract tests

Add tests proving:

- a valid manifest covers every declared validation and test triplet and fold;
- canonical lookup returns the expected symbols, dataset ID, selection range, and test range;
- serialization round-trips exactly;
- byte, field, digest, order, duplicate, and schema tampering are rejected;
- train triplets, missing triplets, extra triplets, symbol substitutions, and reordered members are rejected;
- fold substitutions, truncated ranges, overlaps, and timeline escapes are rejected;
- v2 Stage A plan and legacy request/observation/schedule payloads are rejected.

### PostgreSQL builder tests

Use the maintained fake connection and indicator-artifact fixtures to prove:

- each validation and test triplet is built exactly once over the common timeline;
- the canonical triplet symbol and slot order reaches `build_postgres_market_dataset()`;
- the manifest records the returned real dataset IDs;
- all triplets share one feature identity and source closure;
- metadata, execution-rule, indicator-cache, source-range, or dataset-ID drift is rejected;
- no caller-provided per-cell time range is accepted.

Extend the PostgreSQL Catalog workflow with an integration case that builds at least one validation and one test triplet from the real test database and verifies manifest reload and range closure.

### Orchestrator and execution binding tests

Update existing Stage A tests to prove:

- requests are derived from manifest lookup;
- baseline and policy requests share the same dataset and range for a cell;
- different triplets carry different real dataset IDs;
- different folds carry different ranges while sharing the same triplet dataset ID;
- request, policy-source, replay, producer, store, evaluator, observation, evidence, and sealed-test schedule reject manifest, dataset, and range substitution;
- sealed-test authorization produces one record for every selected test triplet and fold and rejects reopening;
- full validation and selected-only sealed-test orchestration preserve Cartesian closure under v3.

### Repository verification

The merge head must pass, unchanged:

- focused manifest and PostgreSQL tests;
- full pytest and coverage threshold;
- Ruff and Ruff format;
- MyPy;
- Import Linter;
- critical branch-coverage ratchet;
- CLI smoke;
- Windows and Ubuntu compatibility;
- training image and non-root runtime probe;
- PostgreSQL Catalog workflow.

## File boundaries

Create:

- `trade_rl/workflows/stage_a_evaluation_dataset_manifest.py` — typed manifest, lookup, and canonical I/O;
- `trade_rl/integrations/postgres_stage_a_evaluation_datasets.py` — PostgreSQL-backed manifest and dataset construction;
- `tests/workflows/test_stage_a_evaluation_dataset_manifest.py` — manifest contract and tamper tests;
- `tests/integrations/test_postgres_stage_a_evaluation_datasets.py` — builder tests;
- `docs/superpowers/plans/2026-08-01-stage-a-evaluation-dataset-manifest.md` — implementation plan created after this design is accepted.

Modify:

- Stage A plan, contract I/O, observation, evidence, request, schedule, orchestrator, policy-source validation, replay, producer, store, evaluator, and related tests to migrate to v3;
- Stage A runner contracts and artifacts to add Stage A-specific access records while leaving the generic walk-forward ledger unchanged;
- PostgreSQL Catalog tests and workflow fixtures for the real integration case;
- Stage A operations documentation to describe the manifest-backed A6b-2 boundary.

The generic PostgreSQL market dataset builder remains responsible for one exact triplet dataset. The new Stage A integration module owns cross-triplet closure and fold-range binding; those responsibilities must not be merged into `postgres_market_dataset.py`.

## Acceptance criteria

The lane is complete when:

1. no production Stage A contract contains a global dataset identity that claims to identify every triplet dataset;
2. every evaluation cell request contains the exact manifest digest, real triplet dataset ID, and exact evaluation range;
3. every downstream Stage A artifact validates those values transitively;
4. PostgreSQL construction proves exact symbol, source, feature, dataset, and range closure;
5. validation and sealed-test scheduling derive from one immutable manifest;
6. legacy Stage A schemas fail closed;
7. the complete repository verification matrix is green on one unchanged PR head.
