# Stage A Zero-Shot Evaluation Orchestrator Design

## Scope

This specification covers A6a only: a deterministic orchestration layer over the existing Stage A v2 plan, evidence, selection, and sealed-test gate contracts. It does not load real retained checkpoints, construct market datasets, or run the production execution model. Those responsibilities remain in A6b behind a typed evaluator protocol.

## Goals

The orchestrator must:

- evaluate every validation candidate over the exact candidate × triplet × fold × seed Cartesian product;
- evaluate one shared baseline per triplet × fold × seed cell;
- verify every evaluator result against the exact request before constructing a v2 observation;
- select a validation candidate through `select_stage_a_validation_candidate`;
- recompute validation selection before any sealed-test access;
- reserve sealed-test access once per declared fold before evaluating that fold;
- evaluate only the selected candidate and the shared baseline on the test split;
- construct test evidence through the existing v2 contract and decide through `evaluate_stage_a_sealed_test`;
- publish validation and sealed-test packages atomically per phase;
- leave no visible incomplete package when evaluation or publication fails.

## Non-goals

A6a does not resolve checkpoint paths or serving bundles, invoke `canonical_policy_loader`, materialize market or feature datasets, validate a real source execution artifact beyond typed request/result identity closure, or add a CLI or PostgreSQL schema.

## Architecture

### Evaluation cell contracts

`StageAEvaluationCellRequest` is the complete immutable instruction for one policy or baseline evaluation. It contains the plan digest, split, triplet, fold, seed, candidate identity, checkpoint identity, and all dataset/feature/execution/evaluation identities. Baseline requests use `candidate_id=None` and `checkpoint_digest=None`; policy requests require both.

`StageAEvaluationCellResult` contains the originating request digest, the execution-evidence digest, and finite log growth. The orchestrator rejects any result whose request digest differs from the request. A6b must create results only after validating the real source artifact.

`StageAEvaluationCellEvaluator` is the only execution dependency:

```python
class StageAEvaluationCellEvaluator(Protocol):
    def evaluate(
        self, request: StageAEvaluationCellRequest
    ) -> StageAEvaluationCellResult: ...
```

### Test schedule

`StageATestSchedule` binds the plan digest and evaluation identity to one `IndexRange` per declared fold. Its fold set must equal `plan.folds`. The range is used only for the existing `SealedTestLedgerProtocol`; A6b is responsible for proving that the schedule came from the maintained evaluation source represented by `plan.evaluation_identity`.

### Validation phase

`StageAZeroShotEvaluationOrchestrator.evaluate_validation()` iterates in deterministic triplet, fold, seed order. It evaluates the baseline once, then candidates in `plan.candidate_ids` order. The baseline result is reused for every candidate observation in the cell. The method constructs complete v2 validation evidence, then calls `select_stage_a_validation_candidate`. No sealed-test ledger method is called.

### Sealed-test phase

`evaluate_sealed_test(validation_run)` first recomputes the expected validation selection and rejects any supplied mismatch. A failed validation selection raises before ledger or evaluator access.

Before any test evaluation, the orchestrator authorizes every declared fold through `SealedTestLedgerProtocol.authorize_once`, using the exact scheduled range, selected candidate ID, and selected candidate digest. It then evaluates every test triplet × fold × seed cell with one baseline and one selected-candidate result, builds selected-only test evidence, and calls `evaluate_stage_a_sealed_test`.

### Phase outputs

`StageAValidationRun` contains validation evidence and selection. `StageASealedTestRun` contains the validation run, sealed-test access records, test evidence, and final decision. Both runs have content digests and validate their internal identity closure.

### Atomic publication

`StageAZeroShotArtifactPublisher` publishes two independent immutable directories:

- `validation/` containing `evidence.json` and `selection.json`;
- `sealed-test/` containing `evidence.json`, `decision.json`, and `access-records.json`.

Each package is written to a unique sibling staging directory. Files are flushed through maintained atomic writers, then the completed staging directory is renamed to the final directory. Existing final directories are rejected. On any exception, the staging directory is recursively removed.

A completed validation package remains valid if sealed-test evaluation later fails. An incomplete validation or sealed-test package is never visible.

## Error handling

The orchestrator fails closed on undeclared cell identities, result/request digest mismatch, non-finite growth, incomplete evidence closure, forged validation output, validation gate failure, schedule mismatch, repeated ledger authorization, or publication over an existing package. It performs no internal retries.

## Testing strategy

Tests use an in-memory recording evaluator and the existing in-memory sealed-test ledger. They prove exact call counts and order, one shared baseline per cell, complete v2 evidence, request/result rejection, no test access after validation failure, forged-selection rejection, authorization before test evaluation, selected-only test execution, repeated-run rejection, and atomic publication cleanup.

## A6b boundary

A6b will implement `StageAEvaluationCellEvaluator` by loading retained checkpoints through the canonical loader and validating the real execution source before returning `StageAEvaluationCellResult`. It will construct `StageATestSchedule` from maintained evaluation artifacts and provide the PostgreSQL-backed sealed-test ledger.
