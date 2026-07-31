# Stage A Zero-Shot Evaluation Implementation Plan

## Completed contract hardening

### 1. Immutable v2 plan and evidence contracts

Implemented in `trade_rl/evaluation/stage_a_zero_shot_contracts.py`.

- Rejects the flawed v1 plan, observation, and evidence schemas.
- Requires exact candidate × triplet × fold × seed closure.
- Binds dataset, feature, execution, evaluation, checkpoint, policy-execution, and baseline-execution identities.
- Enforces one shared baseline per triplet/fold/seed cell across all candidates.
- Caps bootstrap resamples at 1,000,000.
- Uses strict JSON field closure and content digests.
- Keeps public facades limited to documented public values; internal loaders import private parsing helpers directly from their defining module so static cleanup cannot silently remove required runtime dependencies.
- Keeps string identities and normalized numeric thresholds in distinct typed locals so strict MyPy analysis cannot widen or cross-assign their contracts.

### 2. Robust fold-bootstrap and candidate summaries

Implemented in `trade_rl/evaluation/stage_a_zero_shot_gate.py`.

- Uses fold means as the only bootstrap unit.
- Uses a common derived draw seed for all candidates in the same evidence artifact.
- Generates bootstrap draws in bounded chunks.
- Retains per-triplet and per-seed excess growth.
- Computes worst-triplet excess, worst-seed excess, and non-negative-triplet pass fraction.

### 3. Validation and sealed-test recomputation

- Validation uses every declared candidate and all predeclared thresholds.
- Candidate ranking is deterministic.
- Sealed-test evaluation requires validation evidence and recomputes the supplied selection before accepting it.
- Test evidence contains exactly the selected candidate.
- Selection and decision loaders recompute the complete output and reject mismatches.

### 4. Atomic artifact writes

Implemented through `trade_rl/artifacts/atomic_write.py`.

- Flushes and fsyncs a unique temporary file.
- Atomically replaces the destination.
- Fsyncs the parent directory on supported platforms.
- Removes temporary files on both success and failure.

### 5. Regression coverage

The focused suites cover:

- missing and duplicate Cartesian cells;
- wrong checkpoints and plan identities;
- candidate-dependent baseline substitution;
- forged in-memory validation selections;
- test candidate expansion;
- hidden triplet and seed failures;
- common bootstrap draw seeds and resample caps;
- strict JSON tamper rejection;
- atomic replacement failure preserving the prior file.

## Completed A6a orchestration

Implemented through:

- `trade_rl/workflows/stage_a_zero_shot_runner_contracts.py`;
- `trade_rl/workflows/stage_a_zero_shot_runner.py`;
- `trade_rl/workflows/stage_a_zero_shot_artifacts.py`.

The A6a layer now:

1. binds every evaluation request to the exact plan, split, triplet, fold, seed, candidate, checkpoint, dataset, feature, execution, and evaluation identities;
2. requires each evaluator result to reference the exact request digest and a source execution-evidence digest;
3. evaluates the complete validation candidate × triplet × fold × seed Cartesian product in deterministic order;
4. evaluates one shared baseline per triplet/fold/seed cell and reuses it across all candidates;
5. delegates validation aggregation and selection to the maintained v2 gate;
6. recomputes the complete validation selection before any sealed-test access;
7. rejects failed or forged validation output before ledger or test-evaluator calls;
8. authorizes every declared test fold before the first sealed-test evaluation;
9. evaluates only the selected candidate and the shared baseline on test;
10. builds selected-only v2 test evidence and delegates the final decision to the maintained sealed-test gate;
11. publishes validation and sealed-test outputs as independent immutable directories after all files in the phase are complete;
12. uses an exclusive phase lock and removes only locks acquired by the current publisher;
13. removes incomplete staging directories after publication failures.

The orchestrator depends only on the typed `StageAEvaluationCellEvaluator` protocol. It does not import model frameworks, serving loaders, market adapters, or PostgreSQL.

## Completed A6b-1 verified execution adapter

Implemented through:

- `trade_rl/workflows/stage_a_execution_replay.py`;
- `trade_rl/workflows/stage_a_execution_store.py`;
- `trade_rl/workflows/stage_a_production_evaluator.py`.

The A6b-1 layer now:

1. binds each completed replay to the exact A6a request, plan, split, triplet, fold, seed, candidate, checkpoint, candidate configuration, dataset, feature, execution, and evaluation identities;
2. validates canonical execution-promotion evidence and its exact order-event artifact before publication;
3. requires positive finite equity and exact agreement between reported terminal equity and the event artifact's terminal portfolio value;
4. derives log growth only from the verified replay equity curve;
5. publishes event, evidence, replay, and request-index files under content-addressed canonical paths;
6. permits identical retries but permanently rejects rebinding one request digest to different execution bytes;
7. rejects non-canonical JSON, unsafe relative paths, symlinks, missing files, and event, evidence, replay, or index tampering;
8. validates every request against the immutable Stage A plan before accessing the store;
9. resolves policy candidate configuration from the plan and baseline configuration from a separate immutable identity;
10. returns the complete cell-bound replay digest to A6a, rather than the lower-level promotion digest that lacks triplet/fold/seed identity.

## Next integration stage: A6b-2 execution producer and operations

The remaining A6b-2 work is to:

1. resolve retained checkpoint and serving-bundle paths from maintained artifacts;
2. recompute file and manifest digests before model loading;
3. load policies only through the canonical serving/training loader;
4. materialize the declared triplet, fold, seed, feature, and market inputs;
5. run the maintained conservative execution model and publish the A6b-1 replay package;
6. construct the test schedule from the maintained evaluation source;
7. provide the PostgreSQL-backed one-shot sealed-test ledger;
8. add validation and sealed-test CLI commands plus complete-run artifact wiring.
