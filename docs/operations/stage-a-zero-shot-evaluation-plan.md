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

## Next integration stage

The next PR should add a Stage A runner that:

1. loads retained checkpoints through the canonical serving/training loader;
2. evaluates each declared fold, seed, triplet, and scenario using the maintained execution model;
3. verifies the source execution artifact before constructing each v2 observation;
4. rejects the source artifact before publication when its dataset, feature, execution, evaluation, checkpoint, or evaluation-cell identity differs from the predeclared plan;
5. writes validation evidence and selection;
6. consumes the existing one-shot sealed-test ledger only for the selected candidate;
7. writes the sealed-test evidence and final decision.
