# SB3 Training Boundary Split Design

## Conclusion

`trade_rl.integrations.sb3_training` remains the stable Stable-Baselines3 orchestration entry point, but it no longer owns unrelated runtime policy, vector-environment assembly, behavior-cloning helper logic, or teacher cache/artifact lifecycle.

The implementation is intentionally structural. It preserves algorithm behavior, serialized identities, exception semantics, environment-variable defaults, process start methods, public workflow entry points, and the `StableBaselines3Backend` API.

The selected design is a phased extraction with direct compatibility imports for module-level private helpers and an internal teacher-pipeline base class for cache-oriented backend methods. No forwarding wrappers or duplicate DTOs are introduced.

## Goal

Reduce independent change reasons in `sb3_training.py` while preserving current callers and numerical behavior.

Success means:

1. runtime and resource policy are owned by one focused module;
2. vector-environment construction and training-info filtering are owned by one focused module;
3. coordinator-independent BC helpers are owned by one focused module;
4. teacher generation, immutable cache lookup/publication, and teacher artifact lifecycle are owned by one focused pipeline base;
5. `sb3_training.py` keeps `StableBaselines3Backend`, model/checkpoint/replay coordination, PPO execution, telemetry, and result publication;
6. existing module-level helper imports from `sb3_training` resolve to the same canonical objects;
7. teacher methods are inherited directly from the canonical pipeline base rather than redefined or wrapped;
8. no checkpoint, artifact, reward, BC metric, PPO, execution, or serving behavior changes;
9. architecture tests prevent extracted responsibilities from returning to the coordinator;
10. `sb3_training.py` remains below 60 KiB.

## Non-goals

This change does not:

- redesign the public `StableBaselines3Backend` API;
- alter BC train/validation/purge semantics;
- alter teacher labels, Oracle dynamic programming, or numerical formulas;
- alter Oracle solver selection or fallback behavior;
- alter CUDA determinism, TF32, compilation, memory, or worker defaults;
- alter vector-environment start methods;
- alter checkpoint schemas, policy identity, replay handling, telemetry, or artifact publication;
- modify `trade_rl.rl.environment`;
- extract checkpoint/replay lifecycle in this phase;
- overlap with the Universal Instrument Artifact work in PR #385.

## Approaches considered

### A. Big-bang backend rewrite

Move the complete backend class, model lifecycle, checkpoint lifecycle, teacher pipeline, environment lifecycle, and runtime configuration at once.

Rejected because the review surface would combine structural movement with hidden coupling points. Diagnosis and rollback would be unnecessarily difficult.

### B. Focused extraction with compatibility contracts

Move cohesive helper clusters into owner modules, import module-level helpers directly back into `sb3_training`, and move stateful teacher methods into an internal pipeline base inherited by the backend.

Selected because it removes responsibility concentration immediately while preserving existing call sites and creating stable seams for later checkpoint/replay extraction.

### C. Size ratchet only

Add an architecture test that rejects further growth but leave the implementation in place.

Rejected as insufficient. The size ratchet is useful only after responsibility ownership has been corrected.

## Responsibility boundaries

### `trade_rl.integrations.sb3_runtime`

Owns runtime and resource policy:

- Lagrangian probe worker count;
- Oracle solver environment-variable parsing;
- optional Oracle accelerator resolution;
- teacher worker count;
- Oracle episode sampling adapter;
- Torch CUDA determinism/performance configuration;
- sequence compilation runtime evidence.

It may depend on `learning`, `rl.training`, and `rl.training_modes`, but it must not import `sb3_training` or own training orchestration.

### `trade_rl.integrations.sb3_environment`

Owns the framework adapter around training environments:

- heavy-info compaction;
- the Gym wrapper that filters copied info payloads;
- direct, dummy-vector, and subprocess-vector construction;
- compact hierarchical-sequence worker construction;
- `ParallelSequenceVecEnv` assembly and cleanup;
- structured-export reset normalization;
- effective vector-environment kind selection.

It preserves the existing `spawn` subprocess contract and closes partially constructed workers on failure.

### `trade_rl.integrations.sb3_behavior_cloning`

Owns coordinator-independent BC helpers:

- teacher cache key and lightweight teacher identity;
- relative-improvement validation;
- deterministic BC seed resolution and member-seed restoration;
- pre-PPO BC policy candidate persistence;
- hierarchical actor-head routing;
- hierarchical teacher labels and BC configuration;
- BC gate thresholds, evaluation, and fail-closed enforcement.

It may call existing `learning` and checkpoint persistence contracts, but it must not construct SB3 models or run PPO.

### `trade_rl.integrations.sb3_teacher_pipeline`

Owns teacher computation and immutable teacher-cache lifecycle:

- Oracle episode-batch construction and memoization;
- episode teacher dataset load, generation, validation, and publication;
- flat Oracle target load, generation, validation, and publication;
- trend-baseline target memoization;
- non-episode teacher dataset load, generation, validation, and publication;
- reusable artifact index lookup and registration;
- temporary-directory cleanup and atomic artifact replacement behavior already present in the coordinator.

`_StableBaselines3TeacherPipeline` is an internal base class. It declares the state it consumes from `StableBaselines3Backend` and owns only the five cache-oriented teacher methods. It does not construct models, run PPO, publish final training results, or import `sb3_training`.

### `trade_rl.integrations.sb3_training`

Continues to own:

- `StableBaselines3Backend` construction and public behavior;
- end-to-end training orchestration;
- model construction and algorithm selection;
- checkpoint and replay coordination;
- behavior-cloning invocation and PPO fine-tuning flow;
- progress, architecture, performance, checkpoint, and final-result publication;
- cleanup of resources owned by the training call.

The backend inherits `_StableBaselines3TeacherPipeline`; it does not override or wrap the five extracted teacher methods.

## Compatibility contract

Current tests and internal callers import several underscore-prefixed module helpers from `sb3_training.py`. Removing those names in the same change would mix architecture cleanup with an API migration. Therefore `sb3_training.py` imports each helper directly from its owner module under the same name.

```python
from trade_rl.integrations.sb3_runtime import (
    _configure_sequence_runtime as _configure_sequence_runtime,
    _configure_torch_cuda_runtime as _configure_torch_cuda_runtime,
)
```

The same identity contract applies to runtime, environment, and BC helpers:

```python
getattr(sb3_training, name) is getattr(owner_module, name)
```

Teacher methods use inheritance rather than module aliases:

```python
StableBaselines3Backend._teacher_dataset \
    is _StableBaselines3TeacherPipeline._teacher_dataset
```

Tests that patch dependencies used inside teacher methods patch `sb3_teacher_pipeline`, the canonical owner. Dynamic forwarding through the old coordinator is deliberately not restored.

New production code must import focused private helpers from their owner module, not from the coordinator compatibility surface.

## Data and control flow

```text
StableBaselines3Backend.train
  -> sb3_runtime
       resolve workers / solver / CUDA / sequence runtime
  -> sb3_environment
       build and filter training environments
  -> sb3_teacher_pipeline
       resolve or generate teacher data and immutable cache evidence
  -> sb3_behavior_cloning
       prepare labels / config / gates / candidate persistence
  -> sb3_training coordinator
       model / checkpoint / replay / PPO / telemetry / publication
  -> PolicyTrainingResult
```

Values and exceptions cross module boundaries unchanged. Existing concrete DTOs are reused; no duplicate configuration or result types are introduced.

## Failure behavior

- Invalid environment variables retain their exception types and messages.
- CUDA and sequence compilation failures remain fail-closed.
- Vector-worker construction closes already-created workers before re-raising.
- Missing hierarchical BC fields, invalid losses, failed gates, and missing policy files retain current failures.
- Teacher artifacts retain the existing digest, schema, cache-identity, validation, and replacement rules.
- Teacher temporary directories retain cleanup-on-success and cleanup-on-failure behavior.
- Import cycles are prohibited; none of the four focused modules imports `sb3_training`.

## Enforced architecture contracts

`tests/architecture/test_sb3_training_boundaries.py` locks:

- exact module-level helper ownership;
- absence of extracted definitions from `sb3_training.py`;
- direct compatibility object identity through `sb3_training`;
- exact ownership of the five teacher pipeline methods;
- direct inherited method identity on `StableBaselines3Backend`;
- no reverse import into `sb3_training`;
- a coordinator size ceiling of 61,440 bytes.

`tests/architecture/test_ownership_boundaries.py` reads accelerator-backend ownership from `sb3_teacher_pipeline.py`, matching the new responsibility boundary.

## TDD and focused verification evidence

The extraction was driven by explicit failing contracts:

1. Initial RED: 4 failures and 1,031 passes; failures were only the three missing owner modules and the original 92,927-byte coordinator.
2. First GREEN stage: 82 focused tests passed; the remaining failure was only the 74,392-byte size ratchet.
3. Teacher-pipeline RED: 4 expected failures and 80 passes; failures were missing teacher ownership/identity plus size.
4. First teacher extraction exposed eight tests patching the former coordinator globals. The implementation was not weakened with forwarding wrappers; tests were migrated to the canonical owner.
5. Final focused run: 84 tests passed.
6. MyPy passed for all five affected integration modules.
7. Import Linter kept all 12 contracts with 0 broken.
8. Vulture passed at 100% confidence.
9. The final `sb3_training.py` size is 58,621 bytes.
10. The changed-file set has no overlap with PR #385.

Full repository and cross-platform verification remains an exact-head PR CI requirement before the PR can be marked ready.

## Future phases

A later, separately reviewed change may extract checkpoint/replay lifecycle and final artifact publication from the coordinator. It must preserve the seams and compatibility contracts established here and must not be combined with algorithm or schema changes.
