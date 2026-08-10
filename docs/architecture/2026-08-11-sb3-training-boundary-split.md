# SB3 Training Boundary Split Design

## Conclusion

`trade_rl.integrations.sb3_training` remains the stable orchestration entry point, but it stops owning unrelated runtime, vector-environment, and behavior-cloning helper implementations. The change is intentionally structural: algorithm behavior, serialized identities, exception semantics, public workflow entry points, and `StableBaselines3Backend` remain unchanged.

The selected approach is a phased extraction with compatibility re-exports. A big-bang rewrite would create unnecessary training and checkpoint risk; adding only a file-size guard would document the problem without correcting it.

## Goal

Reduce the number of independent change reasons in `sb3_training.py` while preserving every current caller and test contract.

Success means:

1. runtime/resource parsing is owned by one focused module;
2. vector-environment construction and training-info filtering are owned by one focused module;
3. behavior-cloning helper logic is owned by one focused module;
4. `sb3_training.py` keeps orchestration and `StableBaselines3Backend`;
5. existing imports from `trade_rl.integrations.sb3_training`, including current private test imports, continue to resolve to the same callable objects;
6. no checkpoint, artifact, reward, BC metric, PPO, execution, or serving behavior changes;
7. architecture tests prevent the extracted implementations from returning to the coordinator.

## Non-goals

This change does not:

- redesign `StableBaselines3Backend`;
- alter BC train/validation/purge semantics;
- alter Oracle solver selection or numerical behavior;
- alter CUDA determinism, TF32, compilation, or worker defaults;
- alter vector-environment start methods;
- alter checkpoint schemas, policy identity, replay handling, telemetry, or artifact publication;
- modify `trade_rl.rl.environment`;
- overlap with the Universal Instrument Artifact work in PR #385.

## Approaches considered

### A. Big-bang backend decomposition

Move the backend class, checkpoint lifecycle, teacher pipeline, environment lifecycle, and runtime configuration at once.

Rejected for this change because the review surface would combine structural movement with many hidden coupling points. Exact-head CI would detect regressions, but diagnosis and rollback would be unnecessarily difficult.

### B. Phased helper extraction with compatibility aliases

Move cohesive, mostly stateless helper clusters first. Keep the coordinator class and import the moved symbols back into `sb3_training.py` under their existing names.

Selected because it reduces responsibility immediately, preserves current imports, and creates stable seams for later checkpoint and teacher-pipeline extraction.

### C. Size-budget test only

Add an architecture test that rejects further growth but leave all logic in place.

Rejected as insufficient. A ratchet is useful only after the first responsibility split establishes a smaller baseline.

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

It must preserve the current `spawn` subprocess contract and close partially constructed workers on failure.

### `trade_rl.integrations.sb3_behavior_cloning`

Owns coordinator-independent BC helpers:

- teacher cache key and lightweight teacher identity;
- relative-improvement validation;
- deterministic BC seed resolution and member-seed restoration;
- pre-PPO BC policy candidate persistence;
- hierarchical actor-head routing;
- hierarchical teacher labels and BC configuration;
- BC gate thresholds, evaluation, and fail-closed enforcement.

It may call the existing `learning` and checkpoint persistence contracts, but it must not construct SB3 models or run PPO.

### `trade_rl.integrations.sb3_training`

Continues to own:

- `StableBaselines3Backend`;
- end-to-end training orchestration;
- model/checkpoint/replay coordination;
- teacher generation and artifact lifecycle that are still coupled to the backend method flow;
- progress, architecture, performance, and final result publication.

The coordinator imports the extracted helpers under their existing underscore-prefixed names. This is a compatibility bridge, not duplicate implementation.

## Compatibility contract

Current callers and tests import several private helpers from `sb3_training.py`. Removing those names in the same change would mix architecture cleanup with API migration. Therefore:

```python
from trade_rl.integrations.sb3_runtime import (
    _configure_sequence_runtime,
    _configure_torch_cuda_runtime,
    ...
)
```

The same pattern applies to the environment and BC modules. The imported objects must be identical to the canonical definitions, and `sb3_training.py` must not wrap or redefine them.

No new production code should begin importing these private compatibility names from `sb3_training`. New focused tests should import the owning modules directly.

## Data and control flow

```text
StableBaselines3Backend.train
  -> sb3_runtime: resolve workers / solver / CUDA / sequence runtime
  -> sb3_environment: build and filter training environments
  -> sb3_behavior_cloning: prepare labels/config/gates and candidate persistence
  -> existing model/checkpoint/replay/telemetry contracts
  -> PolicyTrainingResult
```

Values and exceptions cross module boundaries unchanged. The extraction must use the existing concrete DTOs; no duplicate configuration or result types are introduced.

## Failure behavior

- Invalid environment variables retain the current exception type and message.
- CUDA and sequence compilation failures remain fail-closed.
- Vector-worker construction closes already-created workers before re-raising.
- Missing hierarchical BC fields, invalid losses, failed gates, and missing policy files retain current failures.
- Import cycles are prohibited; none of the three new modules imports `sb3_training`.

## Testing strategy

### RED architecture contract

Add a focused test module that initially fails because the new modules do not exist and the helper implementations still live in `sb3_training.py`.

The test locks:

- canonical helper-name ownership by module;
- absence of those function/class definitions from `sb3_training.py`;
- compatibility object identity through `sb3_training`;
- no reverse import into `sb3_training`;
- a post-extraction `sb3_training.py` size ceiling of 60 KiB.

### Focused behavioral verification

Run the existing suites that directly exercise the moved contracts, including:

- sequence runtime acceleration;
- Lagrangian probe and worker selection;
- action-head BC routing;
- BC split/gate/evaluation tests;
- parallel sequence subprocess smoke;
- SB3 training, transfer, cost-critic, and Lagrangian backend tests;
- architecture ownership tests.

### Repository verification

On one final head require:

- Ruff and Ruff format;
- MyPy;
- Import Linter;
- Vulture;
- focused integration tests;
- full pytest and critical branch coverage;
- Windows and Ubuntu compatibility;
- training image and packaged non-root probe;
- frontend and structured-serving checks through normal CI.

## Future phases

After this PR is stable, separate changes may extract checkpoint/replay lifecycle and teacher-generation orchestration. Those changes must use the seams created here and must not be bundled into this first structural PR.
