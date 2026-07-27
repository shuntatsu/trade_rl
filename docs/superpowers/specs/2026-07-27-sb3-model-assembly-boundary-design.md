# SB3 Model Assembly Boundary Design

## Problem

`trade_rl/integrations/sb3_training.py` is the public Stable-Baselines3 adapter, but its `StableBaselines3Backend.train()` method currently owns several independent responsibilities:

1. validating the environment identity;
2. resolving algorithm-specific configuration;
3. selecting policy and feature extractor classes;
4. resolving sequence reconstruction and compact rollout buffers;
5. constructing PPO, Cost Critic PPO, Lagrangian PPO, SAC, TD3, or TQC;
6. validating and loading checkpoint resumes;
7. behavior-cloning preparation;
8. callback construction and learning;
9. architecture, checkpoint, replay, and performance evidence publication.

The model-assembly section alone contains a dense matrix of algorithm, policy, sequence, asset-set, and resume branches. Changes to one model family therefore require editing the same orchestration method that controls evidence publication and training lifecycle cleanup.

## Considered Approaches

### A. Full vertical decomposition

Split runtime setup, environment creation, teacher preparation, model construction, resume, callbacks, learning, and artifact publication into separate modules.

This would produce the smallest files, but it is too broad for one independent remediation PR and would make behavioral equivalence difficult to review.

### B. Extract typed assembly boundaries

Move policy configuration, rollout-buffer selection, and algorithm construction into one typed module, and checkpoint validation/loading into a companion typed module. Keep environment ownership, behavior cloning, callbacks, learning, saving, and result construction in `StableBaselines3Backend`.

This removes the largest branch clusters while preserving the established lifecycle and failure-cleanup boundary. This is the selected approach.

### C. Add an algorithm registry only

Replace the final PPO/SAC/TD3/TQC conditional with a mapping of constructors.

This would reduce a small amount of branching but leave policy kwargs, sequence reconstruction, rollout buffers, constrained-PPO options, and resume identity logic entangled in `train()`.

## Decision

Create two one-way dependencies beneath the backend:

- `trade_rl/integrations/sb3_model_assembly.py` for policy metadata, rollout-buffer selection, and fresh model construction;
- `trade_rl/integrations/sb3_checkpoint_assembly.py` for manifest validation, algorithm-matched loading, constrained-algorithm identity checks, and sequence reconstructor rebinding.

The model assembly module owns:

- `SB3PolicyAssembly`: policy identifier, policy kwargs, sequence metadata, sequence reconstructor, shared-actor declaration, rollout-buffer class/kwargs, and estimated rollout-buffer bytes;
- `resolve_sb3_policy_assembly(...)`;
- `build_sb3_model(...)`.

The checkpoint assembly module owns:

- `LoadedSB3Checkpoint`;
- `load_sb3_checkpoint_model(...)`.

`StableBaselines3Backend.train()` remains the sole owner of environment opening/closing, output paths, teacher caches, behavior cloning, callbacks, learning, performance instrumentation, final saving, replay publication, and `PolicyTrainingResult`.

## Dependency Direction

The assembly modules may depend on:

- typed algorithm configurations;
- maintained policy and rollout-buffer implementations;
- checkpoint contracts;
- the Stable-Baselines3 and sb3-contrib adapters.

They must not depend on:

- behavior-cloning teacher collection;
- TensorBoard callback creation;
- training-performance recording;
- output artifact publication;
- `StableBaselines3Backend`.

`sb3_training` imports both assembly modules, never the reverse. `sb3_checkpoint_assembly` may consume the immutable `SB3PolicyAssembly` contract but does not construct fresh policies.

## Data Flow

1. `StableBaselines3Backend.train()` opens and validates one probe environment.
2. It resolves the typed `AlgorithmConfig` and optional Lagrangian feasibility evidence.
3. `resolve_sb3_policy_assembly()` reads only validated probe metadata and configuration, returning immutable assembly data.
4. The backend constructs the vector environment using its existing lifecycle logic.
5. `build_sb3_model()` creates the selected algorithm from the typed policy assembly.
6. When a checkpoint is configured, `load_sb3_checkpoint_model()` validates manifest identities, loads the matching algorithm class, and rebinds sequence reconstruction when required.
7. The backend continues with architecture evidence, behavior cloning, callbacks, learning, saving, and result publication exactly as before.

## Compatibility

- `StableBaselines3Backend` constructor and `train()` signature remain unchanged.
- Existing imports of `_configure_torch_cuda_runtime`, `_build_training_environment`, `_compact_training_info`, and `_behavior_cloning_quality` remain in `sb3_training.py` during this PR.
- Policy architecture, algorithm kwargs, checkpoint validation, output schemas, and runtime behavior must remain byte-for-byte or value-equivalent.
- No learning hyperparameter, reward, environment, evaluation, Serving, or production-release behavior changes.

## Error Handling

- Missing sequence metadata, dataset-bound reconstruction information, asset-set layout, optional TQC dependency, and checkpoint identity mismatches remain fail-closed with the established messages.
- The backend retains `try/finally` ownership of probe and vector-environment closure.
- Assembly functions do not publish partial artifacts.

## Testing

Regression tests will prove:

1. PPO, Cost Critic PPO, Lagrangian PPO, SAC, TD3, and TQC select the same constructor and kwargs;
2. sequence policies preserve shared-per-asset actor selection and index-backed rollout reconstruction;
3. asset-set policy metadata is unchanged;
4. rollout-buffer budget rejection remains fail-closed;
5. checkpoint resume validates algorithm, seed, environment, training config, and constrained-algorithm identity;
6. sequence checkpoint loading rebinds its reconstructor;
7. `StableBaselines3Backend` still owns environment cleanup and produces the same training result/evidence contracts;
8. architecture tests prevent either assembly module from importing teacher, telemetry, artifact-publication, or backend orchestration modules.

## Scope Exclusions

This PR will not split teacher preparation, callback construction, training-performance instrumentation, architecture evidence serialization, or replay publication. Those can be considered separately only after these assembly boundaries are stable.
