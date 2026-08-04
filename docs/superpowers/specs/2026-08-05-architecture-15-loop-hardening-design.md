# Architecture 15-Loop Hardening Design

## Goal

Perform fifteen review/fix/re-review loops on the maintained `main` architecture without changing trading mathematics, Bellman numerical behavior, model-selection rules, artifact formats, or Production status.

## Scope

The work is limited to architectural boundaries already identified by the current codebase:

1. publish the training-environment identity contract;
2. remove cross-package private imports;
3. enforce that rule generically;
4. separate the generic PostgreSQL artifact catalog from sealed evaluation reservations;
5. share PostgreSQL connection construction through a public adapter utility;
6. remove the RL-to-evaluation dependency for terminal performance calculations;
7. place return-series and performance contracts below both RL and evaluation with compatibility facades;
8. centralize maintained policy identifiers in a standard-library domain module;
9. consume those identifiers from structured export contracts;
10. consume those identifiers from RL policy modules;
11. separate learning-recipe identity from export transport;
12. separate learning-recipe identity from source provenance;
13. retain transport, provenance, resume, and transfer in full run identity;
14. make Oracle accelerator registration explicit rather than import-triggered;
15. align architecture documentation and CI scope with maintained contracts.

## Non-goals

- no Stage B implementation;
- no reward, execution, action, Bellman, PPO, BC, or Lagrangian numerical change;
- no schema-version bump;
- no direct-exchange functionality;
- no broad directory reorganization;
- no CUDA-default change.

## Boundary design

### Public training environment contract

`trade_rl.rl.training_environment_contract` owns `training_environment_identity()` and `validate_training_environment()`. `rl.training` remains a compatibility consumer, while framework adapters import only the public contract.

### Catalog and sealed-test persistence

`PostgresArtifactCatalog` implements only generic artifact-catalog operations. `PostgresSealedTestReservationStore` remains the dedicated sealed-evaluation adapter. Both use `trade_rl.catalog.postgres_connection.default_connection_factory`.

### Performance contracts

`trade_rl.simulation.performance` owns `ReturnKind`, `ReturnSeries`, `PerformanceMetrics`, and `evaluate_performance`. `evaluation.series` and `evaluation.metrics` become compatibility facades. RL imports only the lower simulation contract.

### Policy identifiers

`trade_rl.domain.policy_contracts` owns maintained string/ordering identifiers shared by RL and artifact contracts:

- `SB3_POLICY_IDENTITY_SCHEMA = "sb3_policy_identity_v4"`
- `HIERARCHICAL_SEQUENCE_ENCODER = "hierarchical_sequence_v2"`
- `STRUCTURED_TIMEFRAMES = ("15m", "1h", "4h", "1d")`

The module remains standard-library only.

### Identity separation

`TrainingRunConfig` exposes a recipe payload containing only learning/economic meaning. Full run identity extends that recipe with export transport, Git provenance, resume checkpoint digests, and transfer checkpoint digests. Existing public methods and the maintained schema version remain unchanged.

### Oracle composition

Importing `trade_rl.integrations` must not mutate the learning-layer accelerator registry. A public idempotent registration function is called only from CUDA-capable composition paths.

## Error handling

All new contracts fail closed. Existing validation messages and compatibility imports are preserved where feasible. Missing accelerator registration continues to raise `OracleBackendFailure` rather than silently selecting another backend, except for the already-declared `cuda_or_numpy` fallback.

## Test strategy

A single architecture regression file defines fifteen initially failing contracts. A temporary PR-only workflow runs that file plus Ruff, MyPy, and Import Linter. Each loop reviews the remaining failures, applies one focused correction, and reruns the focused gate. After all loops pass, the temporary workflow is removed and the normal full CI runs on the final PR to `main`.

## Compatibility

Existing imports from `trade_rl.evaluation.metrics`, `trade_rl.evaluation.series`, and private helpers inside `rl.training` remain supported through aliases where needed, but production cross-package consumers move to public contracts. No serialized digest schema changes are introduced; only the membership of candidate recipe identity is corrected to match its documented meaning.
