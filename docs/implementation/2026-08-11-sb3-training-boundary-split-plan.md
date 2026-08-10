# SB3 Training Boundary Split Implementation Plan

> Execution record for the phased responsibility split of `trade_rl.integrations.sb3_training`.

**Goal:** Extract runtime, vector-environment, behavior-cloning, and teacher-cache responsibilities from `trade_rl.integrations.sb3_training` while preserving `StableBaselines3Backend`, numerical behavior, serialized contracts, error semantics, and current call sites.

**Base:** `main` commit `b0664257b7353605342247e62caba27e6965ca0c`

**Implementation branch:** `agent/split-sb3-training-boundaries`

**Pull request:** #386

**Architecture:** Four focused integration owners plus the existing coordinator:

```text
sb3_runtime.py
sb3_environment.py
sb3_behavior_cloning.py
sb3_teacher_pipeline.py
        ↓
sb3_training.py  # StableBaselines3Backend and orchestration
```

Module-level private compatibility names remain direct imported object aliases. Stateful teacher cache methods are inherited directly from `_StableBaselines3TeacherPipeline`; no forwarding wrappers or duplicate implementations are used.

## Global constraints

- [x] Base the work on exact `main` commit `b0664257b7353605342247e62caba27e6965ca0c`.
- [x] Keep the work on an independent branch.
- [x] Do not modify the files changed by Draft PR #385.
- [x] Keep `StableBaselines3Backend` in `trade_rl/integrations/sb3_training.py`.
- [x] Preserve existing module-level underscore-prefixed compatibility imports from `sb3_training.py`.
- [x] Do not add reverse imports from focused modules into `sb3_training`.
- [x] Preserve exception types, exception messages, environment-variable defaults, `spawn` process mode, artifact schemas, and output payloads.
- [x] Keep final `sb3_training.py` below 61,440 bytes.
- [x] Introduce no new external dependency, public DTO, checkpoint schema, or algorithm.
- [x] Do not merge without explicit owner authorization.

## Final file map

### Created

- [x] `trade_rl/integrations/sb3_runtime.py`
- [x] `trade_rl/integrations/sb3_environment.py`
- [x] `trade_rl/integrations/sb3_behavior_cloning.py`
- [x] `trade_rl/integrations/sb3_teacher_pipeline.py`
- [x] `tests/architecture/test_sb3_training_boundaries.py`
- [x] `docs/architecture/2026-08-11-sb3-training-boundary-split.md`
- [x] `docs/implementation/2026-08-11-sb3-training-boundary-split-plan.md`

### Modified

- [x] `trade_rl/integrations/sb3_training.py`
- [x] `tests/architecture/test_ownership_boundaries.py`
- [x] `tests/integrations/test_sb3_training.py`
- [x] `tests/learning/test_episode_teacher_integration.py`

### Explicitly untouched

- [x] `docs/implementation/2026-08-11-universal-instrument-artifact-materialization-plan.md`
- [x] `tests/workflows/test_postgres_universal_instrument_artifacts.py`
- [x] `tests/workflows/test_universal_instrument_artifacts.py`
- [x] `trade_rl/workflows/universal_instrument_artifacts.py`

## Task 1: Establish the RED ownership contract

- [x] Add AST-based owner-module checks for runtime, environment, and BC helpers.
- [x] Require extracted helper definitions to be absent from `sb3_training.py`.
- [x] Require direct object identity between coordinator compatibility imports and canonical owners.
- [x] Reject reverse imports into `sb3_training`.
- [x] Add the 61,440-byte coordinator ratchet.
- [x] Capture initial RED evidence.

Initial RED run:

```text
4 failed, 1,031 passed
```

The failures were limited to the three missing owner modules and the original `sb3_training.py` size of 92,927 bytes.

## Task 2: Extract runtime and resource policy

- [x] Move Lagrangian probe worker parsing.
- [x] Move Oracle solver environment-variable parsing.
- [x] Move optional Oracle accelerator resolution.
- [x] Move teacher worker parsing and Oracle episode-sampling adapter.
- [x] Move Torch CUDA runtime configuration.
- [x] Move sequence compilation runtime evidence.
- [x] Import all seven helpers directly back into `sb3_training.py` under their existing names.
- [x] Preserve lazy CUDA Oracle backend import behavior.

## Task 3: Extract vector-environment ownership

- [x] Move heavy-info compaction.
- [x] Move `_TrainingInfoFilter`.
- [x] Move direct, dummy-vector, and subprocess-vector construction.
- [x] Move compact hierarchical-sequence worker construction.
- [x] Move `ParallelSequenceVecEnv` assembly and partial-construction cleanup.
- [x] Preserve `SubprocVecEnv(..., start_method="spawn")`.
- [x] Move structured-export reset normalization.
- [x] Import all environment helpers directly back into the coordinator.

## Task 4: Extract behavior-cloning helper ownership

- [x] Move teacher cache key and lightweight teacher identity.
- [x] Move relative-improvement validation.
- [x] Move deterministic BC seed resolution and member-seed restoration.
- [x] Move pre-PPO BC policy candidate persistence.
- [x] Move hierarchical actor-head routing.
- [x] Move hierarchical teacher labels and BC configuration.
- [x] Move BC gate threshold construction, evaluation, and fail-closed enforcement.
- [x] Preserve every numerical formula and failure message.
- [x] Import all fourteen helpers directly back into the coordinator.

After runtime, environment, and BC extraction:

```text
82 focused tests passed
```

The only remaining contract failure was coordinator size: 74,392 bytes versus the 61,440-byte ratchet.

## Task 5: Extract teacher cache and artifact lifecycle

The size ratchet showed that helper extraction alone did not remove enough stateful responsibility. The smallest cohesive additional boundary was the cache-oriented teacher pipeline.

- [x] Extend the RED architecture contract with `_StableBaselines3TeacherPipeline`.
- [x] Require direct ownership of:
  - `_oracle_episode_batch`
  - `_episode_teacher_dataset`
  - `_oracle_targets`
  - `_trend_baseline_targets`
  - `_teacher_dataset`
- [x] Move the five existing method implementations without altering their bodies.
- [x] Declare only the backend state consumed by the internal base class.
- [x] Make `StableBaselines3Backend` inherit the pipeline base.
- [x] Require inherited method object identity; do not add wrappers.
- [x] Preserve reusable artifact index lookup/registration.
- [x] Preserve teacher artifact digest, validation, replacement, and temporary-directory cleanup behavior.

Teacher-pipeline RED run:

```text
4 expected failures, 80 passes
```

The failures were missing pipeline ownership/identity and the size ratchet.

## Task 6: Migrate tests to canonical patch ownership

The first teacher-pipeline GREEN attempt exposed eight tests that patched dependencies through the former coordinator module. Restoring dynamic forwarding would have recreated hidden coupling, so the tests were migrated instead.

- [x] Patch `oracle_target_path` in `sb3_teacher_pipeline`.
- [x] Patch `collect_teacher_rollout` in `sb3_teacher_pipeline`.
- [x] Patch `build_episode_oracle_batch` in `sb3_teacher_pipeline`.
- [x] Import `OracleSolverConfig` from its canonical learning module.
- [x] Import `OracleEpisodeSamplingConfig` from its canonical learning module.
- [x] Update the accelerator-backend ownership test to inspect `sb3_teacher_pipeline.py`.
- [x] Keep ordinary backend construction and public calls through `sb3_training`.
- [x] Add no compatibility forwarding wrapper.

## Task 7: Focused GREEN verification

- [x] Ruff check and format all touched implementation/test files.
- [x] Run architecture ownership tests.
- [x] Run sequence runtime acceleration tests.
- [x] Run action-head BC routing tests.
- [x] Run parallel sequence subprocess smoke tests.
- [x] Run core SB3 training tests.
- [x] Run checkpoint-transfer tests.
- [x] Run Lagrangian probe tests.
- [x] Run cost-critic and Lagrangian backend tests.
- [x] Run constraint-info tests.
- [x] Run episode teacher integration tests.
- [x] Run MyPy on all five affected integration modules.
- [x] Run Import Linter.
- [x] Run Vulture at 100% confidence.
- [x] Verify the 60 KiB ratchet.
- [x] Verify no overlap with PR #385.

Focused GREEN evidence:

```text
84 passed
MyPy: success, 5 source files
Import Linter: 12 contracts kept, 0 broken
Vulture: success at 100% confidence
sb3_training.py: 58,621 bytes
PR #385 overlap: none
```

The verified implementation commit was `22cca293bff88da8cb5a07da13f9627ce88ae54b`. Documentation-alignment commits follow it without modifying production code.

## Task 8: Self-review

- [x] Confirm no algorithmic expression was intentionally changed.
- [x] Confirm no failure string was intentionally changed.
- [x] Confirm no backend public method signature changed.
- [x] Confirm module-level compatibility names are direct imported objects.
- [x] Confirm teacher methods are direct inherited objects.
- [x] Confirm no focused owner module imports `sb3_training`.
- [x] Confirm no temporary workflow or transformation script remains in the final tree.
- [x] Confirm only the intended ten files differed before documentation alignment.
- [x] Confirm the implementation remains isolated from PR #385.

## Task 9: Exact-head repository CI

- [ ] Require ordinary PR CI on the final owner-authored head.
- [ ] Require Rebuilt Core success.
- [ ] Require Windows compatibility success.
- [ ] Require Ubuntu compatibility success.
- [ ] Require training image and packaged non-root probe success.
- [ ] Require frontend tests, typecheck, build, and fixed-layout checks.
- [ ] Require Ruff, format, MyPy, Import Linter, and Vulture.
- [ ] Require recovery and structured-serving smoke tests.
- [ ] Require full pytest, coverage, and critical branch coverage.
- [ ] Require package and lock identity.
- [ ] Recheck current `main` and PR mergeability after CI.
- [ ] Mark PR Ready only after all exact-head checks succeed.
- [ ] Do not merge without explicit owner authorization.

The PostgreSQL specialist workflow is not expected to trigger because the changed file set does not match its path filter.

## Remaining risk

This is a structural refactor. It does not prove market profitability, teacher quality, PPO improvement, execution-model realism, or production authorization. Production remains **NO-GO**.

A later separate change may extract checkpoint/replay lifecycle and final artifact publication. It must not be combined with algorithm or schema changes.
