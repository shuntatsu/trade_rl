# SB3 Training Boundary Verification Record

## Scope

This record covers PR #386, which extracts Stable-Baselines3 runtime, vector-environment, behavior-cloning, and teacher-cache responsibilities from `trade_rl.integrations.sb3_training` without changing training algorithms or serialized contracts.

## Focused verification

Verified implementation commit: `22cca293bff88da8cb5a07da13f9627ce88ae54b`

Results:

```text
84 passed
MyPy: success for 5 affected integration modules
Import Linter: 12 contracts kept, 0 broken
Vulture: success at 100% confidence
sb3_training.py: 58,621 bytes
PR #385 changed-file overlap: none
```

The temporary extraction workflow and transformation script were removed before the verified implementation commit was published.

## First full repository CI

Owner-authored exact head: `9ddb7d68cd5ce86f08e7a2e446985acb5897f072`

Workflow run: `31414128674`

Successful jobs and gates before pytest completion:

- Ubuntu compatibility;
- Windows compatibility;
- complete training image and packaged non-root runtime probe;
- frontend test, typecheck, build, and fixed-layout checks;
- workflow security;
- Ruff and Ruff format;
- MyPy for 400 source files;
- Import Linter with 12 contracts kept and 0 broken;
- Vulture at 100% confidence;
- recovery and structured-serving smoke tests.

Full pytest result:

```text
1 failed, 3,256 passed, 26 skipped
coverage: 80.73% (required 80%)
```

The only failure was:

```text
tests/integrations/test_parallel_sequence_environments.py::
test_parallel_sequence_builder_uses_spawn_workers_and_parent_wrapper
```

## Root cause

`_build_parallel_sequence_training_environment` is now canonically defined in `sb3_environment.py`. The test still patched the compatibility alias `sb3_training._build_training_environment`, but the extracted function resolves `_build_training_environment` from its canonical module globals.

This was a test seam ownership mismatch, not an algorithm or runtime regression. Adding a dynamic forwarding wrapper to the coordinator would have recreated the coupling that the refactor removes, so the implementation was not weakened.

## Fix

The test now:

- imports `trade_rl.integrations.sb3_environment` as the canonical owner;
- patches `sb3_environment._build_training_environment`;
- continues to invoke `sb3_training._build_parallel_sequence_training_environment`, proving that the coordinator compatibility alias still resolves to the canonical function.

Focused fix commit: `3c826038ab60cbf1c1f5bf75706ee3988861f7f5`

Focused rerun:

```text
21 passed in 7.83s
```

The rerun covered the complete `test_parallel_sequence_environments.py` module and `test_sb3_training_boundaries.py` architecture contract. Ruff fixed import ordering, Ruff format reported no further changes, and the temporary fix workflow removed itself before the commit was pushed.

## Final gate

A new owner-authored head containing this verification record must pass ordinary exact-head PR CI before PR #386 can be marked Ready. The PR must not be merged without explicit repository-owner authorization.

Production remains **NO-GO**. These results establish structural and regression correctness only; they do not establish market profitability, teacher economic quality, or production authorization.
