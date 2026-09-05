# Universal Trade RL U2 Development Replay Seed Amendment

## Status

This is a pre-results normative amendment for U2 Task 7C-1. It closes one ambiguity discovered while translating the deterministic Development replay design into an implementation plan.

No real Development or Admission numeric source has been opened. Production remains `NO-GO`; Admission remains sealed.

## Problem

The U2 design fixes deterministic policy inference, but the maintained U1 execution path also contains seeded runtime randomness. A replay contract that does not bind that seed would permit otherwise identical candidate/baseline evaluations to use different execution random streams, making paired evidence ambiguous and potentially result-dependent.

## Normative rule

For every U2 Development replay:

```text
evaluation_seed = candidate training seed
```

The allowed values are exactly the preregistered U2 training seeds `(0, 1, 2)`.

The same evaluation seed is used for every policy variant paired to that candidate generation:

```text
candidate
cash
constant_long
constant_short
```

The U1 environment reset receives exactly that seed. A caller may not provide a different evaluation-only RNG seed.

## Pair identity

Every replay request and replay evidence item binds:

- `evaluation_seed`
- `paired_candidate_checkpoint_digest`
- exact scope digest
- exact common-view dataset digest
- U1 contract digest
- U2 contract digest
- policy variant

`paired_candidate_checkpoint_digest` is required for all four variants so that cash and diagnostic static baselines cannot be silently reused across a different candidate generation.

For the `candidate` variant, the model/checkpoint supplied to replay must correspond to the same paired candidate checkpoint identity. Task 7C-1 may use synthetic checkpoint digests in tests; real checkpoint loading remains outside this synthetic-only task.

## Determinism rule

Candidate inference is always called as:

```python
model.predict(observation, deterministic=True)
```

This deterministic inference flag does not replace the execution RNG rule. Policy inference and execution/runtime randomness are separate contracts and both are fixed.

## Paired-baseline rule

Candidate, cash, constant-long, and constant-short evidence are comparable only when all of the following match:

- evaluation seed
- paired candidate checkpoint digest
- scope digest
- evaluation dataset digest
- U1 runtime/economic contract
- normalizer generation
- policy contract

A mismatch is a contract error, not a valid paired comparison.

## Test oracles

Task 7C-1 tests must prove:

1. unregistered evaluation seeds are rejected before numeric loading;
2. reset receives the exact evaluation seed;
3. all four variants use the same requested seed for one paired candidate generation;
4. changing only the evaluation seed changes replay identity;
5. changing only the paired checkpoint digest changes replay identity;
6. no separate evaluation RNG tuning surface exists.

## Non-goals

This amendment does not:

- change U2 training seeds;
- change PPO randomness;
- define gross accounting;
- authorize real Development evaluation;
- authorize Admission access;
- change U1 Risk, Execution, Accounting, normalizer, action, or reward semantics.
