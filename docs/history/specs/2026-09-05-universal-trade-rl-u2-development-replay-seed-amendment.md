# Universal Trade RL U2 Development Replay Seed Amendment

> **2026-09-06 supersession notice:** The seed-coupling rule in this document (`evaluation_seed = candidate training seed`) is superseded by `2026-09-06-universal-trade-rl-u2-pre-development-closure-amendment.md`. U2 V1 Development Selection now derives one **scope-common evaluation seed** from the frozen U2 contract digest and canonical scope digest, and candidate seeds 0/1/2 plus cash/long/short baselines must share that same execution RNG for a given scope. The same-scope pairing, RNG isolation, replay identity, deterministic policy inference, and no-RNG-tuning requirements below remain normative except where they depend on the superseded seed-coupling formula.

## Status

This is a pre-results normative amendment for U2 Task 7C-1. It closes one ambiguity discovered while translating the deterministic Development replay design into an implementation plan.

No real Development or Admission numeric source has been opened. Production remains `NO-GO`; Admission remains sealed.

## Problem

The U2 design fixes deterministic policy inference, but the maintained U1 execution path also contains seeded runtime randomness. A replay contract that does not bind that seed would permit otherwise identical candidate/baseline evaluations to use different execution random streams, making paired evidence ambiguous and potentially result-dependent.

## Normative rule

**Historical rule, superseded only for seed derivation by the 2026-09-06 pre-development closure:**

```text
evaluation_seed = candidate training seed
```

The maintained replay representation still accepts only the preregistered seed values `(0, 1, 2)`, but the authoritative U2 Development boundary must derive which one applies from immutable `(U2 contract, scope)` identity rather than from candidate training seed.

The same derived evaluation seed is used for every policy variant and every training-seed candidate on that scope:

```text
candidate seed 0
candidate seed 1
candidate seed 2
cash
constant_long
constant_short
```

The U1 environment reset receives exactly that scope-common seed. A caller may not substitute a candidate-specific or result-dependent evaluation RNG seed.

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

The maintained low-level replay tests continue to prove request/evidence seed identity and RNG isolation. The authoritative U2 Development boundary additionally proves:

1. unregistered evaluation seeds are rejected before numeric loading;
2. reset receives the exact scope-common evaluation seed;
3. candidate seeds 0/1/2 and all baselines use the same derived seed for one canonical scope;
4. changing only the scope identity may change the derived seed deterministically;
5. changing only the paired checkpoint digest does not change the scope-common evaluation seed;
6. no separate evaluation RNG tuning surface exists.

## Non-goals

This amendment does not:

- change U2 training seeds;
- change PPO randomness;
- define gross accounting;
- authorize real Development evaluation;
- authorize Admission access;
- change U1 Risk, Execution, Accounting, normalizer, action, or reward semantics.
