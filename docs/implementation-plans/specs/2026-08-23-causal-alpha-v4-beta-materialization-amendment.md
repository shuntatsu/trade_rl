# Causal Alpha V4 Beta Materialization Amendment

## Status and precedence

This amendment is part of the approved Causal Alpha V4 design and corrects one dependency-order defect found during implementation-plan self-review.

It applies after `docs/implementation-plans/specs/2026-08-23-causal-alpha-v4-hierarchical-teacher-design.md` and overrides only the sections that imply causal beta is first computed inside the later teacher-fitting/runtime stage. All other V4 design constraints remain unchanged.

No empirical V4 outcome has been read. This is a pre-implementation contract correction, not post-hoc tuning.

## Problem found

The base design requires `causal_beta` to be observable by both Teacher and Student. The first implementation plan, however, exposed beta in the Student observation task before the later task that computed beta for teacher samples.

That ordering permits an invalid implementation in which:

```text
Teacher beta source != Student beta source
```

or in which the routed policy environment has no authoritative beta value when it constructs an observation.

That violates the central V4 information-set invariant.

## Corrected invariant

For every `(target_symbol, decision_index)` in a V4 generation:

```text
Teacher beta
== Student-observation beta
== Serving beta
== beta stored in the immutable V4 target-context artifact
```

There is one authoritative beta series per target context artifact. Teacher fitting and target generation consume that persisted series; they do not silently recompute a second beta series.

## Ownership

Causal beta is part of V4 decision-time market context, not a teacher-private learned state.

The numerical beta primitive therefore belongs with the V4 context contract in:

```text
trade_rl/data/v4_context.py
```

The context materializer computes beta before writing a `V4TargetContext` artifact. Later V4 teacher code validates and consumes it.

## Frozen beta contract

The authored beta contract remains unchanged:

```text
market proxy            = BTCUSDT USD-M perpetual
return clock            = 4h trailing realized log returns
lookback                 = 720h
minimum complete samples = 90
minimum market variance  = 1e-12
clip                     = [-3.0, 3.0]
BTCUSDT beta             = exactly 1.0 when the target is BTCUSDT
```

For a non-BTC target at decision `t`:

```text
beta_s(t)
  = cov(r_s_4h, r_BTC_4h) / var(r_BTC_4h)
```

using only complete historical 4h returns whose source closes are observable at or before decision `t`.

If support is insufficient or BTC variance is below the authored minimum, beta is unavailable. It is not filled with zero, one, a cross-sectional mean, or a fitted fallback.

## Artifact contract change

`V4TargetContext` contains beta explicitly:

```python
@dataclass(frozen=True, slots=True)
class V4TargetContext:
    symbol: str
    local: V4ContextBlock
    global_market: V4ContextBlock
    beta: np.ndarray
    beta_available: np.ndarray
    beta_source_digest: str
    profile_name: str
    digest: str = ""
```

Requirements:

- `beta.shape == (row_count,)` and `beta_available.shape == (row_count,)`;
- `beta` is finite storage data; unavailable rows are semantically controlled by `beta_available`;
- all available beta values lie in `[-3.0, 3.0]`;
- BTC available beta values are exactly `1.0`;
- beta arrays, beta availability, beta config, target/BTC source ranges, and `beta_source_digest` are included in context identity;
- changing any beta input history that affects a row changes the context digest;
- changing future rows after decision `t` cannot change beta at or before `t`.

The `causal_alpha_v4_target_context_artifact_v1` manifest therefore also binds `beta_source_digest` and the beta array identities.

## Student and serving contract change

`V4ContextProvider.resolve` no longer accepts beta from its caller.

Authoritative signature:

```python
class V4ContextProvider:
    def resolve(
        self,
        *,
        symbol: str,
        decision_index: int,
    ) -> V4PolicyContext: ...
```

It resolves local context, global context, `causal_beta`, and `causal_beta_available` from the same immutable target-context artifact.

The routed Student observation continues to expose:

```text
causal_beta
causal_beta_available
```

but those values are now artifact-backed and cannot drift from teacher values.

Serving must use the same context artifact/schema and provider semantics.

## Teacher-runtime change

`CausalAlphaV4SymbolSamples` receives beta from `V4TargetContext` rather than calculating a private beta series.

The V4 runtime must validate:

```text
sample decision indices == context decision indices for the selected rows
sample beta == context beta
sample beta_available == context beta_available
```

before fitting.

A mismatch is identity/correctness failure, not a reason to recompute silently.

## Test Oracle

The beta contract is correct when all of the following are observed:

1. Synthetic target returns `r_s = 2 * r_BTC` produce beta `2.0` after minimum support.
2. BTC produces available beta exactly `1.0`.
3. Mutating target or BTC history strictly after decision `t` leaves beta through `t` unchanged.
4. Mutating an eligible historical return inside the lookback changes beta and the context digest.
5. Insufficient support and near-zero BTC variance produce `beta_available=False`.
6. The Student provider, teacher samples, and reloaded artifact return exactly the same beta array and availability mask.
7. Tampering with only persisted beta bytes is detected by artifact validation.

## Failure modes added

- beta calculated twice with numerically different windows;
- beta observation missing before teacher task exists;
- Student receives a caller-supplied beta that Teacher did not use;
- serving reconstructs beta with a different source range;
- context digest fails to bind beta or beta source identity.

Any one blocks V4 Teacher admission.

## Required implementation-order correction

The implementation order is now:

```text
context formulas + beta primitive
-> context materialization including beta
-> immutable context artifact
-> Student/serving observation provider
-> teacher sample construction that consumes persisted beta
-> fitting / uncertainty / target / gates
```

No later task may restore teacher-private beta calculation.
