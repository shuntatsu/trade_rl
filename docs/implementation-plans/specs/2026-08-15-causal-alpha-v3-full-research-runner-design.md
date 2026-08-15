# Causal Alpha V3 full research runner design

## Objective

Connect the implemented Causal Alpha V3 primitives to one maintained,
artifact-bound real-data research workflow. The workflow must fit and select a
V3 teacher using train/selection data only, replay it through production
execution, evaluate the untouched teacher-admission holdout exactly once, and
only then allow DAgger, behavior cloning, critic warm-start, and anchored
residual RL comparisons.

## Invariants

- Scalar reward remains pure net log growth.
- `max_position_to_market_notional=0.02` remains authoritative.
- Validation/test symbols and teacher-admission holdouts never enter V3 fitting,
  uncertainty calibration, candidate selection, or threshold tuning.
- Historical v2/r3 diagnostics remain `promotion_eligible=false` and cannot be
  resumed as V3 selection evidence.
- Canonical U6 example configs remain unchanged.
- Every long phase is resumable from generator-, config-, dataset-, and scope-
  bound evidence; identity drift fails closed.
- Teacher admission failure prevents BC, DAgger, critic warm-start, and RL.

## Workflow

1. Assemble the existing Universal train bindings, chronological selection
   contracts, causal samples, production execution inputs, and hard-risk
   identity from the runtime manifest.
2. Evaluate a predeclared V3 candidate grid. A candidate binds ridge strength,
   uncertainty multiplier, target magnitudes, rebalance cadence, reversal
   threshold, edge margin, cost multiplier, and max target delta.
3. At every selection contract start, fit V3 only from labels with
   `label_end_index < knowledge_cutoff`. Persist weighted fit identity, forecast
   diagnostics, compiler evidence, and production replay economics.
4. Apply unchanged economic gates: non-negative mean net return, symbol-episode
   lower tail at least `-5%`, turnover no greater than `1.0x/day`, meaningful
   trades, no hard-risk failure, no unexplained execution rejection, and no
   majority-negative gross result. Rank admissible candidates by lower-tail
   net, mean net, turnover, then cost.
5. Freeze the selected V3 candidate and replay each untouched train-symbol
   teacher-admission holdout exactly once. Persist selection, admission, fit,
   batch, and package identities before downstream learning.
6. If and only if admission passes, evaluate random, teacher, BC, DAgger-BC,
   critic warm-start, and anchored residual RL on the same diagnostic segment.
   Record reward, gross/net, baseline excess, cost, turnover, drawdown, command
   changes, and Lagrangian dual mechanics. Promotion remains disabled until
   sealed evaluation succeeds.

## Failure handling

NaN, OOM, identity drift, incomplete checkpoint rows, mixed generator/config
identity, leakage, unexplained execution rejection, or an irreversible economic
gate breach stops the next stage. The runner writes terminal rejection evidence
before returning nonzero. It never relaxes thresholds or changes reward in
response to poor economics.

## Verification

Tests cover cutoff leakage, candidate/config identity, resume identity,
prediction cache reuse, production replay metrics, exact-once holdout access,
admission fail-closed ordering, canonical config non-change, and the admission
guard on DAgger/anchored RL. Real-data evidence is reported separately from
software tests.
