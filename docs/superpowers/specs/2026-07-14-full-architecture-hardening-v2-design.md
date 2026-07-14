# Full Architecture Hardening v2 Design

Date: 2026-07-14
Status: Approved for implementation

## Goal

Remove every architecture defect identified in the July 14 audit while preserving the existing research API where safe. The result must make dataset identity complete, execution quantity semantics correct, point-in-time signals enforceable, walk-forward evidence truthful, serving activation fail-closed, and release approval non-circular.

## Chosen approach

Introduce versioned v2 contracts and migrate all maintained workflows to them. Compatibility shims may read old artifacts, but new artifacts and runtime activation must use the v2 contracts. This avoids an unsafe patchwork while keeping the repository incrementally usable.

## 1. Complete market identity

`MarketDataset` will expose a canonical identity computed from all semantic scalars and every resolved array that can affect observations, eligibility, execution, accounting, or evaluation. The identity includes fee schedules, participation limits, quantity increments, borrow and funding data, activity and direction masks, mark/index prices, corporate actions, cash rates, volume units, contract multipliers, calendar semantics, feature availability, feature age, and missing-reason arrays.

A published dataset must carry a canonical identity payload. Publication and loading independently recompute the ID. Arbitrary caller-supplied IDs and identity-less publication are rejected. Range views use a separate `dataset_view_id`; they do not masquerade as full datasets.

## 2. Unified availability and staleness contract

Replace the ambiguous pair of `feature_staleness_hours` and `feature_staleness` with explicit resolved arrays:

- `feature_age_hours`
- `feature_staleness_ratio`
- `feature_available`
- `feature_missing_reason`
- equivalent global-feature arrays

The observation builder reads the resolved fields actually produced by the builder. Missing global observations are marked unavailable rather than encoded as an available zero.

## 3. Instrument-aware quantity accounting

`InstrumentContract` remains the source of volume-unit and multiplier semantics. Dataset helpers convert raw volume to quote notional and convert quantities to quote notional. Executor capacity, requested notional, filled notional, impact, turnover, margin, borrow, settlement, and portfolio valuation use those helpers.

`BookState` stores per-symbol contract multipliers and values positions as `quantity * mark_price * multiplier`. It validates identity-compatible multipliers on clone, restore, and execution. Quote-notional volume is never multiplied by price twice.

## 4. Fold-local causal signals

Signal artifacts become v2 records containing generator configuration/code digests, fit range, prediction range, per-row `available_at`, validity mask, dataset artifact digest, symbol/factor names, and value arrays. Loading verifies that each requested prediction was generated from an authorized fit range and was available by the decision timestamp.

Walk-forward no longer loads one precomputed signal artifact against `train.start`. A `FoldSignalProviderFactory` produces fold-local training/checkpoint/selection/test views. Training signals must be out-of-fold or expanding predictions inside the train range; validation and OOS predictions must come from a model fit only on the authorized training range.

## 5. Capability-separated walk-forward

Each stage receives only a materialized range-scoped dataset capability. Trainer, checkpoint evaluator, configuration selector, and sealed test evaluator cannot access ranges outside their assigned capability. The sealed test evaluator records an access ledger and permits one authorized opening per experiment-plan digest.

`TrainingRunManifest` and `WalkForwardRunManifest` are distinct schemas. Walk-forward manifests record fold plans, signal lineage, selected checkpoints/configurations, sealed-test access records, and result files.

## 6. Truthful evaluation evidence

Fold results preserve returns plus turnover, cost, funding, borrow, dividends, cash interest, fills, rebalance events, participation, termination reason, opening state, and closing state.

Independent folds are reported as a distribution of fold metrics. They are not presented as a continuous-account total return or drawdown. Continuous-account metrics require contiguous ranges and verified state handoff. Gaps are represented explicitly and elapsed-time annualization is used where appropriate.

Bootstrap inference uses a circular or stationary moving-block method with a data-dependent block length, finite-sample p-value correction, and fold-level resampling for independent folds. Gate evidence records observed value, comparator, threshold, evidence digest, and implementation digest; arbitrary booleans cannot authorize release.

## 7. Reproducible normalization and serving

Observation normalization is split by semantics. Exogenous continuous features are fitted on the training range. Masks, categorical values, weights, actions, risk state, and bounded execution state use pass-through or fixed transforms. Normalizer identity includes dataset artifact digest, train range, observation schema/layout digest, and transform parameters.

Serving loads the observation builder contract and normalizer artifact, constructs and transforms observations, then invokes the policy. Activation runs a deterministic probe corpus through every ensemble member and rejects shape, finite-value, bounds, schema, action-name, or normalizer mismatches before swapping live state.

## 8. Non-circular release attestation

A serving candidate bundle is immutable and contains no release digest. Its digest is computed from its files and runtime contracts. A separate `ReleaseAttestation` binds candidate bundle digest, dataset artifact digest, selection/evaluation evidence, gate evidence, git/dependency/runtime provenance, approver identity, and approval time. Registry activation and runtime startup verify the external attestation.

## 9. Complete provenance and portable artifacts

Run manifests automatically capture git commit, dirty status where available, lockfile digest, Python/Torch/SB3/CUDA versions, platform and hardware metadata, deterministic-seed configuration, and content-addressed artifact references. Local filesystem paths are kept in non-identity diagnostics only.

Off-policy training artifacts include replay-buffer state and resume metadata. Algorithm configuration is represented by typed PPO, SAC, TD3, and TQC configs so ignored parameters cannot silently enter a run identity.

## 10. Session-market and cross-asset support

Dataset build configuration supports continuous and named session calendars. Session datasets derive elapsed time from timestamps. Borrow, cash interest, latency, feature age, and annualization use elapsed-time semantics rather than assuming one equal-duration bar. CLI dataset building accepts calendar and instrument-contract configuration.

## 11. Artifact closure and filesystem safety

Dataset, signal, run, bundle, and release validators enforce an exact allow-list of files, reject undeclared files, reject symlinks and root escapes, validate canonical serialization, and recompute all content identities on load.

## 12. Risk architecture

Retain deterministic pre-trade projection, then add an optional portfolio risk layer for volatility targeting, liquidity-adjusted caps, correlation/factor concentration, beta/net exposure, and scenario stress limits. The v2 interface is pluggable and its complete config and implementation digest are included in environment identity.

## Error handling

All identity, causality, schema, range, quantity, and activation mismatches fail closed with typed `ValueError` or domain-specific validation errors. Economic termination remains separate from code/configuration errors. No validator repairs or silently defaults missing v2 evidence.

## Migration

Old v1/v3 artifacts remain readable only through explicit legacy loaders. Maintained CLI workflows produce v2 artifacts. Legacy action acceptance and ambiguous staleness fields are deprecated and excluded from production release eligibility.

## Verification

Implementation is complete only when:

1. Unit and property tests cover identity mutation for every semantic field.
2. Quantity-accounting tests cover base, quote-notional, and contracts volume.
3. Leakage tests prove stage capabilities cannot escape assigned ranges.
4. Signal tests reject future fit/prediction data and delayed availability.
5. Independent-fold metrics cannot expose continuous-account statistics.
6. Serving activation rejects bad normalizers, member shapes, bounds, and undeclared files before swap.
7. Release attestation verifies without circular hashes.
8. End-to-end CI executes CSV/config build, dataset publish/load, fold-local signal generation, PPO training, checkpoint selection, sealed evaluation, bundle creation, attestation, registry activation, and runtime prediction.
9. Critical branch coverage is at least 90% for artifacts, serving, execution, evaluation, and release code.

## Scope boundary

This hardening establishes a trustworthy research-to-paper-serving architecture. Direct exchange order routing, secrets management, production websocket operations, and broker reconciliation remain a later live-trading integration phase, but the current CLI must report those capabilities as unavailable rather than implying production readiness.
