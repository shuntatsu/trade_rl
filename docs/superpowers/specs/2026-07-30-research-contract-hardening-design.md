# Research Contract Hardening Design

## Goal

Harden the maintained research path so that PostgreSQL and Binance Vision produce the same economic execution semantics, Oracle behavior cloning cannot pass on negligible causal evidence, invalid datasets are rejected before immutable publication, raw Vision cache bytes are content-attested, hierarchical Gate semantics are described and measured as change intensity, and maintained documentation matches config v3 and serving export v2.

## Scope and order

1. Common economic-array contract for PostgreSQL and Vision.
2. Strong causal holdout gate for Oracle BC.
3. Dataset preset validation before publication.
4. Content-attested Binance Vision cache.
5. Change-intensity terminology and effective-action telemetry.
6. README, ARCHITECTURE and CONFIGURATION updates.
7. Audit-language corrections for data ownership, constraint-cost semantics, Quickstart defaults, PR C history, workflow security wording, serving schema and offline signing boundaries.

## Architecture

### Economic semantics

Create `trade_rl.data.economic_semantics` as the canonical constructor for all two-dimensional economic arrays consumed by `MarketDataset`. The returned frozen contract includes active/tradable/information masks, availability timestamps, tick/lot/minimum-notional histories, fee and spread arrays, participation, borrowing permissions and rates, buy/sell permissions, and mark/index prices.

The constructor consumes canonical `InstrumentContract` objects plus explicit source observations. PostgreSQL metadata must first be converted into the same `InstrumentContract` representation used by Vision. Missing required economic metadata is rejected instead of falling through to `MarketDataset` defaults. Both dataset paths call this constructor and pass every resulting array explicitly.

A parity test builds equivalent Vision and PostgreSQL fixtures and compares every economic array.

### Oracle BC gate

Extend the causal holdout evaluation with deterministic bootstrap evidence over per-episode regret. Config v3 adds an explicit bootstrap confidence level and resample count. The mandatory gate requires:

- positive composed-loss relative improvement;
- at least 30 causal holdout trades for maintained Oracle profiles;
- finite point regret within the configured limit;
- one-sided bootstrap upper confidence bound within the same limit;
- existing reconstruction precision, recall, RMSE and activity checks.

The bootstrap seed is derived from immutable evaluation identity, so the gate is reproducible. Maintained target-weight profiles use non-zero relative improvement and non-trivial support.

### Publication order

Extract a pure `validate_maintained_dataset_preset()` helper. `_build_dataset()` performs build, validation, publication, then report construction. A test proves publication is not called for a structurally valid dataset with the wrong bar, symbol or feature contract.

### Raw cache evidence

Each cached Vision payload has an adjacent canonical JSON sidecar containing schema, URL, SHA-256, byte length, acquisition time, downloader identity, ETag and Last-Modified when available. Cache reads require both files and re-hash bytes before returning. Legacy payload-only cache entries fail closed and must be reacquired.

### Gate semantics and telemetry

The hierarchical scalar remains continuous and is renamed in documentation and metrics to `change_intensity`; checkpoint compatibility fields remain unchanged. Telemetry records deterministic composed action, sampled policy action, post-mask action, submitted target, and filled effective weights. The reported effective-change rate is calculated after exploration and downstream processing, so a low change intensity cannot be misread as a hard no-trade event.

### Documentation and audit wording

Documentation states that causal data contracts span contracts, raw series, datasets and sequence observations; constraint costs are separate from scalar reward but are not all hard constraints; Quickstart pins its hybrid reward values explicitly; PR C’s maintained replacement is #193; workflow security validates runner classification, permissions and triggers; structured export is v2; and private-key loading/signing is confined to explicit offline modules with an import-linter dependency boundary rather than an OS-level impossibility claim.

## Error handling

All new integrity checks fail closed with field-specific errors. No compatibility fallback silently fills economic arrays, accepts payload-only Vision caches, publishes an invalid maintained preset, or downgrades a missing BC confidence bound.

## Testing

Add focused unit tests for economic contract construction and cross-source parity, BC support and confidence-bound failures, publication ordering, cache tampering and sidecar absence, action-stage telemetry, explicit Quickstart rewards, documentation schema values, workflow wording, and offline signer boundary wording. Existing full pytest, Ruff, MyPy, import-linter, documentation contracts, serving smoke and package smoke remain mandatory.