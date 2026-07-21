# Conservative OHLC Order Simulator Design

## Status

Approved on 2026-07-21 and self-reviewed for implementation planning.

## Goal

Replace direct per-bar target filling with an explicit order lifecycle that works with OHLCV data while making fills conservative, deterministic, auditable, and harder for an RL policy to exploit.

The simulator preserves `BookState` as the accounting authority. It does not add exchange connectivity and does not claim that OHLCV reproduces a real order book.

## Current Problem

The current executor fills toward target quantities, executes at the next bar open, and uses the preceding bar volume for capacity. Limit orders fill when the bar low or high touches the limit, without durable order state, time in force, trigger timing, residual quantity, or an explicit bar-path assumption.

This creates four research risks:

1. liquidity comes from a different bar than the fill;
2. a late limit touch can consume an unrealistic share of the whole bar volume;
3. residual exposure is implicit and can be resubmitted incorrectly;
4. a policy can learn the OHLC touch rule instead of a robust strategy.

## Principles

- Fail closed on invalid state, arithmetic, identity, or evidence.
- Separate order lifecycle, path interpretation, liquidity allocation, and accounting.
- Use the processing bar's volume for fills in that bar.
- Fix requested quantity using information known at submission time.
- Treat OHLC paths as explicit assumptions, never recovered truth.
- Keep execution deterministic and replayable.
- Require conservative execution for final promotion.
- Preserve the current executor API through a migration adapter.

## Scope

Included:

- market, limit, and stop-market instructions;
- submitted, latency-wait, eligible, triggered, partially-filled, filled, rejected, expired, and cancelled states;
- latency, time in force, cancel-and-replace, partial-fill carry, gap rules, deterministic rejection, and order evidence;
- optimistic, neutral, and conservative OHLC path modes;
- current-bar capacity and reduced available volume for late triggers;
- stateful execution across RL decisions;
- promotion gates requiring conservative evidence.

Excluded:

- L2 reconstruction and inferred exchange queue position;
- hidden liquidity, auctions, iceberg priority, and venue-specific matching;
- real order routing;
- non-replayable stochastic fills;
- any claim that the model equals live execution.

## Architecture

### Order domain

Create `trade_rl/simulation/orders.py`.

`OrderType`:

- `market`
- `limit`
- `stop_market`

`TimeInForce`:

- `ioc`: one eligible processing attempt, then expire the remainder;
- `day`: expire after an explicit dataset index;
- `gtc`: persist until filled, cancelled, superseded, interval termination, or dataset termination.

A persistent market instruction with `gtc` represents a parent execution instruction split over bars, not one literal exchange market order. The legacy target adapter uses IOC market children by default.

`OrderStatus`:

- `submitted`
- `latency_wait`
- `eligible`
- `triggered`
- `partially_filled`
- `filled`
- `rejected`
- `expired`
- `cancelled`

Terminal statuses are filled, rejected, expired, and cancelled.

`OrderIntent` is immutable and contains:

- deterministic order ID;
- symbol index;
- signed requested quantity;
- order type and time in force;
- optional limit or stop price;
- submit, eligible, and optional expiry indices;
- submission reference price and decision equity;
- target-generation identity;
- execution-policy digest;
- optional replaced-order ID.

The requested quantity is fixed at submission using decision-time equity and the latest causally available mark or close. It must not be recalculated from a future eligible-bar open. Latency changes when the order may execute, not how large the submitted order becomes.

Validation requires finite non-zero quantity, valid symbol, valid type-specific prices, causal submission inputs, eligible index not before submit index, and expiry not before eligibility.

`PendingOrder` stores runtime state:

- immutable intent;
- signed remaining quantity;
- cumulative filled quantity and notional;
- current status;
- trigger state and trigger index;
- last processed index;
- terminal reason;
- monotonic evidence version.

Invariant:

`requested_quantity = cumulative_filled_quantity + remaining_quantity`

within the configured quantity tolerance.

### Target-to-order reconciliation

Create `trade_rl/simulation/order_reconciliation.py`.

At each decision:

1. Use decision-time equity and the current causally available price to derive desired quantities.
2. Compare desired quantities with current holdings plus active signed residual quantities.
3. Cancel active orders that no longer move exposure toward the latest target.
4. Submit only the residual delta still required.
5. Link replacements to cancelled order IDs.

Default behavior is cancel-and-replace per symbol. A partial fill and its replacement may not both represent the same residual exposure.

### Bar-path model

Create `trade_rl/simulation/bar_path.py`.

A path is an ordered sequence beginning at open and ending at close. It is recorded as an assumption.

`optimistic` chooses the ordering favorable to the evaluated order and is sensitivity-only.

`neutral` is order independent:

- use `open -> low -> high -> close` when low is closer to open;
- otherwise use `open -> high -> low -> close`;
- ties use low first.

`conservative` delays favorable limit touches behind adverse movement and applies adverse post-trigger movement to stops.

One path is selected per symbol and bar. The engine may not choose a separate favorable or adverse path for every order. When active directions conflict, use the neutral path and record `mixed_direction_fallback`.

### Trigger and gap rules

Market:

- first eligible processing begins at the bar open;
- fill remains subject to admission, capacity, rounding, participation, and cost.

Buy limit:

- open at or below limit: executable at open and priced at open;
- otherwise executable only when the selected path reaches the limit;
- price may not exceed the limit after adverse tick rounding.

Sell limit:

- open at or above limit: executable at open and priced at open;
- otherwise executable only when the selected path reaches the limit;
- price may not fall below the limit after rounding.

Buy stop-market:

- open at or above stop: trigger at open and execute as an adverse market fill;
- otherwise trigger when the path reaches stop;
- execution uses a market-style adverse price bounded by the modeled reachable path and configured impact.

Sell stop-market is symmetric.

A triggered stop remains triggered across later bars until terminal.

### Trigger position and available volume

Capacity uses `dataset.volume[processing_index]`.

Default available-volume fractions:

- executable at open: `1.00`;
- first extreme segment: `0.50`;
- second extreme segment: `0.25`;
- close only: `0.00`.

Fractions are configurable but must be finite, monotonic, and within `[0, 1]`. Final promotion uses these conservative defaults or stricter values.

Orders sharing a symbol and bar consume one symbol-level capacity pool in deterministic priority:

1. previously triggered stop residuals;
2. market instructions;
3. newly triggered stops;
4. older limits;
5. newer limits.

Within a class, sort by eligible index and order ID. This is an explicit allocation convention, not a claim about exchange queue priority.

### Liquidity allocator

Create `trade_rl/simulation/liquidity.py`.

For each symbol and processing bar:

1. Convert current-bar volume to notional with the candidate execution price and contract rules.
2. Apply dataset and configured participation limits.
3. Apply the trigger-position volume fraction.
4. Subtract capacity consumed by higher-priority orders.
5. Cap the fill at remaining order notional.
6. Convert to quantity and apply lot-size rounding.
7. Recompute exact notional from rounded quantity.

If the rounded result falls below minimum notional, record a no-fill reason. Residual quantity remains pending for a later bar when time in force permits.

The allocator must prove that total filled notional never exceeds the symbol-level capacity pool.

### Admission and rejection

Create `OrderAdmissionPolicy`.

Required deterministic rejection reasons:

- inactive asset;
- non-tradable market;
- disabled buy or sell direction;
- unavailable borrow for incremental short exposure;
- invalid tick, lot, or minimum-notional rule;
- zero quantity after rounding;
- failed pre-trade leverage or margin check;
- expired before first eligibility;
- identity mismatch;
- invalid state transition.

A normal inability to fill is represented as no-fill, rejection, expiry, or cancellation evidence rather than an exception.

Seeded stochastic rejection is deferred to a later sensitivity overlay.

### Time in force and state persistence

- IOC expires residual quantity after its first eligible processing attempt.
- Day expires before processing an index later than its explicit expiry index.
- GTC persists only through the stateful execution API.

The compatibility `execute_interval` path remains self-contained and cancels active residuals at interval end with reason `interval_end`.

The RL environment uses the stateful API and carries pending orders between decisions.

### Stateful API

Add:

```python
execute_orders(
    book: BookState,
    order_book: OrderBookState,
    intents: Sequence[OrderIntent],
    *,
    start_index: int,
    bars: int,
) -> StatefulExecutionResult
```

The result contains:

- updated `BookState`;
- updated `OrderBookState`;
- interval accounting metrics;
- ordered `OrderEvent` evidence;
- symbol capacity evidence;
- execution-policy digest;
- selected path mode;
- exact processed range.

The existing `execute_interval(book, target, ...)` becomes an adapter. It converts targets to intents, executes through the new engine, and cancels residual orders at interval end. Deprecation is a separate future change after maintained callers migrate.

### Accounting boundary

`BookState` remains authoritative for cash, quantities, valuation, fees, funding, borrow, dividends, cash interest, splits, delisting settlement, margin, and economic termination.

The order simulator supplies only rounded quantity deltas, fill prices, costs, and evidence. It does not duplicate accounting equations.

The independent P0 accounting oracle remains the verification authority.

### Evidence

Immutable `OrderEvent` records include:

- schema version and event sequence;
- order ID and replacement linkage;
- symbol and event type;
- processing index and timestamp;
- previous and new status;
- requested, remaining, and filled quantities;
- execution price and notional;
- capacity before and after allocation;
- participation rate;
- trigger segment and available-volume fraction;
- rejection, cancellation, expiry, or no-fill reason;
- path mode and path points;
- execution-policy digest and dataset ID.

Events are deterministically ordered and canonically serializable.

`ExecutionResult` gains aggregate counts for submitted, rejected, expired, cancelled, partial, and complete fills. Detailed events remain separate from the compact summary.

### Promotion gate

Selected-final, sealed-test, Serving package, and release evidence must reject execution results unless:

- path mode is conservative;
- processing-bar volume capacity is enabled;
- partial-fill carry is enabled;
- order evidence is complete;
- execution-policy digest matches the experiment plan;
- optimistic-only results are not used as primary evidence.

Neutral and optimistic modes remain development and sensitivity tools.

## Error Handling

Raise explicit domain errors for invalid construction, impossible transitions, duplicate active IDs, inconsistent fill arithmetic, directionally negative residuals, non-finite values, capacity over-allocation, dataset-range violations, policy digest mismatch, and unsupported legacy combinations.

Economic non-fill conditions produce evidence and continue safely.

## Backward Compatibility

Temporarily preserve `ExecutionCostConfig.order_type` and `limit_offset_rate` through the target adapter:

- `market` creates IOC market children after configured latency;
- `limit` creates interval-expiring limit instructions using the configured offset.

New manifest and report fields are versioned additions. Maintained final-evaluation recipes explicitly select conservative stateful execution instead of relying on defaults.

## Testing Strategy

Implementation is test-driven.

Domain tests cover construction, state transitions, invariants, replacement linkage, deterministic IDs, and canonical serialization.

Path tests cover neutral permutations, conservative buy and sell limits, conservative stops, mixed-direction fallback, and open/first-segment/second-segment/close triggers.

Liquidity tests prove use of processing-bar volume, all trigger fractions, shared capacity without over-allocation, lot and minimum-notional effects, residual carry, and deterministic priority.

Lifecycle tests cover latency, IOC expiry, day expiry, GTC persistence, cancel-and-replace, stop trigger persistence, and interval-end cancellation.

Price tests cover favorable and adverse gaps, stop gaps, adverse tick rounding, limit protection, and stop-market reachable-price bounds.

Accounting integration compares every fill path with the independent oracle and preserves fees, splits, delisting, funding, borrow, margin, equity, and exact log reward.

Environment integration proves pending-order observations are causal, Training-Serving parity includes order state, changed targets cannot double-submit residual exposure, and replay is deterministic.

Promotion tests prove optimistic evidence cannot promote a model and incomplete evidence fails closed.

Full verification includes Ruff, format, `mypy .`, full pytest with branch coverage, PostgreSQL evidence persistence, research-to-serving E2E, Docker training-image build, and non-root probe.

## Migration Sequence

1. Add domain types and transition tests.
2. Add bar-path and trigger engine.
3. Add current-bar liquidity allocation and partial-fill carry.
4. Add admission, rejection, expiry, and cancellation.
5. Add stateful API while retaining the adapter.
6. Migrate `ResidualMarketEnv` to carry `OrderBookState`.
7. Extend observations, parity, manifests, and evidence.
8. Enforce conservative promotion.
9. Run oracle, sensitivity, full CI, PostgreSQL, Serving, and Docker verification.
10. Run a deterministic small multi-seed smoke before full-scale training.

## Success Criteria

- No fill uses preceding-bar volume for processing-bar capacity.
- Submitted quantity is fixed using only decision-time information.
- All residual exposure is represented by pending orders.
- Partial fills persist without duplicate submission.
- Market, limit, and stop gap rules are deterministic and tested.
- Path assumptions and volume fractions appear in evidence.
- Conservative mode is required for promotion.
- Accounting and reward invariants remain unchanged.
- Training-Serving parity includes pending-order state.
- One exact head passes static, unit, integration, E2E, PostgreSQL, and Docker verification.
- A smoke training run completes without state divergence or nondeterminism.
