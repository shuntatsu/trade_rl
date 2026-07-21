# Conservative OHLC Order Simulator Design

## Status

Approved for implementation design on 2026-07-21.

## Goal

Replace direct per-bar target filling with an explicit, persistent order lifecycle that remains usable with OHLCV data while making fills materially more conservative, auditable, and resistant to policy exploitation.

The simulator must preserve the existing accounting boundary and research-only posture. It must not introduce exchange connectivity or imply that OHLCV can reproduce a true order book.

## Current Problem

The current executor converts a target directly into desired quantities and repeatedly fills toward those quantities. It uses the next bar open as the execution price but uses the preceding bar volume for capacity. Limit orders are treated as filled when the bar low or high touches the limit, without modeling order state, time in force, trigger timing, residual quantity, or bar-path ambiguity.

This creates four research risks:

1. Liquidity is measured from a different bar than the execution event.
2. A limit touch can consume an unrealistically large portion of the whole bar volume.
3. Unfilled quantity is implicit rather than represented as a durable order state.
4. A policy can learn the deterministic weaknesses of an OHLC touch rule instead of a robust trading strategy.

## Design Principles

- Fail closed when order state, prices, quantities, or evidence are invalid.
- Preserve `BookState` as the authoritative accounting object.
- Keep order lifecycle, bar-path interpretation, liquidity allocation, and accounting as separate components.
- Use the actual processing bar's volume for capacity.
- Treat OHLC path assumptions as explicit evidence, never as recovered truth.
- Use deterministic execution for reproducible research.
- Require conservative execution assumptions for final promotion.
- Keep the current executor API usable during migration through an adapter layer.

## Scope

This design includes:

- market, limit, and stop-market orders;
- explicit submitted, eligible, triggered, partially filled, filled, rejected, expired, and cancelled states;
- latency before an order becomes eligible;
- time-in-force expiry;
- cancel-and-replace when a new target supersedes an existing residual order;
- deterministic optimistic, neutral, and conservative OHLC path modes;
- gap-aware execution prices;
- current-bar liquidity and participation limits;
- reduced available volume when an order becomes executable partway through a bar;
- residual quantity carried across bars;
- deterministic rejection rules;
- order-level evidence and aggregate interval evidence;
- promotion rules requiring conservative mode.

The following are explicitly out of scope:

- historical L2 reconstruction;
- queue position inferred from unavailable data;
- hidden liquidity, iceberg orders, auctions, and exchange-specific matching priority;
- real exchange routing;
- stochastic fills that cannot be replayed from a seed and evidence record;
- claiming that the simulator is equivalent to live execution.

## Architecture

### 1. Order Domain Model

Add a focused order module under `trade_rl/simulation/orders.py`.

#### `OrderType`

Values:

- `market`
- `limit`
- `stop_market`

#### `TimeInForce`

Values:

- `ioc`: execute once on the first eligible bar, then expire any remainder;
- `day`: expire at a configured expiry index;
- `gtc`: persist until filled, cancelled, superseded, or the execution interval ends.

`day` is index-based because the dataset may contain arbitrary bar durations. The caller resolves a timestamp policy into an explicit expiry index before submission.

#### `OrderStatus`

Values:

- `submitted`
- `latency_wait`
- `eligible`
- `triggered`
- `partially_filled`
- `filled`
- `rejected`
- `expired`
- `cancelled`

Terminal statuses are `filled`, `rejected`, `expired`, and `cancelled`.

#### `OrderIntent`

Immutable submission request containing:

- deterministic `order_id`;
- symbol index;
- signed requested quantity;
- order type;
- optional limit price;
- optional stop price;
- submit index;
- eligible index;
- optional expiry index;
- time in force;
- target-generation identity;
- execution-policy digest.

Validation rules:

- quantity must be finite and non-zero;
- symbol index must exist;
- buy and sell direction are derived from quantity sign;
- limit orders require exactly one valid limit price;
- stop-market orders require exactly one valid stop price;
- market orders must not carry limit or stop prices;
- eligible index must not precede submit index;
- expiry index, when present, must not precede eligible index.

#### `PendingOrder`

Mutable runtime state containing:

- original `OrderIntent`;
- remaining signed quantity;
- cumulative filled quantity;
- cumulative execution notional;
- current status;
- trigger state and trigger index;
- last processed index;
- cancellation, rejection, or expiry reason;
- version number for evidence ordering.

The invariant is:

`requested_quantity = cumulative_filled_quantity + remaining_quantity`

within the configured quantity tolerance.

### 2. Target-to-Order Reconciliation

Add `TargetOrderReconciler` under `trade_rl/simulation/order_reconciliation.py`.

At each decision point:

1. Convert target weights into desired quantities using decision equity and the next eligible reference price.
2. Calculate the desired net delta against current holdings.
3. Include the signed remaining quantity of all active orders when computing outstanding exposure.
4. Cancel active orders that no longer move the book toward the latest target.
5. Create replacement orders only for the residual delta required by the latest target.

The reconciler must prevent double counting. A partially filled order and its replacement must never both represent the same residual exposure.

Default behavior is cancel-and-replace per symbol when a new decision target arrives. The cancelled order retains immutable evidence and the replacement receives a new deterministic order ID linked by `replaces_order_id`.

### 3. Bar Path Model

Add `trade_rl/simulation/bar_path.py`.

A bar path is represented as an ordered sequence of price points beginning at open and ending at close. It is an execution assumption, not a reconstructed fact.

#### `optimistic`

Choose the extreme ordering that is most favorable to the order being evaluated. This mode is sensitivity-only and cannot support model promotion.

#### `neutral`

Use a deterministic order-independent path:

- if the low is closer to open than the high, use `open -> low -> high -> close`;
- otherwise use `open -> high -> low -> close`;
- ties use `open -> low -> high -> close`.

This avoids using order direction to choose the path.

#### `conservative`

Choose the extreme ordering that is least favorable to the order:

- buy limit: favorable downward touch is delayed behind the adverse high;
- sell limit: favorable upward touch is delayed behind the adverse low;
- buy stop: upward trigger is followed by the most adverse reachable price before close;
- sell stop: downward trigger is followed by the most adverse reachable price before close.

For multiple active orders in one symbol, path selection must not be independently optimized for each order. The engine derives one conservative path per symbol and bar from the aggregate active order directions. When opposing orders coexist, the fixed neutral path is used and the evidence records `mixed_direction_fallback`.

### 4. Trigger and Gap Rules

#### Market

A market order fills at the first eligible bar open, subject to rejection, tradability, capacity, rounding, and participation constraints.

#### Buy limit

- If open is at or below the limit, execution begins at open.
- Otherwise, the order becomes executable only when the selected path reaches the limit.
- The execution price is no better than the limit and is rounded adversely according to market rules.

#### Sell limit

- If open is at or above the limit, execution begins at open.
- Otherwise, the order becomes executable only when the selected path reaches the limit.
- The execution price is no better than the limit and is rounded adversely.

#### Buy stop-market

- If open is at or above the stop, trigger at open and execute at open plus configured adverse market impact.
- Otherwise, trigger when the path reaches the stop.
- After triggering, execution uses a market-style adverse price bounded by reachable path prices and configured slippage.

#### Sell stop-market

- If open is at or below the stop, trigger at open and execute at open minus configured adverse market impact.
- Otherwise, trigger when the path reaches the stop.
- After triggering, execution uses a market-style adverse price bounded by reachable path prices and configured slippage.

A stop-market order remains triggered across later bars until filled, cancelled, rejected, or expired.

### 5. Available Volume by Trigger Position

Capacity must use `dataset.volume[processing_index]`, not the previous bar's volume.

Available volume is reduced according to when the order becomes executable in the selected path.

Default deterministic fractions:

- executable at open: `1.00`;
- executable on first extreme segment: `0.50`;
- executable on second extreme segment: `0.25`;
- executable only at close: `0.00`.

The fractions are configurable but must be monotonic, finite, and within `[0, 1]`. Final promotion uses the default conservative fractions unless a stricter profile is selected.

For multiple orders competing for the same symbol and bar, the symbol-level capacity is allocated in deterministic order:

1. previously triggered stop-market residuals;
2. market orders;
3. newly triggered stop-market orders;
4. older limit orders;
5. newer limit orders.

Within a class, sort by eligible index and then order ID. This is not an exchange queue model; it is a stable conservative allocation rule.

### 6. Liquidity and Partial Fills

Add `LiquidityAllocator` under `trade_rl/simulation/liquidity.py`.

For each symbol and processing bar:

1. Convert current-bar volume to notional using the candidate execution price and dataset contract rules.
2. Apply dataset and configured maximum participation rates.
3. Multiply by the trigger-position available-volume fraction.
4. Subtract capacity already consumed by higher-priority orders in the same symbol and bar.
5. Fill no more than the order's remaining notional.
6. Convert filled notional to quantity and apply lot-size rounding.
7. Recompute exact notional from the rounded quantity.

If the rounded fill falls below minimum notional, no fill occurs and the evidence records `below_minimum_notional_after_rounding`.

Residual quantity remains on the `PendingOrder`. It is reevaluated on future bars until terminal.

### 7. Deterministic Rejection

Add an `OrderAdmissionPolicy` that evaluates rejection before capacity allocation.

Required rejection reasons include:

- asset inactive;
- market not tradable;
- buy or sell direction disabled;
- borrow unavailable for incremental short exposure;
- invalid tick, lot, or minimum-notional rule;
- order quantity becomes zero after rounding;
- insufficient leverage or margin under configured pre-trade checks;
- order expired before first eligibility;
- invalid state transition.

Random rejection is out of scope for the initial implementation. A later sensitivity overlay may introduce seeded stochastic rejection without changing the order state contract.

### 8. Time in Force and Interval Boundaries

- `ioc`: after the first eligible processing attempt, any residual expires.
- `day`: expires before processing a bar whose index is greater than the explicit expiry index.
- `gtc`: remains active across bars and across calls only when an `OrderBookState` is supplied by the caller.

The existing `execute_interval` compatibility path is self-contained. At interval end, active orders are cancelled with `interval_end` unless the caller uses the new stateful API.

The new stateful API returns both `ExecutionResult` and `OrderBookState`, allowing the RL environment to carry pending orders between decisions.

### 9. Stateful Execution API

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

`StatefulExecutionResult` includes:

- updated `BookState`;
- updated `OrderBookState`;
- interval accounting metrics;
- order events;
- symbol-level capacity evidence;
- execution-policy digest;
- selected path mode;
- exact processed range.

The existing `execute_interval(book, target, ...)` remains as an adapter during migration. It creates intents through the reconciler, executes them through the new engine, and cancels residual orders at interval end. Once all maintained callers use stateful execution, the adapter may be deprecated in a separate change.

### 10. Accounting Boundary

`BookState` remains responsible for:

- cash;
- quantities;
- position values;
- portfolio value;
- fees and execution cost charging;
- funding;
- borrow cost;
- dividends;
- cash interest;
- splits;
- delisting settlement;
- margin state and economic termination.

The order simulator supplies only exact rounded quantity deltas, fill prices, costs, and evidence. It must not reimplement portfolio accounting.

The independent accounting oracle introduced in P0 remains the verification authority for economic state changes.

### 11. Evidence

Add immutable `OrderEvent` records with:

- schema version;
- event sequence;
- order ID and replacement linkage;
- symbol;
- event type;
- processing index and timestamp;
- prior and new status;
- requested, remaining, and filled quantities;
- execution price and notional when applicable;
- capacity before and after allocation;
- participation rate;
- trigger segment and available-volume fraction;
- rejection, cancellation, or expiry reason;
- bar-path mode and path points;
- execution-policy digest;
- dataset ID.

`ExecutionResult` gains aggregate counts for submitted, rejected, expired, cancelled, partial, and full fills. Detailed events are returned separately to avoid making the summary object excessively large.

Evidence ordering must be deterministic and serializable to canonical JSON.

### 12. Promotion Gate

Selected-final, sealed-test, Serving package, and release evidence must reject results unless:

- path mode is `conservative`;
- current-bar volume capacity is enabled;
- partial-fill carry is enabled;
- order evidence is complete;
- execution-policy digest matches the experiment plan;
- no optimistic-only result is used as primary evidence.

Neutral and optimistic modes remain available for sensitivity analysis and development diagnostics.

### 13. Error Handling

The engine raises explicit domain errors for:

- invalid order construction;
- impossible state transitions;
- duplicate active order IDs;
- inconsistent fill arithmetic;
- negative remaining quantity in order direction;
- non-finite prices, quantities, costs, or capacity;
- capacity over-allocation;
- execution outside dataset bounds;
- policy digest mismatch;
- unsupported legacy configuration combinations.

Economic inability to fill is not an exception. It produces a valid no-fill, rejection, expiry, or cancellation event.

### 14. Backward Compatibility

The existing `ExecutionCostConfig.order_type` and `limit_offset_rate` remain temporarily supported by the target adapter.

Mapping:

- `market` creates IOC market intents after configured latency;
- `limit` creates day-style limit intents using the existing offset and interval end as expiry.

New configuration fields must have defaults preserving executable behavior, but maintained final-evaluation recipes will explicitly select conservative stateful execution.

Manifest and report schemas receive versioned additions rather than silent semantic changes.

### 15. Testing Strategy

Implementation is test-driven.

#### Domain tests

- construction validation for every order type and TIF;
- all valid and invalid state transitions;
- invariant preservation after partial fills and replacements;
- deterministic order IDs and canonical evidence serialization.

#### Bar-path tests

- all neutral path permutations;
- conservative buy/sell limit paths;
- conservative buy/sell stop paths;
- mixed-direction fallback;
- open gap, first-segment, second-segment, and close-only triggers.

#### Liquidity tests

- processing-bar volume is used rather than prior-bar volume;
- trigger fractions 1.00, 0.50, 0.25, and 0.00;
- capacity shared across multiple orders without over-allocation;
- lot-size and minimum-notional effects;
- residual quantity carried across bars;
- deterministic priority ordering.

#### Lifecycle tests

- latency wait to eligibility;
- IOC residual expiry;
- day expiry;
- GTC persistence in the stateful API;
- cancel-and-replace after a changed target;
- stop trigger persistence;
- interval-end cancellation in the compatibility adapter.

#### Gap and price tests

- favorable and adverse limit gaps;
- stop gaps through the trigger;
- adverse tick rounding;
- execution price never violates limit protection;
- stop-market price remains within the modeled reachable path plus configured market impact.

#### Accounting integration

- every fill is checked against the independent accounting oracle;
- fees, partial fills, splits, delisting, funding, borrow, and margin termination remain correct;
- exact log reward remains consistent with final equity.

#### Environment integration

- pending-order state appears in observations without future information;
- Training–Serving parity includes pending orders and previous fills;
- target changes cannot double-submit residual exposure;
- deterministic replay produces identical actions, events, and rewards.

#### Sensitivity and promotion

- optimistic performance cannot promote a model;
- conservative performance is primary evidence;
- fee, spread, slippage, capacity, delay, path, and TIF sensitivity are identity-bound;
- incomplete order evidence fails closed.

#### Full verification

- `ruff check .`;
- `ruff format --check .`;
- `mypy .`;
- full `pytest -q` with branch coverage;
- PostgreSQL integration where order evidence is persisted;
- research-to-serving E2E;
- Docker training-image build and non-root probe.

## Migration Sequence

1. Add order domain types and state-transition tests.
2. Add bar-path and trigger engine.
3. Add current-bar liquidity allocator and partial-fill carry.
4. Add admission, rejection, expiry, and cancellation rules.
5. Add stateful execution API while retaining the old adapter.
6. Migrate `ResidualMarketEnv` to carry `OrderBookState`.
7. Extend observation parity and evidence schemas.
8. Add conservative promotion requirements.
9. Run independent oracle, sensitivity, full CI, PostgreSQL, Serving, and Docker verification.
10. Run a small deterministic multi-seed smoke before any full-scale training.

## Success Criteria

The design is complete when all of the following hold:

- no execution path uses preceding-bar volume for a fill occurring in the processing bar;
- all residual exposure is represented by explicit pending orders;
- partial fills persist correctly without double submission;
- market, limit, and stop-market gap rules are deterministic and tested;
- bar-path assumptions and available-volume fractions are present in evidence;
- conservative mode is required for promotion;
- existing accounting and reward invariants remain unchanged;
- Training–Serving parity includes pending-order state;
- full static, unit, integration, E2E, PostgreSQL, and Docker verification passes on one exact head SHA;
- a smoke training run completes without order-state divergence or nondeterminism.
