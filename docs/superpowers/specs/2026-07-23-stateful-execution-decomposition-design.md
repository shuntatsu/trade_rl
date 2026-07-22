# Stateful Execution Decomposition Design

## Context

`trade_rl.simulation.stateful_execution.execute_stateful_orders()` is the maintained
stateful execution path behind `MarketExecutor`. It currently spans roughly 614
lines and owns all of the following in one function:

1. request validation and intent submission;
2. split and inactive-asset handling;
3. order expiry, latency, admission, and eligibility transitions;
4. conservative OHLC path selection and trigger evaluation;
5. symbol-level shared-capacity allocation;
6. partial/full fill accounting and event construction;
7. IOC and disabled-carry remainder expiry;
8. dividends, cash interest, funding, borrow, mark-to-market, and margin updates;
9. insolvency cancellation and flattening;
10. interval metrics and `StatefulExecutionResult` construction.

The behavior is extensively tested, but the current ownership makes small changes
hard to review because event ordering, book mutation, liquidity evidence, carry,
and summary counters are interleaved in one scope.

## Non-goals

This refactor does not:

- change the public `MarketExecutor` or `StatefulExecutionResult` contracts;
- change order, event, evidence, accounting, or execution-policy schemas;
- change conservative OHLC path selection or trigger semantics;
- change capacity sharing, participation, rounding, admission, or fee logic;
- change partial-fill carry, IOC, latency, expiry, split, delisting, or insolvency
  behavior;
- add exchange connectivity or production authorization;
- introduce persisted resumable execution phases.

Production remains `NO-GO`.

## Approaches considered

### A. Private helper extraction in the same module

Move blocks into private functions while keeping all mutable locals in
`stateful_execution.py`.

This would reduce the visible function span, but the same module would still own
admission, fills, settlement, event order, and result accounting. The ownership
boundary would remain difficult to test independently.

### B. One stateful processor class

Move the entire function into a `StatefulExecutionProcessor` with many methods.

This would group mutable fields, but it would mostly relocate the monolith. A
single processor would still own unrelated policies and would be easy to grow
back into another hotspot.

### C. Explicit runtime state plus focused phase services

Create one private execution runtime state and four focused services. The public
function remains the only top-level orchestrator and applies the services in the
existing order.

This is the selected approach. It makes mutation ownership explicit, preserves
ordering, and allows each phase to be tested with narrow inputs.

## Selected architecture

### `StatefulExecutionRuntime`

A private mutable runtime object holds only one invocation's state:

- cloned `BookState` and current `OrderBookState`;
- ordered `OrderEvent` and `SymbolCapacityEvidence` collections;
- starting value and rebalance count;
- requested/filled notional aggregates and per-symbol arrays;
- cost, funding, borrow, dividend, cash-interest, participation, fill, rejection,
  expiry, and gross-factor accumulators.

It is created inside `execute_stateful_orders()` and is never persisted or shared
between calls. It owns sequencing of appended events so all phase services use one
monotonic event order.

It provides the unchanged final `StatefulExecutionResult` construction.

### `StatefulBarLifecycle`

Owns processing-bar boundaries:

- split-driven cancellation and quantity adjustment;
- inactive-asset cancellation and delisting settlement;
- open revaluation, drawdown refresh, and gap return;
- dividend and cash-interest application;
- funding and borrow charging;
- mark-to-market and margin refresh;
- insolvency cancellation and flattening before and after carry.

It returns an immutable bar context containing the previous/processing indices,
period-start value, open prices, effective tick/lot/minimum arrays, and gap return.
It mutates only the invocation-local runtime books and order state.

### `StatefulOrderTransitionProcessor`

Owns state transitions that do not consume symbol capacity:

- expiry before processing;
- latency-wait events;
- admission decisions against a projected book;
- eligibility transitions;
- accepted-order collection;
- IOC and disabled partial-fill-carry remainder expiry.

It preserves sorting by `(eligible_index, order_id)`, existing admission reasons,
projected-book quantity/cash updates, and event ordering.

### `StatefulSymbolFillProcessor`

Owns one processing bar's conservative execution:

- orders grouped by symbol;
- active direction collection and OHLC path selection;
- trigger evaluation and configured segment volume fraction;
- trigger events and no-fill evidence;
- execution-price rounding;
- shared symbol-capacity allocation;
- no-fill, partial-fill, and full-fill transitions;
- book execution, margin update, cost calculation, capacity evidence, and metric
  accumulation.

It does not apply dividends, carry, mark-to-market, or terminal flattening.

### `execute_stateful_orders()`

Remains responsible for:

1. validating the invocation;
2. creating the runtime state;
3. submitting new intents in caller order;
4. iterating processing bars;
5. invoking lifecycle begin, transition admission, symbol fills, remainder expiry,
   and lifecycle end in the existing order;
6. returning the runtime's unchanged result.

The function should be short enough to review as an orchestration protocol rather
than a policy implementation.

## Ordering contract

The following order is immutable:

1. add submitted intents and `submitted` events;
2. apply splits and inactive-asset handling;
3. revalue at the processing open;
4. expire, wait, reject, or mark eligible in stable order;
5. select per-symbol paths, evaluate triggers, allocate capacity, and apply fills;
6. expire attempted remainders when required;
7. flatten immediately if fill accounting causes insolvency;
8. compute gap/intrabar gross return;
9. apply dividend, cash interest, funding, borrow, mark-to-market, and margin;
10. flatten if carry or mark-to-market causes insolvency;
11. finalize interval evidence.

No service may reorder events emitted by an earlier phase.

## Compatibility constraints

The refactor must preserve:

- exception types and messages for invalid intervals, shapes, equity, and missing
  contract multipliers;
- caller intent order and active-order stable sorting;
- event sequence values and every `OrderEvent` field;
- path mode, path points, trigger segment, and volume-fraction evidence;
- requested and filled notional, turnover, fill ratio, participation, cost, and
  count calculations;
- `BookState` and `OrderBookState` final values;
- terminal reason resolution;
- deterministic replay digests and promotion evidence;
- existing Linux and Windows behavior.

Private dataclasses may be added. Existing public schemas and serialization names
must not change.

## Error handling

The phase services do not catch or downgrade domain errors. Invalid market data,
non-finite accounting, stale execution-policy identity, admission failures, and
corrupt order state retain current fail-closed behavior.

A service may raise an internal invariant error only when the existing function
would already fail because required multipliers, metadata, or active-order identity
is missing.

## Testing strategy

1. Add architecture contracts before production modules. They require the runtime
   and three phase-service modules, explicit orchestration calls, and a bounded
   `execute_stateful_orders()` source span.
2. Add focused unit tests for lifecycle ordering, admission/latency/expiry,
   trigger/capacity/fill transitions, and runtime result aggregation.
3. Reuse existing conservative execution, replay, promotion, environment,
   accounting-oracle, and Training-Serving parity tests as behavior evidence.
4. Add exact result-equivalence fixtures which compare all scalar, array, book,
   order-book, event, and capacity fields for multi-bar mixed-order scenarios.
5. Measure branch coverage for the new phase services and add a non-regressing
   critical-coverage group without reducing existing thresholds.
6. Run Ruff, format, Mypy, Import Linter, dead-code analysis, full Pytest and branch
   coverage, CLI smoke, Ubuntu/Windows compatibility, training-image verification,
   and PostgreSQL Catalog on the exact final head.

## Pull-request boundary

The pull request must contain only:

- this design and its implementation plan;
- architecture and focused regression tests;
- the invocation-local runtime and focused phase services;
- the orchestration rewrite in `stateful_execution.py`;
- a measured coverage ratchet;
- final verification evidence.

No unrelated telemetry, Studio, environment, model, training, Serving, release, or
catalog change belongs in this refactor.
