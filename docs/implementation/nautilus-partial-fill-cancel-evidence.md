# Nautilus Partial-Fill and Stale-Working Cancellation Plan

## Goal

Verify the already-approved single-instrument Nautilus migration safety contract when an actual working order is partially filled and a newer target makes the remaining quantity stale.

## Scope

This slice does **not** add Limit/GTC as a maintained Trade RL child-order type. Maintained target-exposure replacement and flattening continue to use the existing Market IOC adapter. A passive GTC limit exists only inside the exact-wheel test fixture so NautilusTrader can produce a real working remainder.

## Contract

1. Run the pinned `nautilus_trader==1.230.0` low-level BacktestEngine with `liquidity_consumption=True`.
2. Seed one passive BUY GTC limit after the initial L1 quote.
3. Move the next quote through the limit with only part of the order quantity displayed. The order must become partially filled and retain a working remainder.
4. Feed `realized_quantity + working_remaining_quantities` into `TargetExposureController` with a changed target. The controller must return `CANCELING_STALE` with no replacement child order.
5. Execute that plan through `submit_target_exposure_plan`, which must cancel the stale working order without submitting a replacement.
6. Replacement submission is allowed only on a later quote after `OrderCanceled` terminal evidence has been observed.
7. The replacement uses the existing Market IOC adapter. A later target-to-zero plan closes the resulting position with a reduce-only Market IOC child order.
8. Final evidence requires no open orders and no open position.

## Why this fixture

NautilusTrader documents that L1 passive limit orders with `liquidity_consumption=True` fill only against displayed liquidity when the market moves through the limit, leaving the remainder open for later fills. This creates a real working remainder without inventing L1 Market-order partial-fill semantics or adding a new maintained Trade RL order type.

## Verification

The dedicated `Nautilus Capability` workflow must run this BacktestEngine scenario in its own pytest process because the pinned runtime has process-global kernel state. Final repository verification still requires Ruff, Format, MyPy, import architecture, compatibility suites, full pytest/coverage, and the dedicated capability workflow on the same final head.
