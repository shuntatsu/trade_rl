# Stateful Funding Boundary Evidence Plan

## Goal

Preserve each actual funding boundary processed by the legacy/reference stateful executor so later historical dual-shadow replay can build canonical funding/equity trace records without reconstructing boundaries from aggregate `interval_funding`.

## Scope

This slice does not change Stage A artifact schemas, promotion authority, or the Nautilus funding adapter. It adds immutable per-boundary evidence to `StatefulExecutionResult` and keeps existing aggregate funding outputs unchanged.

## Contract

For every processing bar where at least one dataset `funding_due` flag is true, stateful execution records one `FundingBoundaryEvidence` containing:

- processing index and exact dataset timestamp;
- per-symbol funding-due flags;
- signed position quantities at the boundary;
- funding-boundary mark prices;
- contract multipliers;
- funding rates;
- total signed funding amount charged for that boundary;
- equity immediately before funding, valued at the same mark prices and after the other pre-funding cash adjustments for that bar;
- equity immediately after funding.

A funding boundary with a zero position is still evidence and must be retained with zero funding. Bars without a funding boundary produce no record. Multi-bar execution preserves every boundary in order; it must not collapse them into only the aggregate `interval_funding`.

## Design

Add a small framework-neutral immutable evidence type under `trade_rl.simulation`. `StatefulExecutionRuntime` owns the invocation-local evidence list. `StatefulBarLifecycle.finish_bar` appends evidence immediately after `BookState.mark_to_market(..., funding_amount=...)` and before margin termination can flatten the book. `StatefulExecutionResult` exposes the tuple unchanged.

The evidence remains decimal/floating execution-source evidence in this slice. Conversion to single-instrument `CanonicalExecutionRecord` will be a separate pure adapter with explicit settlement precision; this slice must not hardcode USDT decimal precision.

## Verification

1. RED tests prove the current result loses individual boundaries.
2. Focused simulation tests verify one boundary, zero-position boundary, and two boundaries inside one multi-bar invocation.
3. Ruff, Format, MyPy, import architecture, compatibility suites, and relevant stateful execution tests run before broad integration.
4. Final repository-wide verification is deferred until the accumulated migration head is ready for the final gate.
