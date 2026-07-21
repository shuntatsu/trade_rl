# Conservative Stateful Order Simulator Verification

## Scope

This record covers the OHLCV-based stateful execution model introduced by PR #75. It does not claim order-book reconstruction, exchange-equivalent fills, or model profitability.

## Required Trust Boundaries

- order quantity is fixed from decision-time information;
- fills use the processing bar's volume and one shared symbol capacity pool;
- partial residuals remain explicit pending orders;
- one deterministic OHLC path is selected per symbol and bar;
- final promotion requires conservative path mode, processing-bar capacity, partial-fill carry, complete order events, and a matching execution-policy digest;
- pending-order state participates in Training–Serving observation parity;
- replay evidence binds dataset, seed, policy, action trace, order events, equity curve, and observation trace.

## Maintained Full-Training Defaults

The maintained Binance multi-timeframe training and walk-forward configurations explicitly select:

```text
path mode: conservative
processing-bar volume capacity: true
partial-fill carry: true
trigger-volume fractions: 1.00 / 0.50 / 0.25 / 0.00
stateful environment time in force: GTC
```

## Focused Verification

The implementation is developed test-first. Focused checks cover order-domain invariants, gap rules, shared capacity, partial carry, target reconciliation, admission, accounting, pending-order observations, Training–Serving parity, promotion evidence, and deterministic replay.

## Exact-Head Evidence

The final exact head, workflow IDs, test count, coverage, source digest, Docker image identity, PostgreSQL result, deterministic replay digest, and three-seed smoke outcome are recorded after repository-wide verification.

## Known Limitations

- OHLCV cannot recover true intrabar order, queue position, hidden liquidity, auctions, or L2 depth.
- The deterministic capacity priority is an explicit research convention, not exchange queue priority.
- Optimistic and neutral modes are sensitivity tools and cannot serve as primary promotion evidence.
- A smoke training result is pipeline evidence only and is not profitability evidence.
