# Causal Alpha V10 Closed-Loop Second Falsification Addendum

This addendum records two additional contract mismatches found after the first closed-loop implementation. It supplements the closed-loop design and its first falsification update.

## Strict no-trade-band boundary

Hard-risk partial executability must match `PreTradeRisk` exactly. An ordinary change is suppressed only when `abs(target - current) < no_trade_band`; equality is executable.

V10 therefore compares the requested delta directly with `no_trade_band`. It must not subtract its observation tolerance from the execution band. With current exposure `0.10`, risk cap `0.0500005`, and no-trade band `0.05`, the partial reduction delta is `0.0499995`; that partial target is not executable and the policy must use `risk_cap_flatten` instead.

## Fit and inference cadence phase

Closed-loop inference is anchored to absolute decision indices. Fitting must use the same phase:

```text
decision_index % horizon_decisions == 0
```

A knowledge cutoff that is not aligned to the horizon must not redefine the cadence phase or eliminate valid absolute-phase training rows.

## Acceptance criteria

- A hard-risk partial reduction strictly below `no_trade_band` uses `risk_cap_flatten`.
- Equality with `no_trade_band` remains executable under the maintained strict `<` semantics.
- Fast fitting and inference share `decision_index % 16 == 0`.
- Slow fitting uses `decision_index % 288 == 0`.
- Selection/Admission thresholds, V8/V9 controls, 4h/72h horizons, and confirmation counts remain unchanged.
