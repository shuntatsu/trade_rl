# Native Multi-Timeframe Research

Trade RL treats multi-timeframe market context as a causal dataset contract. The maintained Binance example makes decisions every 15 minutes while computing each feature only on its own completed native clock (`15m`, `1h`, `4h`, and `1d`). Availability-aware as-of alignment prevents backward filling, incomplete-bar use, and future Ichimoku shifts.

## Maintained contract

The maintained dataset contains 226 ordered point-in-time channels: 59 on 15m, 59 on 1h, 55 on 4h, and 53 on 1d. The policy receives completed native windows of 96, 168, 120, and 60 bars, availability and staleness state, the current market snapshot, execution state, portfolio state, and finite-horizon state. Actions are direct target weights with a one-decision signal delay and hard pre-trade, liquidity, portfolio, and emergency-risk projection.

The structured policy uses timeframe-specific sequence encoders, a shared per-asset actor, and a portfolio-level critic. PPO uses an index-backed rollout: persistent storage contains decision indices and current state, while overlapping histories are reconstructed only for sampled minibatches. Behavior cloning uses an approximate portfolio teacher with executor-compatible minimum-notional, capacity, partial-fill, fee, spread, and delay semantics.

## Evidence protocol

The maintained gate requires six sealed folds covering at least 180 OOS days, a positive circular block-bootstrap lower confidence bound, acceptable drawdown, turnover and costs, and stable recipe selection. Final training retains the fixed three-seed ensemble rather than selecting a lucky seed. A separate fresh confirmation interval is opened only after development choices are frozen and must contain at least 30 sealed days with matching policy identity.

The structured sequence serving runtime restores both normalizers and rebuilds the same Dict observation from a bounded rolling dataset. It rejects symbol order, feature order, cadence, sequence layout, and incomplete-bar drift. Live order routing remains outside the policy artifact.

## Running the complete example

```bash
uv sync --extra dev --extra train-sb3
uv run python examples/binance-multitimeframe/run_full_research.py   --work-root var/binance-multitimeframe-full
```

Production remains NO-GO until the OOS gate, fresh confirmation, CUDA verification, checkpoint recovery, structured serving parity, and paper-trading reconciliation are complete. The research workflow does not authenticate an account or place live orders.
