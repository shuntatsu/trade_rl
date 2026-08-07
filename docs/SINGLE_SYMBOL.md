# Maintained single-symbol workflow

`trade_rl` maintains one instrument per training and evaluation run.

```text
one run
  = one instrument
  = one target-weight action
  = one checkpoint and evidence chain
```

The initial maintained preset is Binance USDS-M perpetual `BTCUSDT`.

## Maintained contract

| Field | Value |
|---|---|
| Symbols | `("BTCUSDT",)` |
| Action mode | `target_weight` |
| Action shape | `(1,)` |
| Action name | `target_weight:BTCUSDT` |
| Range | `[-1.0, 1.0]` |
| Maximum gross exposure | `1.0` |
| Maximum absolute weight | `1.0` |
| Maximum leverage | `1.0x` |
| Decision interval | `15m` |
| Context clocks | `15m`, `1h`, `4h`, `1d` |
| Production status | `NO-GO` |

Capital diversification is performed outside the model by assigning independent
budgets to independent runs. The maintained policy does not allocate between
BTC, ETH and BNB in one action vector.

## Configuration

The maintained profiles are:

- `training-full.json`;
- `training-target-weight-growth-ppo.json`;
- `training-target-weight-constrained-growth.json`;
- `training-target-weight-constrained-growth-discounted.json`;
- `walk-forward-full.json`;
- `walk-forward-target-weight-constrained-growth.json`.

They retain `training_run_config_v4`. A schema bump is not required because the
existing action, symbol-order and policy-identity contracts already represent a
one-element action vector.

The maintained config writer rejects any profile that is not
`target_weight_count=1` before training resources are allocated.

## Data and policy behavior

- Market-data synchronization requests only `BTCUSDT`.
- PostgreSQL dataset assembly supports the maintained one-symbol path while
  preserving the historical three-symbol reader path.
- Observation action width is derived from `dataset.n_symbols`; it is not a
  numeric literal.
- Four native timeframe histories remain active.
- Cross-timeframe fusion remains active.
- The Cross-Asset Transformer module and its parameters are not created when
  `n_symbols == 1`.
- Single-symbol diagnostics report Cross-Asset metrics as non-applicable zeros
  while retaining Cross-Timeframe and gradient diagnostics.
- `single_symbol_bypass_v1` binds that structure into policy identity; inactive
  Asset-Attention settings are omitted from its architecture digest.
- Architecture and asset-binding identity distinguish one-symbol and historical
  three-symbol checkpoints.

## Historical compatibility

Prior multi-asset artifacts remain immutable and readable. They are not
silently converted, resumed or transferred into the maintained one-action
policy. See `docs/implementation/legacy-multi-asset-inventory.md` for the
classification of retained legacy code.

## Execution and safety

This migration does not change the current execution simulator, reward,
constraint, walk-forward, sealed-test or evidence semantics. It does not add
live order submission. Production remains `NO-GO`.
