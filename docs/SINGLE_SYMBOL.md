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

The default full-research workflow is
`walk-forward-target-weight-constrained-growth.json`. It compares these two
single-symbol gamma-one target-weight profiles in order:

- `training-target-weight-growth-ppo.json` — net-growth PPO control;
- `training-target-weight-constrained-growth.json` — constrained-growth Lagrangian PPO candidate.

The 168-hour discounted constrained-growth profile has a distinct environment
boundary and is evaluated through
`walk-forward-target-weight-constrained-growth-discounted.json`. It is an
explicit time-preference ablation, not a candidate in the gamma-one selection
catalog.

`training-full.json` is retained as an explicit legacy mixed-shaping comparison.
It is selected only by naming it through `--training-template training-full.json`;
it is not part of the implicit default candidate catalog.
`walk-forward-full.json` remains available for historical reproducibility and is
not silently substituted for the maintained default.

All training profiles retain `training_run_config_v4`. A schema bump is not
required because the existing action, symbol-order and policy-identity contracts
already represent a one-element action vector.

The maintained config writer rejects any profile that is not
`target_weight_count=1` before training resources are allocated.

Gamma-one growth candidates use `episode_boundary_mode=finite_horizon_termination`,
`finite_horizon_observation=true`, and `liquidate_on_end=false`. Their 720-hour
limit is part of the finite-horizon MDP, so the final mark-to-market state is a
termination and is not bootstrapped. The discounted ablation uses
`episode_boundary_mode=external_truncation`,
`finite_horizon_observation=false`, and `liquidate_on_end=false`, so its time
limit is an external training-window cut and its terminal observation is
bootstrapped.

The two boundary contracts are intentionally not mixed in one
`MarketWalkForwardConfig`: that contract requires all candidates to share the
same environment dynamics. Cross-objective reporting uses common after-cost net
log growth evidence rather than raw training rewards. See
`docs/REWARD_OBJECTIVE.md`.

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

Existing generations remain bound to their recorded source, image, configuration
and artifact identities. Changing the default or the boundary contract affects
only a generation built from a newer source commit; it does not migrate or resume
an existing generation under different objective semantics.

## Execution and safety

The boundary alignment does not add live order submission or weaken the
execution simulator, hard risk, constraint, sealed-test or evidence contracts.
A finite-horizon time limit is not insolvency, does not set the environment's
economic-termination flag, and does not trigger terminal-equity shaping.
Production remains `NO-GO`.
