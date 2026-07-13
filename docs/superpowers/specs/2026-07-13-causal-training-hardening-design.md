# Causal Training Hardening Design

## Status

Approved for implementation from the 2026-07-13 mainline audit.

## Goal

Remove training-only terminal information and non-causal execution inputs, then make PPO resource and rollout settings explicit and reproducible.

## Scope

1. Episode endings distinguish a genuine terminal liquidation from a time-limit truncation.
2. Training defaults to time-limit continuation semantics: no forced liquidation reward and no episode-progress feature.
3. Explicit liquidation remains available for sealed evaluation and is treated as a terminal transition, not a truncation eligible for value bootstrap.
4. Policy observations expose current tradability only. Future `t+1` tradability remains part of transition dynamics and execution, but is not policy input or pre-trade target filtering.
5. Next-open execution uses volume known at decision time (`volume[t]`) as the capacity proxy for fills at `open[t+1]`; it never uses the completed `t+1` bar volume to size an order at its open.
6. Liquidation fails closed when capacity prevents a complete exit.
7. PPO settings are explicit in `ResidualTrainingConfig`; requested and rollout-rounded timesteps are recorded.
8. The Stable-Baselines3 backend accepts the full typed configuration, reports the resolved device and actual timesteps, and supports an explicit device selection.
9. Policy ensemble identity includes the training configuration digest and observation schema.

## Non-goals

- Building the concrete real-data FoldRunner in this change.
- Changing the two-dimensional residual action schema.
- Increasing network size solely to make a GPU busy.
- Claiming production readiness or strategy profitability.

## Environment semantics

`liquidate_on_end=False` is the training default. Reaching `end_index` returns `truncated=True`, keeps the final marked account state, and allows SB3 to bootstrap the terminal observation.

`liquidate_on_end=True` is an explicit sealed-evaluation mode. Reaching `end_index` liquidates both books, includes liquidation costs in the final reward and metrics, returns `terminated=True`, and returns `truncated=False`. Incomplete liquidation raises an error instead of silently resetting with residual positions.

Insolvency remains a true termination. If insolvency and the time limit occur on the same step, termination takes precedence.

## Observation contract

The observation schema becomes `baseline_residual_observation_v2`.

Per-symbol inputs contain current feature availability, current tradability, trend targets, alpha, hybrid weights, shadow weights, and relative weights. Episode progress is removed because live continuous operation cannot reproduce a synthetic random episode boundary.

## Execution causality

At decision close `t`, an order may execute at `open[t+1]`. Capacity for that fill is estimated from `volume[t]`, which is known at the decision. Actual `tradable[t+1]` still determines whether the market accepts a fill. Partial fills may continue across the decision interval, using each just-completed bar's volume as the next-open capacity proxy.

## PPO and GPU policy

A small MLP PPO with one environment is commonly environment/CPU-bound; low GPU utilization is not itself a defect. The code must make device choice observable and configurable, but must not silently enlarge the network or batch merely to increase GPU usage.

The configuration records learning rate, rollout length, batch size, epochs, GAE lambda, clipping, entropy/value coefficients, gradient norm, advantage normalization, policy name, device, and seeds. `actual_timesteps` is the rollout-rounded count for the single-environment backend.

GPU-oriented experiments may explicitly use `device="cuda"`, larger batches, a larger policy network through policy kwargs in a later change, and vectorized environments. Performance comparisons must include wall-clock throughput and evaluation quality, not utilization alone.

## Testing

Regression tests cover:

- time-limit ending without liquidation is truncated only;
- explicit liquidation is terminal only and is fully flat;
- incomplete liquidation fails closed;
- observations contain current, not next, tradability and no progress field;
- pre-trade risk does not inspect future tradability;
- next-open capacity uses prior-bar volume;
- PPO configuration validation and rollout-rounded timesteps;
- backend forwards all explicit PPO settings and reports device/timestep metadata;
- policy manifest digest changes when training configuration changes.

The full authoritative CI remains required before merge.