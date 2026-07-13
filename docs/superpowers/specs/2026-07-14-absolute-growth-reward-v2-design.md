# Absolute Growth Reward v2 Design

## Status

Approved for implementation on 2026-07-14.

## Goal

Train the residual policy to maximize cost-adjusted absolute log wealth growth while lightly discouraging material rolling baseline underperformance and newly worsening drawdown. Hard safety remains outside the scalar reward.

## Reward

For one complete decision interval:

```text
growth = log(V_after / V_before)
shortfall = max(0, rolling_baseline_growth - rolling_policy_growth)
tolerance = full_window_tolerance * min(history, window) / window
hinge = 0                                      before minimum history
        max(0, shortfall - tolerance)          otherwise
baseline_delta = max(0, hinge_after - hinge_before)
drawdown_delta = max(0, severity_after - severity_before)
raw_reward = growth - lambda_baseline * baseline_delta - lambda_drawdown * drawdown_delta
scaled_reward = reward_scale * raw_reward
```

Defaults:

- reward scale: `100.0`
- baseline window: `720 hours` (30 days)
- baseline minimum history: `168 hours` (7 days)
- full-window baseline tolerance: `0.015` log return
- baseline penalty weight: `0.10`
- drawdown penalty weight: `0.05`
- drawdown free zone: `0.05`
- drawdown slope changes: `0.10` and `0.15`
- emergency drawdown stop: `0.20`
- drawdown slopes: `1`, `3`, `8`

Before the full rolling window is available, tolerance scales linearly with observed history. Before minimum history, the baseline hinge is disabled. Neither recovery nor baseline outperformance receives an artificial bonus.

## Drawdown severity

The continuous piecewise-linear severity function is:

```text
0                              DD <= 5%
DD - 5%                        5% < DD <= 10%
5% + 3 * (DD - 10%)           10% < DD <= 15%
20% + 8 * (DD - 15%)          DD > 15%
```

Only positive severity increments are penalized.

## State and observation

The environment derives rolling log growth from base-bar return histories. The observation exposes the complete reward-relevant summary state:

- rolling policy log growth
- rolling baseline log growth
- policy-minus-baseline rolling growth gap
- current baseline shortfall
- current scaled tolerance
- current baseline hinge level
- emergency-deleverage state

The observation schema advances to `baseline_residual_observation_v3`.

## Termination

Random training-window endings remain time-limit truncations without terminal reward shaping. Explicit sealed-evaluation liquidation remains terminal and includes actual liquidation economics.

When policy drawdown reaches 20%, the environment performs an emergency close at the current close, includes realized liquidation cost in wealth growth, requires a complete flatten, and terminates with `termination_reason="drawdown_stop"`. There is no fixed terminal jackpot or penalty. The independent shadow book remains open so policy failure does not impose an artificial cost on the baseline comparator.

Economic minimum-equity failure remains a true termination. Invalid market data remains an exception.

## Diagnostics

Every transition records the unscaled and scaled reward components, rolling growths, tolerance, shortfall, hinge level, before/after drawdown and severity, before/after wealth, peak wealth, termination reason, and any liquidation result used by the transition.

## Identity

All reward parameters, resolved reward-window bars, resolved minimum-history bars, and the new observation schema are included in the environment digest.

## Compatibility

Zero residual action must still reproduce the shadow book exactly, but reward is no longer zero. It equals the baseline book's absolute log growth minus any policy drawdown increment penalty; baseline shortfall and hinge remain zero.

## Validation

Tests cover configuration validation, positive and negative growth, cost sensitivity through wealth, baseline warm-up and tolerance, full-window rollover, baseline outperformance, non-repeated penalties, piecewise drawdown slopes and continuity, observation exposure, zero-action identity, environment identity, truncation, explicit liquidation, emergency drawdown flattening, liquidation-cost inclusion, and fail-closed behavior when emergency liquidation cannot complete.
