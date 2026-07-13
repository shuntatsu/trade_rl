# Absolute Growth Reward v2 Design

## Status

Approved for implementation on 2026-07-13.

## Goal

Replace the baseline-relative step reward with a hierarchical reward whose primary objective is cost-adjusted absolute log wealth growth, while retaining a light rolling baseline non-inferiority penalty and a staged drawdown-increase penalty.

## Reward hierarchy

1. Hard portfolio, liquidity, margin, equity, and drawdown limits remain environment or risk-layer constraints.
2. The primary reward is the hybrid book interval log return after fees, spread, impact, slippage, funding, and liquidation costs.
3. A 30-day rolling baseline hinge penalizes only newly worsening underperformance beyond a 1.5% log-growth tolerance.
4. A drawdown shaping function has a 5% free region and slopes 1, 3, and 8 over the 5-10%, 10-15%, and 15-20% regions. Only increases in shaped severity are penalized.
5. Baseline non-inferiority and risk-adjusted quality remain sealed walk-forward selection gates.

## Formula

For interval hybrid log growth `g_t`, baseline hinge level `B_t`, and drawdown severity `D_t`:

```text
raw_reward_t = g_t
               - baseline_penalty_weight * max(0, B_after - B_before)
               - drawdown_penalty_weight * max(0, D_after - D_before)
reward_t = reward_scale * raw_reward_t
```

Initial defaults:

- reward scale: 100
- baseline window: 720 hours
- baseline minimum history: 168 hours
- baseline tolerance: 0.015 log return at the full window
- baseline penalty weight: 0.10
- drawdown penalty weight: 0.05
- drawdown free threshold: 0.05
- drawdown middle threshold: 0.10
- drawdown high threshold: 0.15
- hard drawdown stop: 0.20
- drawdown slopes: 1, 3, 8

Before the full window is available, the tolerance scales linearly with observed history. The baseline hinge is disabled before minimum history.

## Markov observation contract

The observation schema must expose the reward-relevant rolling state so an MLP policy does not face hidden reward state:

- rolling hybrid log growth
- rolling shadow log growth
- rolling baseline shortfall
- current baseline hinge level
- emergency-deleverage state

The observation schema version changes and remains part of environment and policy identity.

## Environment integration

The environment computes reward context before and after every transition from base-bar return histories. Reward calculation is a pure function returning a typed breakdown. `info` records every raw and scaled component.

Zero residual action continues to produce identical hybrid and shadow books, but its reward is now the baseline strategy's absolute log growth rather than zero.

## Drawdown stop

When hybrid drawdown reaches the configured hard stop, the environment executes an emergency zero-target liquidation at the current close, includes liquidation costs in the final growth reward, requires complete liquidation, and returns a true terminal transition with `termination_reason="drawdown_stop"`. There is no fixed terminal jackpot or penalty.

## Time limits and evaluation

Random training window endings remain truncations without terminal shaping. Explicit sealed-evaluation liquidation remains a true terminal transition and includes actual liquidation costs. Data contract failures remain exceptions; economic safety stops become terminal transitions.

## Diagnostics

Every transition records:

- reward growth raw
- baseline penalty delta and weighted contribution
- drawdown penalty delta and weighted contribution
- raw and scaled total reward
- rolling hybrid and shadow log growth
- baseline shortfall, tolerance, and hinge level
- drawdown before and after, severity before and after
- portfolio values before and after
- termination reason

## Testing

Tests cover pure reward validation, tolerance warm-up, hinge-increase-only semantics, staged drawdown continuity and slopes, no repeated level penalty, zero-action absolute reward, observation shape and reward state, time-limit semantics, emergency liquidation, environment identity changes, and reward diagnostics.