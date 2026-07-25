# Growth-Optimal Reward Design

## Decision

Add a separate growth-optimal research profile without changing the reward schema or the existing full-training profile.

The policy objective is the sum of actual net interval log returns:

```text
reward_t = scale * log(net_equity_after / net_equity_before)
```

Net equity is measured after mark-to-market profit and loss, fees, spread, impact, funding, borrow cost, partial fills, and liquidation execution. With `gamma = 1.0`, the episode reward telescopes to scaled terminal net log growth.

## Reward contract

The profile uses the existing `RewardConfig` with:

```json
{
  "absolute_growth_weight": 1.0,
  "excess_growth_weight": 0.0,
  "incremental_drawdown_weight": 0.0,
  "baseline_underperformance_weight": 0.0,
  "projection_penalty_weight": 0.0,
  "terminal_equity_weight": 0.0,
  "margin_deficit_weight": 0.0
}
```

No additional terminal penalty is used. Economic loss, forced-close cost, and collapse toward zero equity are already represented by the interval net log return. Adding another equity or failure penalty would double count the same consequence and break the terminal-growth identity.

The existing baseline-shaped profile remains unchanged so historical experiments retain their configuration identity.

## Risk and baseline placement

Risk remains enforced outside the scalar training reward through:

- maximum gross exposure;
- maximum per-asset weight;
- market participation and execution capacity;
- margin and insolvency rules;
- the 20% drawdown stop;
- emergency deleveraging.

Baseline uplift, maximum drawdown, turnover, cost fraction, seed stability, and adverse-execution scenarios remain walk-forward selection gates rather than training-reward components.

## Discounting

The 720-hour finite-horizon profile uses `gamma = 1.0` and exposes the remaining horizon in the observation. `gae_lambda = 0.95` remains an estimator bias-variance setting and does not redefine the undiscounted task objective.

## Comparator termination

- Hybrid policy failure remains a true termination.
- Forced final close follows the existing terminal-accounting path.
- Shadow-comparator-only failure is external to the policy and becomes a truncation with reason `shadow_<economic_reason>`.
- When both books fail, hybrid failure takes precedence.

## Artifacts

Add separate profiles:

- `examples/binance-multitimeframe/training-growth-optimal.json`
- `examples/binance-multitimeframe/walk-forward-growth-optimal.json`

The legacy full profiles remain untouched for controlled A/B evaluation.

## Required evidence

- exact reward equals scaled net interval log growth;
- episode reward telescopes to total net log growth;
- terminal equity is not counted twice;
- both profiles parse and expose the same objective;
- shadow-only failure truncates rather than terminates;
- complete exact-head CI passes before merge.
