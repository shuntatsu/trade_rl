# Reward Design Overhaul Design

## Objective

Make reward optimization, execution timing, validation, and release evaluation mathematically consistent across the baseline-residual and legacy direct-weight research paths.

The baseline-residual path remains the preferred architecture. The direct-weight path remains available for diagnostics and comparison, but its reward scaling and decision timing must no longer alter the economic objective accidentally.

## Baseline-residual reward

The environment continues to optimize paired excess log growth:

```text
excess_log_return = log(hybrid_value_after / hybrid_value_before)
                  - log(shadow_value_after / shadow_value_before)
reward = reward_scale * excess_log_return
```

The default residual discount factor becomes `0.99`. The CLI rejects residual training with `gamma < 0.95` unless an explicit research-only override is supplied. This preserves the additive wealth-ratio interpretation over multi-step episodes while still permitting controlled ablations.

Reward scaling is numerical only. It must not change costs, turnover preferences, risk limits, or candidate selection.

## Residual observability

The residual observation includes the state needed to explain the paired reward:

- hybrid minus shadow weights for each symbol;
- log hybrid-to-shadow wealth ratio;
- shadow drawdown;
- shadow gross exposure;
- cumulative hybrid-minus-shadow turnover.

These fields are added only to the residual observation schema. The direct observation schema and serving contract remain unchanged.

## Residual termination

Hybrid insolvency and shadow insolvency are handled separately.

- Hybrid insolvency: preserve the realized paired excess log return for the interval, then add a configurable hybrid-insolvency penalty.
- Shadow insolvency: terminate and mark the episode invalid with `shadow_insolvent=True`; do not assign an uncontrollable fixed penalty to the agent.
- Both conditions must be exposed in `info`.

The evaluation layer fails closed when a shadow-insolvent rollout is encountered.

## Statistical alignment

Checkpoint selection, paired bootstrap inference, candidate selection, and reported excess performance use the same base quantity: per-base-bar excess log return.

Simple-return differences remain available only as descriptive diagnostics and never drive selection or significance tests.

## Direct-weight reward

The direct reward becomes:

```text
reward = reward_scale * (interval_net_log_return - turnover_penalty_rate * interval_turnover)
```

`turnover_penalty_rate` is expressed in return units per unit turnover. Changing `reward_scale` must not change the implied economic turnover penalty.

The legacy `lambda_turnover` CLI option remains as a deprecated alias for one release cycle, but is converted explicitly to `turnover_penalty_rate` and cannot be combined with the new option.

The default direct turnover penalty becomes zero because the execution model already includes fee, spread, and nonlinear impact costs. Nonzero values are research-only stress assumptions.

## Direct decision intervals

One direct environment action advances one complete `decision_every` interval, exactly as in the residual environment.

- The action is processed once at the start of the interval.
- Entry execution cost is charged once.
- The target is held for the interval.
- Funding and mark-to-market returns accrue on every base bar.
- One aggregated reward and one transition are returned.

This removes ignored-action transitions and makes reward attribution valid when `decision_every > 1`.

## DSR boundary

Differential Sharpe reward is removed from the production/control-plane training path. It remains available only through an explicit `--experimental-dsr` research flag.

When enabled:

- `dsr_A` and `dsr_B` are included in the observation;
- the DSR reward is clipped to a configurable finite bound;
- DSR cannot be switched on or off during an episode;
- checkpoint and release reports are marked experimental and ineligible for registration.

## Configuration and manifests

Resolved reports and manifests record:

- requested and effective gamma;
- reward scale;
- direct turnover penalty rate;
- residual hybrid-insolvency penalty;
- whether experimental DSR was enabled;
- decision interval semantics version `interval_v2`.

## Tests

The implementation must include regression tests proving:

1. residual default gamma is at least 0.95 and defaults to 0.99;
2. residual observation exposes paired state and identity action remains exact;
3. shadow-only insolvency does not apply the hybrid penalty and evaluation fails closed;
4. paired inference uses excess log returns;
5. direct reward economics are invariant to reward scaling;
6. direct `decision_every > 1` advances a full interval and charges one entry cost;
7. DSR is unavailable without the explicit research flag and its hidden state is observable when enabled;
8. existing train/eval/serve parity and residual identity contracts remain green.

## Non-goals

- No new alpha model or feature engineering.
- No change to baseline trend construction.
- No production registration of residual candidates.
- No automatic merge into `main` without passing CI and review.