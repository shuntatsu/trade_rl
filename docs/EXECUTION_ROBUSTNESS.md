# Execution robustness evidence

The maintained execution-robustness workflow extends the existing sealed
walk-forward rule-rounding matrix with deterministic adverse assumptions for
fees, spread, market impact, stochastic slippage scale, participation
capacity, order latency, tail slippage, and borrow cost.

## Contract

`ExecutionEnvironmentStress` is an immutable simulation-layer overlay. It
transforms one immutable `ExecutionCostConfig` before either evaluation book
is constructed. The hybrid selected-policy executor and independent shadow
baseline executor receive the same transformed cost object and the same rule
stress, preventing asymmetric evidence.

The overlay is bound into each scenario digest and therefore into the
walk-forward experiment-plan identity. Unknown or invalid values fail closed.
Multipliers are adverse-only (`>= 1`), participation can only decrease, and
latency/tail fields are floors rather than replacements.

## Maintained profile

`walk-forward-target-weight-execution-robustness.json` retains the current
target-weight constrained-growth candidate set and the existing mandatory
`joint_2x` exchange-rule gate. It adds report-only scenarios for:

- two-times fees and spread;
- two-times impact and slippage standard deviation;
- fifty-percent participation capacity;
- at least one bar of order latency;
- one-percent, ten-times tail-slippage floors;
- two-times borrow cost;
- one joint adverse environment combining all dimensions.

Additional execution-environment scenarios remain report-only under
`execution_sensitivity_config_v1`. They do not replace or weaken the required
`joint_2x` gate and are not selected as the default full-research workflow.

## Safety status

This evidence is evaluation-only. It does not change training rewards, PPO or
BC objectives, checkpoint formats, serving, risk limits, sealed-test access,
or live-order behavior. Production remains **NO-GO**.
