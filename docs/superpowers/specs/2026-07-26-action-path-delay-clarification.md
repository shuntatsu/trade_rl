# Action-Path Delay Clarification

Date: 2026-07-26
Status: normative clarification
Applies to: `2026-07-26-constrained-growth-ppo-design.md`

## Decision

The canonical action path contains six representations rather than five whenever signal delay is part of the environment contract:

1. `policy_target`: the target produced by the policy on the current decision;
2. `execution_intent_target`: the target released from the signal-delay queue and presented to the risk pipeline on this transition;
3. `pretrade_target`: the target after emergency flattening, entry/exit hysteresis, no-trade-band handling, and turnover controls;
4. `feasible_target`: the target after hard exposure and portfolio feasibility projection;
5. `submitted_order_target`: the target actually passed to the execution coordinator;
6. `filled_weight`: the realized portfolio weight after fills and any same-transition liquidation.

When signal delay is disabled, `policy_target` and `execution_intent_target` are identical. The explicit stage remains present so delayed and non-delayed runs share one schema.

## Required diagnostics

The environment reports at least:

- `policy_to_execution_intent_l1`;
- `execution_intent_to_pretrade_l1`;
- `pretrade_to_feasible_l1`;
- `feasible_to_submitted_l1`;
- `submitted_to_filled_l1`;
- `execution_intent_to_filled_l1`;
- `policy_to_filled_l1`;
- matching maximum-absolute deviations;
- stage-specific change flags.

The legacy aggregate `policy_to_pretrade` remains available, but it includes both signal-delay displacement and pre-trade controls. It must not be interpreted as a pure risk-projection metric when delay is enabled.

## Constraint-cost attribution

Gross-exposure request excess is calculated from `execution_intent_target`, because that is the target actually presented to the risk mapper on the current transition. The current `policy_target` may not affect execution until a later transition when signal delay is enabled.

Reward remains unchanged. Action-path distances are diagnostic telemetry and never enter the scalar environment reward in PR A.

## Fail-closed invariants

- `execution_intent_target` must equal the proposal recorded by the risk projector;
- all stage vectors must have identical shape and finite values;
- initial capital is supplied explicitly for the stable margin-deficit denominator;
- ordinary execution and same-transition liquidation costs and turnover are combined exactly once;
- positive funding remains separate credit telemetry and does not cancel the non-negative execution safety cost.
