# Constrained Policy Report Core Design

## Goal

Add a deterministic, fail-closed read model for constrained-policy evidence before wiring it into market walk-forward selection.

## Parallel-lane boundary

This change is deliberately isolated from the active C3 and hierarchical sequence-policy lanes. It creates a pure evaluation module and focused tests only. It does not modify CLI, C3 execution/reporting, sequence encoders, SB3 construction, fold execution, market walk-forward orchestration, workflows, Serving, promotion, release, or direct execution.

The subsequent integration change will consume this module after the core contract is merged.

## Input model

A report is built for one candidate configuration across one or more folds. Every fold contains:

- seed-local observations for each evaluated execution scenario;
- one deployable-ensemble observation for each evaluated execution scenario;
- declared and evaluated ensemble member identities;
- return, drawdown, turnover, economic-cost, and raw-to-filled distortion metrics;
- constrained candidates may additionally contain the seven canonical constraint costs and model diagnostics.

The seven costs must preserve `CONSTRAINT_COST_NAMES` order. Unit and completed-episode aggregation are derived from the canonical Lagrangian statistics contract rather than accepted as free-form input.

## Ordinary PPO semantics

Ordinary PPO has no Lagrange multipliers or Cost Critics. Its constraint collection is represented as `None`, never as zero-filled synthetic evidence. Supplying constraint or penalty diagnostics for an ordinary PPO observation is invalid.

## Constrained evidence fields

For every cost observation the core accepts:

- evaluation value;
- budget;
- completed-episode denominator;
- censored-episode count;
- minimum completed-episode support;
- raw and EMA training estimates;
- multiplier mean and maximum;
- upper-cap and lower-bound occupancy fractions;
- Cost Critic explained variance and loss.

Optional model diagnostics remain optional at the input boundary so incomplete evidence can be represented and rejected by eligibility instead of being silently fabricated.

## Eligibility

Eligibility is evaluated only on required scenarios, initially `nominal` and `joint_2x`.

A constrained candidate is ineligible when any required fold/scenario:

- is missing;
- has a different seed set from the other required scenario;
- evaluates an ensemble member set different from the declared set;
- lacks canonical cost evidence;
- lacks required model diagnostics;
- has any seed or ensemble constraint value above budget;
- has completed-episode support below the configured minimum.

Finite-value and range validation occurs during input construction. Lower-bound multiplier occupancy is reported separately and is not saturation. Upper-cap occupancy is reported but does not by itself change eligibility.

Ordinary PPO remains constraint-neutral when required scenarios and member identities are complete.

## Aggregation

The report never concatenates fold equity curves.

For each fold/scenario and aggregate scenario it records:

- deployable-ensemble mean cost;
- worst seed cost;
- worst fold cost;
- conservative minimum completed-episode support;
- maximum censored count;
- complete-only averages for training estimates, multipliers, bound occupancy, and Cost Critic diagnostics;
- mean deployable return, worst seed return, worst fold return, maximum drawdown, maximum turnover, maximum economic cost, and mean raw-to-filled distortion.

If an optional diagnostic is missing from any required source, the aggregate field remains `None`; partial averages are not emitted.

## Identity and determinism

Fold, scenario, seed, cost, and member identities are normalized into stable order. Every public summary exposes a deterministic digest payload, and the top-level report digest is invariant to input fold ordering.

## Non-goals

- No collection of transition costs from environments.
- No checkpoint inspection.
- No modification of candidate selection.
- No paired moving-block inference.
- No artifact writer or CLI.
- No empirical constrained-training claim.

Production remains `NO-GO`.
