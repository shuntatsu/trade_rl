# Lagrangian Advantage Units Correction

Date: 2026-07-26
Status: normative correction
Applies to: PR C clauses in `2026-07-26-constrained-growth-ppo-design.md`, `2026-07-26-constrained-ppo-review-addendum.md`, and `2026-07-26-constrained-ppo-pr-c-lagrangian.md`

## Problem

The earlier PR C draft proposed independently normalizing every cost advantage before applying its Lagrange multiplier. That is incompatible with the maintained dual update, which compares raw completed-episode costs with raw budgets.

For a constraint written as

```text
E[C_i] <= d_i
```

the multiplier has reciprocal units of `C_i`. The actor correction must therefore preserve the product

```text
lambda_i * A_i^c
```

in the original cost units. Independently standardizing `A_i^c` while updating `lambda_i` from raw `C_i - d_i` destroys that unit relationship and makes the actor correction depend on arbitrary measurement scale and rollout variance.

## Corrected actor contract

The authoritative raw combined advantage is

```text
A_raw = A_reward - sum_i(lambda_i * A_cost_i)
```

where reward and cost advantages are the original GAE outputs in canonical batch order.

When PPO advantage normalization is enabled, normalization is applied exactly once to `A_raw`, after all Lagrangian terms have been combined. The normalization uses the same Torch batch statistics and epsilon as the maintained SB3 PPO update.

When every multiplier is zero, the actor path remains bitwise-equivalent to the maintained Cost Critic PPO update after identical RNG reset.

## Unit-invariance requirement

For any positive scale `s`, replacing

```text
A_cost_i <- s * A_cost_i
lambda_i <- lambda_i / s
budget_i <- s * budget_i
cost estimate_i <- s * cost estimate_i
```

must leave the raw actor correction unchanged. Dual learning-rate scaling remains an explicit configuration concern; it is not hidden inside advantage normalization.

## Diagnostics

Independent reward and cost normalization utilities remain available for correlation, critic conditioning, and diagnostic comparisons. They are not used to construct the authoritative Lagrangian actor advantage.

PR C continues to record:

- raw cost covariance and correlation;
- normalized cost-advantage correlation as a diagnostic only;
- raw effective penalty contributions `lambda_i * A_i^c`;
- aggregate raw penalty magnitude relative to reward advantage magnitude.

## Superseded clauses

This correction supersedes only statements that require independently normalized cost advantages in the actor objective. It does not change:

- the scalar all-cost net-log-growth reward;
- per-cost Cost Critic GAE;
- rollout-frozen multipliers;
- completed-episode dual updates;
- per-cost budgets, EMA, warmup, update intervals, or caps;
- the prohibition on automatic decorrelation in PR C.
