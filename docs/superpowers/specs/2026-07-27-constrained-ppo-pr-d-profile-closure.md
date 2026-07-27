# Constrained PPO PR D Profile Closure

## Purpose

Define one fixed, reproducible comparison pack for the first full Constrained PPO study without changing training, evaluation, Serving, release, or direct-execution behavior.

## Candidates

The maintained walk-forward profile contains exactly four candidates with common data, environment, reward, execution, architecture, optimizer schedule, seed, and training-step contracts:

1. ordinary PPO control with undiscounted reward return and reward GAE 0.95;
2. canonical Lagrangian PPO with undiscounted reward return and reward GAE 0.95;
3. Lagrangian PPO with reward GAE 0.97 only;
4. objective-misaligned Lagrangian PPO with reward gamma 0.9995 only.

Cost-return gamma remains 1.0 for every constrained candidate. The GAE and discount ablations do not alter the Cost Critic schema, multiplier configuration, model size, batch size, learning-rate schedule, or execution assumptions.

## Constraint ordering

All vectors follow `CONSTRAINT_COST_NAMES` exactly:

1. drawdown excess;
2. drawdown-stop event;
3. margin-deficit fraction;
4. forced-liquidation event;
5. gross-exposure request excess;
6. daily turnover;
7. execution-cost fraction.

The initial research budgets are fixed at `(0, 0, 0, 0, 0, 1.0, 0.03)`. These are predeclared experiment settings, not empirical claims of optimality or production safety.

Rare-event dual updates require 20 completed episodes for drawdown-stop and forced-liquidation costs. Dense costs use the minimum support of one completed episode. Every multiplier has an explicit learning rate, EMA coefficient, warmup, update interval, initial value, and maximum value.

## Scope boundary

This change provides parseable standalone profiles and one walk-forward comparison profile only. It does not run the experiment, publish empirical results, alter eligibility gates, or establish profitability. Production remains `NO-GO`.
