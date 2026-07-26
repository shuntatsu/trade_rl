# Constrained PPO Design Review Addendum

Date: 2026-07-26
Status: normative addendum
Applies to: `2026-07-26-constrained-growth-ppo-design.md` and `2026-07-26-grn-architecture-ablation.md`

## 1. Purpose

This addendum hardens the planned PR B, PR C, PR D, and GRN ablation against four failure modes:

1. rare-event cost critics being underfit by dense-cost gradients;
2. inappropriate reuse of one reward-side GAE setting for all cost horizons;
3. correlated Lagrange multipliers producing duplicated pressure, oscillation, or excessive suppression;
4. compute-inequitable comparisons and GRN changes that silently alter constrained-learning dynamics.

The primary reward remains exact all-cost net log growth. None of the changes below introduce fixed cost penalties into the environment reward.

## 2. PR B amendments: dense and rare-event cost learning

### 2.1 Cost families

The seven maintained costs are divided for representation and diagnostics:

- continuous or relatively dense: drawdown excess, margin-deficit fraction, gross-exposure request excess, daily turnover, execution-cost fraction;
- terminal rare events: drawdown-stop event and forced-liquidation event.

The first PR B implementation must not assume that one shared trunk is sufficient for both families.

### 2.2 Required architecture comparison

PR B must compare at least:

1. one shared cost trunk with separate heads;
2. a shared low-level representation followed by a continuous-cost adapter and a rare-event adapter;
3. the same family-split design with independent terminal-event heads.

The shared market encoder may remain common, but dense-cost losses must not be allowed to dominate the last representation layers used by rare-event heads without measurement.

The default candidate is:

```text
shared market representation
    |- continuous-cost adapter
    |    |- drawdown excess
    |    |- margin deficit
    |    |- gross request excess
    |    |- daily turnover
    |    `- execution cost
    `- rare-event adapter
         |- drawdown-stop value
         `- forced-liquidation value
```

### 2.3 Loss accounting and gradient diagnostics

Each cost head keeps an independent value loss and target statistics. PR B must report:

- per-head loss and explained variance;
- target mean, standard deviation, non-zero rate, and effective positive sample count;
- gradient norm contributed by each head to its adapter and to the shared encoder;
- the ratio of aggregate dense-cost gradient norm to rare-event gradient norm;
- cosine similarity between dense-family and rare-event gradients when practical;
- event-value calibration, Brier score, and precision-recall diagnostics when the undiscounted target is interpretable as event probability.

A rare-event head is not considered trained merely because its total MSE is small. A near-constant zero predictor can have deceptively low loss under severe class imbalance.

### 2.4 Auxiliary event supervision

The scalar cost-value head remains authoritative for cost GAE. PR B may add an auxiliary event-probability or finite-horizon hazard head to improve representation learning, but:

- it must not replace the cost-value target;
- it must not alter the environment reward;
- its loss coefficient must be explicit and ablated;
- its predictions must be logged separately from cumulative cost values;
- horizon labels must remain causal and fold-local.

Synthetic unsafe environments are required in tests so drawdown-stop and forced-liquidation heads receive known positive examples. Real-data promotion requires reporting the actual number of positive events in each fold and seed.

### 2.5 Cost-specific gamma and lambda

Reward-side `gamma=1.0` remains authoritative for the economic objective. Cost critics receive explicit per-cost `gamma_c` and `lambda_c`; they must not silently inherit one global pair.

For one-time terminal event costs, the canonical semantic setting is:

```text
gamma_c = 1.0
```

because discounting would change an undiscounted event-probability constraint into a preference for avoiding near-term events more than distant events.

`lambda_c` remains an optimization choice and must be tested separately. The initial event-cost comparison is:

```text
lambda_c in {0.95, 0.97, 1.0}
```

with Monte Carlo event returns used as a diagnostic reference when complete episodes are available. Truncated rollouts require explicit bootstrapping rules and may not be treated as negative events.

Continuous costs may retain `gamma_c=1.0` by default, with `lambda_c` selected per family. Any discounted cost profile is an objective-altering ablation and cannot become canonical solely because it trains more easily.

### 2.6 Rare-event promotion gate

A rare-event critic is eligible for PR C only when all of the following hold:

1. positive support is reported for every training fold or the absence is explicitly treated as insufficient evidence;
2. synthetic positive and negative calibration tests pass;
3. a zero-only predictor is beaten on calibration and precision-recall metrics;
4. value estimates remain finite under long episodes and truncation;
5. the rare-event adapter receives non-trivial gradients;
6. the result is stable across seeds.

## 3. PR C amendments: correlated constraints and dual stability

### 3.1 Correlation monitoring

PR C must treat cost correlation as an observable property of the constrained system. Per rollout and per evaluation period, record:

- the correlation and covariance matrix of raw costs;
- the correlation matrix of normalized cost advantages;
- pairwise correlation of effective penalty contributions `lambda_i * A_i^c`;
- aggregate constraint penalty magnitude relative to reward advantage magnitude;
- each multiplier value, update, cap distance, and sign of the constraint residual.

At minimum, explicitly inspect:

- daily turnover versus execution-cost fraction;
- gross-exposure request excess versus margin deficit;
- gross-exposure request excess versus forced liquidation;
- drawdown excess versus drawdown-stop event.

### 3.2 Saturation and oscillation diagnostics

For every multiplier, report:

- fraction of rollouts at `lambda_max`;
- longest consecutive saturation run;
- update sign-change frequency;
- rolling update variance;
- rolling constraint-residual variance;
- violation area above budget;
- time to first sustained constraint satisfaction;
- time spent over-constrained after satisfaction.

Persistent `lambda_max` saturation is a warning, not proof, of infeasibility. Triage must distinguish:

1. genuinely incompatible budgets;
2. an unattainable budget below the environment's feasible floor;
3. inaccurate or underfit cost critics;
4. excessive dual learning rate or insufficient smoothing;
5. duplicated pressure from correlated costs;
6. policy-capacity or exploration failure.

### 3.3 Joint-feasibility witness

Before full training, evaluate at least one deliberately safe witness policy, such as a cash or minimal-exposure policy, against all configured constraints. If the witness cannot jointly satisfy the budgets, the configuration is rejected before PPO training.

This witness does not prove that the return objective and all constraints are jointly easy to optimize, but it prevents obviously impossible constraint sets from being blamed on the dual optimizer.

### 3.4 First-version mitigation policy

The first PR C version must prefer observability and stable dual updates over automatic mathematical decorrelation. It will use:

- independently normalized cost advantages;
- per-cost dual learning rates and EMA coefficients;
- frozen multipliers during PPO epochs;
- slower rollout-level dual updates;
- explicit lambda caps and warm-up;
- optional family-specific update intervals.

It must not silently merge correlated constraints, orthogonalize advantages, or introduce a covariance-preconditioned dual optimizer without a separate ablation. If correlation diagnostics show duplicated suppression, follow-up candidates include family-level constraints, hierarchical multipliers, or covariance-aware dual steps.

### 3.5 PR C promotion gate

The constrained optimizer is not promoted when:

- any required multiplier remains saturated while its violation fails to improve;
- aggregate penalty dominates reward advantage for sustained periods without constraint progress;
- multiplier sign changes and violation residuals show persistent oscillation;
- the safe witness satisfies budgets but PPO repeatedly cannot, without an identified critic, exploration, or capacity cause;
- correlated penalty contributions are near-duplicates and materially reduce growth without additional constraint benefit.

## 4. PR D amendments: step fairness and compute fairness

PR D must report two distinct comparisons.

### 4.1 Fixed environment-interaction budget

Use identical environment steps, folds, seeds, rollout sizes, and evaluation schedules. This measures statistical efficiency and preserves the same market-data interaction budget.

### 4.2 Fixed compute budget

Also compare models at an approximately matched compute budget. Record:

- wall-clock training time on the same device class;
- total GPU-seconds or GPU-hours;
- parameter count;
- optimizer steps and minibatch updates;
- environment steps per second;
- policy updates per second;
- peak GPU memory;
- mean GPU utilization when available;
- checkpoint and evaluation overhead separately.

The report must show both fixed-step and fixed-compute results rather than forcing one definition of fairness. A constrained model may be more statistically efficient but less compute efficient, or vice versa.

The final comparison should present a return-risk-compute Pareto view, not only one scalar score.

## 5. GRN ablation amendments

### 5.1 Two-stage isolation

The GRN comparison is split to avoid changing the policy representation and cost-critic dynamics simultaneously.

Stage 1 keeps the PR C cost-critic architecture fixed and compares:

- current fusion/actor/reward MLPs;
- residual fusion/actor/reward adapters;
- GRN fusion/actor/reward adapters.

Stage 2 keeps the selected policy and reward architecture fixed and compares:

- the PR C cost-critic adapters;
- residual cost-critic adapters;
- GRN cost-critic adapters.

A full-GRN model is evaluated only after both isolated stages. It is not considered a pure policy-architecture comparison.

### 5.2 Frozen dual configuration

The first GRN comparison must keep all PR C dual hyperparameters fixed:

- constraint budgets;
- dual learning rates;
- EMA coefficients;
- warm-up;
- update intervals;
- lambda caps;
- cost normalization rules.

If a GRN variant requires retuned dual hyperparameters, that result is reported as a coupled architecture-optimizer experiment, not evidence that GRN alone is superior.

### 5.3 Lambda-behavior adoption criteria

In addition to the existing GRN adoption gates, require:

- no material increase in multiplier saturation fraction;
- no material increase in multiplier oscillation or sign-change frequency;
- no deterioration in time to sustained constraint satisfaction;
- no increase in aggregate penalty-to-reward-advantage ratio without measurable constraint benefit;
- stable rare-event critic calibration;
- comparable or improved constraint residual variance across folds and seeds.

GRN gate statistics and lambda trajectories must be reviewed together. A gate pattern that improves critic loss but destabilizes dual updates is a rejection condition.

## 6. Required tests and artifacts

The implementation plan for PR B through PR D must include:

- synthetic rare-event environments with deterministic event timing;
- truncation-versus-termination tests for event returns;
- per-cost gamma/lambda identity in checkpoints;
- family-split critic shape and gradient-isolation tests;
- multiplier saturation and oscillation metric tests;
- joint-feasibility witness tests;
- deterministic correlation-matrix reporting tests;
- fixed-step and fixed-compute report schema tests;
- GRN architecture-isolation and frozen-dual-config tests.

## 7. Decision summary

The review concerns are accepted as design requirements.

- Rare-event costs will not rely on unmeasured shared-trunk behavior.
- Event-cost `gamma` remains semantically undiscounted by default, while `lambda` receives an explicit bias-variance ablation.
- Correlated multipliers will be monitored for duplicate pressure, saturation, oscillation, and joint infeasibility.
- PR D will report both sample efficiency and compute efficiency.
- GRN policy/reward changes and GRN cost-critic changes will be isolated, with dual dynamics included in adoption criteria.
