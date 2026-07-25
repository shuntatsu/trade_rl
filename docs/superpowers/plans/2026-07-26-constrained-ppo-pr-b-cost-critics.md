# Constrained Growth PPO — PR B Cost Critics Implementation Plan

> **Execution rule:** Implement with strict RED/GREEN tests. PR B learns and persists cost values but does not alter the actor objective.

**Base:** stacked on `agent/constrained-ppo-design` / PR #188.

**Goal:** Add cost-aware rollout storage, independent cost returns/GAE, family-separated Cost Critics, rare-event diagnostics, and deterministic checkpoint identity while preserving ordinary PPO and the exact net-log-growth reward.

## Non-negotiable invariants

1. Ordinary `ppo` remains unchanged and uses the standard SB3 buffer and policy path.
2. PR B does not subtract cost advantages from reward advantages.
3. The environment scalar reward remains exact all-cost net log growth.
4. Seven optimization costs use the canonical ordering from `CONSTRAINT_COST_NAMES`.
5. Positive funding credit is telemetry only and never receives a Cost Critic head.
6. Reward return/GAE and cost return/GAE remain independently calculated and normalized.
7. Terminal economic events do not bootstrap; time-limit truncations may bootstrap continuous and event values according to explicit terminal observations.
8. Event costs use canonical `gamma_c=1.0`; `lambda_c` is explicit per cost and checkpointed.
9. A missing configured cost, non-finite value, schema mismatch, or reordered head fails closed.
10. PR B saves Cost Critic and cost-rollout identity but introduces no Lagrange multiplier state.

## Cost families

```text
continuous / relatively dense
  - drawdown_excess
  - margin_deficit_fraction
  - gross_exposure_request_excess
  - daily_turnover
  - execution_cost_fraction

rare terminal events
  - drawdown_stop_event
  - forced_liquidation_event
```

The default candidate shares the existing market encoder, then separates the last cost representation layers:

```text
shared critic input
  |- continuous adapter -> five scalar heads
  `- rare-event adapter -> two independent scalar heads
```

A one-trunk/seven-head version remains an ablation, not the default assumption.

## Task 1 — Typed cost schema and per-cost optimization settings

**Create:**
- `trade_rl/rl/cost_learning.py`
- `tests/rl/test_cost_learning.py`

Implement:

- immutable `CostValueSpec` with name, family, `gamma`, `gae_lambda`, value-loss coefficient, and optional auxiliary-event coefficient;
- canonical ordered `CostLearningSchema`;
- explicit default schema with event `gamma=1.0` and initial `lambda` comparison support;
- digest payload suitable for training/checkpoint identity;
- fail-closed duplicate, unknown, missing, or reordered cost names.

RED tests first:

- canonical order equals `CONSTRAINT_COST_NAMES`;
- funding credit cannot become a constraint head;
- event costs reject discounted canonical configuration unless explicitly marked objective-altering;
- schema digest changes when any cost gamma/lambda/loss setting changes.

## Task 2 — Independent cost GAE and aggregation semantics

**Create:**
- `trade_rl/rl/cost_returns.py`
- `tests/rl/test_cost_returns.py`

Implement pure NumPy cost return calculation before SB3 integration:

```python
delta_t = c_t + gamma_c * next_v_t * nonterminal_t - v_t
adv_t = delta_t + gamma_c * lambda_c * nonterminal_t * adv_{t+1}
return_t = adv_t + v_t
```

Required tests:

- independent results for all seven cost columns;
- different `lambda_c` values produce independent expected results;
- event `gamma=1.0` matches Monte Carlo event return on complete episodes;
- true termination does not bootstrap;
- time-limit truncation uses explicitly supplied terminal Cost Critic values;
- one environment ending does not leak into another vector environment;
- zero-event rollout does not imply completed episodes were negative events;
- non-finite costs, values, or bootstrap values fail closed.

## Task 3 — Cost-aware rollout buffer contract

**Create:**
- `trade_rl/integrations/cost_rollout_buffer.py`
- `tests/integrations/test_cost_rollout_buffer.py`

Implement a composable cost storage layer supporting both ordinary and index-backed Dict rollout observations:

- arrays `[n_steps, n_envs, n_costs]` for costs, values, returns, and advantages;
- fixed ordered cost schema;
- `add_cost_transition(...)` validates the compact `info` contract;
- finalization calls the pure cost-return implementation;
- sampled minibatches contain reward fields plus cost values/returns/advantages;
- reset clears every cost array and cached sequence materialization;
- memory estimator includes all cost arrays;
- ordinary rollout state is rejected by constrained buffers and vice versa.

Do not modify SB3's standard `RolloutBuffer` globally.

## Task 4 — Family-separated Cost Critic modules

**Create or extend:**
- `trade_rl/rl/policies.py`
- `trade_rl/rl/cost_critics.py`
- `tests/rl/test_cost_critics.py`
- sequence-policy tests

Implement:

- `ContinuousCostAdapter`;
- `RareEventCostAdapter`;
- one scalar head per enabled cost;
- deterministic ordered tensor output `[batch, n_costs]`;
- optional auxiliary event logits kept separate from cumulative cost values;
- parameter and architecture identity reporting;
- independent per-head losses and diagnostic target statistics.

Initial comparison contract:

1. shared cost trunk + seven heads;
2. shared input + continuous/rare adapters;
3. family split + independent event heads.

Promotion default is family split only after tests show non-trivial rare-event gradients.

Required tests:

- shape/order for flat and sequence policies;
- disabled heads are absent rather than silently zero-filled;
- continuous loss cannot update the rare adapter;
- rare-event loss produces non-zero rare-adapter gradients on synthetic positive samples;
- all-zero event predictor is not accepted solely because MSE is small;
- Brier score, positive support, and calibration-bin diagnostics are deterministic;
- no cost head is included in action or reward-value output.

## Task 5 — Opt-in Cost Critic PPO collector

**Create:**
- `trade_rl/integrations/cost_critic_ppo.py`
- integration tests with deterministic synthetic environments

Add a new opt-in algorithm identifier for PR B, tentatively `cost_critic_ppo`.

The class may subclass SB3 PPO for rollout collection and training reuse, but must:

- instantiate the cost-aware buffer;
- collect `constraint_costs` from compact infos;
- query Cost Critic values without changing action distribution or reward value;
- calculate cost returns after each rollout;
- train reward PPO exactly as before;
- train Cost Critic losses in parallel with explicit coefficients;
- log each head separately;
- never combine reward and cost advantages in PR B.

Synthetic integration tests:

- deterministic continuous costs learn finite values;
- deterministic drawdown-stop and forced-liquidation events provide positive support;
- reward policy update is unchanged when Cost Critic learning is enabled with actor/encoder gradients isolated as configured;
- missing cost info fails closed;
- multi-env event denominators are correct.

## Task 6 — Rare-event supervision and diagnostics

**Create:**
- `trade_rl/rl/cost_diagnostics.py`
- tests for calibration and support metrics

Per head report:

- target mean/std;
- non-zero rate;
- positive sample count;
- value loss and explained variance;
- adapter/head gradient norms;
- dense-to-rare aggregate gradient ratio;
- Brier score and calibration bins for event heads;
- precision-recall inputs without threshold tuning on sealed data.

Optional finite-horizon hazard/event head is allowed only as an explicit ablation. The cumulative event Cost Critic remains authoritative for cost GAE.

## Task 7 — Configuration and checkpoint identity

**Extend:**
- `trade_rl/rl/training.py`
- `trade_rl/rl/algorithm_configs.py`
- `trade_rl/rl/checkpointing.py`
- config/checkpoint tests

Add typed opt-in configuration for:

- enabled ordered costs;
- per-cost `gamma_c` and `lambda_c`;
- value-loss coefficients;
- family architecture variant;
- adapter widths;
- auxiliary-event settings.

Checkpoint/resume requirements:

- algorithm identifier;
- cost schema digest and ordering;
- Cost Critic architecture identity;
- Cost Critic parameters and optimizer state through the saved model;
- cost-rollout schema;
- deterministic rejection of mismatched order, gamma/lambda, or architecture.

No multiplier or EMA dual state belongs in PR B.

## Task 8 — Compute and regression evidence

Record:

- additional parameter count;
- additional rollout-buffer bytes;
- peak memory in smoke tests;
- environment steps/second and update time where practical;
- reward PPO losses unchanged in the controlled no-cost-gradient comparison;
- rare-event positive support in each synthetic test.

Run final verification:

```text
ruff check .
ruff format --check .
mypy .
pytest -q
critical branch coverage
Ubuntu compatibility
Windows compatibility
training image build/probe
```

## Delivery gates

PR B is ready only when:

1. PR A is independently green;
2. standard PPO behavior remains regression-green;
3. all seven cost returns are independently reproducible;
4. event critics beat the zero-only baseline in synthetic calibration tests;
5. no cost advantage enters actor optimization;
6. checkpoint/resume rejects cost identity mismatches;
7. fixed memory ceilings include Cost Critic rollout state;
8. full repository CI succeeds on the exact final head.
