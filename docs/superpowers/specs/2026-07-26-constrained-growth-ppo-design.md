# Constrained Growth PPO Design

Date: 2026-07-26
Status: proposed implementation design
Branch: `agent/constrained-ppo-design`

## 1. Purpose

Extend the existing growth-optimal PPO route without changing its primary economic objective.

The maintained reward remains:

\[
r_t = 100 \log\left(\frac{V^{net}_{t+1}}{V^{net}_t}\right)
\]

The implementation adds independently measured safety and execution costs, dedicated cost-value estimation, action-path diagnostics, and a stabilized Lagrangian policy update. Risk must not be hidden inside fixed reward shaping coefficients.

## 2. Scope decomposition

The full research direction is intentionally split into independent phases so each change can be tested and compared without obscuring causality.

### Phase 1: constrained PPO foundation

This design covers:

1. complete action-path diagnostics;
2. canonical per-transition cost signals;
3. episode-level constraint summaries;
4. one reward critic plus dedicated cost critics;
5. stabilized Lagrange multiplier updates;
6. training telemetry, persistence, and deterministic resume;
7. ablation profiles for `gamma` and `gae_lambda`;
8. strict backward compatibility for ordinary PPO and the existing growth-optimal profile.

### Later independent phases

These are explicitly outside this first implementation:

- state-dependent exploration standard deviation;
- PopArt normalization;
- VALUE/RISK tokens and residual critic adapters;
- execution domain randomization;
- regime-balanced episode sampling;
- auxiliary quantile and hazard heads;
- self-supervised encoder pretraining;
- recurrent decision memory;
- large-model capacity ablation.

They require separate specs and PRs after Phase 1 produces trustworthy diagnostics.

## 3. Design invariants

The following properties are non-negotiable.

1. The growth reward formula and its telescoping property remain unchanged.
2. Ordinary PPO remains the default algorithm and produces identical behavior when constraints are disabled.
3. Constraint costs are diagnostic environment outputs, not edits to the scalar environment reward.
4. Hard pre-trade and emergency limits remain active even when constrained PPO is disabled.
5. No future data may enter observations, costs, sampling decisions, or normalization statistics.
6. Cost limits, multiplier state, normalizer state, and architecture identity are checkpointed and validated on resume.
7. Training, selection, and sealed-test metrics remain separate.
8. A shadow comparator failure must not be treated as a true hybrid MDP termination.

## 4. Canonical action path

Every environment decision records five action representations.

1. `policy_action`: bounded action sampled by the policy.
2. `feasible_target`: result after portfolio and exposure feasibility mapping.
3. `requested_target`: result after no-trade band, entry/exit threshold, emergency override, and tradability handling.
4. `submitted_order_target`: target actually passed to the execution coordinator.
5. `filled_weight`: realized portfolio weight after execution and partial fills.

The environment emits compact scalar diagnostics:

- `policy_to_feasible_l1`;
- `feasible_to_requested_l1`;
- `requested_to_filled_l1`;
- `policy_to_filled_l1`;
- per-stage maximum absolute deviation;
- stage-specific change flags.

These values are never used as reward penalties in Phase 1. They reveal whether PPO is learning against materially different actions from those executed.

## 5. Constraint cost contract

A new immutable transition object, conceptually `ConstraintCostVector`, contains normalized, non-negative cost signals. Each field has one explicit unit and aggregation rule.

### 5.1 Drawdown excess

Per-step continuous cost:

\[
c^{dd}_t = \max(0, DD_t - DD_{budget})
\]

Default budget: 10%.

This measures time and magnitude beyond the soft budget while the existing 20% hard stop remains unchanged.

### 5.2 Drawdown-stop event

Binary terminal-event cost:

\[
c^{ddstop}_t = 1
\]

only on the transition that terminates the hybrid account for `drawdown_stop`; otherwise zero.

The constraint is evaluated as an episode event rate, not a step mean.

### 5.3 Margin deficit

Per-step continuous cost:

\[
c^{margin}_t = \frac{\max(0, margin\ deficit_t)}{\max(V_0, \epsilon)}
\]

It uses initial capital as the stable denominator and is clipped only by an explicit configuration ceiling used for numerical protection.

### 5.4 Forced liquidation event

Binary event cost equal to one when an economically forced liquidation occurs because of insolvency, minimum equity, maintenance margin, or incomplete emergency liquidation. Routine end-of-episode flattening is excluded.

### 5.5 Gross-exposure request excess

Continuous cost calculated before feasibility projection:

\[
c^{gross}_t = \max\left(0, \sum_i |w^{policy}_{i,t}| - G_{max}\right)
\]

This measures how often the policy asks the mapper to repair an infeasible gross exposure. It does not replace the hard projection.

### 5.6 Daily turnover

Incremental cost in daily-equivalent units:

\[
c^{turn}_t = turnover_t \times \frac{24}{decision\ hours}
\]

The episode constraint compares the mean daily turnover to its configured budget. Both requested and filled turnover are logged; filled turnover is authoritative for the constraint.

### 5.7 Execution-cost fraction

Per-step cost:

\[
c^{exec}_t = \frac{fees_t + spread_t + impact_t + funding_t + borrow_t}{\max(V^{net}_{t}, \epsilon)}
\]

Signed rebates are reported separately and cannot silently reduce a non-negative safety cost. The growth reward continues to use actual net equity and therefore still reflects all signed economics.

## 6. Aggregation semantics

Each cost declares one aggregation type:

- `step_sum`: drawdown excess, margin deficit, execution-cost fraction;
- `episode_event`: drawdown stop, forced liquidation;
- `time_normalized_mean`: daily turnover;
- `request_mean`: gross-exposure request excess.

The code must not treat all costs as interchangeable step rewards. The rollout buffer stores transition costs, while the constraint evaluator computes correctly normalized episode or rollout estimates.

Truncated episodes are handled as follows:

- time-limit truncation bootstraps reward and continuous-cost values;
- true economic termination does not bootstrap;
- shadow-only failure remains truncation for the hybrid process;
- event costs are counted only when their hybrid event occurs.

## 7. Constrained PPO architecture

A new opt-in algorithm identifier, `lagrangian_ppo`, reuses the maintained PPO policy and rollout collection but adds cost returns and dual optimization.

### 7.1 Shared feature extractor

The existing causal multi-timeframe TCN and cross-asset encoder remain shared. Phase 1 intentionally avoids changing encoder capacity so improvements can be attributed to the constrained objective.

### 7.2 Reward critic

The existing scalar reward-value head remains authoritative for reward GAE.

### 7.3 Cost critics

Each enabled constraint receives a separate scalar value head and separate optimizer-visible loss. The first implementation uses a shared compact cost trunk followed by one head per constraint:

- input: the same pooled portfolio and global features used by the reward critic;
- shared trunk: two hidden layers using the existing activation convention;
- outputs: one scalar per enabled cost;
- no cost head for disabled constraints.

Cost critics use independent running target statistics for diagnostics, but Phase 1 does not apply PopArt transformations.

Separate cost critics are required because the constraints have different horizons and scales. A single summed-cost critic is prohibited.

## 8. Advantage and policy objective

Reward GAE remains:

\[
A^r_t = GAE(r_t, V^r, \gamma_r, \lambda_r)
\]

Each cost obtains independent GAE:

\[
A^{c_i}_t = GAE(c_{i,t}, V^{c_i}, \gamma_{c_i}, \lambda_{c_i})
\]

The clipped policy surrogate uses:

\[
A^{combined}_t = A^r_t - \sum_i \lambda_i \widetilde{A}^{c_i}_t
\]

where each cost advantage is normalized with its own rollout statistics. Reward and cost advantages must never be concatenated and globally normalized together.

The PPO ratio, clipping, KL guard, entropy term, and reward-value loss retain their existing semantics. Cost-value losses are added with explicit coefficients and reported separately.

## 9. Stabilized dual updates

Each multiplier is represented by a non-negative transformed parameter or explicit projected value.

For constraint `i`:

\[
\bar C_i \leftarrow \beta_i \bar C_i + (1-\beta_i)\hat C_i
\]

\[
\lambda_i \leftarrow clip\left(\lambda_i + \eta_i(\bar C_i-d_i), 0, \lambda^{max}_i\right)
\]

Required stabilization controls:

- independent budget `d_i`;
- independent dual learning rate;
- exponential moving average coefficient;
- warm-up measured in completed rollouts;
- configurable update interval;
- maximum multiplier;
- optional minimum multiplier of zero only;
- finite-value and divergence checks;
- checkpoint persistence;
- deterministic resume tests.

Multiplier updates occur after rollout statistics are finalized, not per minibatch. PPO epochs within a rollout use a frozen multiplier vector.

## 10. Configuration

A new optional section is introduced under training configuration:

```json
{
  "algorithm": "lagrangian_ppo",
  "constraints": {
    "enabled": true,
    "drawdown_excess": {
      "budget": 0.10,
      "limit": 0.0,
      "dual_learning_rate": 0.001,
      "ema_beta": 0.95,
      "lambda_max": 100.0
    }
  }
}
```

The exact serialized schema will follow existing typed configuration conventions. Unknown constraint names fail closed. Enabling `constraints` with ordinary `ppo` is invalid rather than silently ignored.

The growth-optimal profile remains unchanged. A new profile is added for constrained growth PPO so old experiments preserve content identity.

## 11. Gamma and GAE ablation

Phase 1 adds three explicit experiment profiles with otherwise identical configuration:

1. `gamma=1.0`, `gae_lambda=0.95` — canonical economic objective;
2. `gamma=1.0`, `gae_lambda=0.97` — lower bias, potentially higher variance;
3. `gamma=0.9995`, `gae_lambda=0.95` — optimization-stability comparison only.

The third profile is not allowed to replace the canonical profile solely on selection score. Its result must be reported as an objective-misaligned ablation because discounting changes the exact telescoping terminal-wealth objective.

Cost gamma values default to 1.0 for episode-integrated constraints. Any alternative must be explicit per constraint and included in experiment identity.

## 12. Telemetry and artifacts

Training artifacts include:

- constraint schema and budgets;
- cost-critic architecture;
- per-rollout estimated cost returns;
- raw and EMA constraint estimates;
- multiplier values and updates;
- reward and cost explained variance;
- reward and per-cost value losses;
- action-path distances;
- event counts and episode denominators;
- constraint advantage mean and standard deviation;
- gradient norms for encoder, actor, reward critic, and cost trunk;
- PPO KL, clip fraction, entropy, and effective sample statistics.

The checkpoint manifest includes:

- algorithm identifier;
- constraint configuration digest;
- enabled cost ordering;
- multiplier vector;
- EMA state;
- cost-value parameters;
- optimizer states;
- rollout-buffer cost schema.

Resume must reject mismatched cost ordering, budgets, or algorithm mode.

## 13. Selection and sealed-test policy

Constrained training does not weaken existing model-selection gates.

Reports add:

- mean and worst-seed constraint values;
- drawdown-stop and liquidation rates with denominators;
- constraint satisfaction by nominal and adverse execution scenario;
- multiplier saturation frequency;
- performance difference from unconstrained growth PPO;
- raw-to-filled action distortion distribution.

A model is ineligible if a required constraint is violated on the selection period or required `joint_2x` scenario, even if return is higher.

The sealed test is evaluated once under the existing evidence policy. It cannot be used to tune budgets or multiplier parameters.

## 14. Failure handling

The implementation fails closed when:

- a configured cost cannot be produced by the environment;
- a cost contains NaN, infinity, or an invalid negative value;
- episode denominators are zero for an event-rate update;
- checkpoint constraint identity differs;
- multipliers become non-finite;
- a cost critic output shape disagrees with enabled constraints;
- ordinary PPO receives constrained rollout state;
- constrained PPO receives a legacy rollout buffer lacking cost fields.

For sparse events with no completed episode in a rollout, the previous EMA is retained and no event-rate dual update occurs. The condition is logged explicitly.

## 15. Test strategy

### Unit tests

- exact formulas and units for every cost;
- exclusion of routine terminal flattening from forced-liquidation cost;
- correct gross-exposure excess before projection;
- requested versus filled turnover semantics;
- signed economic cost versus non-negative safety cost handling;
- action-path distances;
- true termination versus truncation bootstrapping;
- independent cost GAE;
- projected multiplier update, warm-up, cap, and frozen-per-rollout behavior;
- serialization and resume identity.

### Integration tests

- ordinary PPO produces no constraint artifacts and preserves previous configuration behavior;
- constrained PPO collects reward and all enabled costs from multiple environments;
- one deliberately unsafe synthetic environment increases the corresponding multiplier;
- one safe environment permits multiplier decay toward zero;
- reward remains exact net log growth under constrained PPO;
- shadow-only failure does not create hybrid event costs;
- checkpoint save/resume reproduces the next multiplier update and policy step;
- action diagnostics survive the training info compaction layer.

### Regression tests

- existing growth-optimal reward tests remain unchanged;
- existing PPO, BC-to-PPO, sequence-policy, execution, checkpoint, and walk-forward suites continue to pass;
- parameter and rollout-memory ceilings are enforced;
- deterministic CPU tests cover dual logic without requiring CUDA.

## 16. Delivery plan

Implementation is divided into independent PRs:

1. **PR A — action and cost contracts**
   - action-path diagnostics;
   - canonical environment cost vector;
   - telemetry and unit tests;
   - no algorithm change.

2. **PR B — constrained rollout and critics**
   - cost-aware rollout buffer;
   - cost returns and GAE;
   - cost critic heads;
   - checkpoint schema;
   - no dual policy penalty yet.

3. **PR C — stabilized Lagrangian PPO**
   - combined advantage;
   - frozen-per-rollout multipliers;
   - EMA dual updates;
   - failure handling and resume tests.

4. **PR D — experiment profiles and evaluation**
   - constrained growth profile;
   - gamma/GAE ablations;
   - selection and reporting extensions;
   - nominal and adverse scenario comparisons.

Each PR must be independently green and must not require later PRs to preserve existing functionality.

## 17. Success criteria

Phase 1 is complete only when all of the following hold:

1. existing ordinary PPO and growth-optimal tests pass unchanged;
2. the scalar reward remains exact all-cost net log growth;
3. every enabled constraint is independently measurable and reproducible;
4. cost critics train and expose finite, correctly shaped outputs;
5. multipliers respond in the correct direction on controlled unsafe and safe environments;
6. save/resume preserves constrained-training identity and next-step behavior;
7. action-path distortion is visible in artifacts;
8. walk-forward can compare unconstrained and constrained profiles under identical folds and seeds;
9. required adverse execution evaluation remains enforced;
10. no model-capacity change is mixed into the constrained-objective experiment.

## 18. Rejected alternatives

### Fixed penalties in the reward

Rejected because they change the primary economic objective, duplicate net-equity losses, and require fragile coefficient tuning.

### One summed safety cost

Rejected because event probabilities, turnover, margin deficit, and execution costs have incompatible units and horizons.

### Per-minibatch multiplier updates

Rejected because PPO reuses each rollout for multiple epochs; changing multipliers inside those epochs makes the optimization target inconsistent.

### Immediate CPO implementation

Rejected for the first phase because it introduces a larger optimizer and trust-region change before the environment cost contract is verified.

### Simultaneous model enlargement

Rejected because it would prevent attribution of improvements or regressions to constrained optimization versus capacity.
