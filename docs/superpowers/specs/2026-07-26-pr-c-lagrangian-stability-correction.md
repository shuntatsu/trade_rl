# PR C Lagrangian Stability Correction Design

Date: 2026-07-26
Status: normative
Applies to: PR #191 (`agent/constrained-ppo-lagrangian`)

## 1. Decision

PR C keeps the environment reward exactly equal to all-cost net log growth and keeps seven constraint costs outside the scalar reward.

The initial PR C implementation is corrected in four places before promotion:

1. compose reward and cost advantages in their original units before PPO normalization;
2. separate completed-episode estimation from dual-actuator scheduling;
3. use explicit elapsed-time and censoring semantics for episode aggregation;
4. downgrade the zero-action joint-feasibility witness from a hard proof/gate to a canonical-action diagnostic probe.

The independently normalized actor formula in the original PR C plan is superseded. The hard rejection behavior in the stability addendum's zero-action witness is also superseded.

## 2. Non-negotiable invariants

- The scalar reward remains exact net interval log growth.
- Ordinary `ppo` and PR B `cost_critic_ppo` behavior remain unchanged.
- A zero-multiplier `lagrangian_ppo` update remains exactly equivalent to ordinary PPO after an identical RNG reset.
- Lagrange multipliers are frozen for one complete rollout and all PPO epochs trained from that rollout.
- Actor composition uses raw reward and raw cost advantages in canonical cost order.
- Only the final combined advantage is normalized for PPO.
- Every accepted cost observation is retained during warmup and update-interval skips.
- Shadow-comparator truncation is censored external data, not a safe completed policy episode.
- Unknown truncation reasons, non-positive elapsed time, reordered costs, non-finite values, and incompatible checkpoint state fail closed.
- Integral EMA dual control remains the PR C optimizer baseline. PID and augmented-Lagrangian controllers remain PR D ablations.

## 3. Actor objective

### 3.1 Raw Lagrangian composition

For one rollout-frozen multiplier vector `lambda`, compose each sampled transition as:

```text
A_lagrangian = A_reward - sum_i(lambda_i * A_cost_i)
```

No cost column is independently normalized before multiplication by its Lagrange multiplier.

If PPO advantage normalization is enabled and the minibatch contains more than one sample, normalize only the combined vector using the same Torch expression as pinned SB3 PPO 2.3.2:

```python
A_actor = (A_lagrangian - A_lagrangian.mean()) / (
    A_lagrangian.std() + 1e-8
)
```

If normalization is disabled or the minibatch has one sample, use `A_lagrangian` directly.

This preserves the units of each multiplier. It also preserves ordinary PPO exactly when every multiplier is zero because the composed vector becomes the original reward advantage before the same SB3 normalization.

### 3.2 Cost-unit invariance

For a positive scale factor `k`, replacing one cost by:

```text
A_cost' = k * A_cost
lambda' = lambda / k
```

must leave `A_lagrangian` and the actor update unchanged.

A complete dual-configuration unit conversion additionally requires:

```text
budget' = k * budget
initial_multiplier' = initial_multiplier / k
max_multiplier' = max_multiplier / k
dual_learning_rate' = dual_learning_rate / k^2
```

The repository uses fixed canonical cost units, but pure tests must prove the actor composition invariant and document the complete controller conversion.

### 3.3 Diagnostics versus optimization

Standardized per-cost advantages may still be calculated for correlation and calibration diagnostics. They are observational only and must never feed the actor loss or dual update.

The penalty-to-reward magnitude diagnostic uses the raw contribution:

```text
penalty = sum_i(lambda_i * A_cost_i)
ratio = ||penalty||_2 / max(||A_reward||_2, 1e-12)
```

A second standardized-correlation matrix may be recorded separately, but it cannot be labeled as the effective actor penalty.

## 4. Episode completion and censoring

### 4.1 Completion classification

Each stored transition carries an explicit completion kind:

```text
NONE
ECONOMIC_TERMINATION
TIME_LIMIT_COMPLETION
CENSORED_EXTERNAL_TRUNCATION
```

Classification rules are:

- `terminated=True` -> `ECONOMIC_TERMINATION`;
- `truncated=True` with reason beginning `shadow_` -> `CENSORED_EXTERNAL_TRUNCATION`;
- `truncated=True` with the configured time-limit reason or the maintained time-limit contract -> `TIME_LIMIT_COMPLETION`;
- neither flag -> `NONE`;
- both flags, an unknown truncation reason, or a completion reason inconsistent with the flags -> fail closed.

### 4.2 Aggregation behavior

`ECONOMIC_TERMINATION` and `TIME_LIMIT_COMPLETION` finalize an episode and contribute to constraint estimates.

`CENSORED_EXTERNAL_TRUNCATION` clears that environment's unfinished episode state because the Gymnasium environment resets, but contributes to neither numerator nor denominator. It increments a separate censored-episode counter.

Cost-return and GAE bootstrapping keep their existing truncation semantics. Censoring changes completed-episode dual statistics only; it does not silently rewrite Cost Critic targets.

## 5. Time-aware episode sufficient statistics

Every cost transition stores finite positive `transition_elapsed_hours` from the dataset clock. A constant configured decision interval may not be substituted when the actual transition duration is available.

For each environment, the accumulator maintains:

```text
cost sums
elapsed hours
step count
event occurrence state
```

On valid episode completion, compute one episode statistic per cost:

| Cost | Episode statistic | Unit |
|---|---|---|
| `drawdown_excess` | `sum(drawdown_excess_t * elapsed_hours_t / 24)` | drawdown-days |
| `drawdown_stop_event` | whether the event occurred once | event/episode |
| `margin_deficit_fraction` | `sum(margin_deficit_fraction_t * elapsed_hours_t / 24)` | deficit-fraction-days |
| `forced_liquidation_event` | whether the event occurred once | event/episode |
| `gross_exposure_request_excess` | arithmetic mean across policy decisions | excess/decision |
| `daily_turnover` | `sum(daily_turnover_t * elapsed_hours_t / 24) / sum(elapsed_hours_t / 24)` | turnover/day |
| `execution_cost_fraction` | sum across transitions | fraction/episode |

The maintained transition name `drawdown_excess` remains unchanged in PR C for schema compatibility. Evidence and documentation must label its episode aggregate as `drawdown_excess_area_days`; it must not be described as maximum drawdown.

Maximum drawdown remains a walk-forward selection and promotion metric rather than being inferred from the area statistic.

## 6. Pooled estimator and dual scheduling

### 6.1 Separation of estimator and actuator

For every valid completed episode, add its episode statistic to a per-cost pending numerator and increment the pending denominator. This happens during:

- warmup;
- non-update rollouts;
- rollouts before minimum support is reached;
- rollouts whose multiplier remains frozen for another reason.

The estimator never discards valid observations solely because the dual actuator is not scheduled to move.

### 6.2 Minimum support

Each `LagrangianConstraintSpec` adds:

```text
minimum_completed_episodes: int
```

The value must be positive and becomes part of schema and checkpoint identity.

Canonical PR C defaults are:

- continuous/dense costs: `1` completed episode;
- `drawdown_stop_event`: `20` completed episodes;
- `forced_liquidation_event`: `20` completed episodes.

When support is below the threshold, the update report uses `skip_reason="insufficient_completed_episodes"` and retains all pending numerator and denominator state.

### 6.3 Pooled estimate

At an eligible dual update, calculate:

```text
raw_estimate = pending_numerator / pending_denominator
```

The pending state is reset only after a successful finite dual update. A validation failure leaves the prior committed state untouched.

### 6.4 Denominator-aware EMA

Let `n` be the pending completed-episode denominator consumed by the update. Use:

```text
beta_effective = ema_beta ** n
ema_after = raw_estimate                         # first initialized estimate
ema_after = beta_effective * ema_before
          + (1 - beta_effective) * raw_estimate  # later estimates
```

This prevents a one-episode estimate and a twenty-episode estimate from receiving identical information weight.

### 6.5 Integral dual update

After the EMA update:

```text
constraint_residual = ema_after - budget
lambda_after = clip(
    lambda_before + dual_learning_rate * constraint_residual,
    0,
    max_multiplier,
)
```

One update occurs at most once per cost after one rollout. No update occurs inside an actor or Cost Critic minibatch loop.

## 7. Dual-boundary semantics

`DualUpdateReport` separates lower and upper boundaries:

```text
at_lower_bound: bool
at_upper_cap: bool
```

The term `saturated` is retained only as a compatibility alias for `at_upper_cap` during migration and is then removed from new evidence schemas.

A multiplier equal to zero is not an upper-cap saturation event. Stability metrics named `saturation_fraction` and `longest_saturation_run` count `at_upper_cap` only.

Reports also include:

```text
pending_numerator_before
pending_denominator_before
consumed_denominator
censored_episode_count
raw_estimate
ema_estimate
constraint_residual
multiplier_before
multiplier_after
skip_reason
```

## 8. Canonical-action feasibility probe

The previous `joint-feasibility witness` is renamed `canonical-action feasibility probe`.

The probe records the behavior of one explicit canonical action:

- target-weight mode: zero target weights, representing cash;
- residual mode: zero residual, representing the configured baseline policy.

Because these actions have different economic meaning and neither spans the feasible policy set, probe failure does not prove joint infeasibility.

PR C behavior is therefore:

- always record probe action semantics, estimates, denominators, budgets, violated costs, completion kinds, and digest;
- emit a prominent warning when the probe violates a budget;
- do not reject training solely because the probe fails;
- fail closed only when the probe itself cannot be evaluated deterministically because of malformed action space, missing cost data, unknown completion semantics, non-finite values, or incomplete configured episodes.

A true feasibility search using multiple policies or an optimization oracle belongs to PR D.

## 9. Checkpoint and evidence identity

Checkpoint and architecture identity include:

- raw-composition actor mode identifier;
- cost order;
- every aggregation semantic and unit label;
- `minimum_completed_episodes` per cost;
- pending numerator and denominator state;
- censored-episode counts;
- elapsed-time accumulator state;
- EMA values and initialization state;
- multiplier values and update counts;
- lower-bound and upper-cap semantics version;
- canonical-action probe settings and evidence digest.

Loading an older PR C checkpoint without these fields fails with an explicit schema-version mismatch. Silent migration is not allowed because the optimized objective and estimator semantics changed.

## 10. Required tests

### 10.1 Actor tests

- raw composition matches an explicit NumPy/Torch calculation;
- only the final combined advantage is normalized;
- independent cost normalization is absent from the actor path;
- zero multipliers retain exact ordinary-PPO policy-state parity;
- `A_cost *= k` and `lambda /= k` retain identical combined advantages and actor update;
- multiplier snapshot remains identical across every minibatch and epoch of one rollout;
- non-finite advantages or multipliers fail closed.

### 10.2 Completion tests

- economic termination contributes to numerator and denominator;
- maintained time-limit completion contributes to numerator and denominator;
- `shadow_*` truncation clears state but contributes to neither;
- censored count increments deterministically;
- unknown truncation reason fails closed;
- one vector environment's completion cannot leak into another.

### 10.3 Time-aware aggregation tests

- irregular elapsed intervals produce the exact drawdown-area value;
- turnover/day equals time-weighted turnover rate, not an unweighted step mean;
- margin-deficit area uses elapsed days;
- execution-cost fraction remains an episode sum;
- event costs cannot occur more than once in one episode;
- non-positive or non-finite elapsed time fails closed.

### 10.4 Estimator and controller tests

- warmup observations remain pending and affect the first eligible update;
- update-interval skips retain pending state;
- event updates wait for twenty completed episodes;
- pooled numerator/denominator is invariant to rollout partitioning;
- denominator-aware EMA matches the exact formula;
- pending state resets only after a successful update;
- zero is reported as lower bound, not upper-cap saturation;
- upper-cap saturation metrics ignore lower-bound rollouts;
- save/load reproduces the next estimator and multiplier update exactly.

### 10.5 Probe tests

- target-weight zero action is recorded as cash semantics;
- residual zero action is recorded as baseline semantics;
- budget violation records a warning but does not reject training;
- malformed or non-deterministic probe execution fails closed;
- probe digest changes with action semantics, estimate, denominator, budget, or completion classification.

## 11. Implementation decomposition

The correction is implemented as three reviewable phases on PR #191:

1. **Actor composition correction** — raw combination, final-only PPO normalization, invariance and parity tests.
2. **Episode estimator correction** — elapsed hours, completion classification, censoring, pooled support-aware estimator, denominator-aware EMA, checkpoint migration rejection.
3. **Stability evidence correction** — upper-cap semantics, corrected raw penalty diagnostics, canonical-action probe warning behavior, deterministic evidence.

Each phase must pass its targeted tests before the next phase begins. Full exact-head verification is required after all three phases.

## 12. Out of scope

PR C does not add:

- production budget selection;
- PID, proportional, derivative, momentum, or augmented-Lagrangian control;
- automatic covariance preconditioning or constraint decorrelation;
- CVaR, OCE, quantile, or distributional constraint critics;
- model-capacity or GRN changes;
- sealed-test tuning;
- selection-gate changes;
- scalar reward shaping.

## 13. Self-review

- Placeholder scan: no TBD, TODO, or unspecified behavior remains.
- Objective consistency: the actor uses the raw Lagrangian advantage and only final PPO normalization.
- Unit consistency: every episode aggregate has an explicit unit and elapsed-time rule.
- Completion consistency: policy-complete and externally censored episodes are distinguished.
- State consistency: pending observations, censor counts, elapsed-time state, EMA, and multipliers are checkpointed together.
- Scope consistency: Integral EMA remains the only PR C controller; advanced controllers remain isolated ablations.
