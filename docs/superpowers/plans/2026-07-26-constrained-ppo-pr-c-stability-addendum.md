# Lagrangian PPO Stability Addendum Implementation Plan

> [!IMPORTANT]
> **Superseded for actor composition, episode aggregation, estimator scheduling, and feasibility-probe behavior by:**
> `docs/superpowers/specs/2026-07-26-pr-c-lagrangian-stability-correction.md`
> `docs/superpowers/plans/2026-07-26-pr-c-lagrangian-stability-correction.md`
> Where this document conflicts with those files, the correction specification and plan are normative.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete PR C's normative review requirements by adding deterministic constraint-correlation diagnostics, multiplier saturation/oscillation tracking, and a pre-training canonical-action diagnostic probe without changing the reward or automatically decorrelating constraints.

**Architecture:** Keep the independent per-cost dual optimizer from the main PR C plan. Add pure diagnostic modules that consume raw rollout costs, normalized cost advantages, frozen multipliers, and dual update reports; add a separate canonical-action probe that runs a fresh environment before model construction. Backend integration records warning evidence for budget violations and rejects only malformed execution, metadata, or unsupported action semantics.

**Tech Stack:** Python 3.12, NumPy, Gymnasium 0.29.1, Stable-Baselines3 2.3.2, pytest, Ruff, Mypy.

## Global Constraints

- Do not merge, orthogonalize, reweight, or covariance-precondition cost advantages in PR C.
- Do not change the scalar all-cost net-log-growth reward.
- Correlation diagnostics must be deterministic for constant columns and finite for every accepted input.
- Stability diagnostics are observations and promotion warnings; they do not silently alter dual learning rates, budgets, caps, or update intervals.
- A canonical-action probe uses a fresh environment and the maintained zero-action semantics, never the training environment state.
- A probe must complete the configured number of valid episodes; reaching the maximum step count without completion fails closed.
- A probe estimate must use the same aggregation semantics and cost ordering as the Lagrangian schema.
- Probe settings and probe evidence identity must be included in training/checkpoint identity.
- Ordinary `ppo` and `cost_critic_ppo` cannot accept active probe settings.

---

## File Structure

- `trade_rl/rl/lagrangian_diagnostics.py`: correlation matrices, penalty-to-reward magnitude, and serializable per-cost stability tracker.
- `trade_rl/rl/lagrangian_probe.py`: zero-action episode runner, joint-feasibility decision, and canonical evidence payload.
- `trade_rl/integrations/lagrangian_ppo.py`: capture per-rollout correlation and stability diagnostics; no optimizer change.
- `trade_rl/integrations/sb3_training.py`: execute probe before model construction and persist probe evidence.
- `trade_rl/rl/training.py`: typed probe settings and fail-closed inactive-field validation.
- `trade_rl/rl/algorithm_configs.py`: carry probe settings in `LagrangianPPOConfig`.
- `tests/rl/test_lagrangian_diagnostics.py`: pure matrix and trajectory contracts.
- `tests/rl/test_lagrangian_probe.py`: deterministic safe/unsafe probe contracts.
- `tests/integrations/test_sb3_lagrangian_backend.py`: probe runs before learning and warns on budget violations without rejecting training.

---

### Task A: Deterministic correlation and penalty diagnostics

**Files:**
- Create: `trade_rl/rl/lagrangian_diagnostics.py`
- Test: `tests/rl/test_lagrangian_diagnostics.py`

**Interfaces:**
- Consumes: `cost_names: tuple[str, ...]`, raw costs `[samples, costs]`, normalized cost advantages `[samples, costs]`, frozen multipliers `[costs]`, and reward advantages `[samples]`.
- Produces: `ConstraintCorrelationDiagnostics` and `build_constraint_correlation_diagnostics(...)`.

- [ ] **Step 1: Write failing matrix tests**

```python

def test_constraint_correlation_diagnostics_are_deterministic_for_constant_columns() -> None:
    diagnostics = build_constraint_correlation_diagnostics(
        cost_names=("daily_turnover", "execution_cost_fraction"),
        raw_costs=np.asarray([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]),
        normalized_cost_advantages=np.asarray([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]]),
        multipliers=np.asarray([2.0, 3.0]),
        reward_advantages=np.asarray([1.0, -1.0, 0.0]),
    )

    assert diagnostics.raw_cost_correlation[0, 1] == pytest.approx(1.0)
    assert diagnostics.normalized_cost_advantage_correlation[1, 1] == 0.0
    assert np.isfinite(diagnostics.penalty_contribution_correlation).all()
```

Add tests that assert:

- population covariance and correlation are symmetric;
- a non-constant diagonal is `1.0`;
- a constant-column row and column, including its diagonal, are `0.0`;
- penalty contributions equal `raw_cost_advantages * multipliers[None, :]`;
- `penalty_to_reward_l2_ratio` is `||sum_i lambda_i A_i||_2 / max(||A_r||_2, 1e-12)`;
- zero reward norm with non-zero penalty yields a finite ratio using the fixed denominator;
- shape mismatch, duplicate names, empty input, and non-finite values fail closed.

- [ ] **Step 2: Run and verify RED**

Run: `pytest tests/rl/test_lagrangian_diagnostics.py -q`
Expected: import failure because `trade_rl.rl.lagrangian_diagnostics` does not exist.

- [ ] **Step 3: Implement pure diagnostics**

Implement `_population_covariance(...)` and `_deterministic_correlation(...)`. Correlation uses population standard deviation. When either column has standard deviation `<= 1e-12`, set the pair to `0.0`; otherwise divide covariance by the product of standard deviations and clip numerical drift to `[-1.0, 1.0]`.

`ConstraintCorrelationDiagnostics` must contain:

```python
cost_names: tuple[str, ...]
raw_cost_covariance: np.ndarray
raw_cost_correlation: np.ndarray
normalized_cost_advantage_correlation: np.ndarray
penalty_contribution_correlation: np.ndarray
penalty_to_reward_l2_ratio: float
```

Every returned array is copied and read-only.

- [ ] **Step 4: Run tests and static checks**

```text
pytest tests/rl/test_lagrangian_diagnostics.py -q
ruff check trade_rl/rl/lagrangian_diagnostics.py tests/rl/test_lagrangian_diagnostics.py
ruff format --check trade_rl/rl/lagrangian_diagnostics.py tests/rl/test_lagrangian_diagnostics.py
mypy trade_rl/rl/lagrangian_diagnostics.py
```

- [ ] **Step 5: Commit**

```text
git add trade_rl/rl/lagrangian_diagnostics.py tests/rl/test_lagrangian_diagnostics.py
git commit -m "feat: add constraint correlation diagnostics"
```

---

### Task B: Multiplier saturation and oscillation tracker

**Files:**
- Modify: `trade_rl/rl/lagrangian_diagnostics.py`
- Test: `tests/rl/test_lagrangian_diagnostics.py`

**Interfaces:**
- Consumes: one rollout record per cost with `multiplier_after`, `multiplier_update`, `constraint_residual`, and `max_multiplier`.
- Produces: `DualStabilityTracker.record_rollout(...)`, `DualStabilityReport`, `state_dict()`, and `load_state_dict()`.

- [ ] **Step 1: Write failing trajectory tests**

Use one cost with multipliers `[0, 10, 10, 5, 10]`, updates `[1, 1, 0, -1, 1]`, residuals `[1, 1, -1, -1, 1]`, and cap `10`. Assert:

```python
assert report.saturation_fraction == pytest.approx(3 / 5)
assert report.longest_saturation_run == 2
assert report.update_sign_change_frequency == pytest.approx(2 / 3)
assert report.violation_area == pytest.approx(3.0)
```

Sign-change frequency ignores zero updates and divides sign changes by `nonzero_update_count - 1`; fewer than two non-zero updates yields `0.0`.

Add tests for:

- rolling update variance and residual variance over exactly the configured trailing window;
- `first_sustained_satisfaction_rollout` after three consecutive residuals `<= 0`;
- `over_constrained_rollouts_after_satisfaction` counting positive residuals after that first sustained point;
- independent state for every cost;
- zero-history reports;
- deterministic save/load and exact next report;
- schema/cost-order mismatch and non-finite records failing closed.

- [ ] **Step 2: Run and verify RED**

Run: `pytest tests/rl/test_lagrangian_diagnostics.py -q`
Expected: missing tracker symbols.

- [ ] **Step 3: Implement tracker**

`DualStabilityTracker` constructor accepts ordered `cost_names`, `rolling_window=20`, `sustained_satisfaction_rollouts=3`. It stores complete compact scalar histories because PR C rollouts are orders of magnitude fewer than market transitions.

`DualStabilityReport` contains:

```python
rollout_count: int
saturation_fraction: float
longest_saturation_run: int
update_sign_change_frequency: float
rolling_update_variance: float
rolling_constraint_residual_variance: float
violation_area: float
first_sustained_satisfaction_rollout: int | None
over_constrained_rollouts_after_satisfaction: int
```

Saturation uses `math.isclose(multiplier_after, max_multiplier, rel_tol=0.0, abs_tol=1e-12)`. Variance uses NumPy population variance and returns `0.0` for fewer than two observations.

- [ ] **Step 4: Run tests and static checks**

Run the Task A commands.

- [ ] **Step 5: Commit**

```text
git add trade_rl/rl/lagrangian_diagnostics.py tests/rl/test_lagrangian_diagnostics.py
git commit -m "feat: track Lagrangian multiplier stability"
```

---

### Task C: Pre-training zero-action joint-feasibility probe

**Files:**
- Create: `trade_rl/rl/lagrangian_probe.py`
- Modify: `trade_rl/rl/training.py`
- Modify: `trade_rl/rl/algorithm_configs.py`
- Modify: `trade_rl/integrations/sb3_training.py`
- Test: `tests/rl/test_lagrangian_probe.py`
- Test: `tests/rl/test_lagrangian_training_config.py`
- Test: `tests/integrations/test_sb3_lagrangian_backend.py`

**Interfaces:**
- Consumes: a fresh Gymnasium environment factory, `LagrangianSchema`, `episode_count`, and `max_steps_per_episode`.
- Produces: `JointFeasibilityProbeEvidence` and `run_zero_action_joint_feasibility_probe(...)`.

- [ ] **Step 1: Write failing probe tests**

Create deterministic environments that emit `ConstraintCostVector` in `info`, terminate after three steps, and accept a continuous action. Assert:

```python
safe = run_zero_action_joint_feasibility_probe(
    environment_factory=lambda: ProbeEnvironment(cost=0.0),
    schema=schema,
    episode_count=2,
    max_steps_per_episode=4,
)
assert safe.jointly_feasible is True
assert safe.completed_episodes == 2
```

Add tests that:

- record exactly the zero action on every step;
- reject a probe whose estimate exceeds any budget and list the violated cost names;
- aggregate event rates using completed episode denominators;
- reject missing/negative/non-finite costs;
- reject discrete or malformed action spaces;
- fail if an episode does not complete by `max_steps_per_episode`;
- close the environment on success and failure;
- produce deterministic canonical payload and digest.

- [ ] **Step 2: Run and verify RED**

Run: `pytest tests/rl/test_lagrangian_probe.py -q`
Expected: import failure because `trade_rl.rl.lagrangian_probe` does not exist.

- [ ] **Step 3: Implement probe evaluator**

For each episode, create a new environment from the factory, reset with seed equal to the zero-based episode index, and step `np.zeros(action_space.shape, dtype=action_space.dtype)`. Use `CompletedEpisodeCostAccumulator` with `n_envs=1`; mark the actual `terminated` and `truncated` flags. Require exactly one completed episode per fresh environment.

`JointFeasibilityProbeEvidence` contains ordered estimates, denominators, budgets, violated costs, episode count, maximum steps, action shape, and a content digest. A cost is satisfied when `estimate.value <= budget + 1e-12`.

- [ ] **Step 4: Add typed configuration and backend gate**

Append fields to `ResidualTrainingConfig`:

```python
lagrangian_probe_episodes: int = 0
lagrangian_probe_max_steps_per_episode: int = 0
```

For `lagrangian_ppo`, both must be positive. For every other algorithm, both must be zero. Include both in config digest and `LagrangianPPOConfig`.

In `StableBaselines3Backend.train(...)`, after `_validate_training_environment(...)` and `build_algorithm_config(...)` but before wrapping or constructing the training environment/model, run the probe with `self.environment_factory`. Reject with `ValueError("Lagrangian constraint probe is not jointly feasible: ...")` when `jointly_feasible` is false. Store the probe payload in the training architecture/evidence artifact and expose it on the model as `joint_feasibility_probe_evidence` so checkpoint identity can bind its digest.

- [ ] **Step 5: Run targeted tests and commit**

```text
pytest tests/rl/test_lagrangian_probe.py -q
pytest tests/rl/test_lagrangian_training_config.py -q
pytest tests/integrations/test_sb3_lagrangian_backend.py -q
pytest tests/integrations/test_sb3_cost_critic_backend.py -q
ruff check trade_rl/rl/lagrangian_probe.py trade_rl/rl/training.py trade_rl/rl/algorithm_configs.py trade_rl/integrations/sb3_training.py tests/rl/test_lagrangian_probe.py tests/rl/test_lagrangian_training_config.py tests/integrations/test_sb3_lagrangian_backend.py
ruff format --check trade_rl/rl/lagrangian_probe.py trade_rl/rl/training.py trade_rl/rl/algorithm_configs.py trade_rl/integrations/sb3_training.py tests/rl/test_lagrangian_probe.py tests/rl/test_lagrangian_training_config.py tests/integrations/test_sb3_lagrangian_backend.py
mypy trade_rl/rl/lagrangian_probe.py trade_rl/rl/training.py trade_rl/rl/algorithm_configs.py trade_rl/integrations/sb3_training.py
```

```text
git add trade_rl/rl/lagrangian_probe.py trade_rl/rl/training.py trade_rl/rl/algorithm_configs.py trade_rl/integrations/sb3_training.py tests/rl/test_lagrangian_probe.py tests/rl/test_lagrangian_training_config.py tests/integrations/test_sb3_lagrangian_backend.py
git commit -m "feat: gate Lagrangian training with a safe probe"
```

---

### Task D: Integrate stability evidence without optimizer intervention

**Files:**
- Modify: `trade_rl/integrations/lagrangian_ppo.py`
- Modify: `trade_rl/rl/lagrangian_evidence.py`
- Test: `tests/integrations/test_lagrangian_ppo.py`
- Test: `tests/rl/test_lagrangian_evidence.py`

**Interfaces:**
- Consumes: current rollout raw costs, independently normalized cost advantages, frozen multipliers, reward advantages, and `DualUpdateReport` values.
- Produces: logged correlation matrices, penalty ratio, per-cost stability reports, and evidence digest fields.

- [ ] **Step 1: Write failing integration/evidence tests**

Assert one completed rollout records:

- all three correlation matrices in canonical cost order;
- raw cost covariance;
- penalty-to-reward L2 ratio;
- saturation fraction, longest run, sign-change frequency, rolling variances, violation area, first sustained satisfaction, and post-satisfaction over-constraint count;
- explicit pair entries for turnover/execution cost, gross/margin, gross/liquidation, and drawdown/drawdown-stop;
- digest changes when any matrix element or stability statistic changes.

- [ ] **Step 2: Run and verify RED**

Run the two test files above. Expected: missing diagnostic/evidence fields.

- [ ] **Step 3: Integrate after rollout finalization**

Build correlation diagnostics once from the complete rollout before minibatch shuffling. Record stability only after the one post-rollout dual update. Do not feed diagnostics back into combined advantages or controller updates.

Persist `DualStabilityTracker` state through SB3 save/load and include its configuration in `checkpoint_identity_payload()`.

- [ ] **Step 4: Run targeted regressions**

Run Lagrangian PPO, evidence, save/load, Cost Critic PPO parity, and ordinary PPO parity tests.

- [ ] **Step 5: Commit**

```text
git add trade_rl/integrations/lagrangian_ppo.py trade_rl/rl/lagrangian_evidence.py tests/integrations/test_lagrangian_ppo.py tests/rl/test_lagrangian_evidence.py
git commit -m "feat: record correlated dual stability evidence"
```

---

## Self-Review

- Spec coverage: raw-cost covariance/correlation, normalized-advantage correlation, effective-penalty correlation, penalty/reward magnitude, saturation, longest saturation, sign changes, rolling variances, violation area, sustained satisfaction, post-satisfaction over-constraint, and safe probe are each assigned to a testable task.
- Scope boundary: no automatic decorrelation, hierarchical multiplier, covariance-aware optimizer, production budget selection, or promotion decision is added.
- Type consistency: diagnostics use the same ordered cost names as `LagrangianSchema`; probe aggregation reuses `CompletedEpisodeCostAccumulator`; backend receives probe settings through `LagrangianPPOConfig`.
- Placeholder scan: no TBD/TODO or unspecified error-handling steps remain.
