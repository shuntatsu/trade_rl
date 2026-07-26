# PR C Lagrangian Stability Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct PR #191 so the PPO actor uses a unit-consistent raw Lagrangian advantage, completed-episode estimates are time-aware and censor-aware, dual observations are never discarded by scheduling, and feasibility/stability evidence is diagnostic rather than falsely authoritative.

**Architecture:** Keep the exact all-cost net-log-growth reward, PR B Cost Critics, and pinned SB3 PPO 2.3.2 loop. Replace actor-side independent cost normalization with raw reward-minus-cost composition followed by final PPO normalization; split episode completion and sufficient-statistic logic into `lagrangian_episode.py`; retain schema and dual control in `lagrangian.py`; add separate diagnostic and canonical-action probe modules. Every objective, aggregation unit, pending estimator state, and probe semantic is bound into checkpoint/evidence identity.

**Tech Stack:** Python 3.12, NumPy, PyTorch 2.3.1, Stable-Baselines3 2.3.2, Gymnasium 0.29.1, pytest, Ruff, Mypy.

## Global Constraints

- The scalar reward remains exactly the maintained all-cost net-log-growth reward.
- Ordinary `ppo` and PR B `cost_critic_ppo` behavior remain unchanged.
- A zero-multiplier `lagrangian_ppo` update must remain exactly equivalent to ordinary PPO after an identical RNG reset.
- Lagrange multipliers remain frozen for one complete rollout and all PPO epochs trained from that rollout.
- Actor composition uses raw reward and raw cost advantages in canonical cost order; per-cost standardized advantages are diagnostics only.
- Only the final combined advantage is normalized, using the exact pinned SB3 Torch expression.
- Every accepted completed-episode observation is retained during warmup, update-interval skips, and minimum-support skips.
- Shadow-comparator truncation is censored external data and contributes to neither numerator nor denominator.
- Unknown truncation reasons, malformed completion metadata, non-positive elapsed time, reordered costs, non-finite values, and incompatible checkpoint state fail closed.
- Integral EMA dual control is the only optimizer implemented in PR C. PID, augmented Lagrangian, automatic decorrelation, CVaR, OCE, quantile critics, production budget selection, sealed-test tuning, and reward shaping remain out of scope.
- Do not weaken existing assertions or replace exact parity checks with tolerances unless floating-point serialization makes bitwise comparison impossible and the reason is documented.

---

## File Structure

- `trade_rl/rl/lagrangian_advantages.py`: raw Lagrangian advantage composition and diagnostic-only normalization helpers.
- `trade_rl/integrations/lagrangian_ppo.py`: pinned PPO actor loop, frozen multiplier use, post-rollout estimator/controller update, logger integration, and checkpoint identity.
- `trade_rl/integrations/cost_rollout_buffer.py`: aligned cost, elapsed-time, termination, truncation, and completion-kind storage.
- `trade_rl/rl/environment_info.py`: authoritative `transition_elapsed_hours` and termination-reason step metadata.
- `trade_rl/rl/lagrangian_episode.py`: completion classification, time-aware completed-episode sufficient statistics, censoring, and accumulator state.
- `trade_rl/rl/lagrangian.py`: typed aggregation schema, minimum-support configuration, pooled pending estimator, denominator-aware EMA, integral controller, and deterministic state.
- `trade_rl/rl/training.py`: opt-in Lagrangian vectors, probe settings, inactive-field validation, and training digest.
- `trade_rl/rl/algorithm_configs.py`: `LagrangianPPOConfig` and canonical schema construction.
- `trade_rl/integrations/sb3_training.py`: backend model construction/load, probe execution, architecture metadata, and warning evidence.
- `trade_rl/rl/lagrangian_probe.py`: canonical-action feasibility probe and deterministic payload.
- `trade_rl/rl/lagrangian_diagnostics.py`: raw penalty, correlation, cap-saturation, and oscillation diagnostics.
- `trade_rl/rl/lagrangian_evidence.py`: canonical JSON evidence and digest.
- `tests/rl/test_lagrangian_advantages.py`: raw composition, unit invariance, validation.
- `tests/integrations/test_lagrangian_ppo.py`: actor direction, frozen lambda, update ordering, logging.
- `tests/integrations/test_lagrangian_checkpoint_roundtrip.py`: exact model/controller/accumulator continuation.
- `tests/integrations/test_cost_rollout_buffer.py`: elapsed time and completion metadata alignment.
- `tests/integrations/test_cost_rollout_alignment.py`: vector-environment transition alignment.
- `tests/rl/test_environment_info_service.py`: authoritative elapsed-time info field.
- `tests/rl/test_lagrangian_episode.py`: completion classification, time-aware aggregation, censoring, state round-trip.
- `tests/rl/test_lagrangian.py`: schema and aggregation identity.
- `tests/rl/test_lagrangian_dual.py`: pending estimator, minimum support, EMA, boundaries, state.
- `tests/rl/test_lagrangian_training_config.py`: configuration, inactive defaults, digest.
- `tests/integrations/test_sb3_lagrangian_backend.py`: construction/load/probe behavior.
- `tests/rl/test_lagrangian_probe.py`: cash/baseline semantic probe contracts.
- `tests/rl/test_lagrangian_diagnostics.py`: corrected raw-penalty and stability metrics.
- `tests/rl/test_lagrangian_evidence.py`: evidence schema and digest.
- `docs/verification/2026-07-26-constrained-ppo-pr-c-correction-verification.md`: exact-head verification evidence.

---

### Task 1: Replace actor-side independent normalization with raw composition

**Files:**
- Modify: `trade_rl/rl/lagrangian_advantages.py`
- Modify: `tests/rl/test_lagrangian_advantages.py`

**Interfaces:**
- Consumes: reward advantages `[batch]`, cost advantages `[batch, costs]`, frozen non-negative multipliers `[costs]`.
- Produces: `combine_lagrangian_advantages(*, reward_advantages, cost_advantages, multipliers) -> NDArray[np.float64]` returning `A_reward - cost_advantages @ multipliers` without normalization.
- Retains: `normalize_cost_advantages(...)` only for diagnostics; no actor caller may use it.

- [ ] **Step 1: Replace old expected behavior with failing raw-composition tests**

```python
def test_combine_lagrangian_advantages_preserves_raw_units() -> None:
    reward = np.asarray([3.0, -1.0, 2.0], dtype=np.float64)
    costs = np.asarray(
        [[2.0, 4.0], [1.0, 0.0], [5.0, 2.0]],
        dtype=np.float64,
    )
    multipliers = np.asarray([0.5, 0.25], dtype=np.float64)

    combined = combine_lagrangian_advantages(
        reward_advantages=reward,
        cost_advantages=costs,
        multipliers=multipliers,
    )

    np.testing.assert_allclose(
        combined,
        reward - costs @ multipliers,
        rtol=0.0,
        atol=0.0,
    )
```

```python
@pytest.mark.parametrize("scale", [0.1, 10.0, 1000.0])
def test_actor_composition_is_invariant_to_cost_unit_conversion(scale: float) -> None:
    reward = np.asarray([1.0, -2.0, 4.0], dtype=np.float64)
    costs = np.asarray([[0.2], [0.4], [0.1]], dtype=np.float64)
    multiplier = np.asarray([3.0], dtype=np.float64)

    original = combine_lagrangian_advantages(
        reward_advantages=reward,
        cost_advantages=costs,
        multipliers=multiplier,
    )
    converted = combine_lagrangian_advantages(
        reward_advantages=reward,
        cost_advantages=costs * scale,
        multipliers=multiplier / scale,
    )

    np.testing.assert_array_equal(converted, original)
```

Also update validation cases so the removed `normalize_reward` argument is rejected by Python, zero multipliers return an exact copy of reward advantages, input arrays remain unmodified, and shape/non-finite/negative-multiplier failures retain explicit messages.

- [ ] **Step 2: Run the two focused tests and verify RED**

Run:

```text
pytest tests/rl/test_lagrangian_advantages.py::test_combine_lagrangian_advantages_preserves_raw_units -q
pytest tests/rl/test_lagrangian_advantages.py::test_actor_composition_is_invariant_to_cost_unit_conversion -q
```

Expected: FAIL because the current implementation independently normalizes reward and cost columns.

- [ ] **Step 3: Implement the minimal raw composition**

Replace the body contract with:

```python
def combine_lagrangian_advantages(
    *,
    reward_advantages: ArrayLike,
    cost_advantages: ArrayLike,
    multipliers: ArrayLike,
) -> NDArray[np.float64]:
    reward_vector = _finite_float_array(
        reward_advantages,
        dimensions=1,
        field_name="reward_advantages",
    )
    cost_matrix = _finite_float_array(
        cost_advantages,
        dimensions=2,
        field_name="cost_advantages",
    )
    multiplier_vector = _finite_float_array(
        multipliers,
        dimensions=1,
        field_name="multipliers",
    )
    if cost_matrix.shape[0] != reward_vector.shape[0]:
        raise ValueError("reward and cost batch dimensions must match")
    if multiplier_vector.shape[0] != cost_matrix.shape[1]:
        raise ValueError("multipliers must contain one value per cost column")
    if np.any(multiplier_vector < 0.0):
        raise ValueError("multipliers must be non-negative")
    return np.asarray(
        reward_vector - cost_matrix @ multiplier_vector,
        dtype=np.float64,
    )
```

Keep `normalize_advantage_vector` and `normalize_cost_advantages` exported for diagnostics, but update docstrings to state that neither is authoritative for actor composition.

- [ ] **Step 4: Run unit and static checks**

```text
pytest tests/rl/test_lagrangian_advantages.py -q
ruff check trade_rl/rl/lagrangian_advantages.py tests/rl/test_lagrangian_advantages.py
ruff format --check trade_rl/rl/lagrangian_advantages.py tests/rl/test_lagrangian_advantages.py
mypy trade_rl/rl/lagrangian_advantages.py
```

- [ ] **Step 5: Commit**

```text
git add trade_rl/rl/lagrangian_advantages.py tests/rl/test_lagrangian_advantages.py
git commit -m "fix: preserve raw Lagrangian advantage units"
```

---

### Task 2: Integrate final-only PPO normalization and prove exact parity

**Files:**
- Modify: `trade_rl/integrations/lagrangian_ppo.py`
- Modify: `tests/integrations/test_lagrangian_ppo.py`
- Modify: `tests/integrations/test_lagrangian_checkpoint_roundtrip.py`

**Interfaces:**
- Consumes: Task 1 raw `combine_lagrangian_advantages`, rollout-data reward advantages, aligned cost minibatch, rollout-frozen multiplier vector.
- Produces: `_actor_advantages(...)` whose only normalization is the pinned SB3 Torch expression applied after raw composition.

- [ ] **Step 1: Add failing final-normalization and parity tests**

```python
def test_actor_normalizes_only_the_final_combined_advantage(model: LagrangianPPO) -> None:
    model.frozen_lagrange_multipliers = _readonly(np.asarray([2.0, 0.5]))
    reward = torch.as_tensor([1.0, 3.0, -2.0, 4.0], dtype=torch.float32)
    costs = np.asarray(
        [[0.0, 2.0], [1.0, 0.0], [3.0, 1.0], [2.0, 4.0]],
        dtype=np.float64,
    )
    raw = reward.numpy() - costs @ np.asarray([2.0, 0.5])
    expected = torch.as_tensor(raw, dtype=torch.float32)
    expected = (expected - expected.mean()) / (expected.std() + 1e-8)

    actual = model._actor_advantages(
        reward_advantages=reward,
        cost_advantages=costs,
    )

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
```

Extend the existing ordinary-PPO parity test to compare policy state dictionaries, reward-value optimizer state, `_n_updates`, and logged policy loss after an identical RNG reset. Add a multi-epoch hook that records `frozen_lagrange_multipliers.copy()` for every minibatch and asserts every recorded vector equals the rollout-start snapshot.

- [ ] **Step 2: Run and verify RED**

```text
pytest tests/integrations/test_lagrangian_ppo.py::test_actor_normalizes_only_the_final_combined_advantage -q
pytest tests/integrations/test_lagrangian_ppo.py -k "zero_multiplier and parity" -q
```

Expected: the first test fails because the current path normalizes reward and each cost separately.

- [ ] **Step 3: Change `_actor_advantages` and no other PPO expression**

Use:

```python
combined_numpy = combine_lagrangian_advantages(
    reward_advantages=reward_advantages.detach().cpu().numpy(),
    cost_advantages=cost_advantages,
    multipliers=self.frozen_lagrange_multipliers,
)
advantages = torch.as_tensor(
    combined_numpy,
    dtype=reward_advantages.dtype,
    device=reward_advantages.device,
)
if self.normalize_advantage and len(advantages) > 1:
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
return advantages
```

Keep the existing zero-multiplier fast path only if it is necessary for exact ordinary-PPO parity; both branches must use the same Torch normalization expression. Do not change clipping, entropy, reward-value loss, KL early stop, gradient clipping, permutation behavior, or logger keys.

- [ ] **Step 4: Run integration and regression checks**

```text
pytest tests/integrations/test_lagrangian_ppo.py -q
pytest tests/integrations/test_lagrangian_checkpoint_roundtrip.py -q
pytest tests/integrations/test_cost_critic_ppo.py -q
ruff check trade_rl/integrations/lagrangian_ppo.py tests/integrations/test_lagrangian_ppo.py tests/integrations/test_lagrangian_checkpoint_roundtrip.py
ruff format --check trade_rl/integrations/lagrangian_ppo.py tests/integrations/test_lagrangian_ppo.py tests/integrations/test_lagrangian_checkpoint_roundtrip.py
mypy trade_rl/integrations/lagrangian_ppo.py
```

- [ ] **Step 5: Commit**

```text
git add trade_rl/integrations/lagrangian_ppo.py tests/integrations/test_lagrangian_ppo.py tests/integrations/test_lagrangian_checkpoint_roundtrip.py
git commit -m "fix: normalize only combined Lagrangian advantage"
```

---

### Task 3: Store authoritative elapsed time and completion metadata

**Files:**
- Modify: `trade_rl/rl/environment_info.py`
- Modify: `trade_rl/integrations/cost_rollout_buffer.py`
- Modify: `trade_rl/integrations/cost_critic_ppo.py`
- Create: `trade_rl/rl/lagrangian_episode.py`
- Modify: `tests/rl/test_environment_info_service.py`
- Modify: `tests/integrations/test_cost_rollout_buffer.py`
- Modify: `tests/integrations/test_cost_rollout_alignment.py`

**Interfaces:**
- Produces enum `EpisodeCompletionKind(IntEnum)` with values `NONE=0`, `ECONOMIC_TERMINATION=1`, `TIME_LIMIT_COMPLETION=2`, `CENSORED_EXTERNAL_TRUNCATION=3`.
- Produces `classify_episode_completion(*, terminated: bool, truncated: bool, time_limit_truncated: bool, termination_reason: object | None) -> EpisodeCompletionKind`.
- Extends `CostRolloutStorage` with `elapsed_hours: NDArray[np.float64]` and `completion_kinds: NDArray[np.int8]`, aligned `[steps, n_envs]`.

- [ ] **Step 1: Add failing environment-info test**

```python
def test_step_info_exposes_authoritative_transition_elapsed_hours(builder, request) -> None:
    info = builder.step_info(request)
    assert info["transition_elapsed_hours"] == pytest.approx(
        builder.dataset.elapsed_hours(
            request.hybrid_execution.next_index - request.hybrid_execution.bars_advanced,
            request.hybrid_execution.next_index,
        )
    )
```

The test must also assert a non-positive or non-finite dataset duration raises `RuntimeError("transition duration must be finite and positive")` before returning info.

- [ ] **Step 2: Add failing completion-classification tests**

```python
@pytest.mark.parametrize(
    ("terminated", "truncated", "time_limit", "reason", "expected"),
    [
        (False, False, False, None, EpisodeCompletionKind.NONE),
        (True, False, False, "margin_call", EpisodeCompletionKind.ECONOMIC_TERMINATION),
        (False, True, True, None, EpisodeCompletionKind.TIME_LIMIT_COMPLETION),
        (
            False,
            True,
            True,
            "shadow_minimum_equity",
            EpisodeCompletionKind.CENSORED_EXTERNAL_TRUNCATION,
        ),
    ],
)
def test_episode_completion_classification(...):
    assert classify_episode_completion(...) is expected
```

Add explicit failure cases for both flags true, truncated without `TimeLimit.truncated`, unknown non-shadow truncation reason, shadow reason without truncation, and terminated transition carrying a `shadow_` reason.

- [ ] **Step 3: Add failing storage-alignment test**

Construct two vector environments whose first step has elapsed hours `[0.25, 1.0]`, then one economic termination and one shadow truncation. Assert `CostRolloutStorage.elapsed_hours[0]` and `completion_kinds[0]` preserve environment order exactly after `add_from_infos(...)`.

- [ ] **Step 4: Implement metadata production and storage**

In `EnvironmentInfoBuilder.step_info`, calculate the decision duration once and add:

```python
"transition_elapsed_hours": transition_elapsed_hours,
```

Use the same value when constructing `ConstraintCostRequest.decision_hours` so telemetry and cost calculation cannot disagree.

In `CostRolloutStorage.add_from_infos`, for each environment read:

```python
elapsed = float(info["transition_elapsed_hours"])
reason = info.get("termination_reason")
time_limit = bool(info.get("TimeLimit.truncated", False))
kind = classify_episode_completion(
    terminated=bool(terminated_array[index]),
    truncated=bool(truncated_array[index]),
    time_limit_truncated=time_limit,
    termination_reason=reason,
)
```

Reject missing/non-finite/non-positive elapsed time. Store elapsed time as `float64` and enum values as `int8`. Keep existing `terminated`, `truncated`, and terminal-cost bootstrap arrays unchanged for Cost Critic GAE.

Update `estimate_cost_rollout_storage_bytes` to include exactly one float64 and one int8 array per transition; add tests with an explicit byte calculation.

- [ ] **Step 5: Run targeted checks and commit**

```text
pytest tests/rl/test_environment_info_service.py -q
pytest tests/integrations/test_cost_rollout_buffer.py -q
pytest tests/integrations/test_cost_rollout_alignment.py -q
pytest tests/integrations/test_cost_critic_ppo.py -q
ruff check trade_rl/rl/environment_info.py trade_rl/rl/lagrangian_episode.py trade_rl/integrations/cost_rollout_buffer.py trade_rl/integrations/cost_critic_ppo.py tests/rl/test_environment_info_service.py tests/integrations/test_cost_rollout_buffer.py tests/integrations/test_cost_rollout_alignment.py
ruff format --check trade_rl/rl/environment_info.py trade_rl/rl/lagrangian_episode.py trade_rl/integrations/cost_rollout_buffer.py trade_rl/integrations/cost_critic_ppo.py tests/rl/test_environment_info_service.py tests/integrations/test_cost_rollout_buffer.py tests/integrations/test_cost_rollout_alignment.py
mypy trade_rl/rl/environment_info.py trade_rl/rl/lagrangian_episode.py trade_rl/integrations/cost_rollout_buffer.py
```

```text
git add trade_rl/rl/environment_info.py trade_rl/rl/lagrangian_episode.py trade_rl/integrations/cost_rollout_buffer.py trade_rl/integrations/cost_critic_ppo.py tests/rl/test_environment_info_service.py tests/integrations/test_cost_rollout_buffer.py tests/integrations/test_cost_rollout_alignment.py
git commit -m "feat: store constraint completion and elapsed-time metadata"
```

---

### Task 4: Implement time-aware completed-episode statistics and censoring

**Files:**
- Modify: `trade_rl/rl/lagrangian_episode.py`
- Modify: `trade_rl/rl/lagrangian.py`
- Create: `tests/rl/test_lagrangian_episode.py`
- Modify: `tests/rl/test_lagrangian.py`
- Modify: `trade_rl/integrations/lagrangian_ppo.py`

**Interfaces:**
- Extends `ConstraintAggregation` with `EPISODE_TIME_AREA`, `EPISODE_DECISION_MEAN`, `EPISODE_TIME_WEIGHTED_MEAN`, `EPISODE_EVENT_RATE`, and `EPISODE_SUM`.
- Produces `CompletedEpisodeBatch(estimates: dict[str, ConstraintEstimate | None], completed_episode_count: int, censored_episode_count: int)`.
- Produces `CompletedEpisodeCostAccumulator.ingest_rollout(*, costs, elapsed_hours, completion_kinds) -> CompletedEpisodeBatch`.

- [ ] **Step 1: Write failing irregular-time aggregation test**

```python
def test_completed_episode_statistics_are_time_aware() -> None:
    accumulator = CompletedEpisodeCostAccumulator(n_envs=1, schema=canonical_schema())
    costs = _seven_cost_rollout(
        drawdown_excess=[0.10, 0.20],
        margin_deficit_fraction=[0.04, 0.08],
        gross_exposure_request_excess=[0.30, 0.10],
        daily_turnover=[2.0, 4.0],
        execution_cost_fraction=[0.001, 0.002],
    )
    result = accumulator.ingest_rollout(
        costs=costs,
        elapsed_hours=np.asarray([[6.0], [18.0]]),
        completion_kinds=np.asarray(
            [[EpisodeCompletionKind.NONE], [EpisodeCompletionKind.TIME_LIMIT_COMPLETION]],
            dtype=np.int8,
        ),
    )

    assert result.estimates["drawdown_excess"].value == pytest.approx(0.175)
    assert result.estimates["margin_deficit_fraction"].value == pytest.approx(0.07)
    assert result.estimates["gross_exposure_request_excess"].value == pytest.approx(0.20)
    assert result.estimates["daily_turnover"].value == pytest.approx(3.5)
    assert result.estimates["execution_cost_fraction"].value == pytest.approx(0.003)
```

The expected drawdown value is `0.10 * 6/24 + 0.20 * 18/24`; turnover is `(2.0 * 6/24 + 4.0 * 18/24) / (24/24)`.

- [ ] **Step 2: Write failing censoring test**

```python
def test_shadow_truncation_clears_state_without_safe_denominator() -> None:
    accumulator = CompletedEpisodeCostAccumulator(n_envs=1, schema=canonical_schema())
    result = accumulator.ingest_rollout(
        costs=_unsafe_two_step_rollout(),
        elapsed_hours=np.asarray([[1.0], [1.0]]),
        completion_kinds=np.asarray(
            [[EpisodeCompletionKind.NONE], [EpisodeCompletionKind.CENSORED_EXTERNAL_TRUNCATION]],
            dtype=np.int8,
        ),
    )

    assert result.completed_episode_count == 0
    assert result.censored_episode_count == 1
    assert all(value is None for value in result.estimates.values())
    assert accumulator.state_dict()["episode_step_counts"] == [0]
```

Add tests proving economic termination and time-limit completion contribute; event cost can occur at most once; one environment's completion does not clear another; rollout-boundary carry includes elapsed time; state round-trip reproduces the next batch; old accumulator state version fails closed; unknown enum values and non-positive elapsed hours fail.

- [ ] **Step 3: Implement explicit canonical aggregation mapping**

Use:

```python
{
    "drawdown_excess": ConstraintAggregation.EPISODE_TIME_AREA,
    "drawdown_stop_event": ConstraintAggregation.EPISODE_EVENT_RATE,
    "margin_deficit_fraction": ConstraintAggregation.EPISODE_TIME_AREA,
    "forced_liquidation_event": ConstraintAggregation.EPISODE_EVENT_RATE,
    "gross_exposure_request_excess": ConstraintAggregation.EPISODE_DECISION_MEAN,
    "daily_turnover": ConstraintAggregation.EPISODE_TIME_WEIGHTED_MEAN,
    "execution_cost_fraction": ConstraintAggregation.EPISODE_SUM,
}
```

Add `canonical_constraint_unit(name)` returning respectively `drawdown_excess_area_days`, `event_per_episode`, `margin_deficit_fraction_days`, `event_per_episode`, `excess_per_decision`, `turnover_per_day`, and `execution_cost_fraction_per_episode`. Include aggregation and unit in `LagrangianConstraintSpec.digest_payload()`.

- [ ] **Step 4: Implement accumulator sufficient statistics**

Maintain per environment:

```text
episode_cost_sums[n_envs, n_costs]
episode_time_weighted_sums[n_envs, n_costs]
episode_elapsed_hours[n_envs]
episode_step_counts[n_envs]
```

For every transition, add raw costs, `cost * elapsed_hours / 24`, elapsed hours, and one step. On valid completion:

```python
if aggregation is EPISODE_TIME_AREA:
    contribution = time_weighted_sum
elif aggregation is EPISODE_DECISION_MEAN:
    contribution = raw_sum / step_count
elif aggregation is EPISODE_TIME_WEIGHTED_MEAN:
    contribution = time_weighted_sum / (elapsed_hours / 24.0)
elif aggregation is EPISODE_SUM:
    contribution = raw_sum
else:
    contribution = min(raw_sum, 1.0)
```

On censored completion, clear only that environment and increment the censored count without adding a contribution. Return one `ConstraintEstimate` per cost only when at least one valid episode completed.

- [ ] **Step 5: Integrate the new batch into `LagrangianPPO` and run checks**

Pass `cost_rollout_storage.costs`, `.elapsed_hours`, and `.completion_kinds`; store `last_completed_episode_batch`; do not yet change controller scheduling in this task.

```text
pytest tests/rl/test_lagrangian_episode.py -q
pytest tests/rl/test_lagrangian.py -q
pytest tests/integrations/test_lagrangian_ppo.py -q
ruff check trade_rl/rl/lagrangian_episode.py trade_rl/rl/lagrangian.py trade_rl/integrations/lagrangian_ppo.py tests/rl/test_lagrangian_episode.py tests/rl/test_lagrangian.py
ruff format --check trade_rl/rl/lagrangian_episode.py trade_rl/rl/lagrangian.py trade_rl/integrations/lagrangian_ppo.py tests/rl/test_lagrangian_episode.py tests/rl/test_lagrangian.py
mypy trade_rl/rl/lagrangian_episode.py trade_rl/rl/lagrangian.py trade_rl/integrations/lagrangian_ppo.py
```

```text
git add trade_rl/rl/lagrangian_episode.py trade_rl/rl/lagrangian.py trade_rl/integrations/lagrangian_ppo.py tests/rl/test_lagrangian_episode.py tests/rl/test_lagrangian.py
git commit -m "fix: aggregate time-aware completed constraint episodes"
```

---

### Task 5: Separate pooled estimator state from dual-actuator scheduling

**Files:**
- Modify: `trade_rl/rl/lagrangian.py`
- Modify: `tests/rl/test_lagrangian_dual.py`
- Modify: `trade_rl/integrations/lagrangian_ppo.py`
- Modify: `tests/integrations/test_lagrangian_ppo.py`

**Interfaces:**
- Adds `minimum_completed_episodes: int` to `LagrangianConstraintSpec` and `canonical_lagrangian_schema(...)`.
- `LagrangianDualController.update_after_rollout(estimates, *, censored_episode_count: int) -> dict[str, DualUpdateReport]` accumulates observations before applying schedule gates.
- `DualUpdateReport` contains pending/consumed support, residual, and separate lower/upper boundary flags.

- [ ] **Step 1: Replace the old warmup-discard test with a failing pooled-state test**

```python
def test_warmup_observations_feed_first_eligible_dual_update() -> None:
    controller = LagrangianDualController(
        _schema(
            minimum_completed_episodes=3,
            warmup_rollouts=1,
            update_interval_rollouts=2,
            ema_beta=0.5,
            budget=0.0,
            dual_learning_rate=1.0,
        )
    )

    warmup = controller.update_after_rollout(
        {"drawdown_excess": _estimate("drawdown_excess", value=1.0, denominator=1)},
        censored_episode_count=0,
    )["drawdown_excess"]
    interval = controller.update_after_rollout(
        {"drawdown_excess": _estimate("drawdown_excess", value=2.0, denominator=1)},
        censored_episode_count=0,
    )["drawdown_excess"]
    updated = controller.update_after_rollout(
        {"drawdown_excess": _estimate("drawdown_excess", value=3.0, denominator=1)},
        censored_episode_count=0,
    )["drawdown_excess"]

    assert warmup.skip_reason == "warmup"
    assert interval.skip_reason == "update_interval"
    assert updated.consumed_denominator == 3
    assert updated.raw_estimate == pytest.approx(2.0)
    assert updated.ema_estimate == pytest.approx(2.0)
```

- [ ] **Step 2: Add failing denominator-aware EMA, partition, and boundary tests**

```python
def test_denominator_aware_ema_uses_beta_power_support() -> None:
    controller = LagrangianDualController(_schema(ema_beta=0.9, budget=0.0))
    first = controller.update_after_rollout(
        {"drawdown_excess": _estimate("drawdown_excess", 0.2, denominator=1)},
        censored_episode_count=0,
    )["drawdown_excess"]
    second = controller.update_after_rollout(
        {"drawdown_excess": _estimate("drawdown_excess", 0.6, denominator=4)},
        censored_episode_count=0,
    )["drawdown_excess"]
    expected = (0.9**4) * 0.2 + (1.0 - 0.9**4) * 0.6
    assert first.ema_estimate == pytest.approx(0.2)
    assert second.ema_estimate == pytest.approx(expected)
```

Add one test feeding the same five episodes as `1+4`, `2+3`, and `5` support partitions and assert identical pooled raw estimate before an eligible update. Add tests that event costs wait for `minimum_completed_episodes=20`, skipped updates retain pending state, a successful update resets it, validation failure leaves state unchanged, λ=0 sets `at_lower_bound=True` and `at_upper_cap=False`, λ=max sets the reverse, and `saturated` equals `at_upper_cap` during compatibility migration.

- [ ] **Step 3: Extend schema and report types**

Add to each spec:

```python
minimum_completed_episodes: int
```

Validate it as a positive non-boolean integer and include it in schema digest. Canonical config defaults are supplied by Task 6, not hidden inside the schema constructor.

Extend `DualUpdateReport` with:

```text
pending_numerator_before: float
pending_denominator_before: int
consumed_denominator: int
censored_episode_count: int
constraint_residual: float | None
at_lower_bound: bool
at_upper_cap: bool
```

Retain `saturated` as a read-only property returning `at_upper_cap` until evidence migration completes.

- [ ] **Step 4: Implement estimator-first update ordering**

For each cost, first add an available estimate's numerator and denominator to pending arrays. Then apply gates in this order:

```text
warmup
update_interval
insufficient_completed_episodes
missing_estimate_or_pending_support
eligible update
```

At an eligible update:

```python
raw_estimate = pending_numerator / pending_denominator
beta_effective = spec.ema_beta ** pending_denominator
ema_after = (
    raw_estimate
    if previous_ema is None
    else beta_effective * previous_ema
    + (1.0 - beta_effective) * raw_estimate
)
residual = ema_after - spec.budget
multiplier_after = float(
    np.clip(
        multiplier_before + spec.dual_learning_rate * residual,
        0.0,
        spec.max_multiplier,
    )
)
```

Commit multipliers, EMA, update count, and pending reset only after every calculated value is finite. Censored episode counts are cumulative controller state and appear in every report.

Bump controller state to `lagrangian_dual_controller_v2`; include pending numerator/denominator and censored count. Reject v1 state explicitly.

- [ ] **Step 5: Integrate logging and run checks**

Log pending support, consumed support, residual, lower bound, upper cap, and censored count. Do not log λ=0 as saturation.

```text
pytest tests/rl/test_lagrangian_dual.py -q
pytest tests/integrations/test_lagrangian_ppo.py -q
pytest tests/rl/test_lagrangian.py tests/rl/test_lagrangian_episode.py -q
ruff check trade_rl/rl/lagrangian.py trade_rl/integrations/lagrangian_ppo.py tests/rl/test_lagrangian_dual.py tests/integrations/test_lagrangian_ppo.py
ruff format --check trade_rl/rl/lagrangian.py trade_rl/integrations/lagrangian_ppo.py tests/rl/test_lagrangian_dual.py tests/integrations/test_lagrangian_ppo.py
mypy trade_rl/rl/lagrangian.py trade_rl/integrations/lagrangian_ppo.py
```

```text
git add trade_rl/rl/lagrangian.py trade_rl/integrations/lagrangian_ppo.py tests/rl/test_lagrangian_dual.py tests/integrations/test_lagrangian_ppo.py
git commit -m "fix: pool constraint estimates before dual scheduling"
```

---

### Task 6: Bind corrected semantics into configuration, backend, and checkpoint identity

**Files:**
- Modify: `trade_rl/rl/training.py`
- Modify: `trade_rl/rl/algorithm_configs.py`
- Modify: `trade_rl/integrations/sb3_training.py`
- Modify: `trade_rl/integrations/lagrangian_ppo.py`
- Modify: `tests/rl/test_lagrangian_training_config.py`
- Modify: `tests/integrations/test_sb3_lagrangian_backend.py`
- Modify: `tests/integrations/test_lagrangian_checkpoint_roundtrip.py`
- Modify: `tests/rl/test_lagrangian_checkpoint_identity.py`

**Interfaces:**
- Extends opt-in vectors with `lagrangian_minimum_completed_episodes`.
- Produces `LagrangianPPOConfig(CostCriticPPOConfig)` containing `lagrangian_schema`, probe settings, and actor-composition semantic version.
- Checkpoint identity includes aggregation units, minimum support, actor mode, controller/accumulator state schema versions, and completion semantics.

- [ ] **Step 1: Add failing configuration and inactive-field tests**

At the end of `ResidualTrainingConfig`, define:

```python
lagrangian_budgets: tuple[float, ...] = ()
lagrangian_dual_learning_rates: tuple[float, ...] = ()
lagrangian_ema_betas: tuple[float, ...] = ()
lagrangian_initial_multipliers: tuple[float, ...] = ()
lagrangian_max_multipliers: tuple[float, ...] = ()
lagrangian_warmup_rollouts: tuple[int, ...] = ()
lagrangian_update_interval_rollouts: tuple[int, ...] = ()
lagrangian_minimum_completed_episodes: tuple[int, ...] = ()
lagrangian_probe_episodes: int = 0
lagrangian_probe_max_steps_per_episode: int = 0
```

For `algorithm="lagrangian_ppo"`, require all eight vectors to have exactly seven entries in `CONSTRAINT_COST_NAMES` order. Require both probe integers to be positive. For every other algorithm, require empty vectors and zero probe integers. Set canonical test fixture support to `(1, 20, 1, 20, 1, 1, 1)`.

Assert every vector and probe field changes `ResidualTrainingConfig.digest_payload()`, while ordinary PPO's default digest remains unchanged.

- [ ] **Step 2: Add failing checkpoint semantic-mismatch tests**

Save a model after one partial episode and at least one pending dual estimate. Change each of the following independently and assert load/identity validation rejects it:

```text
actor composition mode
constraint aggregation
constraint unit label
minimum completed episodes
completion semantics version
accumulator state version
controller state version
probe action semantic
```

Assert an old checkpoint missing corrected fields raises an explicit schema-version or algorithm-identity mismatch rather than silently resetting state.

- [ ] **Step 3: Implement typed config and backend construction**

Add `lagrangian_ppo` to PPO-like algorithm validation and rounded timestep logic. Cost Critic settings are active for both `cost_critic_ppo` and `lagrangian_ppo`.

Add:

```python
@dataclass(frozen=True, slots=True)
class LagrangianPPOConfig(CostCriticPPOConfig):
    lagrangian_schema: LagrangianSchema
    probe_episodes: int
    probe_max_steps_per_episode: int
    actor_composition_mode: str = "raw_lagrangian_then_sb3_normalize_v1"
```

`build_algorithm_config()` constructs `canonical_lagrangian_schema(...)` using all vectors and returns this type only for `lagrangian_ppo`.

In `StableBaselines3Backend`, construct/load `LagrangianPPO`, reuse PR B memory accounting plus Task 3 metadata bytes, reject replay-buffer resume as PPO-family, and include the full schema digest payload and actor mode in architecture JSON.

- [ ] **Step 4: Extend checkpoint payload and deterministic serialization**

`LagrangianPPO.checkpoint_identity_payload()` must include:

```python
{
    **super().checkpoint_identity_payload(),
    "algorithm": "lagrangian_ppo",
    "actor_composition_mode": "raw_lagrangian_then_sb3_normalize_v1",
    "completion_semantics": "economic_time_limit_censored_shadow_v1",
    "lagrangian_schema": self.lagrangian_schema.digest_payload(),
    "lagrangian_schema_digest": self.lagrangian_schema.digest,
    "accumulator_state_version": self.completed_episode_cost_accumulator.state_version,
    "controller_state_version": self.lagrangian_controller.state_version,
}
```

Keep runtime controller and accumulator attributes serialized by SB3. Normalize restored NumPy arrays and read-only multiplier snapshot in `_setup_model()`. Checkpoint round-trip must reproduce the next completed-episode batch, next report, and next frozen multiplier exactly.

- [ ] **Step 5: Run config/backend/checkpoint regressions and commit**

```text
pytest tests/rl/test_lagrangian_training_config.py -q
pytest tests/integrations/test_sb3_lagrangian_backend.py -q
pytest tests/integrations/test_lagrangian_checkpoint_roundtrip.py -q
pytest tests/rl/test_lagrangian_checkpoint_identity.py -q
pytest tests/rl/test_cost_critic_training_config.py -q
pytest tests/integrations/test_sb3_cost_critic_backend.py -q
pytest tests/integrations/test_sb3_training.py -q
ruff check trade_rl/rl/training.py trade_rl/rl/algorithm_configs.py trade_rl/integrations/sb3_training.py trade_rl/integrations/lagrangian_ppo.py tests/rl/test_lagrangian_training_config.py tests/integrations/test_sb3_lagrangian_backend.py tests/integrations/test_lagrangian_checkpoint_roundtrip.py tests/rl/test_lagrangian_checkpoint_identity.py
ruff format --check trade_rl/rl/training.py trade_rl/rl/algorithm_configs.py trade_rl/integrations/sb3_training.py trade_rl/integrations/lagrangian_ppo.py tests/rl/test_lagrangian_training_config.py tests/integrations/test_sb3_lagrangian_backend.py tests/integrations/test_lagrangian_checkpoint_roundtrip.py tests/rl/test_lagrangian_checkpoint_identity.py
mypy trade_rl/rl/training.py trade_rl/rl/algorithm_configs.py trade_rl/integrations/sb3_training.py trade_rl/integrations/lagrangian_ppo.py
```

```text
git add trade_rl/rl/training.py trade_rl/rl/algorithm_configs.py trade_rl/integrations/sb3_training.py trade_rl/integrations/lagrangian_ppo.py tests/rl/test_lagrangian_training_config.py tests/integrations/test_sb3_lagrangian_backend.py tests/integrations/test_lagrangian_checkpoint_roundtrip.py tests/rl/test_lagrangian_checkpoint_identity.py
git commit -m "feat: bind corrected Lagrangian semantics into training identity"
```

---

### Task 7: Replace hard feasibility witness with canonical-action diagnostic probe

**Files:**
- Create: `trade_rl/rl/lagrangian_probe.py`
- Create: `tests/rl/test_lagrangian_probe.py`
- Modify: `trade_rl/integrations/sb3_training.py`
- Modify: `tests/integrations/test_sb3_lagrangian_backend.py`
- Modify: `trade_rl/rl/lagrangian_evidence.py` if already present

**Interfaces:**
- Produces `CanonicalActionSemantic(str, Enum)` with `TARGET_WEIGHT_CASH` and `RESIDUAL_BASELINE`.
- Produces `CanonicalActionProbeEvidence` and `run_canonical_action_feasibility_probe(...)`.
- Probe budget violations emit warning evidence but do not reject training; malformed execution still fails closed.

- [ ] **Step 1: Write failing semantic and warning tests**

```python
def test_target_weight_zero_action_is_recorded_as_cash() -> None:
    evidence = run_canonical_action_feasibility_probe(
        environment_factory=target_weight_environment_factory,
        schema=schema,
        episode_count=2,
        max_steps_per_episode=8,
    )
    assert evidence.action_semantic is CanonicalActionSemantic.TARGET_WEIGHT_CASH
    assert evidence.action.tolist() == [0.0, 0.0, 0.0]
```

```python
def test_residual_zero_action_is_recorded_as_baseline() -> None:
    evidence = run_canonical_action_feasibility_probe(
        environment_factory=residual_environment_factory,
        schema=schema,
        episode_count=2,
        max_steps_per_episode=8,
    )
    assert evidence.action_semantic is CanonicalActionSemantic.RESIDUAL_BASELINE
```

```python
def test_probe_budget_violation_warns_without_rejecting_training(backend) -> None:
    result = backend.train(config=_violating_probe_config())
    assert result is not None
    assert result.architecture["lagrangian_probe"]["violated_costs"]
    assert result.architecture["lagrangian_probe"]["warning"] is True
```

Also assert malformed Box shape, discrete action space, missing costs, non-finite elapsed time, unknown completion kind, and failure to complete configured episodes raise before model construction. Probe digest must change with action semantic, action vector, estimate, denominator, budget, completion count, censor count, or warning flag.

- [ ] **Step 2: Implement explicit action-semantic resolution**

Inspect the environment's maintained action specification. For target-weight mode, step `np.zeros(action_space.shape, dtype=action_space.dtype)` and record cash semantics. For residual mode, step the same zero vector and record baseline semantics. Reject any unsupported action mode instead of guessing.

Create a fresh environment for each episode, reset with seed equal to the zero-based episode index, close it on success and failure, and use the same `CompletedEpisodeCostAccumulator` and completion classification as training. Require exactly the configured number of valid completed episodes; censored episodes do not satisfy the count and the probe continues within the configured step cap.

- [ ] **Step 3: Integrate as warning evidence before model construction**

Run the probe after environment/config validation and before constructing the training model. Store its canonical payload and digest in architecture/evidence and expose it on the model for checkpoint identity. When `violated_costs` is non-empty, record a prominent warning but continue training. Remove or supersede any code path that raises solely because `estimate > budget`.

- [ ] **Step 4: Run tests and commit**

```text
pytest tests/rl/test_lagrangian_probe.py -q
pytest tests/integrations/test_sb3_lagrangian_backend.py -q
pytest tests/rl/test_lagrangian_episode.py -q
ruff check trade_rl/rl/lagrangian_probe.py trade_rl/integrations/sb3_training.py tests/rl/test_lagrangian_probe.py tests/integrations/test_sb3_lagrangian_backend.py
ruff format --check trade_rl/rl/lagrangian_probe.py trade_rl/integrations/sb3_training.py tests/rl/test_lagrangian_probe.py tests/integrations/test_sb3_lagrangian_backend.py
mypy trade_rl/rl/lagrangian_probe.py trade_rl/integrations/sb3_training.py
```

```text
git add trade_rl/rl/lagrangian_probe.py trade_rl/integrations/sb3_training.py trade_rl/rl/lagrangian_evidence.py tests/rl/test_lagrangian_probe.py tests/integrations/test_sb3_lagrangian_backend.py
git commit -m "fix: make canonical feasibility probe diagnostic"
```

---

### Task 8: Correct stability diagnostics and evidence semantics

**Files:**
- Create or modify: `trade_rl/rl/lagrangian_diagnostics.py`
- Create or modify: `trade_rl/rl/lagrangian_evidence.py`
- Modify: `trade_rl/integrations/lagrangian_ppo.py`
- Create or modify: `tests/rl/test_lagrangian_diagnostics.py`
- Create or modify: `tests/rl/test_lagrangian_evidence.py`
- Modify: `tests/integrations/test_lagrangian_ppo.py`

**Interfaces:**
- Diagnostics consume raw costs, optional standardized cost advantages for correlation only, raw cost advantages, raw reward advantages, frozen multipliers, and dual reports.
- Effective penalty is always `raw_cost_advantages * multipliers[None, :]`.
- Saturation metrics count `at_upper_cap` only.

- [ ] **Step 1: Add failing raw-penalty diagnostic test**

```python
def test_effective_penalty_diagnostics_use_raw_cost_advantages() -> None:
    raw_cost_advantages = np.asarray([[1.0, 10.0], [3.0, 30.0]])
    normalized = normalize_cost_advantages(raw_cost_advantages)
    multipliers = np.asarray([2.0, 0.5])
    diagnostics = build_constraint_correlation_diagnostics(
        cost_names=("drawdown_excess", "execution_cost_fraction"),
        raw_costs=np.asarray([[0.1, 0.01], [0.2, 0.02]]),
        raw_cost_advantages=raw_cost_advantages,
        normalized_cost_advantages=normalized,
        multipliers=multipliers,
        reward_advantages=np.asarray([4.0, -2.0]),
    )

    np.testing.assert_array_equal(
        diagnostics.penalty_contributions,
        raw_cost_advantages * multipliers[None, :],
    )
```

Assert `penalty_to_reward_l2_ratio` equals `||sum_i lambda_i A_ci||_2 / max(||A_r||_2, 1e-12)`. Keep normalized-advantage correlation as a separately named observational matrix.

- [ ] **Step 2: Add failing upper-cap-only stability test**

Record multipliers `[0, 0, 10, 10, 5]` with cap `10`. Assert `saturation_fraction == 2/5`, `longest_saturation_run == 2`, and lower-bound rollouts are counted in a separate `lower_bound_fraction`. Keep update sign-change frequency, rolling variances, violation area, sustained satisfaction, and post-satisfaction over-constraint definitions deterministic.

- [ ] **Step 3: Implement diagnostics and evidence payload**

Evidence must include:

```text
actor_composition_mode
raw reward advantage statistics
raw cost advantage statistics per cost
raw effective penalty statistics per cost
raw aggregate penalty/reward L2 ratio
raw-cost covariance/correlation
normalized-cost-advantage correlation (diagnostic only)
pending and consumed denominators
censored episode counts
beta_effective
constraint residual
at_lower_bound
at_upper_cap
probe semantic, warning, payload, and digest
schema aggregation and unit labels
```

Remove labels implying normalized penalty contributions are the actor's effective penalty. Evidence digest changes when any listed field changes. Matrices use deterministic zero rows/columns for constant inputs and all arrays are copied/read-only before serialization.

- [ ] **Step 4: Integrate once per finalized rollout**

Build correlation and advantage diagnostics before minibatch shuffling from the complete rollout. Record stability after the one post-rollout dual update. Diagnostics never modify advantages, multipliers, budgets, learning rates, or scheduling.

- [ ] **Step 5: Run diagnostics/evidence regressions and commit**

```text
pytest tests/rl/test_lagrangian_diagnostics.py -q
pytest tests/rl/test_lagrangian_evidence.py -q
pytest tests/integrations/test_lagrangian_ppo.py -q
pytest tests/rl/test_cost_diagnostics.py tests/rl/test_cost_evidence.py -q
ruff check trade_rl/rl/lagrangian_diagnostics.py trade_rl/rl/lagrangian_evidence.py trade_rl/integrations/lagrangian_ppo.py tests/rl/test_lagrangian_diagnostics.py tests/rl/test_lagrangian_evidence.py tests/integrations/test_lagrangian_ppo.py
ruff format --check trade_rl/rl/lagrangian_diagnostics.py trade_rl/rl/lagrangian_evidence.py trade_rl/integrations/lagrangian_ppo.py tests/rl/test_lagrangian_diagnostics.py tests/rl/test_lagrangian_evidence.py tests/integrations/test_lagrangian_ppo.py
mypy trade_rl/rl/lagrangian_diagnostics.py trade_rl/rl/lagrangian_evidence.py trade_rl/integrations/lagrangian_ppo.py
```

```text
git add trade_rl/rl/lagrangian_diagnostics.py trade_rl/rl/lagrangian_evidence.py trade_rl/integrations/lagrangian_ppo.py tests/rl/test_lagrangian_diagnostics.py tests/rl/test_lagrangian_evidence.py tests/integrations/test_lagrangian_ppo.py
git commit -m "fix: record raw Lagrangian stability evidence"
```

---

### Task 9: Exact-head verification, documentation reconciliation, and PR metadata

**Files:**
- Create: `docs/verification/2026-07-26-constrained-ppo-pr-c-correction-verification.md`
- Modify: `docs/superpowers/plans/2026-07-26-constrained-ppo-pr-c-lagrangian.md`
- Modify: `docs/superpowers/plans/2026-07-26-constrained-ppo-pr-c-stability-addendum.md`
- Modify: PR #191 body.

**Interfaces:**
- Produces exact-head evidence and removes superseded claims that independent normalization or zero-action hard rejection remain normative.

- [ ] **Step 1: Run targeted correction suite**

```text
pytest tests/rl/test_lagrangian_advantages.py -q
pytest tests/rl/test_lagrangian_episode.py -q
pytest tests/rl/test_lagrangian.py -q
pytest tests/rl/test_lagrangian_dual.py -q
pytest tests/rl/test_lagrangian_probe.py -q
pytest tests/rl/test_lagrangian_diagnostics.py -q
pytest tests/rl/test_lagrangian_evidence.py -q
pytest tests/integrations/test_cost_rollout_buffer.py -q
pytest tests/integrations/test_cost_rollout_alignment.py -q
pytest tests/integrations/test_lagrangian_ppo.py -q
pytest tests/integrations/test_lagrangian_checkpoint_roundtrip.py -q
pytest tests/integrations/test_sb3_lagrangian_backend.py -q
```

Record command, exit code, test count, and commit SHA.

- [ ] **Step 2: Run full local/CI-equivalent verification**

```text
ruff check .
ruff format --check .
mypy .
pytest -q
```

Then run repository-maintained critical branch coverage, Ubuntu compatibility, Windows compatibility, and training-image build/probe workflows. Record exact workflow run identifiers and the verified head SHA. Do not claim a platform passed without a completed result tied to that SHA.

- [ ] **Step 3: Record controlled behavioral evidence**

The verification document must contain exact results for:

```text
zero-multiplier ordinary-PPO policy/optimizer parity
raw cost-unit conversion invariance
frozen multipliers across every minibatch and epoch
irregular-time drawdown and turnover aggregation
shadow censoring excluded from denominator
warmup observations consumed by first eligible update
20-episode rare-event minimum support
denominator-aware EMA formula
lower-bound versus upper-cap reporting
checkpoint continuation equality
cash versus baseline probe semantics
probe violation warning without training rejection
ordinary PPO and Cost Critic PPO regressions
```

State explicitly that PR C does not establish production budgets, production safety, or sealed-test superiority.

- [ ] **Step 4: Reconcile superseded documents**

Add a prominent header to both older PR C plans:

```text
Superseded for actor composition, episode aggregation, estimator scheduling,
and feasibility-probe behavior by:
docs/superpowers/specs/2026-07-26-pr-c-lagrangian-stability-correction.md
docs/superpowers/plans/2026-07-26-pr-c-lagrangian-stability-correction.md
```

Update any remaining statement that says reward and each cost advantage are independently normalized for optimization. Update the stability addendum so the canonical-action probe is diagnostic and no longer rejects solely on budget violation.

- [ ] **Step 5: Update PR #191 body and commit documentation**

The PR body must describe the implemented corrected contracts, exact test evidence, current head SHA, stacked dependency on PR #190, and remaining non-goals. It must not retain `Current state: implementation plan committed; RED tests follow` after implementation begins.

```text
git add docs/verification/2026-07-26-constrained-ppo-pr-c-correction-verification.md docs/superpowers/plans/2026-07-26-constrained-ppo-pr-c-lagrangian.md docs/superpowers/plans/2026-07-26-constrained-ppo-pr-c-stability-addendum.md
git commit -m "docs: verify corrected Lagrangian PPO semantics"
```

---

## Self-Review

- **Spec coverage:** Tasks 1-2 cover raw actor composition, final-only normalization, cost-unit invariance, frozen multipliers, and zero-λ parity. Tasks 3-4 cover elapsed time, completion classification, censoring, explicit units, and state carry. Task 5 covers pooled support, warmup retention, denominator-aware EMA, boundaries, and deterministic state. Task 6 covers configuration, backend, identity, and old-checkpoint rejection. Task 7 covers canonical-action semantics and warning-only budget violations. Task 8 covers corrected raw penalty and upper-cap-only stability evidence. Task 9 covers exact-head verification and superseded-document reconciliation.
- **Placeholder scan:** No TBD, TODO, “implement later,” implicit edge handling, or unnamed test task remains.
- **Type consistency:** `EpisodeCompletionKind`, `CompletedEpisodeBatch`, `LagrangianConstraintSpec.minimum_completed_episodes`, `LagrangianDualController.update_after_rollout(..., censored_episode_count=...)`, `DualUpdateReport.at_lower_bound`, `DualUpdateReport.at_upper_cap`, `LagrangianPPOConfig`, and `CanonicalActionProbeEvidence` have one spelling and one owning module throughout.
- **Objective consistency:** The environment reward and Cost Critic returns are unchanged; only actor composition and completed-episode dual statistics change.
- **Unit consistency:** Drawdown and margin deficit use fraction-days, turnover uses turnover/day, gross excess uses per decision, execution cost uses per episode, and events use event/episode.
- **Scope consistency:** PID, augmented Lagrangian, covariance-aware optimization, distributional risk, budget selection, selection gates, model-capacity changes, and sealed-test tuning remain outside this plan.
