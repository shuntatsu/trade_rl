# Stabilized Lagrangian PPO Implementation Plan

> [!IMPORTANT]
> **Superseded for actor composition, episode aggregation, estimator scheduling, and feasibility-probe behavior by:**
> `docs/superpowers/specs/2026-07-26-pr-c-lagrangian-stability-correction.md`
> `docs/superpowers/plans/2026-07-26-pr-c-lagrangian-stability-correction.md`
> Where this document conflicts with those files, the correction specification and plan are normative.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in `lagrangian_ppo` algorithm that constrains the PPO actor with raw Cost Critic advantages, final-only combined normalization, and stabilized per-cost dual variables, without changing ordinary PPO, PR B `cost_critic_ppo`, or the exact all-cost net-log-growth reward.

**Architecture:** Keep PR B's market encoder, reward critic, Cost Critics, and cost rollout storage unchanged. Add a pure typed Lagrangian layer for completed-episode aggregation, EMA/capped dual updates, and advantage composition; then add `LagrangianPPO`, a `CostCriticPPO` subclass whose PPO update uses frozen rollout multipliers and whose dual update runs once after the rollout's actor and Cost Critic updates. Configuration, telemetry, checkpoint identity, and save/load state remain explicit and fail closed.

**Tech Stack:** Python 3.12, NumPy, PyTorch 2.3.1, Stable-Baselines3 2.3.2, Gymnasium 0.29.1, pytest, Ruff, Mypy.

## Global Constraints

- The scalar reward remains exactly the maintained all-cost net-log-growth reward.
- Ordinary `ppo` and PR B `cost_critic_ppo` remain behaviorally unchanged.
- No multiplier may change during PPO epochs for a rollout.
- The actor composes raw reward and raw cost advantages first; only the final combined vector uses the pinned SB3 normalization. Per-cost standardization is diagnostics-only.
- A dual update occurs once after rollout statistics are finalized, never per minibatch.
- Event estimates use completed-episode denominators; a zero denominator skips the update and retains previous EMA and multiplier.
- True economic terminations and time-limit truncations both complete an episode for aggregation, but only actual event cost values enter event numerators.
- Unknown costs, reordered costs, non-finite estimates, non-finite multipliers, mismatched checkpoint identity, and incompatible algorithm state fail closed.
- Lagrangian schema, multiplier state, EMA state, episode accumulators, update counters, and architecture identity survive deterministic save/load.
- PR C does not choose production budgets or change evaluation gates; production profiles and selection policy remain PR D.

---

## File Structure

- `trade_rl/rl/lagrangian.py`: typed constraint aggregation, completed-episode accumulator, dual controller, state and digest identity.
- `trade_rl/rl/lagrangian_advantages.py`: pure raw reward-minus-cost composition plus diagnostics-only normalization helpers.
- `trade_rl/integrations/lagrangian_ppo.py`: SB3-pinned PPO training loop with aligned cost minibatches, frozen multipliers, Cost Critic update, and one post-rollout dual update.
- `trade_rl/rl/algorithm_configs.py`: typed `LagrangianPPOConfig` view.
- `trade_rl/rl/training.py`: opt-in vector configuration and fail-closed validation.
- `trade_rl/integrations/sb3_training.py`: backend construction, memory accounting reuse, architecture metadata, checkpoint resume class, and replay rejection.
- `trade_rl/rl/checkpointing.py`: no schema bump; existing algorithm-identity mechanism validates the new Lagrangian identity.
- `tests/rl/test_lagrangian.py`: pure aggregation and dual-state contracts.
- `tests/rl/test_lagrangian_advantages.py`: normalization and composition contracts.
- `tests/integrations/test_lagrangian_ppo.py`: actor behavior, frozen multiplier, dual update, and save/load contracts.
- `tests/rl/test_lagrangian_training_config.py`: configuration and digest contracts.
- `tests/integrations/test_sb3_lagrangian_backend.py`: backend model selection and memory/replay behavior.
- `docs/verification/2026-07-26-constrained-ppo-pr-c-verification.md`: exact-head verification evidence.

---

### Task 1: Typed aggregation and dual schema

**Files:**
- Create: `trade_rl/rl/lagrangian.py`
- Test: `tests/rl/test_lagrangian.py`

**Interfaces:**
- Consumes: `CONSTRAINT_COST_NAMES` and `CostLearningSchema.names`.
- Produces: `ConstraintAggregation`, `LagrangianConstraintSpec`, `LagrangianSchema`, and `canonical_lagrangian_schema(...)`.

- [ ] **Step 1: Write failing schema tests**

```python
from trade_rl.rl.lagrangian import (
    ConstraintAggregation,
    LagrangianConstraintSpec,
    LagrangianSchema,
)


def test_lagrangian_schema_preserves_canonical_order_and_identity() -> None:
    schema = LagrangianSchema(
        (
            LagrangianConstraintSpec(
                name="drawdown_excess",
                aggregation=ConstraintAggregation.EPISODE_SUM,
                budget=0.0,
                dual_learning_rate=0.05,
                ema_beta=0.9,
                initial_multiplier=0.0,
                max_multiplier=10.0,
                warmup_rollouts=2,
                update_interval_rollouts=3,
            ),
            LagrangianConstraintSpec(
                name="drawdown_stop_event",
                aggregation=ConstraintAggregation.EPISODE_EVENT_RATE,
                budget=0.0,
                dual_learning_rate=0.1,
                ema_beta=0.95,
                initial_multiplier=0.0,
                max_multiplier=20.0,
                warmup_rollouts=1,
                update_interval_rollouts=1,
            ),
        )
    )

    assert schema.names == ("drawdown_excess", "drawdown_stop_event")
    assert len(schema.digest) == 64
```

Add tests that reject duplicate, unknown, reordered, wrong aggregation, negative budget, non-positive learning rate/cap/interval, `ema_beta` outside `[0, 1)`, and initial multiplier above its cap.

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest tests/rl/test_lagrangian.py -q`
Expected: import failure because `trade_rl.rl.lagrangian` does not exist.

- [ ] **Step 3: Implement immutable typed schema**

Implement:

```python
class ConstraintAggregation(str, Enum):
    EPISODE_SUM = "episode_sum"
    EPISODE_MEAN = "episode_mean"
    EPISODE_EVENT_RATE = "episode_event_rate"
```

Canonical aggregation must be:

```python
{
    "drawdown_excess": EPISODE_SUM,
    "drawdown_stop_event": EPISODE_EVENT_RATE,
    "margin_deficit_fraction": EPISODE_SUM,
    "forced_liquidation_event": EPISODE_EVENT_RATE,
    "gross_exposure_request_excess": EPISODE_MEAN,
    "daily_turnover": EPISODE_MEAN,
    "execution_cost_fraction": EPISODE_SUM,
}
```

`LagrangianSchema` must preserve canonical cost order, provide `names`, `digest_payload()`, `digest`, and `__getitem__(name)`.

- [ ] **Step 4: Run unit tests and static checks**

Run:

```text
pytest tests/rl/test_lagrangian.py -q
ruff check trade_rl/rl/lagrangian.py tests/rl/test_lagrangian.py
ruff format --check trade_rl/rl/lagrangian.py tests/rl/test_lagrangian.py
mypy trade_rl/rl/lagrangian.py
```

Expected: all pass.

- [ ] **Step 5: Commit**

```text
git add trade_rl/rl/lagrangian.py tests/rl/test_lagrangian.py
git commit -m "feat: add typed Lagrangian constraint schema"
```

---

### Task 2: Completed-episode cost aggregation

**Files:**
- Modify: `trade_rl/rl/lagrangian.py`
- Test: `tests/rl/test_lagrangian.py`

**Interfaces:**
- Consumes: cost matrices `[steps, envs, costs]`, `terminated`, and `truncated` from `CostRolloutStorage`.
- Produces: `ConstraintEstimate` and `CompletedEpisodeCostAccumulator.ingest_rollout(...)`.

- [ ] **Step 1: Write failing aggregation tests**

Use two environments with episodes crossing rollout boundaries. Assert:

```python
estimates = accumulator.ingest_rollout(
    costs=costs,
    terminated=terminated,
    truncated=truncated,
)
assert estimates["drawdown_excess"].value == pytest.approx(0.06)
assert estimates["gross_exposure_request_excess"].value == pytest.approx(0.15)
assert estimates["drawdown_stop_event"].value == pytest.approx(0.5)
assert estimates["drawdown_stop_event"].denominator == 2
```

Add tests that:

- preserve unfinished per-environment sums into the next rollout;
- count both termination and truncation as completed episodes;
- reject a transition marked both terminated and truncated;
- return `None` for every estimate when no episode completed;
- reject shape mismatch, negative costs, and non-finite values;
- restore accumulator state exactly from `state_dict()`.

- [ ] **Step 2: Run test and verify RED**

Run: `pytest tests/rl/test_lagrangian.py -q`
Expected: missing accumulator symbols.

- [ ] **Step 3: Implement aggregation**

`CompletedEpisodeCostAccumulator` must keep per-environment cost sums and step counts. On each done transition:

- `EPISODE_SUM`: add that episode's accumulated sum to the rollout numerator;
- `EPISODE_MEAN`: add `episode_sum / episode_steps`;
- `EPISODE_EVENT_RATE`: add the episode's event sum, requiring it to remain within `[0, 1]`;
- increment the completed-episode denominator once;
- clear only that environment's accumulator.

`ConstraintEstimate.value` is `numerator / denominator` and must be finite and non-negative.

- [ ] **Step 4: Run tests and static checks**

Run the Task 1 commands plus `pytest tests/rl/test_lagrangian.py -q`.

- [ ] **Step 5: Commit**

```text
git add trade_rl/rl/lagrangian.py tests/rl/test_lagrangian.py
git commit -m "feat: aggregate completed episode constraint costs"
```

---

### Task 3: Stabilized dual controller

**Files:**
- Modify: `trade_rl/rl/lagrangian.py`
- Test: `tests/rl/test_lagrangian.py`

**Interfaces:**
- Consumes: `LagrangianSchema` and one optional `ConstraintEstimate` per constraint after each rollout.
- Produces: immutable frozen multiplier snapshots, `DualUpdateReport`, deterministic `state_dict()`, and `load_state_dict()`.

- [ ] **Step 1: Write failing dual tests**

Cover:

```python
frozen = controller.begin_rollout()
report = controller.update_after_rollout({"drawdown_excess": estimate})
assert frozen["drawdown_excess"] == 0.0
assert report["drawdown_excess"].multiplier_after > 0.0
assert controller.begin_rollout()["drawdown_excess"] == report["drawdown_excess"].multiplier_after
```

Also assert:

- multiplier increases above budget and decreases below budget;
- multiplier clips at zero and its independent cap;
- EMA follows `ema = beta * previous + (1-beta) * raw`, initializing from the first raw estimate;
- warmup and update interval skip without mutating EMA or multiplier;
- zero-denominator/missing estimate skips and retains state;
- one cost update cannot alter another cost;
- non-finite estimates or resulting multipliers fail closed;
- state save/load reproduces the next update exactly;
- schema digest mismatch rejects state load.

- [ ] **Step 2: Run test and verify RED**

Run: `pytest tests/rl/test_lagrangian.py -q`.

- [ ] **Step 3: Implement controller**

Use the exact update:

```python
lambda_after = np.clip(
    lambda_before + spec.dual_learning_rate * (ema_cost - spec.budget),
    0.0,
    spec.max_multiplier,
)
```

`begin_rollout()` returns a copied read-only multiplier vector in schema order. `update_after_rollout()` increments one rollout counter and performs at most one update per eligible constraint.

- [ ] **Step 4: Run tests and static checks**

Run all Task 1 static commands and `pytest tests/rl/test_lagrangian.py -q`.

- [ ] **Step 5: Commit**

```text
git add trade_rl/rl/lagrangian.py tests/rl/test_lagrangian.py
git commit -m "feat: add stabilized per-cost dual controller"
```

---

### Task 4: Independent advantage normalization and composition

**Files:**
- Create: `trade_rl/rl/lagrangian_advantages.py`
- Test: `tests/rl/test_lagrangian_advantages.py`

**Interfaces:**
- Consumes: reward advantages `[batch]`, cost advantages `[batch, costs]`, and frozen multipliers `[costs]`.
- Produces: `normalize_advantage_vector(...)`, `normalize_cost_advantages(...)`, and `combine_lagrangian_advantages(...)`.

- [ ] **Step 1: Write failing numerical tests**

Assert each cost column is independently zero-mean/unit-variance, a constant column becomes zeros, and:

```python
combined = combine_lagrangian_advantages(
    reward_advantages=np.asarray([1.0, -1.0]),
    cost_advantages=np.asarray([[2.0, 4.0], [0.0, 2.0]]),
    multipliers=np.asarray([0.5, 0.25]),
    normalize_reward=True,
)
```

matches the explicitly calculated reward-normalized minus independently cost-normalized result.

Add tests for zero multipliers, disabled reward normalization, shape mismatch, empty arrays, and non-finite values.

- [ ] **Step 2: Run test and verify RED**

Run: `pytest tests/rl/test_lagrangian_advantages.py -q`.

- [ ] **Step 3: Implement pure functions**

Normalization uses population standard deviation and epsilon `1e-8`; if standard deviation is at or below epsilon, return zeros for that vector. Never normalize a flattened multi-cost matrix.

- [ ] **Step 4: Run tests and static checks**

Run:

```text
pytest tests/rl/test_lagrangian_advantages.py -q
ruff check trade_rl/rl/lagrangian_advantages.py tests/rl/test_lagrangian_advantages.py
ruff format --check trade_rl/rl/lagrangian_advantages.py tests/rl/test_lagrangian_advantages.py
mypy trade_rl/rl/lagrangian_advantages.py
```

- [ ] **Step 5: Commit**

```text
git add trade_rl/rl/lagrangian_advantages.py tests/rl/test_lagrangian_advantages.py
git commit -m "feat: compose independently normalized Lagrangian advantages"
```

---

### Task 5: Lagrangian PPO actor update

**Files:**
- Create: `trade_rl/integrations/lagrangian_ppo.py`
- Modify: `trade_rl/integrations/cost_critic_ppo.py` only to expose protected reusable Cost Critic update/diagnostic hooks if required; do not change PR B behavior.
- Test: `tests/integrations/test_lagrangian_ppo.py`

**Interfaces:**
- Consumes: PR B `CostCriticPPO`, `CostRolloutStorage`, Lagrangian schema/controller/accumulator, and pinned SB3 PPO 2.3.2 rollout internals.
- Produces: `LagrangianPPO` with algorithm identifier `lagrangian_ppo`.

- [ ] **Step 1: Write failing integration tests**

Build deterministic one- and two-environment synthetic environments. Test:

- `lambda=0` produces byte-for-byte equal ordinary PPO policy state after identical RNG reset;
- positive multiplier changes the policy update in the direction induced by the matching synthetic cost advantage;
- the multiplier used by every minibatch/epoch of one rollout is identical;
- dual update occurs once after the actor and Cost Critic update, and becomes visible only to the next rollout;
- missing cost info fails closed through the inherited collector;
- a rollout with no completed episode skips dual updates;
- unsafe completed episodes increase only the matching multiplier;
- safe completed episodes decrease an existing multiplier;
- event multiplier skips when there is no completed-episode denominator.

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest tests/integrations/test_lagrangian_ppo.py -q`.

- [ ] **Step 3: Implement the aligned PPO training loop**

Subclass `CostCriticPPO`. At rollout start, capture `self.frozen_lagrange_multipliers = self.lagrangian_controller.begin_rollout()`.

Copy the pinned SB3 2.3.2 PPO minibatch loop into a private method and change only advantage calculation:

```python
reward_advantages = rollout_data.advantages
if self.normalize_advantage and len(reward_advantages) > 1:
    reward_advantages = normalize_advantage_vector(reward_advantages)

cost_batch = self.cost_rollout_storage.sample(batch_indices)
combined_advantages = combine_lagrangian_advantages(
    reward_advantages=reward_advantages,
    cost_advantages=cost_batch.cost_advantages,
    multipliers=self.frozen_lagrange_multipliers,
    normalize_reward=False,
)
```

Generate one permutation and pass identical `batch_indices` to `rollout_buffer._get_samples(...)` and `cost_rollout_storage.sample(...)`. Preserve every remaining SB3 expression, KL early stop, clipping, entropy, reward-value loss, optimizer step, and logger key.

After actor update:

1. call inherited `_train_cost_critic()`;
2. aggregate the just-finished cost rollout;
3. call `update_after_rollout(...)` exactly once;
4. log raw estimate, denominator, EMA, budget, multiplier before/after, saturation, and skip reason per cost.

- [ ] **Step 4: Run integration and regression tests**

Run:

```text
pytest tests/integrations/test_lagrangian_ppo.py -q
pytest tests/integrations/test_cost_critic_ppo.py -q
pytest tests/rl/test_lagrangian.py tests/rl/test_lagrangian_advantages.py -q
```

Expected: all pass and PR B equivalence remains unchanged.

- [ ] **Step 5: Commit**

```text
git add trade_rl/integrations/lagrangian_ppo.py trade_rl/integrations/cost_critic_ppo.py tests/integrations/test_lagrangian_ppo.py
git commit -m "feat: apply frozen Lagrangian advantages to PPO"
```

---

### Task 6: Deterministic save/load and checkpoint identity

**Files:**
- Modify: `trade_rl/integrations/lagrangian_ppo.py`
- Test: `tests/integrations/test_lagrangian_ppo.py`
- Test: `tests/rl/test_cost_checkpoint_identity.py`

**Interfaces:**
- Consumes: existing SB3 model serialization and checkpoint `algorithm_identity` validation.
- Produces: complete Lagrangian state and identity through `checkpoint_identity_payload()`.

- [ ] **Step 1: Write failing round-trip tests**

Save a model with non-zero multipliers, EMA, accumulator carry, rollout count, and update counts. Load it and assert:

- identity equality;
- multiplier/EMA/counter equality;
- unfinished episode accumulator equality;
- Cost Critic and optimizer equality;
- the next `update_after_rollout()` report is identical before and after load.

Change budget, cap, aggregation, schema order, or architecture and assert checkpoint identity rejection.

- [ ] **Step 2: Run test and verify RED**

Run the two test files above.

- [ ] **Step 3: Extend identity and serialization**

`checkpoint_identity_payload()` must include:

```python
{
    **super().checkpoint_identity_payload(),
    "algorithm": "lagrangian_ppo",
    "lagrangian_schema_digest": self.lagrangian_schema.digest,
    "lagrangian_cost_names": list(self.lagrangian_schema.names),
}
```

Keep controller and accumulator as serializable model attributes. Normalize restored tuple/array containers in `_setup_model()` before rebuilding runtime helpers.

- [ ] **Step 4: Run tests and static checks**

Run Lagrangian integration tests, checkpoint tests, Ruff, format, and Mypy for modified files.

- [ ] **Step 5: Commit**

```text
git add trade_rl/integrations/lagrangian_ppo.py tests/integrations/test_lagrangian_ppo.py tests/rl/test_cost_checkpoint_identity.py
git commit -m "feat: persist Lagrangian PPO dual state"
```

---

### Task 7: Typed configuration and SB3 backend

**Files:**
- Modify: `trade_rl/rl/training.py`
- Modify: `trade_rl/rl/algorithm_configs.py`
- Modify: `trade_rl/integrations/sb3_training.py`
- Test: `tests/rl/test_lagrangian_training_config.py`
- Test: `tests/integrations/test_sb3_lagrangian_backend.py`

**Interfaces:**
- Consumes: canonical Cost Critic schema and vector settings aligned in canonical order.
- Produces: `ResidualTrainingConfig(algorithm="lagrangian_ppo", ...)`, `LagrangianPPOConfig`, and backend construction.

- [ ] **Step 1: Write failing config tests**

Add opt-in fields at the end of `ResidualTrainingConfig` to preserve positional compatibility:

```python
lagrangian_budgets: tuple[float, ...] = ()
lagrangian_dual_learning_rates: tuple[float, ...] = ()
lagrangian_ema_betas: tuple[float, ...] = ()
lagrangian_initial_multipliers: tuple[float, ...] = ()
lagrangian_max_multipliers: tuple[float, ...] = ()
lagrangian_warmup_rollouts: tuple[int, ...] = ()
lagrangian_update_interval_rollouts: tuple[int, ...] = ()
```

For `lagrangian_ppo`, every vector must contain exactly one value per enabled cost. For other algorithms, every vector must be empty. Tests must prove digest changes for every vector and ordinary PPO defaults/digest stay unchanged.

- [ ] **Step 2: Run tests and verify RED**

Run the two new test files.

- [ ] **Step 3: Implement typed config view**

Add `LagrangianPPOConfig(CostCriticPPOConfig)` carrying `lagrangian_schema`. `build_algorithm_config()` must return it only for `algorithm="lagrangian_ppo"`.

Backend behavior:

- construct `LagrangianPPO` instead of PPO/CostCriticPPO;
- reuse PR B's PPO plus cost-rollout memory estimator;
- reject replay-buffer resume as PPO-family;
- load checkpoints with `LagrangianPPO`;
- include Lagrangian schema digest and constraint settings in architecture JSON.

- [ ] **Step 4: Run targeted regressions**

Run:

```text
pytest tests/rl/test_lagrangian_training_config.py -q
pytest tests/integrations/test_sb3_lagrangian_backend.py -q
pytest tests/rl/test_cost_critic_training_config.py -q
pytest tests/integrations/test_sb3_cost_critic_backend.py -q
```

- [ ] **Step 5: Commit**

```text
git add trade_rl/rl/training.py trade_rl/rl/algorithm_configs.py trade_rl/integrations/sb3_training.py tests/rl/test_lagrangian_training_config.py tests/integrations/test_sb3_lagrangian_backend.py
git commit -m "feat: expose opt-in Lagrangian PPO training"
```

---

### Task 8: Telemetry and deterministic dual evidence

**Files:**
- Modify: `trade_rl/integrations/lagrangian_ppo.py`
- Create: `trade_rl/rl/lagrangian_evidence.py`
- Test: `tests/rl/test_lagrangian_evidence.py`
- Test: `tests/integrations/test_lagrangian_ppo.py`

**Interfaces:**
- Consumes: dual reports, frozen multiplier snapshot, Lagrangian identity, and PR B compute evidence.
- Produces: canonical JSON evidence with a content digest.

- [ ] **Step 1: Write failing evidence tests**

Assert evidence contains:

- schema and cost ordering;
- budgets and caps;
- frozen rollout multipliers;
- raw and EMA estimates;
- denominators and skip reasons;
- multiplier before/after and saturation;
- rollout/update counters;
- per-cost normalized advantage mean/std;
- reward and combined advantage mean/std;
- content digest that changes when any budget, multiplier, EMA, estimate, denominator, or skip reason changes.

- [ ] **Step 2: Run test and verify RED**

Run: `pytest tests/rl/test_lagrangian_evidence.py -q`.

- [ ] **Step 3: Implement evidence and logger integration**

Use `canonical_json_bytes` and `content_digest`. Do not include sealed-test decisions or tune thresholds in evidence generation.

- [ ] **Step 4: Run targeted tests**

Run Lagrangian evidence and integration files plus PR B cost evidence tests.

- [ ] **Step 5: Commit**

```text
git add trade_rl/rl/lagrangian_evidence.py trade_rl/integrations/lagrangian_ppo.py tests/rl/test_lagrangian_evidence.py tests/integrations/test_lagrangian_ppo.py
git commit -m "feat: record Lagrangian dual evidence"
```

---

### Task 9: Full regression and verification record

**Files:**
- Create: `docs/verification/2026-07-26-constrained-ppo-pr-c-verification.md`
- Modify tests only if a real regression is found; do not weaken assertions or thresholds.

**Interfaces:**
- Consumes: final exact PR C head and CI artifacts.
- Produces: review-ready PR C with explicit claims and non-claims.

- [ ] **Step 1: Run the full local/CI-equivalent suite**

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

- [ ] **Step 2: Record controlled behavioral evidence**

Document:

- zero-multiplier exact ordinary-PPO parity;
- unsafe synthetic multiplier increase;
- safe synthetic multiplier decrease;
- zero-event-denominator skip;
- frozen multiplier across all rollout epochs;
- deterministic save/load and next-update equality;
- ordinary PPO and Cost Critic PPO regression status;
- no production-budget recommendation and no sealed-test tuning.

- [ ] **Step 3: Self-review against the design spec**

Confirm every PR C claim has a test or artifact. Confirm no Lagrange setting is accepted by ordinary PPO or `cost_critic_ppo`. Confirm no dual update occurs inside a minibatch loop.

- [ ] **Step 4: Commit verification record**

```text
git add docs/verification/2026-07-26-constrained-ppo-pr-c-verification.md
git commit -m "docs: record Lagrangian PPO verification"
```

- [ ] **Step 5: Trigger exact-head CI and prepare the stacked PR**

Temporarily retarget to `main` only when required to trigger CI, then restore the base to `agent/constrained-ppo-cost-critics`. Mark ready only after exact-head CI succeeds and review threads are empty.

---

## Self-Review

- Spec coverage: heterogeneous aggregation, event denominators, independent EMA/caps/warmup/interval, frozen rollout multipliers, independent normalization, post-rollout update, checkpoint state, telemetry, ordinary PPO compatibility, and fail-closed validation are each assigned to a task.
- Scope boundary: production budget selection, adverse-regime evaluation, and promotion gates remain PR D.
- Type consistency: every later task consumes names introduced in Tasks 1–4; `LagrangianPPOConfig` extends the existing `CostCriticPPOConfig`; backend and checkpoint identity use the same ordered schema.
- Placeholder scan: the plan contains no TBD/TODO steps or unspecified error handling.
