# Stage A SB3 Evaluation Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute one manifest-bound Stage A validation or sealed-test cell through the maintained `ResidualMarketEnv`, using the complete PostgreSQL-backed triplet dataset for causal observation history while scoring only the exact request range.

**Architecture:** Add a framework-independent bounded executor in `trade_rl.workflows` that resolves the exact dataset, builds an identity-bound `ResidualMarketEnv`, resets it at `evaluation_range.start`, and advances it until `evaluation_range.stop`. The adapter accepts either the canonical loaded policy or a baseline request; policy observations are passed directly to `predict()`, while baseline requests use an environment-owned action that reproduces the shadow baseline. Every action, observation digest, equity value, order event, dataset identity, execution identity, and processing index is validated before returning `StageAEvaluationEpisodeResult`.

**Tech Stack:** Python 3.12, NumPy, Gymnasium, maintained `ResidualMarketEnv`, canonical serving-policy interfaces, pytest, Ruff, Mypy, Import Linter.

## Global Constraints

- Keep Stable-Baselines3 and framework-specific assembly outside this executor; `trade_rl.workflows` remains framework independent and receives the maintained environment through a Protocol.
- Never materialize or truncate a `MarketDataset` at the evaluation boundary.
- Reset portfolio, reward, execution, and order state exactly at `evaluation_range.start`.
- The environment may read pre-range bars only for causal feature and sequence observation construction.
- Emit no action, reward, equity transition, or order event before the authorized range.
- Stop after the final interval whose resulting index equals `evaluation_range.stop`.
- Require deterministic policy inference and reject unsupported policy call signatures.
- Require exact dataset ID, feature identity, execution-policy digest, range, candidate-config digest, and request digest closure.
- Reject order events whose `processing_index` or timestamp does not belong to the authorized interval.
- Do not weaken existing Stage A artifact, replay, or evidence validation.
- Keep maintained documentation under `docs/operations`; `docs/superpowers` is prohibited by repository contract.

---

## File Structure

- Create `trade_rl/workflows/stage_a_sb3_evaluation.py`: concrete dataset resolver, environment handle/factory protocols, observation hashing, deterministic policy dispatch, and `StageASB3EvaluationEpisodeExecutor`.
- Modify `trade_rl/rl/environment.py`: expose one deterministic `baseline_action()` method that encodes the current shadow baseline under both residual and direct target-weight action modes.
- Create `tests/workflows/test_stage_a_sb3_evaluation.py`: focused adapter tests for range reset, deterministic action dispatch, observation/equity/event collection, and fail-closed identity checks.
- Modify `tests/rl/test_environment_timing.py`: prove `baseline_action()` keeps hybrid and shadow execution identical for residual and target-weight modes.
-
### Task 1: Environment-Owned Baseline Action

**Files:**
- Modify: `trade_rl/rl/environment.py`
- Modify: `tests/rl/test_environment_timing.py`

**Interfaces:**
- Consumes: `ResidualMarketEnv.action_spec`, `_market_inputs()`, `ActionMode`.
- Produces: `ResidualMarketEnv.baseline_action() -> np.ndarray`.

- [ ] **Step 1: Write failing residual-mode test**

Add a test that resets a residual environment, calls `baseline_action()`, verifies the vector is finite with exact action-space shape, steps once, and proves hybrid quantities equal shadow quantities.

- [ ] **Step 2: Run the focused test and verify RED**

Run: `uv run pytest -q tests/rl/test_environment_timing.py -k baseline_action`

Expected: failure because `ResidualMarketEnv` has no `baseline_action` method.

- [ ] **Step 3: Implement minimal baseline encoding**

Implement:

```python
def baseline_action(self) -> np.ndarray:
    trends, _, _ = self._market_inputs()
    if self.action_spec.mode is ActionMode.TARGET_WEIGHT:
        action = np.asarray(trends.base, dtype=np.float32)
    else:
        action = np.zeros(self.action_spec.size, dtype=np.float32)
    self.action_spec.parse(action, mode=ActionValidationMode.FAIL_CLOSED)
    return action.copy()
```

- [ ] **Step 4: Add and pass target-weight test**

Construct a target-weight environment, verify `baseline_action()` equals the current trend baseline, step once, and prove hybrid and shadow quantities are identical.

- [ ] **Step 5: Run environment regression tests**

Run: `uv run pytest -q tests/rl/test_environment_timing.py tests/rl/test_target_weight_action.py`

Expected: all pass.

### Task 2: Manifest-Bound Executor Contract

**Files:**
- Create: `trade_rl/workflows/stage_a_sb3_evaluation.py`
- Create: `tests/workflows/test_stage_a_sb3_evaluation.py`

**Interfaces:**
- Consumes: `StageAEvaluationCellRequest`, `StageAEvaluationEpisodeResult`, `MarketDataset`, `ResidualMarketEnv`.
- Produces:
  - `StageAEvaluationDatasetResolver.resolve(request) -> MarketDataset`
  - `StageAEvaluationEnvironmentHandle(environment, candidate_config_digest)`
  - `StageAEvaluationEnvironmentFactory.build(request, dataset, candidate_config_digest) -> StageAEvaluationEnvironmentHandle`
  - `StageASB3EvaluationEpisodeExecutor.execute(...) -> StageAEvaluationEpisodeResult`

- [ ] **Step 1: Write failing exact-range execution test**

Use a small fake environment implementing the executor-facing protocol. Assert that execution calls reset exactly once with:

```python
{
    "start_idx": request.evaluation_range.start,
    "episode_bars": request.evaluation_range.stop - request.evaluation_range.start,
    "initial_state_mode": "cash",
}
```

Assert no dataset view or sliced dataset is supplied.

- [ ] **Step 2: Verify RED**

Run: `uv run pytest -q tests/workflows/test_stage_a_sb3_evaluation.py::test_executor_uses_full_dataset_and_exact_request_range`

Expected: import failure because the integration module does not exist.

- [ ] **Step 3: Implement resolver and environment closure**

The executor must:

1. call `request.validate_manifest(plan, manifest)`;
2. resolve the full dataset;
3. require `dataset.dataset_id == request.dataset_id`;
4. require `dataset.feature_config_digest == request.feature_identity`;
5. require `0 <= start < stop < dataset.n_bars`;
6. build an environment handle;
7. require handle candidate-config digest equals the requested producer digest;
8. require environment dataset ID and execution-policy digest equal the request;
9. require `environment.minimum_start_index <= start`.

- [ ] **Step 4: Pass exact-range test**

Run the focused test and confirm GREEN.

### Task 3: Deterministic Policy and Baseline Rollout

**Files:**
- Modify: `trade_rl/workflows/stage_a_sb3_evaluation.py`
- Modify: `tests/workflows/test_stage_a_sb3_evaluation.py`

**Interfaces:**
- Consumes: policy objects with `predict(observation) -> np.ndarray`, environment `baseline_action()`.
- Produces: exact action tuple, observation-digest tuple, equity curve, and terminal states.

- [ ] **Step 1: Write failing policy rollout test**

Provide a policy whose `predict()` records each observation and returns finite actions. Assert:

- policy is called exactly once per environment decision;
- the initial observation is hashed before the first action;
- each next observation is hashed after its transition;
- `len(observation_digests) == len(actions) + 1`;
- equity begins with reset-time portfolio value and ends with terminal portfolio value.

- [ ] **Step 2: Write failing baseline rollout test**

Pass `policy=None` and assert the executor calls `environment.baseline_action()` for every decision and never invokes a policy.

- [ ] **Step 3: Implement canonical observation hashing**

Hash ndarray or mapping observations with explicit dtype, shape, and finite numeric values:

```python
{
    "schema_version": "stage_a_policy_observation_digest_v1",
    "components": {
        key: {"dtype": str(array.dtype), "shape": list(array.shape), "values": array.tolist()}
    },
}
```

Sort mapping keys through canonical JSON and reject unsupported, empty, or non-finite observations.

- [ ] **Step 4: Implement rollout loop**

At each decision:

1. choose `environment.baseline_action()` or `policy.predict(observation)`;
2. normalize to a finite one-dimensional float32 action matching `environment.action_space.shape`;
3. append the action;
4. call `environment.step(action)`;
5. collect `hybrid_execution.order_events` and optional liquidation events when exposed;
6. append next observation digest and `environment.hybrid.portfolio_value`;
7. require termination or truncation occurs exactly at the authorized stop, otherwise fail closed.

- [ ] **Step 5: Run focused rollout tests**

Run: `uv run pytest -q tests/workflows/test_stage_a_sb3_evaluation.py -k 'rollout or baseline or policy'`

Expected: all pass.

### Task 4: Evidence and Range Hardening

**Files:**
- Modify: `trade_rl/workflows/stage_a_sb3_evaluation.py`
- Modify: `tests/workflows/test_stage_a_sb3_evaluation.py`

**Interfaces:**
- Consumes: collected `OrderEvent` values and dataset timestamps.
- Produces: a fully validated `StageAEvaluationEpisodeResult`.

- [ ] **Step 1: Add failing identity-drift tests**

Cover dataset ID, feature identity, candidate-config digest, environment execution-policy digest, environment dataset ID, and insufficient minimum start history.

- [ ] **Step 2: Add failing range-escape tests**

Reject:

- evaluation ranges ending at or beyond `dataset.n_bars`;
- early environment termination before the range stop;
- environment advancement beyond the range stop;
- event `processing_index < start` or `processing_index > stop`;
- event timestamp unequal to the dataset timestamp at its processing index;
- duplicate or non-contiguous global event sequence after combining decisions.

- [ ] **Step 3: Implement event normalization**

Re-sequence collected events globally with `dataclasses.replace(event, sequence=index)` only after all original per-step streams are internally contiguous. Validate each event against request identities, processing bounds, and exact dataset timestamps.

- [ ] **Step 4: Construct and self-validate result**

Build `StageAEvaluationEpisodeResult` from the normalized values, then call `validate_against()` before returning it.

- [ ] **Step 5: Run complete adapter tests**

Run: `uv run pytest -q tests/workflows/test_stage_a_sb3_evaluation.py tests/workflows/test_stage_a_execution_producer.py`

Expected: all pass.

### Task 5: Architecture and Full Verification

**Files:**
- Modify: `docs/operations/stage-a-sb3-evaluation-environment-implementation-plan.md`

**Interfaces:**
- Produces: lazy `StageASB3EvaluationEpisodeExecutor` export.

- [ ] **Step 1: Verify the workflow boundary**

Ensure the bounded executor imports no Stable-Baselines3 or Gymnasium modules and passes the repository responsibility-layer contract.

- [ ] **Step 2: Run focused verification**

Run:

```bash
uv run pytest -q \
  tests/workflows/test_stage_a_sb3_evaluation.py \
  tests/workflows/test_stage_a_execution_producer.py \
  tests/rl/test_environment_timing.py \
  tests/rl/test_target_weight_action.py
uv run ruff check trade_rl/workflows/stage_a_sb3_evaluation.py tests/workflows/test_stage_a_sb3_evaluation.py trade_rl/rl/environment.py tests/rl/test_environment_timing.py
uv run ruff format --check trade_rl/workflows/stage_a_sb3_evaluation.py tests/workflows/test_stage_a_sb3_evaluation.py trade_rl/rl/environment.py tests/rl/test_environment_timing.py
uv run mypy trade_rl/workflows/stage_a_sb3_evaluation.py trade_rl/rl/environment.py
uv run lint-imports
```

- [ ] **Step 3: Run exact-head full verification**

Run the repository CI and PostgreSQL Catalog workflows on one unchanged head. Require all pytest, branch coverage, Studio, Windows/Ubuntu compatibility, training image, Mypy, Ruff, Format, Import Linter, and PostgreSQL tests to pass.

- [ ] **Step 4: Update PR evidence and merge**

Record the exact head SHA and test totals in the PR body. Merge only after the exact-head workflows succeed and no review thread remains unresolved.

## Self-Review

- Spec coverage: full-dataset warm-up, exact range reset, deterministic baseline/policy actions, event/equity/observation closure, identity drift, range escape, framework boundary, and exact-head verification are each assigned to a task.
- Placeholder scan: no deferred implementation or unspecified error-handling step remains.
- Type consistency: the executor implements the existing `StageAEvaluationEpisodeExecutor` protocol and returns the existing `StageAEvaluationEpisodeResult`; new factory and resolver protocols are defined in the integration module only.
