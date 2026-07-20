# RTX 4070 Ti SUPER Full Training Speedup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the full-capacity `oracle-bc-ppo-15m-target` model while reducing repeated CPU-side teacher and sequence preparation work during multi-seed walk-forward training.

**Architecture:** Keep one oracle candidate and preserve the original full sequence encoder and PPO update depth. Add a vectorized batch gather API to `SequencePolicyPlane`, route behavior-cloning mini-batches through that precomputed plane, and cache the immutable supervised teacher dataset inside each fold-scoped `StableBaselines3Backend` so all three seeds reuse one rollout.

**Tech Stack:** Python 3.12, NumPy, PyTorch, Stable-Baselines3 PPO, pytest.

## Global Constraints

- The maintained full run contains only `oracle-bc-ppo-15m-target`.
- Full training uses three seeds and the existing six-fold walk-forward contract.
- Restore `sequence_d_model=336`, 8 attention heads, 2 attention layers, policy `[384, 256, 128]`, value `[512, 384, 256]`, and `max_policy_parameters=12_000_000`.
- Restore PPO `batch_size=128` and `n_epochs=10`; do not obtain speed by reducing optimization work.
- Preserve causal sequence alignment, normalizer identity, dataset identity, environment identity, and teacher artifact validation.
- Keep the compact model only for smoke tests.
- Remove `.github/workflows/source-export-temp.yml` before final publication.

---

### Task 1: Restore the full maintained training recipe

**Files:**
- Modify: `examples/binance-multitimeframe/training-full.json`
- Modify: `examples/binance-multitimeframe/walk-forward-full.json`
- Modify: `tests/examples/test_binance_multitimeframe_full_assets.py`

**Interfaces:**
- Consumes: `TrainingRunConfig.from_json` and `MarketWalkForwardConfig.from_json`.
- Produces: one full-capacity oracle candidate configuration.

- [ ] **Step 1: Strengthen the existing config tests**

Add assertions for `sequence_capacity == "standard"`, `sequence_d_model == 336`, 8 heads, 2 layers, `max_policy_parameters == 12_000_000`, `batch_size == 128`, and `n_epochs == 10` in both direct and walk-forward full configurations.

- [ ] **Step 2: Run the focused example test and verify RED**

Run: `pytest tests/examples/test_binance_multitimeframe_full_assets.py -q`

Expected: failure because the current walk-forward configuration is compact and uses the reduced PPO settings.

- [ ] **Step 3: Restore the full configuration values**

Replace the compact encoder and reduced network fields in both JSON files with the full values listed in Global Constraints.

- [ ] **Step 4: Run the focused example test and verify GREEN**

Run: `pytest tests/examples/test_binance_multitimeframe_full_assets.py -q`

Expected: all tests pass.

### Task 2: Add vectorized sequence-plane batch gathering

**Files:**
- Modify: `trade_rl/rl/sequence_observations.py`
- Modify: `tests/rl/test_sequence_normalization.py`

**Interfaces:**
- Consumes: `SequencePolicyPlane.values`, `.available`, `.staleness`, `.steps`, and `.windows`.
- Produces: `SequencePolicyPlane.batch_components(decision_indices: np.ndarray) -> dict[str, np.ndarray]`.

- [ ] **Step 1: Write a failing batch-equivalence test**

Create a test that calls `batch_components(np.asarray([128, 129]))` and proves each returned sample equals `components(128)` and `components(129)`. Also reject empty, non-rank-one, and out-of-range index arrays.

- [ ] **Step 2: Run the focused sequence test and verify RED**

Run: `pytest tests/rl/test_sequence_normalization.py -q`

Expected: failure because `batch_components` does not exist.

- [ ] **Step 3: Implement vectorized gathers**

Validate a one-dimensional integer index array, construct `[batch, window]` source rows per timeframe, gather `[batch, native_time, symbol, feature]`, transpose to `[batch, symbol, native_time, feature]`, and return values/availability/staleness. Make scalar `components()` delegate to the batch implementation and strip the leading dimension.

- [ ] **Step 4: Run the focused sequence test and verify GREEN**

Run: `pytest tests/rl/test_sequence_normalization.py -q`

Expected: all tests pass.

### Task 3: Use the sequence plane in behavior cloning

**Files:**
- Modify: `trade_rl/learning/teacher_artifact.py`
- Modify: `trade_rl/integrations/sb3_training.py`
- Modify: `tests/learning/test_structured_teacher_artifact.py`

**Interfaces:**
- Consumes: `SequencePolicyPlane.batch_components`.
- Produces: `StructuredTeacherObservationProvider(..., policy_plane: SequencePolicyPlane | None = None)`.

- [ ] **Step 1: Write a failing provider test**

Construct a structured environment, pass its `sequence_policy_plane` to the provider, replace `sequence_builder.build` with a forbidden implementation, and prove `provider.get()` still reconstructs the exact direct policy observations.

- [ ] **Step 2: Run the focused teacher test and verify RED**

Run: `pytest tests/learning/test_structured_teacher_artifact.py -q`

Expected: failure because the provider does not accept or use `policy_plane`.

- [ ] **Step 3: Implement the fast provider path**

Type-check and identity-check the plane at construction. In `get()`, gather compact observation fields, call `policy_plane.batch_components(decision_indices)`, merge the result, and retain the existing builder loop as a compatibility fallback.

- [ ] **Step 4: Wire the environment plane into SB3 behavior cloning**

Pass `unwrapped_teacher.sequence_policy_plane` when constructing `StructuredTeacherObservationProvider`.

- [ ] **Step 5: Run the focused teacher tests and verify GREEN**

Run: `pytest tests/learning/test_structured_teacher_artifact.py tests/integrations/test_sb3_training.py -q`

Expected: all tests pass.

### Task 4: Cache one immutable teacher rollout per fold backend

**Files:**
- Modify: `trade_rl/integrations/sb3_training.py`
- Modify: `tests/integrations/test_sb3_training.py`

**Interfaces:**
- Produces: `StableBaselines3Backend._teacher_dataset(...) -> SupervisedPolicyDataset` with an identity-bound in-memory cache.

- [ ] **Step 1: Write a failing cache test**

Monkeypatch `collect_teacher_rollout`, call `_teacher_dataset` twice with the same dataset/range/environment/action/teacher identities, and assert one collection call and object identity. Change one identity and assert a new collection.

- [ ] **Step 2: Run the focused integration test and verify RED**

Run: `pytest tests/integrations/test_sb3_training.py -q`

Expected: failure because `_teacher_dataset` and its cache do not exist.

- [ ] **Step 3: Implement the cache**

Store read-only `SupervisedPolicyDataset` instances in a backend dictionary keyed by dataset ID, train start/stop, environment digest, action-spec digest, and teacher-config digest. Call this helper from `train()` instead of calling `collect_teacher_rollout` directly.

- [ ] **Step 4: Run the focused integration test and verify GREEN**

Run: `pytest tests/integrations/test_sb3_training.py -q`

Expected: all tests pass.

### Task 5: Verify and publish a clean product diff

**Files:**
- Delete: `.github/workflows/source-export-temp.yml`
- Modify only if required by checks: files touched in Tasks 1–4.

**Interfaces:**
- Produces: a branch containing only product code, tests, configuration, and this implementation plan.

- [ ] **Step 1: Remove the temporary source-export workflow**

Delete `.github/workflows/source-export-temp.yml`.

- [ ] **Step 2: Run focused tests**

Run:

```bash
pytest tests/examples/test_binance_multitimeframe_full_assets.py \
  tests/rl/test_sequence_normalization.py \
  tests/learning/test_structured_teacher_artifact.py \
  tests/integrations/test_sb3_training.py -q
```

Expected: all tests pass.

- [ ] **Step 3: Run static checks on changed Python files**

Run:

```bash
ruff check trade_rl/rl/sequence_observations.py \
  trade_rl/learning/teacher_artifact.py \
  trade_rl/integrations/sb3_training.py \
  tests/rl/test_sequence_normalization.py \
  tests/learning/test_structured_teacher_artifact.py \
  tests/integrations/test_sb3_training.py
ruff format --check trade_rl/rl/sequence_observations.py \
  trade_rl/learning/teacher_artifact.py \
  trade_rl/integrations/sb3_training.py \
  tests/rl/test_sequence_normalization.py \
  tests/learning/test_structured_teacher_artifact.py \
  tests/integrations/test_sb3_training.py
mypy trade_rl/rl/sequence_observations.py \
  trade_rl/learning/teacher_artifact.py \
  trade_rl/integrations/sb3_training.py
```

Expected: all checks pass.

- [ ] **Step 4: Review the final diff**

Confirm no temporary workflow, generated artifact, model checkpoint, or unrelated refactor remains.
