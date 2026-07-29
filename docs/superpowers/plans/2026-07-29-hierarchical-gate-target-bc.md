# Hierarchical Gate-Target BC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the collapse-prone single-MSE BC warm start with a hierarchical Gate + Target actor while preserving the continuous target-weight action contract through BC, PPO, checkpointing, export, and serving.

**Architecture:** The structured sequence policy receives an explicit current-weight vector, produces per-asset gate logits and bounded target proposals, composes them in action space, and converts the result back to squashed-Gaussian mean logits. BC trains gate, event-target, and composed-action losses separately; mandatory teacher-reconstruction and causal non-collapse gates fail closed.

**Tech Stack:** Python 3.12, PyTorch, Stable-Baselines3, Gymnasium, NumPy, pytest, Ruff, MyPy, Docker/CUDA.

## Global Constraints

- Keep `Box(-1, 1, (n_symbols,))` absolute target-weight actions unchanged.
- Keep dataset symbol ordering exact and fail closed on identity mismatch.
- Use one actor structure for BC, PPO, CostCriticPPO, LagrangianPPO, checkpoint reload, export, and serving.
- Do not delete, reorder, or post-select Oracle rows by realized outcome.
- Do not use a hard gate inside PPO action generation.
- Current effective weights must be explicit causal observation state, not inferred from previous actions.
- Bump configuration identity to `training_run_config_v3`; reject v2 instead of silently defaulting structural fields.
- Every task uses test-first development and ends in an independently reviewable commit.

---

## File Map

**Create**

- `trade_rl/learning/hierarchical_teacher_labels.py`: immutable gate/target/event labels and diagnostics.
- `trade_rl/learning/hierarchical_bc_metrics.py`: teacher reconstruction and collapse metrics.
- `tests/learning/test_hierarchical_teacher_labels.py`
- `tests/learning/test_hierarchical_bc_metrics.py`

**Modify**

- `trade_rl/rl/observations.py`: expose stable current-weight field identity.
- `trade_rl/rl/environment_observation_contract.py`: add current-weight metadata to structured observations.
- `trade_rl/rl/sequence_observations.py`: reconstruct current weights in structured batches.
- `trade_rl/integrations/compact_rollout_buffer.py`: retain per-step current weights.
- `trade_rl/rl/policies.py`: add Gate + Target head and policy output contract.
- `trade_rl/integrations/sb3_model_assembly.py`: select and digest the hierarchical head.
- `trade_rl/learning/behavior_cloning.py`: add hierarchical BC configuration/result fields.
- `trade_rl/integrations/behavior_cloning.py`: train and validate three BC losses.
- `trade_rl/integrations/sb3_training.py`: generate labels, enforce gates, and export evidence.
- `trade_rl/workflows/training_run.py`: parse v3 config and reject v2.
- `trade_rl/integrations/sb3_serving.py`: validate and load the new actor identity.
- `examples/binance-multitimeframe/walk-forward-full.json`: migrate full run to v3 and C actor.
- `docs/CONFIGURATION.md`, `docs/ARCHITECTURE.md`, `START.md`: document configuration and run order.

**Tests to extend**

- `tests/rl/test_sequence_policy_core.py`
- `tests/rl/test_environment_observation_contract.py`
- `tests/rl/test_sequence_observations.py`
- `tests/integrations/test_sb3_model_assembly.py`
- `tests/integrations/test_behavior_cloning.py`
- `tests/integrations/test_sb3_training.py`
- `tests/workflows/test_training_run.py`
- `tests/serving/test_sb3_loader.py`
- `tests/serving/test_observation_parity.py`
- `tests/architecture/test_architecture_audit_fixes.py`

---

### Task 1: Immutable Hierarchical Teacher Labels

**Files:**
- Create: `trade_rl/learning/hierarchical_teacher_labels.py`
- Create: `tests/learning/test_hierarchical_teacher_labels.py`
- Modify: `trade_rl/learning/teacher_artifact.py`

**Interfaces:**
- Produces: `HierarchicalTeacherLabels`, `TeacherActionEvent`, and `build_hierarchical_teacher_labels(...)`.
- Consumes: chronological teacher targets, explicit current weights, active masks, source artifact digest, and a positive finite change threshold.

- [ ] **Step 1: Write failing construction and event-classification tests**

```python
import numpy as np

from trade_rl.learning.hierarchical_teacher_labels import (
    TeacherActionEvent,
    build_hierarchical_teacher_labels,
)


def test_labels_classify_enter_exit_reverse_without_reordering() -> None:
    current = np.array([[0.0], [0.4], [0.4], [-0.3]], dtype=np.float32)
    target = np.array([[0.4], [0.0], [-0.3], [-0.3]], dtype=np.float32)
    active = np.ones_like(target, dtype=np.bool_)

    labels = build_hierarchical_teacher_labels(
        teacher_targets=target,
        current_weights=current,
        active_mask=active,
        change_threshold=0.01,
        source_teacher_digest="a" * 64,
    )

    assert labels.gate_labels[:, 0].tolist() == [True, True, True, False]
    assert labels.events[:, 0].tolist() == [
        TeacherActionEvent.ENTER,
        TeacherActionEvent.EXIT,
        TeacherActionEvent.REVERSE,
        TeacherActionEvent.HOLD,
    ]
    np.testing.assert_array_equal(labels.target_actions, target)
```

- [ ] **Step 2: Run the test and verify import failure**

Run: `pytest tests/learning/test_hierarchical_teacher_labels.py -q`

Expected: collection fails because `hierarchical_teacher_labels` does not exist.

- [ ] **Step 3: Implement validated immutable labels**

```python
class TeacherActionEvent(IntEnum):
    HOLD = 0
    ENTER = 1
    RESIZE = 2
    EXIT = 3
    REVERSE = 4


@dataclass(frozen=True, slots=True)
class HierarchicalTeacherLabels:
    gate_labels: np.ndarray
    current_weights: np.ndarray
    target_actions: np.ndarray
    active_mask: np.ndarray
    events: np.ndarray
    source_teacher_digest: str
    label_config_digest: str
```

`build_hierarchical_teacher_labels` must validate equal two-dimensional shapes, finite values, `[-1, 1]` bounds, chronological row preservation, positive threshold, and SHA-256 source identity. Event rules are:

```python
changed = active & (np.abs(target - current) >= threshold)
enter = changed & (np.abs(current) < threshold) & (np.abs(target) >= threshold)
exit_ = changed & (np.abs(current) >= threshold) & (np.abs(target) < threshold)
reverse = changed & (current * target < 0.0)
resize = changed & ~(enter | exit_ | reverse)
```

All arrays are copied, marked read-only, and included in `label_config_digest`.

- [ ] **Step 4: Add validation tests**

Cover inactive assets, non-finite values, shape mismatch, target outside bounds, threshold zero, and source digest mismatch.

- [ ] **Step 5: Run focused tests**

Run: `pytest tests/learning/test_hierarchical_teacher_labels.py tests/learning/test_structured_teacher_artifact.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add trade_rl/learning/hierarchical_teacher_labels.py trade_rl/learning/teacher_artifact.py tests/learning/test_hierarchical_teacher_labels.py
git commit -m "feat: add hierarchical BC teacher labels"
```

---

### Task 2: Explicit Current-Weight Observation Contract

**Files:**
- Modify: `trade_rl/rl/observations.py`
- Modify: `trade_rl/rl/environment_observation_contract.py`
- Modify: `trade_rl/rl/sequence_observations.py`
- Modify: `tests/rl/test_environment_observation_contract.py`
- Modify: `tests/rl/test_sequence_observations.py`
- Modify: `tests/serving/test_observation_parity.py`

**Interfaces:**
- Produces: structured observation key `current_weights`, shape `(n_symbols,)`, dtype `float32`.
- Produces metadata: `current_weight_source="effective_book_weights"`.
- Consumes: effective `BookState.weights` at the decision instant.

- [ ] **Step 1: Write a failing observation-contract test**

```python
def test_sequence_observation_exposes_effective_current_weights(env) -> None:
    observation, _ = env.reset()
    assert observation["current_weights"].shape == (env.dataset.n_symbols,)
    np.testing.assert_allclose(
        observation["current_weights"],
        env.unwrapped.book.weights,
        atol=1e-7,
    )
    assert env.unwrapped.sequence_layout_metadata["current_weight_source"] == (
        "effective_book_weights"
    )
```

- [ ] **Step 2: Run focused tests and verify missing-key failure**

Run: `pytest tests/rl/test_environment_observation_contract.py tests/rl/test_sequence_observations.py -q`

Expected: FAIL because `current_weights` is absent.

- [ ] **Step 3: Add the explicit structured field**

Add to sequence space:

```python
"current_weights": spaces.Box(
    low=-1.0,
    high=1.0,
    shape=(self.dataset.n_symbols,),
    dtype=np.float32,
)
```

Populate it from the effective book used to build the same observation. Do not use `previous_action` or `pending_target`.

- [ ] **Step 4: Bind field identity into schemas and digests**

Increment the structured observation schema version, include `current_weight_source` and field shape in the sequence layout digest, and ensure normalizers treat this field as passthrough.

- [ ] **Step 5: Add parity and mutation-isolation tests**

Verify reconstructed and serving observations contain identical weights, returned arrays do not alias mutable book arrays, and partial-fill state differs from the previously submitted target when expected.

- [ ] **Step 6: Run focused tests**

Run: `pytest tests/rl/test_environment_observation_contract.py tests/rl/test_sequence_observations.py tests/serving/test_observation_parity.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add trade_rl/rl/observations.py trade_rl/rl/environment_observation_contract.py trade_rl/rl/sequence_observations.py tests/rl/test_environment_observation_contract.py tests/rl/test_sequence_observations.py tests/serving/test_observation_parity.py
git commit -m "feat: expose current weights in structured observations"
```

---

### Task 3: Preserve Current Weights in Compact PPO Rollouts

**Files:**
- Modify: `trade_rl/integrations/compact_rollout_buffer.py`
- Modify: `trade_rl/rl/rollout_memory.py`
- Modify: `tests/integrations/test_compact_rollout_buffer.py`
- Modify: `tests/rl/test_rollout_memory.py`

**Interfaces:**
- Produces: per-step compact state sufficient to reconstruct `current_weights` exactly.
- Consumes: structured observation batches from Task 2.

- [ ] **Step 1: Write a failing round-trip test**

```python
def test_compact_rollout_round_trip_preserves_current_weights(buffer_fixture) -> None:
    expected = buffer_fixture.observation["current_weights"].copy()
    buffer_fixture.add_current_observation()
    reconstructed = buffer_fixture.sample_first_observation()
    np.testing.assert_array_equal(reconstructed["current_weights"], expected)
```

- [ ] **Step 2: Run focused tests**

Run: `pytest tests/integrations/test_compact_rollout_buffer.py tests/rl/test_rollout_memory.py -q`

Expected: FAIL because the stateful field is not stored.

- [ ] **Step 3: Add minimal stateful storage**

Store a float32 tensor of shape `(buffer_size, n_envs, n_symbols)` for current weights. Include its exact bytes in rollout-memory estimation:

```python
current_weight_bytes = n_steps * n_envs * n_symbols * np.dtype(np.float32).itemsize
```

Do not materialize market sequence arrays per step.

- [ ] **Step 4: Add fail-closed shape and dtype checks**

Reject missing key, non-float data, non-finite values, wrong symbol count, or reconstruction without initialized state.

- [ ] **Step 5: Run focused tests**

Run: `pytest tests/integrations/test_compact_rollout_buffer.py tests/rl/test_rollout_memory.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add trade_rl/integrations/compact_rollout_buffer.py trade_rl/rl/rollout_memory.py tests/integrations/test_compact_rollout_buffer.py tests/rl/test_rollout_memory.py
git commit -m "feat: preserve current weights in compact rollouts"
```

---

### Task 4: Hierarchical Gate-Target Actor Head

**Files:**
- Modify: `trade_rl/rl/policies.py`
- Modify: `tests/rl/test_sequence_policy_core.py`

**Interfaces:**
- Produces: `HierarchicalActorOutputs` and `SharedPerAssetGateTargetHead`.
- Policy method: `hierarchical_actor_outputs(observations) -> HierarchicalActorOutputs`.
- Final `action_net` output remains the pre-tanh mean consumed by `MaskedSharedSquashedDiagGaussianDistribution`.

- [ ] **Step 1: Write mathematical composition tests**

```python
def test_gate_target_composition_matches_distribution_mode() -> None:
    head = SharedPerAssetGateTargetHead(
        n_symbols=2,
        context_dim=5,
        hidden_dims=(8,),
        temperature=1.0,
    )
    outputs = head.compose(
        gate_logits=torch.tensor([[0.0, 20.0]]),
        target_logits=torch.tensor([[0.0, -0.5]]),
        current_weights=torch.tensor([[0.4, 0.2]]),
        active_mask=torch.tensor([[True, True]]),
    )
    expected = torch.tensor(
        [[0.2, torch.tanh(torch.tensor(-0.5)).item()]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(outputs.composed_actions, expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        torch.tanh(outputs.mean_logits), outputs.composed_actions, atol=1e-6, rtol=1e-6
    )
```

Also test gate near zero preserves current weight, inactive assets output zero, gradients reach both heads, and values near ±1 remain finite after `atanh` clamping.

- [ ] **Step 2: Run the policy tests and verify missing-class failure**

Run: `pytest tests/rl/test_sequence_policy_core.py -q`

Expected: FAIL because the hierarchical head does not exist.

- [ ] **Step 3: Implement output contract and shared head**

```python
@dataclass(frozen=True, slots=True)
class HierarchicalActorOutputs:
    gate_logits: torch.Tensor
    gate_probabilities: torch.Tensor
    target_actions: torch.Tensor
    composed_actions: torch.Tensor
    mean_logits: torch.Tensor
    current_weights: torch.Tensor
    active_mask: torch.Tensor
```

The shared network has one common trunk and two scalar heads per asset. `compose` uses sigmoid temperature, tanh proposal, interpolation from current weight, clamp with `eps=1e-6`, and `torch.atanh`.

- [ ] **Step 4: Extend feature and actor latent layouts**

Append raw `current_weights` after active flags in `SequenceAssetFeatureExtractor`. Update `features_dim`, `SharedAssetActorCriticExtractor._parts`, `actor_context_dim`, and actor contexts to include one current-weight scalar. Keep the critic input unchanged.

- [ ] **Step 5: Integrate into `SharedPerAssetActorCriticPolicy`**

Replace `SharedPerAssetActionHead` only when constructor field `shared_actor_head="hierarchical_gate_target_v1"` is selected. Save constructor parameters including head identity and temperature. `_get_action_dist_from_latent` passes `mean_logits`; deterministic mode therefore equals `composed_actions`.

- [ ] **Step 6: Run focused tests**

Run: `pytest tests/rl/test_sequence_policy_core.py tests/integrations/test_sb3_model_assembly.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add trade_rl/rl/policies.py tests/rl/test_sequence_policy_core.py
git commit -m "feat: add hierarchical gate-target actor"
```

---

### Task 5: Hierarchical BC Loss and Metrics

**Files:**
- Create: `trade_rl/learning/hierarchical_bc_metrics.py`
- Create: `tests/learning/test_hierarchical_bc_metrics.py`
- Modify: `trade_rl/learning/behavior_cloning.py`
- Modify: `trade_rl/integrations/behavior_cloning.py`
- Modify: `tests/integrations/test_behavior_cloning.py`

**Interfaces:**
- Produces: `HierarchicalBehaviorCloningMetrics`.
- Extends `BehaviorCloningResult` with initial/final component losses, validation metrics, event support, and collapse flags.
- Consumes `HierarchicalTeacherLabels` and policy `hierarchical_actor_outputs`.

- [ ] **Step 1: Write failing metric tests**

```python
def test_all_hold_prediction_is_reported_as_collapse() -> None:
    metrics = hierarchical_bc_metrics(
        gate_probabilities=np.zeros((8, 2), dtype=np.float32),
        proposal_actions=np.zeros((8, 2), dtype=np.float32),
        composed_actions=np.zeros((8, 2), dtype=np.float32),
        labels=labels_with_positive_events(),
        gate_threshold=0.5,
    )
    assert metrics.all_hold_collapse is True
    assert metrics.gate_recall == 0.0
    assert metrics.positive_support > 0
```

- [ ] **Step 2: Run tests and verify failure**

Run: `pytest tests/learning/test_hierarchical_bc_metrics.py tests/integrations/test_behavior_cloning.py -q`

Expected: FAIL because metrics and hierarchical training are absent.

- [ ] **Step 3: Implement numerically stable component losses**

```python
gate_loss = F.binary_cross_entropy_with_logits(
    outputs.gate_logits[active],
    gate_labels[active].float(),
    pos_weight=positive_weight,
)

target_loss = F.smooth_l1_loss(
    outputs.target_actions[event_mask],
    teacher_targets[event_mask],
)

composed_loss = F.smooth_l1_loss(
    outputs.composed_actions[active],
    teacher_targets[active],
)
```

When `event_mask` has zero support, `target_loss` is an exact differentiable zero and metrics record insufficient support. Compute positive class weight from training indices only and clamp to `[1.0, config.max_positive_class_weight]`.

- [ ] **Step 4: Add deterministic chronological validation**

Use the existing chronological split. Early stopping score is the configured weighted sum of validation component losses. Restore the complete policy state at the best epoch.

- [ ] **Step 5: Add metric calculations**

Calculate precision, recall, F1, positive support, predicted-positive support, active-event target RMSE, composed RMSE, teacher and policy activity rates, activity ratio, constant-action collapse, all-hold collapse, and all-trade collapse.

- [ ] **Step 6: Verify the old failure mode is caught**

Add a synthetic hold-dominated dataset where plain MSE improves with zero predicted events. Assert the hierarchical result reports collapse and fails the configured reconstruction gate.

- [ ] **Step 7: Run focused tests**

Run: `pytest tests/learning/test_hierarchical_bc_metrics.py tests/integrations/test_behavior_cloning.py -q`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add trade_rl/learning/hierarchical_bc_metrics.py trade_rl/learning/behavior_cloning.py trade_rl/integrations/behavior_cloning.py tests/learning/test_hierarchical_bc_metrics.py tests/integrations/test_behavior_cloning.py
git commit -m "feat: train hierarchical BC objectives"
```

---

### Task 6: Mandatory Reconstruction and Causal Non-Collapse Gates

**Files:**
- Modify: `trade_rl/integrations/sb3_training.py`
- Modify: `trade_rl/learning/evaluation.py`
- Modify: `tests/integrations/test_sb3_training.py`

**Interfaces:**
- Produces structured gate payloads with `required=true`, support, threshold, observed value, and reason.
- Consumes Task 5 reconstruction metrics plus existing causal holdout path metrics.

- [ ] **Step 1: Write failing zero-trade rejection test**

```python
def test_required_bc_gate_rejects_zero_trade_causal_holdout(
    configured_training_case,
) -> None:
    configured_training_case.teacher_positive_support = 12
    configured_training_case.causal_holdout_trade_count = 0
    with pytest.raises(RuntimeError, match="zero-trade collapse"):
        configured_training_case.run_behavior_cloning_gate()
```

- [ ] **Step 2: Run focused test**

Run: `pytest tests/integrations/test_sb3_training.py -q`

Expected: FAIL because holdout reproduction is currently diagnostic-only.

- [ ] **Step 3: Implement separate gate groups**

`teacher_reconstruction_gate` requires configured composed-loss improvement, gate precision, gate recall, active target RMSE, activity-ratio bounds, and no collapse.

`causal_non_collapse_gate` requires minimum executed trades when teacher support is nonzero, non-constant submitted actions, and after-cost regret no worse than the explicit catastrophic threshold. It does not require 80% hindsight-Oracle agreement.

- [ ] **Step 4: Make insufficient support explicit**

A metric with support below its configured minimum receives status `insufficient_support`; the aggregate gate fails unless the configuration explicitly disables that metric. Omitted v3 fields are configuration errors, not implicit disables.

- [ ] **Step 5: Export explainable collapse evidence**

Include gate-positive rate, proposal-distance rate, downstream no-trade suppression, execution rejection, inactive-mask rate, submitted changes, executed changes, and trade count in JSON and Markdown reports.

- [ ] **Step 6: Run focused tests**

Run: `pytest tests/integrations/test_sb3_training.py tests/learning/test_evaluation.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add trade_rl/integrations/sb3_training.py trade_rl/learning/evaluation.py tests/integrations/test_sb3_training.py
git commit -m "feat: enforce BC reconstruction and non-collapse gates"
```

---

### Task 7: Configuration v3 and Architecture Identity

**Files:**
- Modify: `trade_rl/rl/training.py`
- Modify: `trade_rl/workflows/training_run.py`
- Modify: `trade_rl/integrations/sb3_model_assembly.py`
- Modify: `tests/workflows/test_training_run.py`
- Modify: `tests/integrations/test_sb3_model_assembly.py`
- Modify: `examples/binance-multitimeframe/walk-forward-full.json`

**Interfaces:**
- Produces required `training_run_config_v3` fields from the design spec.
- Produces architecture digest containing actor-head identity, temperature, current-weight observation schema, and all structural widths.

- [ ] **Step 1: Write failing schema tests**

```python
def test_v2_training_config_is_rejected_with_migration_message(tmp_path) -> None:
    payload = valid_training_payload(schema_version="training_run_config_v2")
    with pytest.raises(ValueError, match="training_run_config_v3"):
        load_training_run_config(write_json(tmp_path, payload))


def test_v3_requires_explicit_actor_head(tmp_path) -> None:
    payload = valid_training_payload(schema_version="training_run_config_v3")
    del payload["policy_actor_head"]
    with pytest.raises(ValueError, match="policy_actor_head"):
        load_training_run_config(write_json(tmp_path, payload))
```

- [ ] **Step 2: Run focused tests**

Run: `pytest tests/workflows/test_training_run.py tests/integrations/test_sb3_model_assembly.py -q`

Expected: FAIL because v3 is unknown.

- [ ] **Step 3: Add validated configuration fields**

Require exact enum `hierarchical_gate_target_v1`, temperature `> 0`, non-negative loss weights with positive total, change threshold in `(0, 1]`, positive class cap `>= 1`, precision/recall in `[0, 1]`, RMSE `>= 0`, activity bounds with `0 <= min <= max`, non-negative trade count, and non-negative catastrophic regret.

- [ ] **Step 4: Bind assembly and digest**

Pass:

```python
"shared_actor_head": config.policy_actor_head,
"shared_actor_gate_temperature": config.hierarchical_gate_temperature,
```

into `SharedPerAssetActorCriticPolicy`. Include both in architecture payload and checkpoint manifest.

- [ ] **Step 5: Migrate full research configuration**

Set schema v3, hierarchical actor head, explicit loss weights, and explicit gate thresholds. Keep BC epochs at 15 and the current teacher period unchanged for the first verification run.

- [ ] **Step 6: Run focused tests**

Run: `pytest tests/workflows/test_training_run.py tests/integrations/test_sb3_model_assembly.py tests/workflows/test_signal_digest.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add trade_rl/rl/training.py trade_rl/workflows/training_run.py trade_rl/integrations/sb3_model_assembly.py examples/binance-multitimeframe/walk-forward-full.json tests/workflows/test_training_run.py tests/integrations/test_sb3_model_assembly.py tests/workflows/test_signal_digest.py
git commit -m "feat: bind hierarchical actor to training config v3"
```

---

### Task 8: Checkpoint, Export, and Serving Parity

**Files:**
- Modify: `trade_rl/integrations/sb3_serving.py`
- Modify: `trade_rl/serving/package.py`
- Modify: `tests/serving/test_sb3_loader.py`
- Modify: `tests/serving/test_observation_snapshot_fail_closed.py`
- Modify: `tests/serving/test_observation_parity.py`

**Interfaces:**
- Consumes architecture and observation digests from Task 7.
- Produces identical deterministic target-weight output after save/load and package load.

- [ ] **Step 1: Write failing save/load parity test**

```python
def test_hierarchical_actor_checkpoint_round_trip_preserves_outputs(model, obs, tmp_path) -> None:
    before = model.policy.hierarchical_actor_outputs(obs)
    path = tmp_path / "model.zip"
    model.save(path)
    loaded = load_canonical_sb3_model(path)
    after = loaded.policy.hierarchical_actor_outputs(obs)
    torch.testing.assert_close(after.gate_logits, before.gate_logits)
    torch.testing.assert_close(after.target_actions, before.target_actions)
    torch.testing.assert_close(after.composed_actions, before.composed_actions)
```

- [ ] **Step 2: Run serving tests**

Run: `pytest tests/serving/test_sb3_loader.py tests/serving/test_observation_snapshot_fail_closed.py tests/serving/test_observation_parity.py -q`

Expected: FAIL until constructor and manifest fields are supported.

- [ ] **Step 3: Validate actor and observation identities before loading weights**

Reject missing or mismatched actor head, temperature, current-weight schema, action names, symbol order, architecture digest, and observation contract digest.

- [ ] **Step 4: Add structured output parity diagnostics**

For test/debug export only, expose gate logits, gate probabilities, target proposals, current weights, composed actions, and final deterministic action. Production output remains only the continuous target-weight vector.

- [ ] **Step 5: Run serving tests**

Run: `pytest tests/serving/test_sb3_loader.py tests/serving/test_observation_snapshot_fail_closed.py tests/serving/test_observation_parity.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add trade_rl/integrations/sb3_serving.py trade_rl/serving/package.py tests/serving/test_sb3_loader.py tests/serving/test_observation_snapshot_fail_closed.py tests/serving/test_observation_parity.py
git commit -m "feat: serve hierarchical actor with fail-closed parity"
```

---

### Task 9: Documentation, Architecture Tests, and Full Verification

**Files:**
- Modify: `docs/CONFIGURATION.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `START.md`
- Modify: `tests/architecture/test_architecture_audit_fixes.py`

**Interfaces:**
- Produces final operator guidance and architecture boundary enforcement.

- [ ] **Step 1: Add architecture tests before documentation edits**

Assert learning-domain label/metric modules do not import SB3, integrations depend inward on learning contracts, serving does not import BC trainers, and policy modules do not import workflow configuration.

- [ ] **Step 2: Run architecture tests**

Run: `pytest tests/architecture/test_architecture_audit_fixes.py -q`

Expected: PASS after correcting any forbidden imports.

- [ ] **Step 3: Document the C actor and run sequence**

Document Gate + Target composition, why current weights are explicit, distinction between teacher reconstruction and causal holdout, every v3 field, migration failure behavior, and the rule that epochs/data range are unchanged until BC gates pass.

- [ ] **Step 4: Run static and full CPU verification**

Run:

```bash
ruff check .
mypy trade_rl
pytest -q
```

Expected: all commands exit 0.

- [ ] **Step 5: Run CPU BC-to-PPO smoke**

Run the repository's maintained short training command with v3 config and assert:

- BC reports all three losses;
- reconstruction gate passes on the synthetic/short fixture;
- causal holdout is not zero-trade when teacher support exists;
- PPO starts from the same architecture digest;
- checkpoint reload reproduces deterministic action.

- [ ] **Step 6: Run CUDA smoke**

Run the maintained Docker CUDA smoke with the hierarchical sequence model. Confirm CUDA device use, finite gate/target/composed losses, checkpoint save/load, and no rollout-memory-limit regression.

- [ ] **Step 7: Commit**

```bash
git add docs/CONFIGURATION.md docs/ARCHITECTURE.md START.md tests/architecture/test_architecture_audit_fixes.py
git commit -m "docs: document hierarchical BC actor and gates"
```

- [ ] **Step 8: Open implementation PR**

PR title:

```text
feat: prevent BC collapse with hierarchical gate-target actor
```

PR body must include exact Ruff, MyPy, pytest, CPU smoke, and CUDA smoke outputs; before/after BC metrics; architecture digest evidence; and explicit confirmation that the environment action contract is unchanged.

---

## Self-Review

- Spec coverage: actor structure, explicit current state, BC objective, two evaluation classes, mandatory gates, v3 identity, rollout parity, serving parity, documentation, and full verification each have a task.
- Placeholder scan: no deferred implementation instructions or unspecified error handling remain.
- Type consistency: `HierarchicalTeacherLabels`, `HierarchicalActorOutputs`, `HierarchicalBehaviorCloningMetrics`, `hierarchical_actor_outputs`, and `hierarchical_gate_target_v1` are used consistently across tasks.
- Scope control: Oracle optimality auditing is intentionally separate; this plan changes how Oracle trajectories are labelled and imitated, not the Oracle optimizer itself.
