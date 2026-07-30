# Action-Head Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `hierarchical_gate_target_v1` and `shared_target_v1` controlled, identity-bound, telemetry-comparable sequence-policy candidates and provide a canonical paired walk-forward experiment.

**Architecture:** Keep the external target-weight and SB3 distribution contracts unchanged. Generalize sequence actor selection, expose one head-independent action-stage API, upgrade policy identity to v4 with head-specific coupling, and make Oracle causal holdout mandatory for both hierarchical and direct BC. Add paired run files whose complete payloads differ only by actor head.

**Tech Stack:** Python 3.12, PyTorch, Stable-Baselines3, Gymnasium, NumPy, pytest, Ruff, MyPy, import-linter, JSON workflow profiles.

## Global Constraints

- The external action contract remains ordered continuous `target_weight:<symbol>` values.
- `hierarchical_gate_target_v1` remains the default sequence actor head.
- Both heads use `MaskedSharedSquashedDiagGaussianDistribution`, one shared scalar `log_std`, tanh squashing, and `use_sde=false`.
- No reward, risk, no-trade-band, execution, liquidation, PPO objective, observation encoder, or exchange-routing behavior changes.
- `hierarchical_gate_temperature` must equal `1.0` when `shared_target_v1` is selected.
- Oracle BC causal non-collapse evidence is mandatory for both actor heads.
- Direct exchange execution remains NO-GO.

---

## File Structure

- Modify `trade_rl/rl/training.py`: validate the two supported sequence actor heads and inactive direct-head gate temperature.
- Modify `trade_rl/rl/policies.py`: add `ActionStageOutputs`, expose current weights from the direct head, and provide a common policy action-stage method.
- Modify `trade_rl/rl/policy_identity.py`: implement `sb3_policy_identity_v4` for both actor heads.
- Modify `trade_rl/rl/tensorboard_logging.py`: consume the common action-stage method for comparable metrics.
- Modify `trade_rl/learning/evaluation.py`: factor causal non-collapse metrics and add direct reconstruction gate evaluation.
- Modify `trade_rl/integrations/sb3_training.py`: derive teacher-change support for both heads and enforce the direct Oracle BC gate.
- Modify `trade_rl/integrations/sb3_model_assembly.py`: preserve exact head and active/inactive gate-temperature identity inputs.
- Modify structured export and serving fixtures that serialize policy identity v3.
- Create `examples/binance-multitimeframe/training-action-head-ablation-gate.json`.
- Create `examples/binance-multitimeframe/training-action-head-ablation-direct.json`.
- Create `examples/binance-multitimeframe/walk-forward-action-head-ablation.json`.
- Create `tests/examples/test_action_head_ablation_profiles.py`.
- Add focused tests in the existing training, policy, identity, BC, telemetry, export, and serving test modules.

---

### Task 1: Lock the Configuration and Profile Contract with Failing Tests

**Files:**
- Modify: `tests/workflows/test_training_run_config.py`
- Create: `tests/examples/test_action_head_ablation_profiles.py`

**Interfaces:**
- Consumes: `TrainingRunConfig.from_mapping(raw: object) -> TrainingRunConfig`.
- Produces: the required accepted head set and paired-profile equality rule used by later implementation tasks.

- [ ] **Step 1: Add a failing sequence-head acceptance test**

```python
@pytest.mark.parametrize(
    "actor_head",
    ("hierarchical_gate_target_v1", "shared_target_v1"),
)
def test_sequence_training_accepts_supported_action_ablation_heads(actor_head: str) -> None:
    raw = _mapping()
    raw["action"] = {"alpha_enabled": False, "n_factors": 0}
    raw["training"] = {
        **raw["training"],
        "policy": "MultiInputPolicy",
        "observation_encoder": "hierarchical_sequence_v2",
        "policy_actor_head": actor_head,
    }
    raw["environment"] = {
        **raw["environment"],
        "structured_sequence_observation": True,
        "sequence_windows": [["15m", 1], ["1h", 1], ["4h", 1], ["1d", 1]],
    }

    config = TrainingRunConfig.from_mapping(raw)

    assert config.training.policy_actor_head == actor_head
```

- [ ] **Step 2: Add fail-closed tests for unsupported heads and direct gate temperature**

```python
def test_sequence_training_rejects_unknown_action_ablation_head() -> None:
    raw = _sequence_mapping()
    raw["training"]["policy_actor_head"] = "discrete_buy_sell_hold_v1"
    with pytest.raises(ValueError, match="policy_actor_head"):
        TrainingRunConfig.from_mapping(raw)


def test_direct_sequence_head_rejects_active_gate_temperature() -> None:
    raw = _sequence_mapping(actor_head="shared_target_v1")
    raw["training"]["hierarchical_gate_temperature"] = 0.5
    with pytest.raises(ValueError, match="inactive.*shared_target_v1"):
        TrainingRunConfig.from_mapping(raw)
```

- [ ] **Step 3: Add the paired-profile integrity test before the files exist**

```python
def test_action_head_ablation_training_profiles_differ_only_by_actor_head() -> None:
    root = Path(__file__).parents[2] / "examples" / "binance-multitimeframe"
    gate = json.loads((root / "training-action-head-ablation-gate.json").read_text())
    direct = json.loads((root / "training-action-head-ablation-direct.json").read_text())
    assert gate["training"].pop("policy_actor_head") == "hierarchical_gate_target_v1"
    assert direct["training"].pop("policy_actor_head") == "shared_target_v1"
    assert gate == direct
```

Add a second test that loads `walk-forward-action-head-ablation.json`, requires exactly two candidates, and resolves their `run_file` values to the two training files.

- [ ] **Step 4: Run the focused tests and verify RED**

Run:

```bash
pytest -q \
  tests/workflows/test_training_run_config.py::test_sequence_training_accepts_supported_action_ablation_heads \
  tests/workflows/test_training_run_config.py::test_direct_sequence_head_rejects_active_gate_temperature \
  tests/examples/test_action_head_ablation_profiles.py
```

Expected: `shared_target_v1` is rejected and the three new JSON files are missing.

- [ ] **Step 5: Commit the failing tests**

```bash
git add tests/workflows/test_training_run_config.py tests/examples/test_action_head_ablation_profiles.py
git commit -m "test: define action-head ablation contract"
```

---

### Task 2: Implement Sequence Actor-Head Configuration

**Files:**
- Modify: `trade_rl/rl/training.py`
- Test: `tests/workflows/test_training_run_config.py`

**Interfaces:**
- Consumes: `ResidualTrainingConfig.policy_actor_head: str | None` and `hierarchical_gate_temperature: float`.
- Produces: normalized `policy_actor_head` in `{hierarchical_gate_target_v1, shared_target_v1}` for sequence policies.

- [ ] **Step 1: Replace the single expected sequence head with an explicit supported set**

```python
_SEQUENCE_ACTOR_HEADS = frozenset(
    {"hierarchical_gate_target_v1", "shared_target_v1"}
)
```

Default `None` to `hierarchical_gate_target_v1`. For sequence policies, reject values outside `_SEQUENCE_ACTOR_HEADS`. For non-sequence policies, continue to require `standard_continuous_v1`.

- [ ] **Step 2: Enforce inactive temperature for the direct head**

```python
if actor_head == "shared_target_v1" and self.hierarchical_gate_temperature != 1.0:
    raise ValueError(
        "hierarchical_gate_temperature is inactive for policy_actor_head=shared_target_v1"
    )
```

Keep positive/finite validation for the hierarchical head.

- [ ] **Step 3: Run the focused configuration tests**

```bash
pytest -q tests/workflows/test_training_run_config.py
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add trade_rl/rl/training.py tests/workflows/test_training_run_config.py
git commit -m "feat: allow direct sequence action head"
```

---

### Task 3: Add a Common Action-Stage Policy API

**Files:**
- Modify: `trade_rl/rl/policies.py`
- Modify: `tests/rl/test_sequence_policy_core.py`

**Interfaces:**
- Produces: `ActionStageOutputs` and `SharedPerAssetActorCriticPolicy.action_stage_outputs(observations)`.
- Preserves: `hierarchical_actor_outputs(observations) -> HierarchicalActorOutputs`.

- [ ] **Step 1: Write failing tests for both actor heads**

Construct minimal policy-head latents with known current weights and active masks. Require:

```python
outputs = policy.action_stage_outputs(observations)
assert outputs.current_weights.shape == outputs.deterministic_actions.shape
assert outputs.active_mask.dtype == torch.bool
assert torch.count_nonzero(outputs.deterministic_actions[~outputs.active_mask]) == 0
```

For `hierarchical_gate_target_v1`, require non-null `change_intensity`. For `shared_target_v1`, require `change_intensity is None`.

- [ ] **Step 2: Run the focused test and verify RED**

```bash
pytest -q tests/rl/test_sequence_policy_core.py -k action_stage_outputs
```

Expected: FAIL because `ActionStageOutputs` or `action_stage_outputs` does not exist.

- [ ] **Step 3: Add the immutable output type**

```python
@dataclass(frozen=True, slots=True)
class ActionStageOutputs:
    current_weights: torch.Tensor
    deterministic_actions: torch.Tensor
    active_mask: torch.Tensor
    change_intensity: torch.Tensor | None
```

- [ ] **Step 4: Add `current_weights()` to `SharedPerAssetActionHead`**

The direct head uses the same context layout as the hierarchical head. Return context column `-2`, masked by column `-1`.

- [ ] **Step 5: Implement `action_stage_outputs()` once in the policy**

Extract features and actor latent once. For a hierarchical head, reuse `SharedPerAssetGateTargetHead.outputs()` and return composed actions. For a direct head, use `torch.tanh(self.action_net(latent_pi))`, `active_mask()`, and `current_weights()`.

Refactor `deterministic_actions()` to return `action_stage_outputs(...).deterministic_actions`.

- [ ] **Step 6: Run the policy tests**

```bash
pytest -q tests/rl/test_sequence_policy_core.py tests/integrations/test_sb3_model_assembly.py
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add trade_rl/rl/policies.py tests/rl/test_sequence_policy_core.py
git commit -m "feat: expose common action-stage outputs"
```

---

### Task 4: Upgrade Policy Identity to v4

**Files:**
- Modify: `trade_rl/rl/policy_identity.py`
- Replace or modify: `tests/integrations/test_sb3_policy_identity_v3.py`
- Modify: `tests/rl/test_structured_export.py`
- Modify: `tests/serving/test_sb3_loader.py`
- Modify: `tests/serving/test_structured_policy_loader.py`

**Interfaces:**
- Produces: `SB3_POLICY_IDENTITY_SCHEMA = "sb3_policy_identity_v4"`.
- Accepts sequence heads: `hierarchical_gate_target_v1`, `shared_target_v1`.
- Rejects serialized v2 identities and reads existing hierarchical v3 identities only for migration compatibility.

- [ ] **Step 1: Add failing identity tests for direct and hierarchical heads**

Require both payloads to include the exact actor head and current-weight identity. Require head-specific coupling:

```python
assert hierarchical["exploration_contract"]["mean_coupling"] == (
    "post_composition_gate_independent_v1"
)
assert direct["exploration_contract"]["mean_coupling"] == "direct_target_mean_v1"
assert hierarchical["gate_temperature"] == 1.0
assert direct["gate_temperature"] is None
```

Require cross-head validation to raise `ValueError("SB3 policy architecture identity mismatch")` and v3 to raise a migration error.

- [ ] **Step 2: Run identity and serving/export fixture tests and verify RED**

```bash
pytest -q \
  tests/integrations/test_sb3_policy_identity_v3.py \
  tests/rl/test_structured_export.py \
  tests/serving/test_sb3_loader.py \
  tests/serving/test_structured_policy_loader.py
```

Expected: FAIL on schema v4 and direct-head identity requirements.

- [ ] **Step 3: Implement the v4 constants and validation**

Use:

```python
SB3_POLICY_IDENTITY_SCHEMA = "sb3_policy_identity_v4"
LEGACY_SB3_POLICY_IDENTITY_SCHEMAS = frozenset(
    {"sb3_policy_identity_v2", "sb3_policy_identity_v3"}
)
POLICY_ARCHITECTURE_SCHEMA = "shared_target_weight_policy_v3"
SUPPORTED_SEQUENCE_ACTOR_HEADS = frozenset(
    {"hierarchical_gate_target_v1", "shared_target_v1"}
)
```

Build the exploration payload from the actor head. Keep distribution, scalar `log_std`, gSDE, and tanh checks common. Validate `gate_temperature` as positive only for the hierarchical head and exactly `None` for the direct head.

- [ ] **Step 4: Bind actual model and assembly head exactly**

Reject when `policy.shared_actor_head != assembly.policy_actor_head`. Read the actual gate temperature only for the hierarchical head. Include it as `None` for the direct head.

- [ ] **Step 5: Update all serialized test fixtures to v4**

Update schema strings, architecture schema strings, exploration keys, and policy-architecture digests in the structured export and serving tests. Do not weaken digest checks.

- [ ] **Step 6: Run identity, checkpoint, export, and serving tests**

```bash
pytest -q \
  tests/integrations/test_sb3_policy_identity_v3.py \
  tests/rl/test_structured_export.py \
  tests/serving/test_sb3_loader.py \
  tests/serving/test_structured_policy_loader.py \
  tests/rl/test_checkpointing.py
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add trade_rl/rl/policy_identity.py tests/integrations/test_sb3_policy_identity_v3.py \
  tests/rl/test_structured_export.py tests/serving/test_sb3_loader.py \
  tests/serving/test_structured_policy_loader.py tests/rl/test_checkpointing.py
git commit -m "feat: bind both action heads into policy identity v4"
```

---

### Task 5: Make Action Telemetry Comparable

**Files:**
- Modify: `trade_rl/rl/tensorboard_logging.py`
- Modify: the existing TensorBoard callback test module located by searching `build_tensorboard_metrics_callback`.

**Interfaces:**
- Consumes: `policy.action_stage_outputs(observations) -> ActionStageOutputs`.
- Produces: common action-stage TensorBoard tags for both heads; Gate intensity only when available.

- [ ] **Step 1: Add a failing direct-head callback test**

Use a fake policy whose `action_stage_outputs()` returns current weights, deterministic actions, active mask, and `change_intensity=None`. Feed sampled actions and action-path info. Require records for:

```text
trade_rl/deterministic_change_l1_mean
trade_rl/exploration_l1_mean
trade_rl/sampled_change_l1_mean
trade_rl/submission_l1_mean
trade_rl/effective_action_l1_mean
```

Require no `trade_rl/change_intensity_mean` record.

- [ ] **Step 2: Verify RED**

```bash
pytest -q tests/rl -k tensorboard_logging
```

Expected: direct stage metrics are absent because the callback only probes `hierarchical_actor_outputs`.

- [ ] **Step 3: Switch the callback to the common API**

Probe `action_stage_outputs`. Read `deterministic_actions` and `current_weights`. Record `change_intensity_mean` only when the returned value is not `None`. Keep `hierarchical_action_stage_metrics` as the single calculation owner.

- [ ] **Step 4: Run telemetry tests**

```bash
pytest -q tests/rl -k "tensorboard or action_telemetry"
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add trade_rl/rl/tensorboard_logging.py tests/rl
git commit -m "feat: compare action stages across actor heads"
```

---

### Task 6: Enforce Oracle Causal Holdout for the Direct Head

**Files:**
- Modify: `trade_rl/learning/evaluation.py`
- Modify: `trade_rl/integrations/sb3_training.py`
- Modify: `tests/learning/test_learning_evaluation.py`
- Modify: `tests/integrations/test_sb3_training.py`

**Interfaces:**
- Produces: `evaluate_direct_behavior_cloning_gates(...) -> BehaviorCloningGateEvaluation`.
- Reuses: the same causal non-collapse metric builder as hierarchical BC.

- [ ] **Step 1: Write a failing direct-gate unit test**

Create direct MSE values with adequate improvement and a causal holdout whose evidence has zero executed and submitted changes. Require:

```python
evaluation = evaluate_direct_behavior_cloning_gates(...)
assert evaluation.teacher_reconstruction_gate.passed
assert not evaluation.causal_non_collapse_gate.passed
with pytest.raises(RuntimeError, match="zero-trade collapse"):
    evaluation.require_passed()
```

Add a passing case with non-constant submitted actions, enough executed changes, and regret below both thresholds.

- [ ] **Step 2: Verify RED**

```bash
pytest -q tests/learning/test_learning_evaluation.py -k direct_behavior_cloning
```

Expected: FAIL because the direct evaluator does not exist.

- [ ] **Step 3: Factor the causal metric construction**

Move the existing causal metric tuple from `evaluate_behavior_cloning_gates()` into a private helper accepting:

```python
def _causal_non_collapse_metrics(
    *,
    holdout: BehaviorCloningHoldoutEvaluation | object | None,
    teacher_change_support: int | None,
    thresholds: BehaviorCloningGateThresholds,
) -> tuple[BehaviorCloningGateMetric, ...]:
```

Keep every current threshold and failure reason unchanged.

- [ ] **Step 4: Implement direct reconstruction evaluation**

Compute MSE relative improvement from finite non-negative initial/final values. Create one required teacher metric named `action_mse_relative_improvement`, then attach the shared causal group using teacher change support from chronological teacher labels.

- [ ] **Step 5: Derive teacher labels independently of actor type**

Refactor the SB3 adapter so chronological teacher labels are built whenever structured `active` and `current_weights` observations exist. Pass them to `pretrain_policy` only for the hierarchical actor. For a direct Oracle actor, use `labels.diagnostics.gate_positive_count` as teacher change support and call the direct gate evaluator after the same episode holdout is computed.

- [ ] **Step 6: Persist and enforce the direct evaluation**

Write `behavior-cloning-gates.json`, include its digest and payload in `behavior-cloning.json`, set `quality_passed` from the evaluation, and call `require_passed()`. Trend-baseline direct BC retains the existing MSE-only gate because it has no Oracle episode holdout.

- [ ] **Step 7: Run BC and training adapter tests**

```bash
pytest -q \
  tests/learning/test_learning_evaluation.py \
  tests/learning/test_behavior_cloning.py \
  tests/integrations/test_sb3_training.py
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add trade_rl/learning/evaluation.py trade_rl/integrations/sb3_training.py \
  tests/learning/test_learning_evaluation.py tests/integrations/test_sb3_training.py
git commit -m "feat: gate direct Oracle BC on causal holdout"
```

---

### Task 7: Add Canonical Paired Experiment Profiles

**Files:**
- Create: `examples/binance-multitimeframe/training-action-head-ablation-gate.json`
- Create: `examples/binance-multitimeframe/training-action-head-ablation-direct.json`
- Create: `examples/binance-multitimeframe/walk-forward-action-head-ablation.json`
- Test: `tests/examples/test_action_head_ablation_profiles.py`
- Modify: `docs/BINANCE.md`

**Interfaces:**
- Produces: two training run files and one two-candidate market walk-forward file.

- [ ] **Step 1: Copy the maintained target-weight PPO profile twice**

Use `training-target-weight-growth-ppo.json` as the complete source. Preserve all fields and values. Set only:

```json
"policy_actor_head": "hierarchical_gate_target_v1"
```

in the Gate file and:

```json
"policy_actor_head": "shared_target_v1"
```

in the direct file.

- [ ] **Step 2: Add the two-candidate walk-forward profile**

Copy the execution-sensitivity, fold, and selection contract from `walk-forward-target-weight-constrained-growth.json`, but define exactly:

```json
"candidates": [
  {
    "name": "target-weight-gate-head-ppo",
    "run_file": "training-action-head-ablation-gate.json"
  },
  {
    "name": "target-weight-direct-head-ppo",
    "run_file": "training-action-head-ablation-direct.json"
  }
]
```

- [ ] **Step 3: Document the command and interpretation**

In `docs/BINANCE.md`, add the maintained CLI invocation using `walk-forward-action-head-ablation.json`. State that both candidates share the complete economic and training contract except actor head and that the result does not authorize live routing.

- [ ] **Step 4: Run profile tests**

```bash
pytest -q tests/examples/test_action_head_ablation_profiles.py \
  tests/examples/test_target_weight_constrained_growth_profiles.py \
  tests/workflows/test_market_walk_forward.py
```

Expected: all tests pass and both `run_file` references resolve.

- [ ] **Step 5: Commit**

```bash
git add examples/binance-multitimeframe/training-action-head-ablation-*.json \
  examples/binance-multitimeframe/walk-forward-action-head-ablation.json \
  tests/examples/test_action_head_ablation_profiles.py docs/BINANCE.md
git commit -m "feat: add paired action-head ablation profiles"
```

---

### Task 8: Full Verification and Publication

**Files:**
- Review all modified files.

**Interfaces:**
- Produces: a verified draft pull request against `main`.

- [ ] **Step 1: Run formatting and static analysis**

```bash
ruff check .
ruff format --check .
mypy trade_rl
lint-imports
```

Expected: exit code 0 for every command.

- [ ] **Step 2: Run the complete test suite with the maintained coverage gate**

```bash
pytest
```

Expected: zero failures and coverage at or above the repository threshold.

- [ ] **Step 3: Run targeted serving and workflow smoke tests**

```bash
pytest -q \
  tests/rl/test_structured_export.py \
  tests/serving/test_sb3_loader.py \
  tests/serving/test_structured_policy_loader.py \
  tests/workflows/test_market_walk_forward.py
```

Expected: all tests pass.

- [ ] **Step 4: Review the exact branch diff**

Confirm that no reward coefficient, action name/order, risk threshold, execution cost, PPO loss, observation encoder, or live-routing path changed.

- [ ] **Step 5: Push the branch and open a draft PR**

```bash
git push -u origin agent/action-head-ablation-v1
gh pr create --draft --base main --head agent/action-head-ablation-v1 \
  --title "Add controlled action-head ablation" \
  --body-file /tmp/action-head-ablation-pr.md
```

The PR body must include RED evidence, GREEN evidence, exact test counts, coverage, identity migration notes, paired-profile integrity, and the explicit live-routing NO-GO boundary.

- [ ] **Step 6: Verify hosted CI on the exact PR head**

Require every mandatory GitHub Actions check to complete successfully. If a check fails, inspect the exact job log, add a failing regression test for the root cause when applicable, and repeat the TDD cycle before updating the PR.
