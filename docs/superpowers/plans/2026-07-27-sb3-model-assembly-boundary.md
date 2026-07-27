# SB3 Model Assembly Boundary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract typed Stable-Baselines3 policy, rollout-buffer, algorithm-construction, and checkpoint-loading decisions from `StableBaselines3Backend.train()` without changing training behavior.

**Architecture:** Add `sb3_model_assembly.py` as a one-way dependency beneath `sb3_training.py`. The new module returns immutable policy assembly metadata and owns model construction/loading, while the backend retains environment lifecycle, behavior cloning, callbacks, learning, evidence publication, saving, and result construction.

**Tech Stack:** Python 3.12, Stable-Baselines3, sb3-contrib, Gymnasium, PyTorch, dataclasses, pytest, Ruff, Mypy.

## Global Constraints

- Keep the `StableBaselines3Backend` constructor and `train()` signature unchanged.
- Do not change learning hyperparameters, policy architecture, rollout-buffer selection, checkpoint identity, output schemas, or environment cleanup.
- Do not move behavior cloning, callback construction, performance instrumentation, architecture serialization, replay publication, or final saving in this PR.
- `sb3_model_assembly.py` must not import teacher, TensorBoard, telemetry, training-performance, artifact-publication, or `StableBaselines3Backend` modules.
- Production remains `NO-GO`.

---

### Task 1: Add RED policy-assembly contracts

**Files:**
- Create: `tests/integrations/test_sb3_model_assembly.py`
- Create later: `trade_rl/integrations/sb3_model_assembly.py`
- Reuse: `tests/integrations/test_sb3_training.py`

**Interfaces:**
- Consumes: `ResidualTrainingConfig`, typed `AlgorithmConfig`, validated environment identity, probe environment
- Produces: `SB3PolicyAssembly` and `resolve_sb3_policy_assembly(...)`

- [ ] **Step 1: Write failing policy assembly tests**

Cover these exact contracts:

```python
assembly = resolve_sb3_policy_assembly(
    probe=probe,
    identity=identity,
    config=config,
    algorithm_config=build_algorithm_config(config),
)
assert assembly.policy_identifier == config.policy
assert assembly.policy_kwargs["net_arch"] == {
    "pi": list(config.policy_net_arch),
    "vf": list(config.value_net_arch),
}
```

Also assert:

- PPO includes `log_std_init`;
- off-policy algorithms use `qf` rather than `vf`;
- asset-set metadata remains exact;
- sequence configuration selects `SharedPerAssetActorCriticPolicy` only for direct target-weight actions;
- sequence configuration returns `SequenceRolloutReconstructor` and index-backed rollout-buffer metadata;
- estimated PPO plus cost-rollout bytes fail when exceeding `max_rollout_buffer_bytes`.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
uv run pytest tests/integrations/test_sb3_model_assembly.py -q
```

Expected: collection succeeds and tests fail because `SB3PolicyAssembly` and `resolve_sb3_policy_assembly` do not exist.

- [ ] **Step 3: Commit RED evidence**

Commit only the new tests and update the draft PR with the exact failing reason.

---

### Task 2: Implement immutable policy assembly

**Files:**
- Create: `trade_rl/integrations/sb3_model_assembly.py`
- Test: `tests/integrations/test_sb3_model_assembly.py`

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True, slots=True)
class SB3PolicyAssembly:
    policy_identifier: object
    policy_kwargs: Mapping[str, object]
    rollout_buffer_bytes: int | None
    sequence_metadata: Mapping[str, object] | None
    sequence_reconstructor: object | None
    uses_shared_asset_actor: bool
```

```python
def resolve_sb3_policy_assembly(
    *,
    probe: object,
    identity: Mapping[str, object],
    config: ResidualTrainingConfig,
    algorithm_config: AlgorithmConfig,
) -> SB3PolicyAssembly: ...
```

- [ ] **Step 1: Implement rollout-budget resolution**

Move the maintained PPO and Cost Critic buffer estimators without changing formulas or error text.

- [ ] **Step 2: Implement sequence assembly**

Move sequence metadata validation, `SequenceRolloutReconstructor`, feature extractor kwargs, shared actor kwargs, and index-backed rollout declaration.

- [ ] **Step 3: Implement ordinary and asset-set policy kwargs**

Preserve PPO `vf`, off-policy `qf`, `log_std_init`, and asset-set extractor metadata exactly.

- [ ] **Step 4: Run focused tests**

```bash
uv run pytest tests/integrations/test_sb3_model_assembly.py -q
uv run ruff check trade_rl/integrations/sb3_model_assembly.py tests/integrations/test_sb3_model_assembly.py
uv run mypy trade_rl/integrations/sb3_model_assembly.py
```

Expected: PASS.

---

### Task 3: Add RED algorithm-construction contracts

**Files:**
- Modify: `tests/integrations/test_sb3_model_assembly.py`
- Modify later: `trade_rl/integrations/sb3_model_assembly.py`

**Interfaces:**
- Produces `build_sb3_model(...)`

```python
def build_sb3_model(
    *,
    environment: object,
    seed: int,
    config: ResidualTrainingConfig,
    algorithm_config: AlgorithmConfig,
    policy: SB3PolicyAssembly,
    verbose: int,
    output_root: Path,
    canonical_action_probe_evidence: object | None,
) -> object: ...
```

- [ ] **Step 1: Write constructor-spy tests**

Assert exact class and kwargs for:

- PPO;
- Cost Critic PPO;
- Lagrangian PPO including canonical feasibility evidence;
- SAC;
- TD3;
- TQC and missing optional dependency failure.

- [ ] **Step 2: Run tests and verify RED**

Expected: tests fail because `build_sb3_model` does not exist.

- [ ] **Step 3: Implement model construction**

Move common learning-rate schedule, gamma, policy kwargs, seed, device, verbosity, TensorBoard root, PPO-specific options, constrained-PPO options, and off-policy options unchanged.

- [ ] **Step 4: Run focused tests and static checks**

Expected: PASS.

---

### Task 4: Add RED checkpoint-loading contracts

**Files:**
- Modify: `tests/integrations/test_sb3_model_assembly.py`
- Modify later: `trade_rl/integrations/sb3_model_assembly.py`

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True, slots=True)
class LoadedSB3Checkpoint:
    model: object
    manifest: CheckpointManifest
```

```python
def load_sb3_checkpoint_model(
    *,
    checkpoint_root: Path,
    environment: object,
    seed: int,
    config: ResidualTrainingConfig,
    identity: Mapping[str, object],
    algorithm_config: AlgorithmConfig,
    policy: SB3PolicyAssembly,
    fresh_model: object,
) -> LoadedSB3Checkpoint: ...
```

- [ ] **Step 1: Write checkpoint contract tests**

Assert fail-closed behavior for algorithm, seed, environment digest, training-config digest, timestep, and constrained-algorithm identity mismatches. Assert sequence reconstructor rebinding and `rollout_buffer_kwargs` restoration.

- [ ] **Step 2: Verify RED**

Expected: tests fail because the typed checkpoint loader does not exist.

- [ ] **Step 3: Implement checkpoint loading**

Move the existing algorithm-class selection and identity checks without changing messages.

- [ ] **Step 4: Run focused tests and static checks**

Expected: PASS.

---

### Task 5: Integrate the backend and add architecture ratchets

**Files:**
- Modify: `trade_rl/integrations/sb3_training.py`
- Modify: `tests/integrations/test_sb3_training.py`
- Modify: `tests/integrations/test_sb3_model_assembly.py`

**Interfaces:**
- Consumes: all Task 2–4 assembly APIs
- Produces: unchanged `StableBaselines3Backend.train()` behavior

- [ ] **Step 1: Replace inline policy/model/resume branches**

The backend must call:

```python
policy_assembly = resolve_sb3_policy_assembly(...)
model = build_sb3_model(...)
loaded = load_sb3_checkpoint_model(...)  # only when configured
```

Use returned metadata for architecture evidence and sequence inspection.

- [ ] **Step 2: Preserve monkeypatch compatibility intentionally**

Keep existing public helpers in `sb3_training.py`. Update tests to patch the new assembly seam where model construction is the subject, while environment lifecycle tests continue to patch the backend module.

- [ ] **Step 3: Add dependency-boundary test**

Assert `sb3_model_assembly.py` does not contain imports from:

```text
trade_rl.learning
trade_rl.integrations.behavior_cloning
trade_rl.rl.tensorboard_logging
trade_rl.rl.training_performance
trade_rl.artifacts.store
trade_rl.integrations.sb3_training
```

- [ ] **Step 4: Run integration regression suite**

```bash
uv run pytest tests/integrations/test_sb3_model_assembly.py tests/integrations/test_sb3_training.py -q
```

Expected: PASS with unchanged lifecycle and evidence assertions.

---

### Task 6: Exact-head verification and merge

**Files:**
- Update: PR description with exact head and RED/GREEN evidence

- [ ] **Step 1: Run complete verification**

Require successful results for Ruff, format, Mypy, Import Linter, dead-code, full Pytest and coverage, critical branch coverage, CLI smoke, PostgreSQL Catalog, Ubuntu, Windows, and training image probe.

- [ ] **Step 2: Review final diff**

Confirm no behavior-cloning, callback, performance, replay, Serving, reward, environment, or release semantics changed.

- [ ] **Step 3: Mark ready and squash merge exact head**

Preserve production `NO-GO`.
