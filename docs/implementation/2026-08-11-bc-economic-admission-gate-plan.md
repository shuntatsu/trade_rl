# Behavior-Cloning Economic Admission Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Require statistically supported causal after-cost performance before an Oracle behavior-cloning warm start may proceed to PPO.

**Architecture:** Extend the existing causal BC evidence path rather than adding a second training pipeline. Episode holdout evaluation will publish a deterministic one-sided lower confidence bound for causal policy net return; the existing mandatory causal gate will require both configured episode support and a configured lower-bound floor. Configuration remains backward-compatible, while the three maintained target-weight profiles opt into stronger thresholds explicitly.

**Tech Stack:** Python 3.12, NumPy, dataclasses, pytest, Stable-Baselines3 integration contracts, JSON training profiles, GitHub Actions.

## Global Constraints

- Do not modify any file changed by open PR #385 or PR #387.
- Do not change PPO, Lagrangian PPO, reward, action-space, execution, episode-boundary, checkpoint, serving, or live-order behavior.
- Preserve historical `training_run_config_v4` readability through dataclass defaults; do not add the new fields to `_REQUIRED_V4_TRAINING_FIELDS`.
- Oracle agreement and Oracle regret remain hindsight diagnostics, never production-generalization evidence.
- Every production behavior change must be preceded by a regression test observed failing for the expected missing contract.
- Production remains `NO-GO`.

---

### Task 1: Commit the complete RED contract suite

**Files:**
- Create: `tests/learning/test_bc_economic_admission_gate.py`
- Create: `tests/rl/test_bc_economic_admission_config.py`
- Modify: `tests/learning/test_episode_teacher_integration.py`
- Modify: `tests/learning/test_oracle_bc_causal_gate_contract.py`

**Interfaces:**
- Consumes: current `BehaviorCloningGateThresholds`, `evaluate_behavior_cloning_gates`, `evaluate_episode_behavior_cloning_holdout`, `ResidualTrainingConfig`, and maintained JSON profiles.
- Produces: failing contracts for the lower-bound helper, evidence field, mandatory metric, authored configuration, and maintained thresholds.

- [ ] **Step 1: Add the statistical and gate regression file**

Create `tests/learning/test_bc_economic_admission_gate.py` with real `PathPerformanceMetrics`, `ActionPathCollapseEvidence`, and record-like episode evidence. The tests must require:

```python
from trade_rl.learning.evaluation import (
    BehaviorCloningGateThresholds,
    deterministic_bootstrap_lower_bound,
    evaluate_behavior_cloning_gates,
)
```

Required test cases:

```python
def test_bootstrap_lower_bound_is_deterministic_one_sided_and_accepts_losses():
    values = np.asarray([-0.08, -0.02, 0.01, 0.04, 0.07])
    first = deterministic_bootstrap_lower_bound(
        values,
        confidence_level=0.95,
        resamples=2_000,
        seed_material="a" * 64,
    )
    second = deterministic_bootstrap_lower_bound(...same arguments...)
    assert first == second
    assert first <= float(np.mean(values))
```

```python
def test_bc_gate_rejects_insufficient_complete_episode_support():
    thresholds = BehaviorCloningGateThresholds(
        ...existing passing values...,
        minimum_causal_holdout_episodes=3,
        minimum_causal_holdout_net_return_lower_bound=-0.05,
    )
    holdout = episode_holdout_with_two_records_and_passing_action_evidence()
    gates = evaluate_behavior_cloning_gates(...)
    metric = causal_metric(gates, "causal_net_return_lower_confidence_bound")
    assert metric.status == "insufficient_support"
    assert gates.passed is False
```

```python
def test_bc_gate_rejects_after_cost_lower_bound_below_floor():
    thresholds = ...minimum episodes 2, lower-bound floor -0.05...
    holdout = episode_holdout_with_two_records(
        causal_net_return_lower_confidence_bound=-0.08,
    )
    gates = evaluate_behavior_cloning_gates(...)
    metric = causal_metric(gates, "causal_net_return_lower_confidence_bound")
    assert metric.status == "failed"
    with pytest.raises(RuntimeError, match="lower confidence bound"):
        gates.require_passed()
```

```python
def test_legacy_single_path_holdout_uses_observed_net_return_as_bound():
    thresholds = existing defaults
    holdout = BehaviorCloningHoldoutEvaluation(...positive causal net return...)
    assert evaluate_behavior_cloning_gates(...).passed is True
```

- [ ] **Step 2: Add authored configuration validation tests**

Create `tests/rl/test_bc_economic_admission_config.py` with a helper that constructs the minimum valid `ResidualTrainingConfig`:

```python
def config(**overrides: object) -> ResidualTrainingConfig:
    values = {
        "timesteps": 128,
        "gamma": 1.0,
        "seeds": (0,),
        "behavior_cloning_min_causal_holdout_episodes": 1,
        "behavior_cloning_min_causal_holdout_net_return_lower_bound": -1.0,
    }
    values.update(overrides)
    return ResidualTrainingConfig(**values)
```

Test:

```python
@pytest.mark.parametrize("value", (0, -1, True, 1.5))
def test_config_rejects_invalid_minimum_causal_holdout_episode_count(value): ...
```

```python
@pytest.mark.parametrize("value", (float("nan"), float("inf"), -1.0001))
def test_config_rejects_invalid_causal_net_return_lower_bound_floor(value): ...
```

```python
def test_config_defaults_preserve_legacy_bc_admission_contract():
    resolved = ResidualTrainingConfig(timesteps=128, gamma=1.0, seeds=(0,))
    assert resolved.behavior_cloning_min_causal_holdout_episodes == 1
    assert resolved.behavior_cloning_min_causal_holdout_net_return_lower_bound == -1.0
```

- [ ] **Step 3: Add real episode-holdout evidence coverage**

Extend `tests/learning/test_episode_teacher_integration.py` by importing:

```python
from trade_rl.learning.episode_oracle_bc import (
    evaluate_episode_action_path,
    evaluate_episode_behavior_cloning_holdout,
)
from trade_rl.learning.episode_behavior_cloning import BehaviorCloningSplit
```

Add a deterministic zero-action model:

```python
class _ZeroPolicy:
    def predict(self, observation: object, deterministic: bool = True):
        del observation, deterministic
        return np.zeros(1, dtype=np.float32), None
```

Add a test using the existing real `_environment()` and `_episode_batch()` helpers, with both episodes assigned to validation and a non-empty dummy training partition:

```python
def test_episode_bc_holdout_persists_causal_net_return_lower_bound(tmp_path):
    batch = _episode_batch(_environment())
    split = BehaviorCloningSplit(
        train_indices=np.asarray([0]),
        validation_indices=np.asarray([1, 2]),
        train_episode_ids=np.asarray([99]),
        validation_episode_ids=np.asarray([0, 1]),
    )
    audit, holdout = evaluate_episode_behavior_cloning_holdout(
        environment_factory=_environment,
        model=_ZeroPolicy(),
        batch=batch,
        split=split,
        output_root=tmp_path,
        bootstrap_confidence_level=0.95,
        bootstrap_resamples=2_000,
    )
    assert holdout is not None
    observed = [record.causal_policy_performance.net_return for record in holdout.records]
    assert holdout.causal_net_return_lower_confidence_bound <= np.mean(observed)
    assert audit["causal_net_return_lower_confidence_bound"] == holdout.causal_net_return_lower_confidence_bound
    payload = json.loads((tmp_path / "behavior-cloning-holdout.json").read_text())
    assert payload["causal_net_return_lower_confidence_bound"] == holdout.causal_net_return_lower_confidence_bound
    assert payload["schema_version"] == "episode_oracle_bc_evaluation_v2"
```

- [ ] **Step 4: Extend maintained-profile contracts**

Update `tests/learning/test_oracle_bc_causal_gate_contract.py` so all three target-weight profiles require:

```python
assert training["behavior_cloning_min_causal_holdout_episodes"] >= 5
assert (
    training["behavior_cloning_min_causal_holdout_net_return_lower_bound"]
    >= -0.05
)
```

- [ ] **Step 5: Commit RED tests**

Commit only the four test files:

```text
test: require causal BC economic admission evidence
```

- [ ] **Step 6: Open a Draft PR and verify RED on exact test head**

Run through GitHub Actions on the exact commit. Expected failure causes:

- import error for `deterministic_bootstrap_lower_bound`;
- unexpected `BehaviorCloningGateThresholds` keyword;
- unexpected `ResidualTrainingConfig` keywords;
- missing holdout evidence field;
- missing maintained JSON fields.

No production implementation may be committed before this RED result is observed.

---

### Task 2: Implement the deterministic lower-bound primitive and configuration

**Files:**
- Modify: `trade_rl/learning/evaluation.py`
- Modify: `trade_rl/rl/training.py`

**Interfaces:**
- Consumes: existing `_finite_vector`, `content_digest`, NumPy deterministic bootstrap pattern.
- Produces: `deterministic_bootstrap_lower_bound(...)` and two validated training fields.

- [ ] **Step 1: Add the minimal lower-bound helper**

Add beside `deterministic_bootstrap_upper_bound`:

```python
def deterministic_bootstrap_lower_bound(
    values: object,
    *,
    confidence_level: float,
    resamples: int,
    seed_material: str,
) -> float:
    sample = _finite_vector(values, field="bootstrap values")
    if not math.isfinite(confidence_level) or not 0.5 < confidence_level < 1.0:
        raise ValueError("bootstrap confidence_level must be within (0.5, 1)")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1_000:
        raise ValueError("bootstrap resamples must be an integer of at least 1000")
    if not isinstance(seed_material, str) or not seed_material:
        raise ValueError("bootstrap seed_material must be non-empty")
    seed = int(content_digest({"seed_material": seed_material})[:16], 16)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(sample), size=(resamples, len(sample)))
    means = sample[indices].mean(axis=1, dtype=np.float64)
    return float(np.quantile(means, 1.0 - confidence_level, method="lower"))
```

Export it from `evaluation.__all__`.

- [ ] **Step 2: Add authored configuration fields**

Add to `ResidualTrainingConfig` immediately after the existing causal holdout fields:

```python
behavior_cloning_min_causal_holdout_episodes: int = 1
behavior_cloning_min_causal_holdout_net_return_lower_bound: float = -1.0
```

Validate:

```python
if (
    isinstance(self.behavior_cloning_min_causal_holdout_episodes, bool)
    or not isinstance(self.behavior_cloning_min_causal_holdout_episodes, int)
    or self.behavior_cloning_min_causal_holdout_episodes <= 0
):
    raise ValueError("behavior_cloning_min_causal_holdout_episodes must be positive")
```

```python
if (
    not math.isfinite(self.behavior_cloning_min_causal_holdout_net_return_lower_bound)
    or self.behavior_cloning_min_causal_holdout_net_return_lower_bound < -1.0
):
    raise ValueError(
        "behavior_cloning_min_causal_holdout_net_return_lower_bound must be finite and at least -1"
    )
```

- [ ] **Step 3: Run focused tests**

```bash
uv run pytest \
  tests/learning/test_bc_economic_admission_gate.py \
  tests/rl/test_bc_economic_admission_config.py -q
```

Expected: helper/config tests pass; evidence/gate tests may remain RED until Tasks 3–4.

- [ ] **Step 4: Commit**

```text
feat: add causal BC confidence configuration
```

---

### Task 3: Persist complete-episode causal return evidence

**Files:**
- Modify: `trade_rl/learning/episode_oracle_bc.py`

**Interfaces:**
- Consumes: `deterministic_bootstrap_lower_bound`, per-record `causal_policy_performance.net_return`.
- Produces: immutable episode holdout evidence v2 field `causal_net_return_lower_confidence_bound`.

- [ ] **Step 1: Import the lower-bound helper and advance the schema**

```python
EPISODE_ORACLE_BC_EVALUATION_SCHEMA = "episode_oracle_bc_evaluation_v2"
```

- [ ] **Step 2: Add the evidence field and validation**

Add to `EpisodeBehaviorCloningHoldoutEvaluation`:

```python
causal_net_return_lower_confidence_bound: float
```

Include it in finite-value validation and `to_dict()`.

- [ ] **Step 3: Compute the lower bound from complete episode records**

After `resolved_records` is built:

```python
causal_return_lower = deterministic_bootstrap_lower_bound(
    np.asarray(
        [record.causal_policy_performance.net_return for record in resolved_records],
        dtype=np.float64,
    ),
    confidence_level=bootstrap_confidence_level,
    resamples=bootstrap_resamples,
    seed_material=content_digest(
        {
            "batch_digest": batch.digest,
            "scope": "causal_policy_net_return",
            "validation_episode_ids": validation_ids,
        }
    ),
)
```

Pass it into the dataclass and expose it in the returned audit payload.

- [ ] **Step 4: Run the real episode evidence test**

```bash
uv run pytest \
  tests/learning/test_episode_teacher_integration.py::test_episode_bc_holdout_persists_causal_net_return_lower_bound -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```text
feat: persist causal BC episode return confidence
```

---

### Task 4: Enforce the mandatory economic gate and maintained thresholds

**Files:**
- Modify: `trade_rl/learning/evaluation.py`
- Modify: `trade_rl/integrations/sb3_behavior_cloning.py`
- Modify: `examples/binance-multitimeframe/training-target-weight-growth-ppo.json`
- Modify: `examples/binance-multitimeframe/training-target-weight-constrained-growth.json`
- Modify: `examples/binance-multitimeframe/training-target-weight-constrained-growth-discounted.json`

**Interfaces:**
- Consumes: evidence field from Task 3 and authored fields from Task 2.
- Produces: behavior-cloning gate v2 with mandatory lower-bound metric and explicit maintained policy.

- [ ] **Step 1: Extend the gate threshold contract**

Add to `BehaviorCloningGateThresholds`:

```python
minimum_causal_holdout_net_return_lower_bound: float = -1.0
```

Validate it as finite and at least `-1.0`. Advance:

```python
BEHAVIOR_CLONING_GATE_SCHEMA = "behavior_cloning_gate_evaluation_v2"
```

- [ ] **Step 2: Add the mandatory causal metric**

Inside `evaluate_behavior_cloning_gates`, resolve:

```python
causal_net_return_lower = (
    None
    if holdout is None
    else getattr(
        holdout,
        "causal_net_return_lower_confidence_bound",
        holdout.causal_policy_performance.net_return,
    )
)
```

Append a causal metric:

```python
_gate_metric(
    name="causal_net_return_lower_confidence_bound",
    observed=causal_net_return_lower,
    comparison=">=",
    threshold=thresholds.minimum_causal_holdout_net_return_lower_bound,
    support=causal_episode_support,
    minimum_support=thresholds.minimum_causal_holdout_episodes,
    passed=(
        causal_net_return_lower is not None
        and causal_net_return_lower
        >= thresholds.minimum_causal_holdout_net_return_lower_bound
    ),
    failure_reason=(
        "causal after-cost net-return lower confidence bound is below the required floor"
    ),
)
```

Keep every existing causal metric mandatory.

- [ ] **Step 3: Map authored configuration in the SB3 integration**

Replace the hard-coded episode count in `_behavior_cloning_gate_thresholds`:

```python
minimum_causal_holdout_episodes=(
    config.behavior_cloning_min_causal_holdout_episodes
),
minimum_causal_holdout_net_return_lower_bound=(
    config.behavior_cloning_min_causal_holdout_net_return_lower_bound
),
```

- [ ] **Step 4: Update maintained target-weight profiles**

Add to each of the three maintained profile `training` objects:

```json
"behavior_cloning_min_causal_holdout_episodes": 5,
"behavior_cloning_min_causal_holdout_net_return_lower_bound": -0.05
```

Do not modify legacy profiles or default workflow candidate composition.

- [ ] **Step 5: Run the complete focused suite**

```bash
uv run pytest \
  tests/learning/test_bc_economic_admission_gate.py \
  tests/learning/test_episode_teacher_integration.py \
  tests/learning/test_learning_evaluation.py \
  tests/learning/test_oracle_bc_causal_gate_contract.py \
  tests/rl/test_bc_economic_admission_config.py \
  tests/integrations/test_sb3_training.py \
  tests/workflows/test_training_run_config.py -q
```

Expected: all focused tests pass.

- [ ] **Step 6: Commit**

```text
feat: enforce causal BC economic admission gate
```

---

### Task 5: Documentation, self-review, and exact-head verification

**Files:**
- Modify only when needed: `docs/CONFIGURATION.md`
- Modify only when needed: `docs/BINANCE.md`
- Update: Draft PR body

**Interfaces:**
- Consumes: completed implementation and exact GitHub Actions evidence.
- Produces: auditable PR ready for owner review, without merging.

- [ ] **Step 1: Document authored fields**

Document the two fields as BC admission controls, explicitly stating that they do not establish profitability and that walk-forward/sealed OOS remain authoritative.

- [ ] **Step 2: Run static and architecture checks**

```bash
uv run ruff check trade_rl tests
uv run ruff format --check trade_rl tests
uv run mypy trade_rl
uv run lint-imports
```

- [ ] **Step 3: Run complete repository verification**

Use the repository's exact-head CI on the final commit. Require success for:

- full Python tests and branch coverage;
- Ruff and formatting;
- MyPy;
- Import Linter and dead-code checks;
- frontend verification;
- Ubuntu and Windows compatibility;
- complete training image;
- PostgreSQL Catalog and Nautilus Capability where triggered.

- [ ] **Step 4: Review the entire diff as a reviewer**

Verify:

- no file overlaps open PR #385/#387;
- no unrelated refactor;
- defaults preserve legacy config readability;
- maintained profiles explicitly opt in;
- the lower-bound evidence uses causal after-cost policy returns, not Oracle returns;
- every new artifact field is digest-bound;
- temporary scripts/workflows are absent;
- Production remains `NO-GO`.

- [ ] **Step 5: Update the Draft PR body**

Include What, Why, design decisions, RED evidence, GREEN focused tests, exact-head CI, changed-file scope, compatibility, and remaining risks.

- [ ] **Step 6: Do not merge**

Leave the PR open for owner review. Mark it Ready only after every required exact-head check succeeds and the final diff is self-reviewed.