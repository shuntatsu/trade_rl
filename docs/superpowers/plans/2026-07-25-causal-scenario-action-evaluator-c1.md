# Causal Scenario Action Evaluator C1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the evaluation-only C1 contracts, deterministic residual-candidate generator, artificial-scenario action evaluator, and immutable `causal_scenario_value_artifact_v1` without connecting the feature to training, walk-forward selection, Serving, or production paths.

**Architecture:** `trade_rl.evaluation.causal_scenario_values` owns validated immutable inputs, candidate projection/deduplication, scenario rollout orchestration, mean/CVaR/regret statistics, deterministic confidence intervals, and tie-breaking. `trade_rl.evaluation.causal_scenario_artifact` owns deterministic two-file persistence and fail-closed loading. A rollout factory protocol keeps C1 independent of historical scenario construction and guarantees a fresh rollout object for every scenario-candidate pair.

**Tech Stack:** Python 3.12, NumPy 1.26, dataclasses, typing `Protocol`, canonical SHA-256 identities, deterministic NPZ/JSON artifacts, Pytest, pytest-cov, Ruff, MyPy.

## Global Constraints

- C1 is research evaluation only and must not modify `examples/binance-multitimeframe/walk-forward-full.json`, maintained PPO fitting, checkpoint/configuration selection, Serving, promotion, release, or direct-execution code.
- The evaluator receives frozen scenario references and an immutable causal query snapshot; it never receives the query's realized future bars.
- Every candidate applies one residual action at the first decision and zero residual for the remaining `horizon_decisions - 1` decisions.
- Default configuration is exactly: `action_dimension=n_assets`, `scenario_count=64`, `horizon_decisions=96`, `cvar_alpha=0.10`, `cvar_penalty=0.25`, `bootstrap_resamples=256`, `confidence_level=0.90`, `score_tolerance=1e-8`, and `max_candidates=32`.
- Candidate raw actions are finite, rank-one, dimension-matched, and bounded in `[-1, 1]`.
- Zero residual must be present after semantic projection deduplication.
- Scenario probabilities are version-one uniform probabilities and must sum to one within `1e-12`.
- Every scenario-candidate rollout is created from a fresh rollout object returned by the factory.
- Arrays exposed by public contracts are C-contiguous, signed-zero normalized, and read-only.
- Non-finite numeric values, inconsistent replay evidence, invalid digests, malformed artifact closure, or nondeterministic ordering fail closed.
- New production modules require 100% focused statement and branch coverage before merge.
- Production classification remains `NO-GO`.

---

## File Structure

- Create `trade_rl/evaluation/causal_scenario_values.py`: public C1 configuration, query/scenario/candidate/rollout/result contracts, candidate generation, evaluation statistics, and deterministic selection.
- Create `trade_rl/evaluation/causal_scenario_artifact.py`: canonical `manifest.json` plus deterministic `arrays.npz` writer/loader.
- Modify `trade_rl/evaluation/__init__.py`: export only the stable public C1 API.
- Create `tests/evaluation/test_causal_scenario_contracts.py`: configuration, immutable arrays, query/scenario validation, and candidate-generation tests.
- Create `tests/evaluation/test_causal_scenario_evaluator.py`: artificial-market rankings, CVaR, regret, confidence intervals, and tie-breaking tests.
- Create `tests/evaluation/test_causal_scenario_fail_closed.py`: malformed evidence, state isolation, deduplication, determinism, and numeric rejection tests.
- Create `tests/evaluation/test_causal_scenario_artifact.py`: deterministic write/load, file closure, digest tampering, shape/dtype tampering, and public export tests.

---

### Task 1: Add immutable C1 contracts and deterministic residual candidates

**Files:**
- Create: `trade_rl/evaluation/causal_scenario_values.py`
- Test: `tests/evaluation/test_causal_scenario_contracts.py`

**Interfaces:**
- Consumes: `trade_rl.artifacts.hashing.content_digest` and lowercase SHA-256 identity strings.
- Produces:
  - `CAUSAL_SCENARIO_EVALUATOR_SCHEMA: Final[str]`
  - `CausalScenarioEvaluatorConfig`
  - `CausalQuerySnapshot`
  - `CausalScenarioSet`
  - `ProjectedResidualCandidate`
  - `ScenarioRolloutEvidence`
  - `ScenarioRollout`
  - `ScenarioRolloutFactory`
  - `generate_residual_candidates(...) -> tuple[np.ndarray, ...]`

- [ ] **Step 1: Write failing configuration and immutable-array tests**

Create tests equivalent to:

```python
from __future__ import annotations

import numpy as np
import pytest

from trade_rl.evaluation.causal_scenario_values import (
    CAUSAL_SCENARIO_EVALUATOR_SCHEMA,
    CausalQuerySnapshot,
    CausalScenarioEvaluatorConfig,
    CausalScenarioSet,
    generate_residual_candidates,
)


def sha(char: str) -> str:
    return char * 64


def test_default_config_is_digest_stable() -> None:
    config = CausalScenarioEvaluatorConfig(action_dimension=3)

    assert CAUSAL_SCENARIO_EVALUATOR_SCHEMA == (
        "causal_scenario_action_evaluator_v1"
    )
    assert config.scenario_count == 64
    assert config.horizon_decisions == 96
    assert config.cvar_alpha == 0.10
    assert config.cvar_penalty == 0.25
    assert config.bootstrap_resamples == 256
    assert config.confidence_level == 0.90
    assert config.score_tolerance == 1e-8
    assert config.max_candidates == 32
    assert config.digest == CausalScenarioEvaluatorConfig(
        action_dimension=3
    ).digest


def test_query_and_scenario_arrays_are_read_only() -> None:
    query = CausalQuerySnapshot(
        dataset_id=sha("a"),
        fold_digest=sha("b"),
        train_start=0,
        train_stop=10_000,
        query_index=10_100,
        query_timestamp_ns=1_800_000_000_000_000_000,
        source_commit="c" * 40,
        query_digest=sha("1"),
        state_snapshot_digest=sha("2"),
        observation_digest=sha("3"),
        environment_digest=sha("4"),
        action_spec_digest=sha("5"),
        execution_policy_digest=sha("6"),
        risk_digest=sha("7"),
        trend_digest=sha("8"),
        starting_equity=100_000.0,
        baseline_target=np.asarray([0.1, -0.2, 0.0]),
    )
    scenarios = CausalScenarioSet(
        scenario_ids=tuple(f"scenario-{index:02d}" for index in range(64)),
        probabilities=np.full(64, 1.0 / 64.0),
        anchor_indices=np.full(64, -1, dtype=np.int64),
        distances=np.arange(64, dtype=np.float64),
        query_condition=np.asarray([0.5, -0.5]),
        anchor_conditions=np.zeros((64, 2), dtype=np.float64),
        library_digest=sha("9"),
    )

    assert not query.baseline_target.flags.writeable
    assert not scenarios.probabilities.flags.writeable
    assert not scenarios.anchor_indices.flags.writeable
    assert not scenarios.distances.flags.writeable
    assert not scenarios.query_condition.flags.writeable
    assert not scenarios.anchor_conditions.flags.writeable

    with pytest.raises(ValueError):
        query.baseline_target[0] = 0.0
```

Add parameterized rejection tests for:

- non-positive or Boolean `action_dimension`;
- non-positive or Boolean `scenario_count`;
- `horizon_decisions <= 0`;
- `cvar_alpha` outside `(0, 1]`;
- negative `cvar_penalty`;
- non-positive `bootstrap_resamples`;
- `confidence_level` outside `(0, 1)`;
- negative `score_tolerance`;
- `max_candidates < 1` or `max_candidates > 32`;
- malformed SHA-256 identities;
- malformed 40-character lowercase source commit;
- invalid train/query indices or timestamps;
- non-positive `starting_equity`;
- query target dimension mismatch or non-finite values;
- duplicate/empty scenario IDs;
- nonuniform, negative, non-finite, or incorrectly summed probabilities;
- scenario array shape mismatches;
- non-finite distances/conditions;
- `library_digest` that is not lowercase SHA-256.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
pytest tests/evaluation/test_causal_scenario_contracts.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'trade_rl.evaluation.causal_scenario_values'`.

- [ ] **Step 3: Implement validation and read-only normalization helpers**

Implement these private helpers in `causal_scenario_values.py`:

```python
def _finite_float(name: str, value: object) -> float: ...
def _positive_int(name: str, value: object) -> int: ...
def _readonly_float_array(
    name: str,
    value: object,
    *,
    ndim: int,
    shape: tuple[int | None, ...] | None = None,
) -> np.ndarray: ...
def _readonly_int_array(
    name: str,
    value: object,
    *,
    ndim: int,
    shape: tuple[int | None, ...] | None = None,
) -> np.ndarray: ...
def _require_digest(name: str, value: object) -> str: ...
def _canonical_array_payload(value: np.ndarray) -> dict[str, object]: ...
```

Normalization requirements:

```python
array = np.asarray(value, dtype=np.float64).copy(order="C")
array[array == 0.0] = 0.0
array.setflags(write=False)
```

Integer arrays use `np.int64`. Boolean values must never pass integer or real-number validation.

- [ ] **Step 4: Implement the public immutable contracts**

Use frozen slot dataclasses with explicit validation:

```python
CAUSAL_SCENARIO_EVALUATOR_SCHEMA: Final = (
    "causal_scenario_action_evaluator_v1"
)


@dataclass(frozen=True, slots=True)
class CausalScenarioEvaluatorConfig:
    action_dimension: int
    scenario_count: int = 64
    horizon_decisions: int = 96
    cvar_alpha: float = 0.10
    cvar_penalty: float = 0.25
    bootstrap_resamples: int = 256
    confidence_level: float = 0.90
    score_tolerance: float = 1e-8
    max_candidates: int = 32
    replay_tolerance: float = 1e-10
    probability_tolerance: float = 1e-12
    schema_version: str = CAUSAL_SCENARIO_EVALUATOR_SCHEMA

    def digest_payload(self) -> dict[str, object]: ...

    @property
    def digest(self) -> str:
        return content_digest(self.digest_payload())
```

```python
@dataclass(frozen=True, slots=True)
class CausalQuerySnapshot:
    dataset_id: str
    fold_digest: str
    train_start: int
    train_stop: int
    query_index: int
    query_timestamp_ns: int
    source_commit: str
    query_digest: str
    state_snapshot_digest: str
    observation_digest: str
    environment_digest: str
    action_spec_digest: str
    execution_policy_digest: str
    risk_digest: str
    trend_digest: str
    starting_equity: float
    baseline_target: np.ndarray

    @property
    def action_dimension(self) -> int:
        return int(self.baseline_target.shape[0])
```

```python
@dataclass(frozen=True, slots=True)
class CausalScenarioSet:
    scenario_ids: tuple[str, ...]
    probabilities: np.ndarray
    anchor_indices: np.ndarray
    distances: np.ndarray
    query_condition: np.ndarray
    anchor_conditions: np.ndarray
    library_digest: str

    @property
    def scenario_count(self) -> int:
        return len(self.scenario_ids)

    @property
    def digest(self) -> str: ...
```

```python
@dataclass(frozen=True, slots=True)
class ProjectedResidualCandidate:
    raw_action: np.ndarray
    projected_target: np.ndarray
    execution_intent_digest: str
    candidate_digest: str
    expected_turnover_hint: float
    is_zero: bool
```

`candidate_digest` must be recomputed and verified from:

```python
{
    "execution_intent_digest": execution_intent_digest,
    "projected_target": projected_target.tolist(),
    "schema_version": "projected_residual_candidate_v1",
}
```

```python
@dataclass(frozen=True, slots=True)
class ScenarioRolloutEvidence:
    terminal_equity: float
    reported_log_return: float
    filled_turnover: float
    interval_cost: float
    fill_ratio: float
    feasible: bool
    termination_reason: str
    evidence_digest: str
```

`evidence_digest` must bind every economic field through schema `scenario_rollout_evidence_v1`.

Define protocols:

```python
class ScenarioRollout(Protocol):
    def run(
        self,
        candidate: ProjectedResidualCandidate,
        *,
        horizon_decisions: int,
        zero_residual_after_first: bool,
    ) -> ScenarioRolloutEvidence: ...


class ScenarioRolloutFactory(Protocol):
    def project_candidate(
        self,
        query: CausalQuerySnapshot,
        raw_action: np.ndarray,
    ) -> ProjectedResidualCandidate: ...

    def create_rollout(
        self,
        query: CausalQuerySnapshot,
        scenario_index: int,
        scenario_id: str,
    ) -> ScenarioRollout: ...
```

- [ ] **Step 5: Write failing candidate-generation tests**

Add tests equivalent to:

```python
def test_generate_residual_candidates_has_stable_mandatory_order() -> None:
    actions = generate_residual_candidates(
        np.asarray([0.3, -0.2, 0.0], dtype=np.float64),
        external_actions=(
            np.asarray([0.2, 0.0, -0.1], dtype=np.float64),
        ),
        max_candidates=32,
    )

    np.testing.assert_array_equal(actions[0], np.zeros(3))
    np.testing.assert_array_equal(actions[1], np.asarray([-1.0, 0.0, 0.0]))
    np.testing.assert_array_equal(actions[2], np.asarray([-0.5, 0.0, 0.0]))
    np.testing.assert_array_equal(actions[3], np.asarray([0.5, 0.0, 0.0]))
    np.testing.assert_array_equal(actions[4], np.asarray([1.0, 0.0, 0.0]))
    assert any(
        np.array_equal(action, np.asarray([-0.5, 0.5, 0.0]))
        for action in actions
    )
    assert any(
        np.array_equal(action, np.asarray([0.2, 0.0, -0.1]))
        for action in actions
    )
    assert all(not action.flags.writeable for action in actions)
```

Add tests proving:

- isolated candidates are emitted asset-major and magnitude-major;
- portfolio reduction candidates equal `-sign(trend_target) * magnitude`;
- a zero Trend coordinate stays zero in portfolio-reduction candidates;
- duplicate external actions are removed by raw-array identity before projection;
- an external action outside `[-1, 1]`, wrong dimension, non-finite, or non-array sequence fails;
- more than `max_candidates` fails instead of truncating;
- signed zero produces the same candidate ordering and digest inputs.

- [ ] **Step 6: Implement `generate_residual_candidates`**

Use this exact deterministic construction:

```python
def generate_residual_candidates(
    trend_target: np.ndarray,
    *,
    external_actions: Sequence[np.ndarray] = (),
    max_candidates: int = 32,
) -> tuple[np.ndarray, ...]:
    target = _readonly_float_array("trend_target", trend_target, ndim=1)
    actions: list[np.ndarray] = [np.zeros_like(target)]
    for asset_index in range(target.size):
        for magnitude in (-1.0, -0.5, 0.5, 1.0):
            action = np.zeros_like(target)
            action[asset_index] = magnitude
            actions.append(action)
    for magnitude in (0.5, 1.0):
        actions.append(-np.sign(target) * magnitude)
    actions.extend(
        np.asarray(action, dtype=np.float64) for action in external_actions
    )
```

Canonicalize signed zero, validate bounds/dimensions, deduplicate by `content_digest({"raw_action": action.tolist(), "schema_version": "raw_residual_candidate_v1"})`, preserve first occurrence, reject count above `max_candidates`, and return read-only arrays.

- [ ] **Step 7: Run contracts tests GREEN and commit**

Run:

```bash
pytest tests/evaluation/test_causal_scenario_contracts.py -q
```

Expected: all contract and candidate-generation tests pass.

Commit:

```bash
git add trade_rl/evaluation/causal_scenario_values.py \
  tests/evaluation/test_causal_scenario_contracts.py
git commit -m "feat: add causal scenario evaluator contracts"
```

---

### Task 2: Implement projection deduplication, scenario statistics, and deterministic selection

**Files:**
- Modify: `trade_rl/evaluation/causal_scenario_values.py`
- Create: `tests/evaluation/test_causal_scenario_evaluator.py`

**Interfaces:**
- Consumes:
  - Task 1 contracts and `ScenarioRolloutFactory`.
  - `generate_residual_candidates(...)`.
- Produces:
  - `CausalScenarioEvaluationResult`
  - `evaluate_causal_scenario_actions(...) -> CausalScenarioEvaluationResult`

- [ ] **Step 1: Write artificial rollout helpers and failing ranking tests**

Define test-only helpers inside `test_causal_scenario_evaluator.py`:

```python
@dataclass
class ArtificialScenarioRollout:
    starting_equity: float
    scenario_return: float
    action_coefficients: np.ndarray
    instance_id: int

    def run(
        self,
        candidate: ProjectedResidualCandidate,
        *,
        horizon_decisions: int,
        zero_residual_after_first: bool,
    ) -> ScenarioRolloutEvidence:
        assert horizon_decisions == 96
        assert zero_residual_after_first is True
        log_return = float(
            self.scenario_return
            + np.dot(self.action_coefficients, candidate.raw_action)
        )
        terminal = self.starting_equity * math.exp(log_return)
        turnover = float(np.abs(candidate.raw_action).sum())
        payload = {
            "terminal_equity": terminal,
            "reported_log_return": log_return,
            "filled_turnover": turnover,
            "interval_cost": 0.001 * turnover,
            "fill_ratio": 1.0,
            "feasible": True,
            "termination_reason": "horizon",
            "schema_version": "scenario_rollout_evidence_v1",
        }
        return ScenarioRolloutEvidence(
            **{
                key: value
                for key, value in payload.items()
                if key != "schema_version"
            },
            evidence_digest=content_digest(payload),
        )
```

Create a factory whose `project_candidate` maps `projected_target = query.baseline_target + 0.25 * raw_action`, clips to `[-0.45, 0.45]`, and creates semantic candidate digests. Its `create_rollout` increments an instance counter and returns a fresh object.

Add known-ranking tests:

```python
def test_monotonic_up_scenarios_select_positive_first_asset() -> None:
    result = evaluate_causal_scenario_actions(
        query=query_snapshot(),
        scenarios=scenario_set(
            coefficients=np.tile(
                np.asarray([0.02, 0.0, 0.0]),
                (64, 1),
            )
        ),
        config=CausalScenarioEvaluatorConfig(action_dimension=3),
        rollout_factory=ArtificialFactory(...),
    )

    np.testing.assert_array_equal(
        result.raw_candidate_actions[result.selected_candidate_index],
        np.asarray([1.0, 0.0, 0.0]),
    )
    assert result.score[result.selected_candidate_index] > result.score[
        result.zero_candidate_index
    ]
```

Also add:

- monotonic-down selects `[-1, 0, 0]`;
- flat scenarios select zero through tie-break;
- high cost makes zero best;
- asymmetric downside makes the lower-mean but safer candidate win after CVaR;
- candidate projection aliases deduplicate to one semantic candidate while retaining zero;
- every candidate's `regret` equals `max(score) - score`;
- the selected candidate has regret zero within tolerance.

- [ ] **Step 2: Run evaluator tests and verify RED**

Run:

```bash
pytest tests/evaluation/test_causal_scenario_evaluator.py -q
```

Expected: import fails because `CausalScenarioEvaluationResult` and `evaluate_causal_scenario_actions` do not exist.

- [ ] **Step 3: Implement the immutable evaluation result**

Add:

```python
@dataclass(frozen=True, slots=True)
class CausalScenarioEvaluationResult:
    config: CausalScenarioEvaluatorConfig
    dataset_id: str
    fold_digest: str
    train_start: int
    train_stop: int
    query_index: int
    query_timestamp_ns: int
    source_commit: str
    query_digest: str
    state_snapshot_digest: str
    observation_digest: str
    environment_digest: str
    action_spec_digest: str
    execution_policy_digest: str
    risk_digest: str
    trend_digest: str
    starting_equity: float
    candidate_generator_digest: str
    scenario_set_digest: str
    scenario_library_digest: str
    scenario_ids: tuple[str, ...]
    candidate_digests: tuple[str, ...]
    execution_intent_digests: tuple[str, ...]
    termination_reasons: tuple[str, ...]
    raw_candidate_actions: np.ndarray
    projected_targets: np.ndarray
    scenario_probabilities: np.ndarray
    scenario_anchor_indices: np.ndarray
    scenario_distances: np.ndarray
    query_condition: np.ndarray
    anchor_conditions: np.ndarray
    terminal_equity: np.ndarray
    gross_log_returns: np.ndarray
    baseline_relative_advantages: np.ndarray
    filled_turnover: np.ndarray
    interval_cost: np.ndarray
    fill_ratio: np.ndarray
    feasible_mask: np.ndarray
    termination_codes: np.ndarray
    mean_advantage: np.ndarray
    loss_cvar: np.ndarray
    score: np.ndarray
    regret: np.ndarray
    confidence_lower: np.ndarray
    confidence_upper: np.ndarray
    expected_filled_turnover: np.ndarray
    selected_candidate_index: int
    zero_candidate_index: int
    tie_candidate_indices: tuple[int, ...]
    result_digest: str
    schema_version: str = CAUSAL_SCENARIO_EVALUATOR_SCHEMA
```

Validation must recompute:

- `config.digest` and every copied query/scenario identity;
- all array shapes from `(scenario_count, candidate_count, action_dimension)`;
- zero-candidate raw action;
- `baseline_relative_advantages == gross_log_returns - gross_log_returns[:, zero_candidate_index, None]` with correct broadcasting;
- `mean_advantage`, `loss_cvar`, `score`, `regret`;
- selected index and tie set;
- `result_digest`.

Use integer `termination_codes` with a sorted unique `termination_reasons` vocabulary. Arrays are read-only.

- [ ] **Step 4: Implement projection and semantic deduplication**

Add:

```python
def _project_candidates(
    query: CausalQuerySnapshot,
    raw_actions: tuple[np.ndarray, ...],
    rollout_factory: ScenarioRolloutFactory,
    *,
    max_candidates: int,
) -> tuple[ProjectedResidualCandidate, ...]: ...
```

Requirements:

- call `project_candidate` in raw-action order;
- verify returned raw action exactly matches the supplied raw action;
- verify projected target dimension equals query action dimension;
- verify candidate and execution-intent digests;
- semantic dedup key is `candidate.candidate_digest`;
- if a non-zero raw candidate collides with zero, preserve the zero candidate;
- reject a second distinct projection with the same candidate digest but different projected target or execution-intent digest;
- zero must remain and projected count must not exceed `max_candidates`.

- [ ] **Step 5: Implement statistics helpers**

Add these pure functions and cover them directly:

```python
def _loss_cvar(
    advantages: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    tail_count = int(math.ceil(alpha * advantages.shape[0]))
    losses = -advantages
    ordered = np.sort(losses, axis=0)
    return ordered[-tail_count:].mean(axis=0)
```

```python
def _bootstrap_mean_intervals(
    advantages: np.ndarray,
    *,
    query_digest: str,
    config_digest: str,
    resamples: int,
    confidence_level: float,
) -> tuple[np.ndarray, np.ndarray]:
    seed_digest = content_digest(
        {
            "config_digest": config_digest,
            "query_digest": query_digest,
            "schema_version": "causal_scenario_bootstrap_seed_v1",
        }
    )
    seed = int(seed_digest[:16], 16)
    generator = np.random.Generator(np.random.Philox(seed))
    indices = generator.integers(
        0,
        advantages.shape[0],
        size=(resamples, advantages.shape[0]),
        endpoint=False,
    )
    sample_means = advantages[indices].mean(axis=1)
    tail = (1.0 - confidence_level) / 2.0
    return (
        np.quantile(sample_means, tail, axis=0, method="linear"),
        np.quantile(sample_means, 1.0 - tail, axis=0, method="linear"),
    )
```

Use the same resampling index matrix for all candidates.

- [ ] **Step 6: Implement `evaluate_causal_scenario_actions`**

Signature:

```python
def evaluate_causal_scenario_actions(
    *,
    query: CausalQuerySnapshot,
    scenarios: CausalScenarioSet,
    config: CausalScenarioEvaluatorConfig,
    rollout_factory: ScenarioRolloutFactory,
    external_actions: Sequence[np.ndarray] = (),
) -> CausalScenarioEvaluationResult:
```

Algorithm:

1. Verify query action dimension and scenario count match config.
2. Generate raw candidates from `query.baseline_target`.
3. Project and semantically deduplicate.
4. Locate the single zero candidate.
5. Compute and store a canonical `candidate_generator_digest` from the generator schema, action dimension, baseline Trend target, mandatory magnitude lists, external raw actions, generated raw candidate actions, and maximum candidate count.
6. Allocate `(scenario_count, candidate_count)` float64 evidence matrices plus Boolean feasibility and integer termination-code matrices.
7. For every scenario-candidate pair, call `rollout_factory.create_rollout(...)` separately, then call `run(...)` with `zero_residual_after_first=True`.
8. Independently recompute `log(terminal_equity / query.starting_equity)` and compare with `reported_log_return` using `config.replay_tolerance`.
9. Reject any `feasible=False` evidence in C1 and encode termination reasons deterministically.
10. Compute baseline-relative advantages, mean, CVaR, score, regret, bootstrap interval, and expected turnover.
11. Build the tie set where `max(score) - score <= score_tolerance`.
12. Select lexicographically by:
    - expected filled turnover;
    - raw residual L1 norm;
    - zero candidate first;
    - candidate digest.
13. Construct and validate the immutable result.

- [ ] **Step 7: Run evaluator tests GREEN and commit**

Run:

```bash
pytest tests/evaluation/test_causal_scenario_contracts.py \
  tests/evaluation/test_causal_scenario_evaluator.py -q
```

Expected: all tests pass.

Commit:

```bash
git add trade_rl/evaluation/causal_scenario_values.py \
  tests/evaluation/test_causal_scenario_evaluator.py
git commit -m "feat: evaluate causal scenario action values"
```

---

### Task 3: Add fail-closed evidence and state-isolation coverage

**Files:**
- Modify: `trade_rl/evaluation/causal_scenario_values.py`
- Create: `tests/evaluation/test_causal_scenario_fail_closed.py`

**Interfaces:**
- Consumes: Task 2 evaluator.
- Produces: no new public API; strengthens the evaluator trust boundary.

- [ ] **Step 1: Write malicious-factory and malformed-evidence tests**

Add test doubles that independently return:

- terminal equity `<= 0`;
- NaN or infinity in every numeric evidence field;
- negative turnover or interval cost;
- fill ratio outside `[0, 1]`;
- `feasible=False`;
- empty termination reason;
- incorrect evidence digest;
- reported log return differing from independent reconstruction;
- a projection with mismatched raw action;
- wrong projected-target dimension;
- malformed candidate or execution-intent digest;
- nondeterministic candidate digest for identical projection;
- a reused rollout object.

For fresh-state enforcement, implement a rollout object that raises when called twice:

```python
class SingleUseRollout:
    def __init__(self, evidence: ScenarioRolloutEvidence) -> None:
        self._evidence = evidence
        self._used = False

    def run(
        self,
        candidate,
        *,
        horizon_decisions,
        zero_residual_after_first,
    ):
        if self._used:
            raise AssertionError("rollout object was reused")
        self._used = True
        return self._evidence
```

The factory must return a new `SingleUseRollout` for every pair; assert the number created equals `scenario_count * candidate_count`.

- [ ] **Step 2: Run fail-closed tests and verify RED**

Run:

```bash
pytest tests/evaluation/test_causal_scenario_fail_closed.py -q
```

Expected: failures expose missing or insufficient validation.

- [ ] **Step 3: Add independent validation at every protocol boundary**

Before accepting `ProjectedResidualCandidate` or `ScenarioRolloutEvidence`, recompute their digest payloads and reject mismatch.

For each rollout evidence, enforce:

```python
if terminal_equity <= 0.0:
    raise ValueError("scenario rollout terminal_equity must be positive")
if filled_turnover < 0.0 or interval_cost < 0.0:
    raise ValueError(
        "scenario rollout costs and turnover must be non-negative"
    )
if not 0.0 <= fill_ratio <= 1.0:
    raise ValueError("scenario rollout fill_ratio must be in [0, 1]")
if feasible is not True:
    raise ValueError("C1 requires every scenario rollout to be feasible")
replayed = math.log(terminal_equity / query.starting_equity)
if not math.isclose(
    replayed,
    reported_log_return,
    rel_tol=0.0,
    abs_tol=config.replay_tolerance,
):
    raise ValueError(
        "scenario rollout log return does not match terminal equity"
    )
```

Do not catch protocol exceptions and substitute baseline/cash behavior.

- [ ] **Step 4: Add deterministic replay and signed-zero tests**

Evaluate identical inputs twice with independent factories and assert exact array equality and identical result digests. Repeat with raw/projected arrays containing `-0.0` and assert canonical `+0.0` and unchanged identity.

Add a test that changes query digest and proves bootstrap intervals and result digest change while economic means remain identical.

- [ ] **Step 5: Run all C1 in-memory tests GREEN with focused coverage**

Run:

```bash
pytest \
  tests/evaluation/test_causal_scenario_contracts.py \
  tests/evaluation/test_causal_scenario_evaluator.py \
  tests/evaluation/test_causal_scenario_fail_closed.py \
  --cov=trade_rl.evaluation.causal_scenario_values \
  --cov-branch \
  --cov-report=term-missing \
  --cov-fail-under=100
```

Expected: tests pass and the module reaches 100% statement and branch coverage.

Commit:

```bash
git add trade_rl/evaluation/causal_scenario_values.py \
  tests/evaluation/test_causal_scenario_fail_closed.py
git commit -m "test: harden causal scenario evaluator evidence"
```

---

### Task 4: Add deterministic state-action value artifact persistence

**Files:**
- Create: `trade_rl/evaluation/causal_scenario_artifact.py`
- Create: `tests/evaluation/test_causal_scenario_artifact.py`

**Interfaces:**
- Consumes: `CausalScenarioEvaluationResult`.
- Produces:
  - `CAUSAL_SCENARIO_VALUE_ARTIFACT_SCHEMA`
  - `CAUSAL_SCENARIO_MANIFEST_NAME`
  - `CAUSAL_SCENARIO_ARRAYS_NAME`
  - `CausalScenarioValueArtifactManifest`
  - `write_causal_scenario_value_artifact(...) -> str`
  - `load_causal_scenario_value_artifact(...) -> CausalScenarioEvaluationResult`

- [ ] **Step 1: Write failing deterministic round-trip tests**

Create a valid result through the artificial evaluator, then test:

```python
def test_artifact_round_trip_is_deterministic(tmp_path: Path) -> None:
    result = valid_result()
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_digest = write_causal_scenario_value_artifact(first, result)
    second_digest = write_causal_scenario_value_artifact(second, result)

    assert first_digest == second_digest
    assert (first / "manifest.json").read_bytes() == (
        second / "manifest.json"
    ).read_bytes()
    assert (first / "arrays.npz").read_bytes() == (
        second / "arrays.npz"
    ).read_bytes()

    loaded = load_causal_scenario_value_artifact(first)
    assert loaded.result_digest == result.result_digest
    np.testing.assert_array_equal(
        loaded.baseline_relative_advantages,
        result.baseline_relative_advantages,
    )
    assert not loaded.raw_candidate_actions.flags.writeable
```

Also test:

- destination must be absent or empty;
- exact closure rejects undeclared files, directories, and symlinks;
- manifest digest tampering;
- arrays digest tampering;
- missing or extra array;
- shape/dtype mismatch;
- scenario/candidate metadata length mismatch;
- result digest mismatch after array mutation;
- schema mismatch;
- deterministic ZIP timestamps and file modes;
- `allow_pickle=False` loading.

- [ ] **Step 2: Run artifact tests and verify RED**

Run:

```bash
pytest tests/evaluation/test_causal_scenario_artifact.py -q
```

Expected: import fails because the artifact module does not exist.

- [ ] **Step 3: Implement deterministic encoding primitives**

In `causal_scenario_artifact.py`, use:

```python
CAUSAL_SCENARIO_VALUE_ARTIFACT_SCHEMA: Final = (
    "causal_scenario_value_artifact_v1"
)
CAUSAL_SCENARIO_MANIFEST_NAME: Final = "manifest.json"
CAUSAL_SCENARIO_ARRAYS_NAME: Final = "arrays.npz"
_FIXED_ZIP_TIMESTAMP: Final = (1980, 1, 1, 0, 0, 0)
_ALLOWED_FILES = frozenset(
    {CAUSAL_SCENARIO_MANIFEST_NAME, CAUSAL_SCENARIO_ARRAYS_NAME}
)
```

Implement local, focused helpers matching repository artifact behavior:

```python
def _sha256_bytes(payload: bytes) -> str: ...
def _npy_bytes(array: np.ndarray) -> bytes: ...
def _deterministic_npz(arrays: Mapping[str, np.ndarray]) -> bytes: ...
def _atomic_write(path: Path, payload: bytes) -> None: ...
def _verify_exact_files(root: Path) -> None: ...
```

Use ZIP_STORED, sorted names, fixed timestamps, Unix regular-file mode `0o100644 << 16`, and `np.lib.format.write_array(..., allow_pickle=False)`.

- [ ] **Step 4: Implement the manifest and writer**

Define:

```python
@dataclass(frozen=True, slots=True)
class CausalScenarioValueArtifactManifest:
    artifact_digest: str
    arrays_digest: str
    result_digest: str
    query_digest: str
    config_digest: str
    scenario_set_digest: str
    scenario_ids: tuple[str, ...]
    candidate_digests: tuple[str, ...]
    execution_intent_digests: tuple[str, ...]
    termination_reasons: tuple[str, ...]
    config_payload: dict[str, object]
    dataset_id: str
    fold_digest: str
    train_start: int
    train_stop: int
    query_index: int
    query_timestamp_ns: int
    source_commit: str
    state_snapshot_digest: str
    observation_digest: str
    environment_digest: str
    action_spec_digest: str
    execution_policy_digest: str
    risk_digest: str
    trend_digest: str
    starting_equity: float
    candidate_generator_digest: str
    scenario_library_digest: str
    selected_candidate_index: int
    zero_candidate_index: int
    tie_candidate_indices: tuple[int, ...]
    array_metadata: dict[str, dict[str, object]]
    schema_version: str = CAUSAL_SCENARIO_VALUE_ARTIFACT_SCHEMA
```

Store every NumPy field from `CausalScenarioEvaluationResult` in `arrays.npz`. Store strings, indices, identities, and shapes/dtypes in the manifest. Compute:

```python
base_manifest = {
    ...,
    "arrays_digest": arrays_digest,
    "arrays_file": CAUSAL_SCENARIO_ARRAYS_NAME,
    "schema_version": CAUSAL_SCENARIO_VALUE_ARTIFACT_SCHEMA,
}
artifact_digest = content_digest(base_manifest)
```

Write arrays first, manifest second, using atomic replacement.

- [ ] **Step 5: Implement the fail-closed loader**

Loader sequence:

1. verify exact file closure and reject symlinks;
2. parse manifest as a mapping;
3. pop and verify `artifact_digest`;
4. verify schema and arrays filename;
5. hash raw NPZ bytes;
6. load with `allow_pickle=False`;
7. verify exact array names, shapes, and dtypes;
8. reconstruct `CausalScenarioEvaluatorConfig` from the exact `config_payload` and verify its digest;
9. reconstruct `CausalScenarioEvaluationResult`;
10. rely on its full invariant recomputation, including terminal-equity/log-return, scenario metadata, statistics, tie set, and result identity;
11. verify reconstructed `result_digest` matches manifest.

Do not accept unknown manifest keys or silently cast incompatible dtypes.

- [ ] **Step 6: Run artifact tests GREEN with focused coverage**

Run:

```bash
pytest tests/evaluation/test_causal_scenario_artifact.py \
  --cov=trade_rl.evaluation.causal_scenario_artifact \
  --cov-branch \
  --cov-report=term-missing \
  --cov-fail-under=100
```

Expected: tests pass and artifact module reaches 100% statement and branch coverage.

Commit:

```bash
git add trade_rl/evaluation/causal_scenario_artifact.py \
  tests/evaluation/test_causal_scenario_artifact.py
git commit -m "feat: persist causal scenario value artifacts"
```

---

### Task 5: Publish the stable API and enforce architecture boundaries

**Files:**
- Modify: `trade_rl/evaluation/__init__.py`
- Modify: `tests/evaluation/test_causal_scenario_artifact.py`
- Modify: `tests/architecture/test_complete_trust_boundary_remediation.py` only if an existing dependency assertion is the canonical location; otherwise create `tests/architecture/test_causal_scenario_evaluator_boundary.py`.

**Interfaces:**
- Consumes: Tasks 1-4 public classes/functions.
- Produces: stable imports from `trade_rl.evaluation`.

- [ ] **Step 1: Write failing public-import and dependency-boundary tests**

Add imports:

```python
from trade_rl.evaluation import (
    CAUSAL_SCENARIO_EVALUATOR_SCHEMA,
    CAUSAL_SCENARIO_VALUE_ARTIFACT_SCHEMA,
    CausalQuerySnapshot,
    CausalScenarioEvaluationResult,
    CausalScenarioEvaluatorConfig,
    CausalScenarioSet,
    ProjectedResidualCandidate,
    ScenarioRolloutEvidence,
    evaluate_causal_scenario_actions,
    generate_residual_candidates,
    load_causal_scenario_value_artifact,
    write_causal_scenario_value_artifact,
)
```

Assert each exported object is the same object as its defining module.

Add an architecture test that scans maintained training, Serving, release, promotion, and example configuration modules and rejects imports of:

```text
trade_rl.evaluation.causal_scenario_values
trade_rl.evaluation.causal_scenario_artifact
```

The evaluation package may export C1; training/Serving may not depend on it.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
pytest tests/evaluation/test_causal_scenario_artifact.py \
  tests/architecture/test_causal_scenario_evaluator_boundary.py -q
```

Expected: public imports fail until `__init__.py` is updated.

- [ ] **Step 3: Export the C1 API**

Modify `trade_rl/evaluation/__init__.py` with explicit imports and sorted `__all__`. Do not export private validation/statistics helpers or the rollout factory protocol if the repository's public API convention excludes protocols; the stable minimum is the list in Step 1.

- [ ] **Step 4: Run focused C1 and architecture suites**

Run:

```bash
pytest tests/evaluation/test_causal_scenario_*.py \
  tests/architecture/test_causal_scenario_evaluator_boundary.py -q
```

Expected: all pass.

Commit:

```bash
git add trade_rl/evaluation/__init__.py \
  tests/evaluation/test_causal_scenario_artifact.py \
  tests/architecture/test_causal_scenario_evaluator_boundary.py
git commit -m "feat: publish causal scenario evaluator API"
```

---

### Task 6: Complete verification and prepare the isolated C1 pull request

**Files:**
- Modify: `docs/superpowers/plans/2026-07-25-causal-scenario-action-evaluator-c1.md` to check completed steps only after fresh evidence exists.
- Do not modify training configuration, Serving bundles, workflow behavior, or production gates.

**Interfaces:**
- Consumes: complete C1 implementation.
- Produces: exact-head verification evidence and a reviewable PR.

- [ ] **Step 1: Run focused tests and both-module branch coverage**

Run:

```bash
pytest tests/evaluation/test_causal_scenario_*.py -q
pytest tests/evaluation/test_causal_scenario_*.py \
  --cov=trade_rl.evaluation.causal_scenario_values \
  --cov=trade_rl.evaluation.causal_scenario_artifact \
  --cov-branch \
  --cov-report=term-missing \
  --cov-fail-under=100
```

Expected: all focused tests pass and both production modules report 100% statement and branch coverage.

- [ ] **Step 2: Run static and architecture gates**

Run:

```bash
ruff check .
ruff format --check .
mypy .
python scripts/check_import_architecture.py
python scripts/report_dead_code.py
python -m compileall -q trade_rl tests
```

Expected: every command exits zero.

- [ ] **Step 3: Run complete repository tests**

Run:

```bash
pytest -q
```

Expected: complete suite passes with no new skip or xfail introduced by C1.

- [ ] **Step 4: Run packaged and cross-platform CI gates**

On the exact PR head, require:

- standard CI Rebuilt Core all steps;
- Windows compatibility;
- Ubuntu compatibility;
- complete training-image build and packaged non-root probe;
- PostgreSQL Catalog workflow when triggered by the changed paths;
- critical branch-coverage ratchets;
- CLI smoke.

Record workflow run IDs, exact head SHA, test counts, total coverage, focused coverage, and artifact digests in the PR body.

- [ ] **Step 5: Review the diff for prohibited dependencies**

Run:

```bash
git diff --name-only main...HEAD
git grep -n "causal_scenario" -- \
  trade_rl/rl trade_rl/serving trade_rl/release trade_rl/workflows \
  examples/binance-multitimeframe/walk-forward-full.json
```

Expected:

- changed production paths are limited to `trade_rl/evaluation`;
- no maintained training, Serving, release, promotion, or configuration path imports or invokes C1;
- design and plan documents are the only documentation changes.

- [ ] **Step 6: Mark plan steps complete, commit evidence, and open draft PR**

After all local gates are green, update checkboxes supported by actual evidence and commit:

```bash
git add docs/superpowers/plans/2026-07-25-causal-scenario-action-evaluator-c1.md
git commit -m "docs: record causal scenario C1 verification"
```

Open a draft PR titled:

```text
feat: add causal scenario action evaluator contracts
```

The PR body must state:

- C1 is evaluation-only;
- no historical scenario library or walk-forward integration is included;
- no Phase A teacher, BC, PPO, Serving, or production integration exists;
- exact-head test and coverage evidence;
- production remains `NO-GO`.

- [ ] **Step 7: Merge only after exact-head checks and review are clean**

Before merge:

- no unresolved review threads;
- PR is mergeable against latest `main`;
- exact head SHA still matches the verified SHA;
- all required checks conclude `success`;
- no new commit appeared after verification.

Use squash merge and record the resulting `main` commit SHA.
