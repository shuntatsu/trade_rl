# Universal Cost-Aware Causal Teacher Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add decision-time cost-aware target admission, signal-quality diagnostics, correctly classified execution evidence, and strict train-only economic selection to the Universal causal-alpha teacher without changing reward or weakening any downstream gate.

**Architecture:** Keep model fitting and pure controller state transitions in `trade_rl.learning`; keep production-environment extraction, chronological replay, selection, checkpoints, and artifacts in `trade_rl.workflows`. Preserve the historical v1 r3 checkpoint as read-only evidence, introduce explicit v2 identities for corrected candidates, and require a new provenance-bound Docker generation before any corrected real-data result is admitted.

**Tech Stack:** Python 3.12, NumPy, Gymnasium-compatible Universal environment, dataclasses, canonical SHA-256 content digests, pytest, Ruff, mypy, Docker/CUDA.

## Global Constraints

- Reward stays exactly `100 * log(net_equity_after / net_equity_before)`; no scalar cost, baseline, or drawdown penalty is added.
- Fee, spread, impact, funding, and borrow costs continue to enter reward exactly once through net equity.
- No minimum holding duration is introduced; a qualified strong reversal may execute on the next decision.
- Fitting, diagnostics, controller selection, BC, and critic warm-start use train symbols only; the latest train-symbol episodes, validation symbols, and test symbols remain untouched by selection.
- Corrected selection uses 12 predeclared candidates, mean net return `>= 0`, minimum symbol-episode net return `>= -0.05`, mean turnover/day `<= 1.0`, zero unexplained execution rejections, meaningful trades, no hard-risk failure, and no majority-negative gross result.
- An inactive cash policy cannot pass through low turnover alone.
- r3 artifacts remain historical v1 evidence and cannot resume under corrected v2 candidate or checkpoint identity.
- A failed teacher or BC gate performs zero critic warm-start and zero PPO updates.
- Windows `uv` commands are serialized against the shared `.venv`.

---

### Task 1: Implement the pure cost-aware controller contract

**Files:**
- Modify: `trade_rl/learning/causal_alpha_teacher.py`
- Modify: `tests/learning/test_causal_alpha_teacher.py`

**Interfaces:**
- Consumes: existing `CausalAlphaControllerConfig`, combined score arrays, one-way execution-cost-rate arrays, initial episode weight, and actionable mask.
- Produces: `CausalAlphaCostAwareConfig`, `CausalAlphaCostAwareTargetPath`, and `causal_alpha_cost_aware_target_path(...)` for corrected candidate replay.

- [ ] **Step 1: Write failing validation and digest tests**

Add tests that require this public contract:

```python
config = CausalAlphaCostAwareConfig(
    execution_cost_multiplier=1.5,
    edge_margin=0.001,
    confirmation_count=2,
    strong_reversal_threshold=0.02,
    max_abs_target=0.5,
)
assert config.digest == CausalAlphaCostAwareConfig(**asdict(config)).digest
with pytest.raises(ValueError):
    CausalAlphaCostAwareConfig(
        execution_cost_multiplier=0.0,
        edge_margin=0.001,
        confirmation_count=2,
        strong_reversal_threshold=0.02,
        max_abs_target=0.5,
    )
```

- [ ] **Step 2: Run the focused test and prove RED**

Run:

```powershell
uv run pytest tests/learning/test_causal_alpha_teacher.py -q
```

Expected: collection or assertion failure because `CausalAlphaCostAwareConfig` does not exist.

- [ ] **Step 3: Add immutable v1 cost-aware configuration and path evidence**

Implement these exact fields and validation:

```python
CAUSAL_ALPHA_COST_AWARE_SCHEMA: Final = "causal_alpha_cost_aware_v1"

@dataclass(frozen=True, slots=True)
class CausalAlphaCostAwareConfig:
    execution_cost_multiplier: float
    edge_margin: float
    confirmation_count: int
    strong_reversal_threshold: float
    max_abs_target: float
    schema_version: str = CAUSAL_ALPHA_COST_AWARE_SCHEMA

    @property
    def digest(self) -> str:
        return content_digest(self)

@dataclass(frozen=True, slots=True)
class CausalAlphaCostAwareTargetPath:
    initial_weight: float
    targets: np.ndarray
    proposed_turnover: np.ndarray
    predicted_incremental_edge: np.ndarray
    estimated_cost_hurdle: np.ndarray
    edge_to_cost_ratio: np.ndarray
    confirmation_state: np.ndarray
    cost_suppressed_change_count: int
    submitted_change_count: int
    strong_reversal_count: int
    sign_flip_count: int
    actionable_mask: np.ndarray
    digest: str = ""
```

Reject non-finite values, multiplier `<= 0`, margin `< 0`, confirmation count `< 1`, strong threshold `<= 0`, and exposure cap outside `(0, 1]`. Copy arrays, make them read-only, and include every field in the digest payload.

- [ ] **Step 4: Write failing economic-admission state-machine tests**

Cover these exact behaviors:

```python
path = causal_alpha_cost_aware_target_path(
    np.asarray([0.003, 0.003, -0.004, -0.03]),
    one_way_cost_rates=np.full(4, 0.0009),
    controller=base_controller,
    economic=config,
    initial_weight=0.0,
)
assert path.targets[0] == 0.0              # confirmation 1/2
assert path.targets[1] > 0.0               # confirmed entry
assert path.targets[2] >= 0.0              # weak reversal suppressed
assert path.targets[3] < path.targets[2]   # immediate strong reversal
assert path.cost_suppressed_change_count >= 1
```

Also prove that a marginal score is suppressed when
`score * delta <= abs(delta) * (cost_rate * multiplier + edge_margin)`, a stronger score is admitted, inactive decisions hold state, exposure never exceeds `max_abs_target`, and there is no hidden elapsed-time/24h lock.

- [ ] **Step 5: Implement the minimal deterministic state machine**

Use the existing `_desired_target` for the unconstrained desired exposure, clip it to the economic exposure cap, and evaluate:

```python
delta = desired - previous
incremental_edge = float(score) * delta
hurdle = abs(delta) * (
    float(cost_rate) * economic.execution_cost_multiplier
    + economic.edge_margin
)
qualifies = delta != 0.0 and incremental_edge > hurdle
```

Maintain consecutive same-direction qualification state. Ordinary changes require `confirmation_count`; a sign reversal with `abs(score) >= strong_reversal_threshold` bypasses only the confirmation count, not the cost hurdle, no-trade band, max target delta, exposure cap, actionable mask, or production risk projection.

- [ ] **Step 6: Verify GREEN and focused quality gates**

Run:

```powershell
uv run pytest tests/learning/test_causal_alpha_teacher.py -q
uv run ruff check trade_rl/learning/causal_alpha_teacher.py tests/learning/test_causal_alpha_teacher.py
uv run mypy trade_rl/learning/causal_alpha_teacher.py
```

Expected: all commands exit 0.

- [ ] **Step 7: Commit Task 1**

Run the maintained commit helper:

```powershell
& 'C:\Program Files\Git\bin\bash.exe' 'C:/Users/shun/.codex/skills/git-pushing/scripts/smart_commit.sh' 'feat: add cost-aware causal target controller'
```

---

### Task 2: Separate execution rejection and risk-projection evidence

**Files:**
- Modify: `trade_rl/learning/evaluation.py`
- Modify: `trade_rl/learning/rollout_evaluation.py`
- Modify: `trade_rl/learning/episode_oracle_bc.py`
- Modify: `tests/learning/test_learning_evaluation.py`
- Modify: `tests/learning/test_rollout_evaluation.py`
- Modify: `tests/learning/test_episode_bc_holdout_aggregation.py`

**Interfaces:**
- Consumes: `StatefulExecutionResult.order_events`, `StatefulExecutionResult.rejected_count`, and `info["hybrid_risk"].reasons`.
- Produces: deterministic reason-count tuples on `ActionPathCollapseEvidence`; candidate selection consumes them in Task 4.

- [ ] **Step 1: Write failing reason-classification tests**

Extend the rollout fixture with one rejected order event carrying `reason="minimum_notional"` and one projected target carrying `reasons=("no_trade_band",)`. Assert:

```python
assert evidence.execution_rejection_count == 1
assert evidence.execution_rejection_reason_counts == (("minimum_notional", 1),)
assert evidence.risk_projection_reason_counts == (("no_trade_band", 1),)
assert evidence.hard_risk_violation is False
```

Add a mismatch test where rejected events do not sum to `rejected_count`; it must raise rather than fabricate an `unknown` reason.

- [ ] **Step 2: Run tests and prove RED**

```powershell
uv run pytest tests/learning/test_rollout_evaluation.py tests/learning/test_learning_evaluation.py -q
```

Expected: failure because the reason fields are absent.

- [ ] **Step 3: Add backward-compatible fields and canonical count helpers**

Append fields with empty defaults so existing diagnostic constructors remain source compatible:

```python
execution_rejection_reason_counts: tuple[tuple[str, int], ...] = ()
risk_projection_reason_counts: tuple[tuple[str, int], ...] = ()
hard_risk_violation: bool = False
```

Normalize counts by non-empty reason, sort lexically, reject duplicate keys, non-positive counts, and any execution-count mismatch. Include the fields in `to_dict()`.

- [ ] **Step 4: Collect exact per-step reasons in rollout evaluation**

For each step, inspect `execution.order_events` and count only events whose `event_type == "rejected"`. Require `event.reason` to be non-empty. Count `hybrid_risk.reasons` separately. Do not infer a hard-risk failure from projections or rejected orders; a thrown hard-risk invariant exception still fails the replay immediately.

- [ ] **Step 5: Preserve reason counts through aggregate BC evidence**

Update `_aggregate_collapse_evidence` and the other explicit constructors in `episode_oracle_bc.py` to merge reason tuples by key and OR `hard_risk_violation`. Add aggregation tests proving two episodes with the same reason produce a summed count.

- [ ] **Step 6: Verify GREEN**

```powershell
uv run pytest tests/learning/test_rollout_evaluation.py tests/learning/test_learning_evaluation.py tests/learning/test_episode_bc_holdout_aggregation.py -q
uv run ruff check trade_rl/learning/evaluation.py trade_rl/learning/rollout_evaluation.py trade_rl/learning/episode_oracle_bc.py tests/learning
uv run mypy trade_rl/learning/evaluation.py trade_rl/learning/rollout_evaluation.py trade_rl/learning/episode_oracle_bc.py
```

- [ ] **Step 7: Commit Task 2**

```powershell
& 'C:\Program Files\Git\bin\bash.exe' 'C:/Users/shun/.codex/skills/git-pushing/scripts/smart_commit.sh' 'feat: classify causal execution evidence'
```

---

### Task 3: Persist train-only signal diagnostics and v2 candidate evidence

**Files:**
- Create: `trade_rl/learning/causal_alpha_diagnostics.py`
- Create: `tests/learning/test_causal_alpha_diagnostics.py`
- Modify: `trade_rl/workflows/universal_causal_alpha_contracts.py`
- Modify: `trade_rl/workflows/universal_causal_alpha_teacher.py`
- Modify: `tests/workflows/test_universal_causal_alpha_selection.py`
- Modify: `tests/workflows/test_universal_causal_alpha_teacher.py`

**Interfaces:**
- Consumes: 24h/72h predictions, realized selection-episode labels, target-path diagnostics, and Task 2 reason counts.
- Produces: `CausalAlphaSignalDiagnostics`, `CausalAlphaCandidateEpisodeMetricsV2`, `CausalAlphaCandidateEvidenceV2`, `CausalAlphaSelectionEvidenceV2`, `write_causal_alpha_selection_checkpoint_metric_v2(...)`, `load_causal_alpha_selection_checkpoint_v2(...)`, and digest-bound progress payloads.

- [ ] **Step 1: Write failing deterministic diagnostics tests**

Define fixed quantiles `(0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)` and five fixed score bins based on prediction quantiles. Assert exact count conservation, means, sign accuracy, Pearson correlation, average-rank Spearman correlation with ties, and explicit undefined-correlation reason for a constant prediction.

```python
diagnostics = evaluate_causal_alpha_signal_diagnostics(predicted, realized)
assert sum(item.count for item in diagnostics.bins) == predicted.size
assert diagnostics.direction_accuracy == pytest.approx(expected_accuracy)
assert diagnostics.undefined_correlation_reason is None
```

- [ ] **Step 2: Run diagnostics tests and prove RED**

```powershell
uv run pytest tests/learning/test_causal_alpha_diagnostics.py -q
```

- [ ] **Step 3: Implement the pure diagnostics module**

Create frozen digest-bearing `CausalAlphaSignalBin` and `CausalAlphaSignalDiagnostics`. Implement average ranks in NumPy, with stable mergesort ordering and average tied ranks. Never replace an undefined coefficient with zero; store `pearson_correlation=None`, `rank_correlation=None`, and a non-empty reason.

- [ ] **Step 4: Write failing v2 evidence and checkpoint tests**

Require corrected metrics to contain:

```python
signal_24h: CausalAlphaSignalDiagnostics
signal_72h: CausalAlphaSignalDiagnostics
cost_suppressed_change_count: int
submitted_change_count: int
strong_reversal_count: int
command_sign_flip_count: int
execution_rejection_count: int
execution_rejection_reason_counts: tuple[tuple[str, int], ...]
risk_projection_reason_counts: tuple[tuple[str, int], ...]
hard_risk_violation: bool
```

Prove that a v1 r3 checkpoint can be loaded only through a historical loader and is rejected as resume input for a v2 grid. Prove that changing any diagnostic changes the metric and checkpoint digests.

- [ ] **Step 5: Add explicit v2 contracts and atomic checkpoint schema**

Keep the existing v1 class/loader available for historical r3 reporting. Introduce `CausalAlphaCandidateEpisodeMetricsV2`, `CausalAlphaCandidateEvidenceV2`, and `CausalAlphaSelectionEvidenceV2` plus checkpoint schema `causal_alpha_selection_checkpoint_metric_v2`. Implement explicit `write_causal_alpha_selection_checkpoint_metric_v2(...)` and `load_causal_alpha_selection_checkpoint_v2(...)` functions; the latter accepts only v2 rows and the expected grid digest. Do not silently coerce v1 `risk_violation` into a v2 hard-risk flag. Extend progress `episode_metric` with the complete v2 payload before the atomic progress write.

- [ ] **Step 6: Compute diagnostics only on earlier completed selection episodes**

In the selection replay, align realized 24h/72h labels with the exact prediction decision indices, reject incomplete labels, and assert every metric episode belongs to `selection_contracts`. The latest causal holdout digest, validation symbols, and test symbols must be unavailable to the diagnostics function; retain the existing leakage-sentinel tests and add a v2 sentinel.

- [ ] **Step 7: Verify GREEN**

```powershell
uv run pytest tests/learning/test_causal_alpha_diagnostics.py tests/workflows/test_universal_causal_alpha_selection.py tests/workflows/test_universal_causal_alpha_teacher.py -q
uv run ruff check trade_rl/learning/causal_alpha_diagnostics.py trade_rl/workflows/universal_causal_alpha_contracts.py trade_rl/workflows/universal_causal_alpha_teacher.py tests/learning/test_causal_alpha_diagnostics.py tests/workflows/test_universal_causal_alpha_selection.py tests/workflows/test_universal_causal_alpha_teacher.py
uv run mypy trade_rl/learning/causal_alpha_diagnostics.py trade_rl/workflows/universal_causal_alpha_contracts.py trade_rl/workflows/universal_causal_alpha_teacher.py
```

- [ ] **Step 8: Commit Task 3**

```powershell
& 'C:\Program Files\Git\bin\bash.exe' 'C:/Users/shun/.codex/skills/git-pushing/scripts/smart_commit.sh' 'feat: persist causal signal diagnostics'
```

---

### Task 4: Add the corrected 12-candidate grid and strict selection gates

**Files:**
- Modify: `trade_rl/workflows/universal_causal_alpha_selection.py`
- Modify: `trade_rl/workflows/universal_causal_alpha_fitting.py`
- Modify: `trade_rl/workflows/universal_causal_alpha_teacher.py`
- Modify: `trade_rl/workflows/universal_causal_alpha_contracts.py`
- Modify: `tests/workflows/test_universal_causal_alpha_selection.py`
- Modify: `tests/workflows/test_universal_causal_alpha_prediction_availability.py`

**Interfaces:**
- Consumes: Task 1 cost-aware controller, Task 2 execution evidence, Task 3 signal/v2 metrics, `ExecutionCostConfig`, dataset fee/spread arrays, and chronological contracts.
- Produces: `default_cost_aware_causal_alpha_candidate_grid(...)`, `CausalAlphaSelectionThresholds`, `rank_cost_aware_causal_alpha_candidates(...)`, and v2 selection/rejection evidence.

- [ ] **Step 1: Write failing cost-rate timing tests**

Add a synthetic regular-cadence dataset with distinct fee/spread values at decision and first-executable rows. Require the helper to use the first executable row from the maintained signal delay, never the decision row or a later holdout row.

```python
rates = causal_alpha_one_way_cost_rates(
    dataset,
    execution_cost,
    decision_indices=np.asarray([10, 14]),
    signal_delay_decisions=1,
    decision_bars=4,
)
assert rates[0] == pytest.approx(expected_row_15_rate)
```

The conservative deterministic rate includes configured and dataset fee rate, market/taker fee, full spread for market orders (half for limit), and configured impact at `sqrt(max_participation_rate)`. Stochastic slippage is excluded from teacher prediction and remains accounted by production replay.

- [ ] **Step 2: Run the timing test and prove RED**

```powershell
uv run pytest tests/workflows/test_universal_causal_alpha_prediction_availability.py -q
```

- [ ] **Step 3: Implement cost-rate extraction and wire Task 1 path generation**

Add `causal_alpha_one_way_cost_rates(...) -> np.ndarray` with explicit shape/finite/range checks. Pass the environment dataset and execution config into corrected target generation. Retain prediction caching; controller-only variants must not refit ridge or recompute predictions.

- [ ] **Step 4: Write the exact grid identity test**

Assert these 12 unique names and one-factor differences:

```python
assert tuple(item.name for item in grid) == (
    "cost-aware-baseline", "horizon-24h", "horizon-72h",
    "cost-multiplier-high", "edge-margin-high", "confirmation-one",
    "confirmation-three", "strong-reversal-low", "scale-low",
    "exposure-low", "no-trade-high", "delta-low",
)
```

Baseline values are equal horizon, multiplier `1.5`, margin `0.001`, confirmation `2`, strong reversal `0.02`, scale `25`, exposure cap `0.5`, no-trade `0.05`, delta `0.125`. Assert every non-baseline member changes exactly one serialized field.

- [ ] **Step 5: Implement candidate v2 identity without rewriting r3 history**

Add optional `economic_controller: CausalAlphaCostAwareConfig | None` to the candidate contract. Preserve schema-v1 digest behavior when it is `None`; use `causal_alpha_candidate_v2` and include the economic config digest when present. Corrected canonical selection requires all candidates to be v2.

- [ ] **Step 6: Write failing gate-reason tests**

Create one isolated fixture for each rejection reason:

```python
expected = {
    "hard_risk_violation",
    "unexplained_execution_rejection",
    "no_meaningful_trades",
    "negative_mean_net_return",
    "lower_tail_net_return_below_floor",
    "turnover_per_day_above_maximum",
    "majority_negative_gross_return",
}
```

Also prove that a zero-trade cash candidate fails, a high-net/high-turnover candidate fails, and ranking remains lower-tail net, mean net, turnover, cost only after admissibility.

- [ ] **Step 7: Implement immutable selection thresholds and fail-closed ranking**

```python
@dataclass(frozen=True, slots=True)
class CausalAlphaSelectionThresholds:
    minimum_mean_net_return: float = 0.0
    minimum_symbol_episode_net_return: float = -0.05
    maximum_mean_turnover_per_day: float = 1.0
    maximum_unexplained_execution_rejections: int = 0
```

Include its digest in v2 grid and selection evidence. Implement `rank_cost_aware_causal_alpha_candidates(...)` as the only v2 ranker; retain `rank_causal_alpha_candidates(...)` for historical v1 evidence/tests. Keep downstream teacher admission and BC thresholds unchanged.

- [ ] **Step 8: Verify GREEN and cache behavior**

```powershell
uv run pytest tests/workflows/test_universal_causal_alpha_selection.py tests/workflows/test_universal_causal_alpha_prediction_availability.py tests/workflows/test_universal_causal_alpha_teacher.py -q
uv run ruff check trade_rl/workflows/universal_causal_alpha_selection.py trade_rl/workflows/universal_causal_alpha_fitting.py trade_rl/workflows/universal_causal_alpha_teacher.py trade_rl/workflows/universal_causal_alpha_contracts.py tests/workflows
uv run mypy trade_rl/workflows/universal_causal_alpha_selection.py trade_rl/workflows/universal_causal_alpha_fitting.py trade_rl/workflows/universal_causal_alpha_teacher.py trade_rl/workflows/universal_causal_alpha_contracts.py
```

Add assertions that 12 controller candidates over the same symbol/contract use one prediction computation per ridge/horizon identity and do not regress to per-candidate pooled fits.

- [ ] **Step 9: Commit Task 4**

```powershell
& 'C:\Program Files\Git\bin\bash.exe' 'C:/Users/shun/.codex/skills/git-pushing/scripts/smart_commit.sh' 'feat: select cost-aware causal teachers'
```

---

### Task 5: Integrate v2 evidence into monitoring, artifacts, and report generation

**Files:**
- Modify: `trade_rl/operations/universal_training_monitor.py`
- Modify: `trade_rl/workflows/universal_causal_alpha_teacher.py`
- Modify: `trade_rl/integrations/universal_pretraining.py`
- Modify: `tests/operations/test_universal_training_monitor.py`
- Modify: `tests/workflows/test_universal_causal_alpha_teacher.py`
- Modify: `tests/integrations/test_universal_pretraining_bundle.py`
- Modify: `report/universal-real-data-training-2026-08-12.md`

**Interfaces:**
- Consumes: v2 progress/checkpoint/selection/rejection/admission/package artifacts.
- Produces: bounded live summaries and durable branch/commit/result/fix history; preserves zero-RL-on-failure.

- [ ] **Step 1: Write failing monitor and artifact tests**

Require bounded incremental reading of v2 checkpoint rows and trend summaries for score correlation, direction accuracy, predicted edge, cost hurdle, cost suppressions, execution rejection reasons, mean/lower-tail net, turnover, and cost. Assert the monitor never rereads the whole growing JSONL file after its cursor advances.

- [ ] **Step 2: Run tests and prove RED**

```powershell
uv run pytest tests/operations/test_universal_training_monitor.py tests/workflows/test_universal_causal_alpha_teacher.py tests/integrations/test_universal_pretraining_bundle.py -q
```

- [ ] **Step 3: Implement bounded v2 monitoring and immediate persistence**

Extend the existing cursor/streaming aggregate rather than adding a second scanner. Write v2 selection rejection immediately after all candidates, and teacher admission/package immediately after holdout replay. Verify every artifact digest before displaying it.

- [ ] **Step 4: Preserve pretraining fail-closed order**

Add integration assertions that failed teacher selection/admission writes all shared evidence and then raises before `pretrain_universal_policy`, critic warm-start, or SB3 `learn`. A passing teacher still reuses one shared package across PPO, Lagrangian, and Discounted Lagrangian.

- [ ] **Step 5: Update the report checkpoint format**

Append a generated/manual verified section containing branch, exact commits, image/source/lock/manifest digests, r3 historical outcome, corrected generation outcome, candidate and symbol tables, score diagnostics, rejection reasons, reward contract, admission decision, and whether random/BC/critic/RL ran.

- [ ] **Step 6: Verify GREEN**

```powershell
uv run pytest tests/operations/test_universal_training_monitor.py tests/workflows/test_universal_causal_alpha_teacher.py tests/integrations/test_universal_pretraining_bundle.py -q
uv run ruff check trade_rl/operations/universal_training_monitor.py trade_rl/workflows/universal_causal_alpha_teacher.py trade_rl/integrations/universal_pretraining.py tests/operations tests/workflows/test_universal_causal_alpha_teacher.py tests/integrations/test_universal_pretraining_bundle.py
uv run mypy trade_rl/operations/universal_training_monitor.py trade_rl/workflows/universal_causal_alpha_teacher.py trade_rl/integrations/universal_pretraining.py
git diff --check
```

- [ ] **Step 7: Commit Task 5**

```powershell
& 'C:\Program Files\Git\bin\bash.exe' 'C:/Users/shun/.codex/skills/git-pushing/scripts/smart_commit.sh' 'feat: report cost-aware teacher evidence'
```

---

### Task 6: Verify, rebuild with provenance, and continue real-data training gates

**Files:**
- Modify only if failures prove a root cause: files implicated by the failing test/runtime evidence.
- Update: `report/universal-real-data-training-2026-08-12.md`

**Interfaces:**
- Consumes: exact committed corrected head and existing runtime manifest `6726b3737df9fbacf6787f3d02894e846c512a840bec4dd037538a02af1480b0`.
- Produces: verified Docker image, corrected selection/admission artifacts, conditional CUDA stage smoke, final report commits, and pushed branch.

- [ ] **Step 1: Run the complete corrected causal surface**

```powershell
uv run pytest tests/learning/test_causal_alpha_teacher.py tests/learning/test_causal_alpha_diagnostics.py tests/learning/test_rollout_evaluation.py tests/learning/test_learning_evaluation.py tests/learning/test_episode_bc_holdout_aggregation.py tests/workflows/test_universal_causal_alpha_selection.py tests/workflows/test_universal_causal_alpha_prediction_availability.py tests/workflows/test_universal_causal_alpha_teacher.py tests/operations/test_universal_training_monitor.py tests/integrations/test_universal_pretraining_bundle.py -q
uv run ruff check .
uv run ruff format --check .
uv run mypy trade_rl
uv run lint-imports
```

Read every command's exit code and complete failure count. Do not describe the branch as green if a repository-wide gate fails; distinguish new failures from already documented platform-specific failures with exact evidence.

- [ ] **Step 2: Run the maintained full test/coverage gate**

Run the exact coverage command maintained in `.github/workflows/ci.yml`:

```powershell
uv run pytest -q --cov=trade_rl --cov-branch --cov-report=term-missing --cov-report=json:coverage.json
```

Record passed, skipped, failed, and coverage totals in the report.

- [ ] **Step 3: Commit and push the exact source head before image build**

```powershell
& 'C:\Program Files\Git\bin\bash.exe' 'C:/Users/shun/.codex/skills/git-pushing/scripts/smart_commit.sh' 'feat: complete cost-aware causal teacher correction'
```

Require a clean worktree and upstream equality before computing provenance.

- [ ] **Step 4: Build and verify a provenance-bound training image**

Compute exact Git commit, source-tree digest, `uv.lock` digest, and runtime-manifest digest. Build `Dockerfile.training` target `training-runtime` with all four labels, inspect them from the resulting image, and record the image ID. Do not reuse the dirty r3 image for corrected evidence.

- [ ] **Step 5: Preserve r3, then launch exactly one corrected teacher generation**

Allow r3 to complete or stop it only after its durable checkpoint and latest progress are copied/verified. Check for any existing corrected container/generation before launch. Mount the same `trade-rl-training-data` volume and read-only Universal artifacts, keep CVAT stopped while Docker memory remains 7.63 GiB, and launch one new container name/generation.

- [ ] **Step 6: Monitor corrected selection to completion**

At each checkpoint, verify completed/total, symbol, episode, fit/prediction cache counts, score correlations, direction accuracy, edge/hurdle suppressions, gross/net, lower-tail, turnover/day, cost, trade count, rejection reasons, hard-risk state, CPU/RAM, OOM, and digest continuity. Resume only the same generation from a verified v2 checkpoint after interruption.

- [ ] **Step 7: Apply the conditional admission boundary**

If no candidate is admissible, verify `causal-teacher-selection-rejected.json`, aggregate the exact rejection reasons, update report/commit/push, and form the next evidence-based hypothesis without starting BC or RL. If selection passes, verify selection, admission, and package digests and require teacher admission before CUDA smoke.

- [ ] **Step 8: Run real-data CUDA random -> BC -> critic -> RL smoke only after admission**

Evaluate the same segment at random initialization, after BC, after critic warm-start, and after every PPO rollout/checkpoint. Record reward mean, `reward_growth_raw`, portfolio/baseline values, rolling and baseline excess, gross/net PnL, interval cost, filled turnover, fills/trades, drawdown, target delta/sign flips, command target delta/sign flips, NaN/OOM/constraints, and for Lagrangian completed episodes, warm-up, dual-update count, and multiplier history.

- [ ] **Step 9: Decide canonical training from economic evidence**

Do not use reward mean alone. Require non-worsening baseline gap, sane cost/turnover, no BC-to-RL net degradation, no NaN/OOM/constraint anomaly, and valid Lagrangian updates. Only then run three algorithms x three seeds x 524,288 timesteps through the maintained audits.

- [ ] **Step 10: Final report, verification, commit, and push**

Update `report/universal-real-data-training-2026-08-12.md` with the final branch, every corrective commit, result that motivated it, artifacts/digests, candidate and stage time series, gate decisions, and final status. Run `git diff --check`, verify the report facts against artifacts, commit with the maintained helper, push `codex/universal-real-data-training`, and verify upstream equality.
