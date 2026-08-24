# Causal Alpha V6 Fast-First Research Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run an artifact-bound V6 fast-first teacher research path through fixed Signal, paired economic Selection, and untouched Admission.

**Architecture:** Reuse the immutable V4 causal fit and forecast. Compile two per-symbol target candidates (`fast_only` and `fast_slow_retention`) that differ only in the slow retention filter, replay them under identical execution inputs, select by after-cost wealth and robustness, and open holdout only after upstream gates pass. V4 and V5 do not import V6.

**Tech Stack:** Python 3.12, NumPy, pytest, Ruff, Mypy, import-linter, PostgreSQL-backed Binance runtime, Docker training image.

**Spec:** `docs/implementation-plans/specs/2026-08-25-causal-alpha-v6-fast-first-design.md`

## Global Constraints

- Reward remains `scale * log(net_equity_after / net_equity_before)` after execution costs.
- Long/short state, PnL, reward, execution, liquidity, and risk remain independent per symbol.
- Train scope remains exactly nine symbols and Signal remains eight independent episodes / 72 scopes.
- V4 and V5 code, artifacts, schemas, gates, and results remain unchanged.
- No symbol-ID lookup, symbol-specific intercept, cross-sectional action ranking, holdout tuning, or one-minute prediction input.
- Fixed V6 target values are exactly those in the spec.
- Admission failure performs zero BC and RL updates and publishes no package.
- Docker runs as non-root with two numerical threads and at most 7.63 GiB memory.
- Every code task follows red-green-refactor and ends with a focused commit.

---

## File structure

- `trade_rl/learning/causal_alpha_v6.py`: V6 enums, frozen config, target-path artifact contract.
- `trade_rl/learning/causal_alpha_v6_target.py`: deterministic fast proposal, slow retention filter, state machine, and target compilation.
- `trade_rl/workflows/universal_causal_alpha_v6_signal.py`: target-liveness scope metrics and V4-fast-bound Signal gate.
- `trade_rl/workflows/universal_causal_alpha_v6_replay.py`: per-symbol/episode economic and holding evidence.
- `trade_rl/workflows/universal_causal_alpha_v6_selection.py`: candidate summaries, common eligibility, and paired selection.
- `trade_rl/workflows/universal_causal_alpha_v6_admission.py`: selected-vs-baseline untouched holdout gate.
- `trade_rl/workflows/universal_causal_alpha_v6_artifact_store.py`: V6-named immutable store and run lock.
- `trade_rl/workflows/universal_causal_alpha_v6_pipeline.py`: fail-closed stage order and package publication.
- `trade_rl/workflows/universal_causal_alpha_v6_runner.py`: exact config parser and stable CLI exit codes.
- `trade_rl/workflows/universal_causal_alpha_v6_stage_entry.py`: DB/artifact preparation, sequential cutoff fits, replays, and diagnostics.
- `trade_rl/workflows/universal_causal_alpha_v6_stage_execution.py`: narrow import boundary for the concrete entry.
- `scripts/run_universal_causal_alpha_v6_research.py`: executable CLI.
- `examples/binance/universal-causal-alpha-v6-research.json`: exact authored V6 config.
- `tests/learning/test_causal_alpha_v6.py`: contract validation and digest tests.
- `tests/learning/test_causal_alpha_v6_target.py`: hand-computable transition tests.
- `tests/workflows/test_universal_causal_alpha_v6_{signal,replay,selection,admission,pipeline,runner,stage_entry}.py`: workflow evidence and orchestration tests.
- `tests/architecture/test_causal_alpha_v6_boundaries.py`: version/import/training-boundary enforcement.

---

### Task 1: V6 configuration and target artifact contracts

**Files:**
- Create: `trade_rl/learning/causal_alpha_v6.py`
- Create: `tests/learning/test_causal_alpha_v6.py`

**Interfaces:**
- Produces: `CausalAlphaV6Candidate`, `CausalAlphaV6SlowState`, `CausalAlphaV6TargetConfig`, `CausalAlphaV6TargetPath`.
- `CausalAlphaV6TargetPath` exposes readonly decision, target, objective, cap, confirmation, slow-state, and reason evidence plus a content digest.

- [ ] **Step 1: Write failing exact-config and malformed-artifact tests**

```python
def test_v6_target_config_is_exact_and_digest_stable() -> None:
    config = CausalAlphaV6TargetConfig()
    assert config.target_magnitudes == (0.0, 0.025, 0.05, 0.10, 0.25)
    assert config.maximum_absolute_target == 0.25
    assert config.maximum_target_delta == 0.125
    assert config.fast_rebalance_decisions == 4
    assert config.slow_context_decisions == 16
    assert config.confirmation_count == 2
    assert config.strong_reversal_threshold == 0.02
    assert config.digest == CausalAlphaV6TargetConfig().digest


def test_v6_target_path_rejects_unaccounted_reason() -> None:
    with pytest.raises(ValueError, match="reasons"):
        _path(reasons=("not_a_v6_reason",))
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `uv run pytest -q tests/learning/test_causal_alpha_v6.py`

Expected: collection fails because `trade_rl.learning.causal_alpha_v6` does not exist.

- [ ] **Step 3: Implement frozen types and payload validation**

```python
class CausalAlphaV6Candidate(str, Enum):
    FAST_ONLY = "fast_only"
    FAST_SLOW_RETENTION = "fast_slow_retention"


class CausalAlphaV6SlowState(str, Enum):
    FLAT = "flat"
    SUPPORTIVE = "supportive"
    MIXED = "mixed"
    OPPOSED = "opposed"


@dataclass(frozen=True, slots=True)
class CausalAlphaV6TargetConfig:
    target_magnitudes: tuple[float, ...] = (0.0, 0.025, 0.05, 0.10, 0.25)
    maximum_absolute_target: float = 0.25
    maximum_target_delta: float = 0.125
    fast_rebalance_decisions: int = 4
    slow_context_decisions: int = 16
    uncertainty_multiplier: float = 1.0
    execution_cost_multiplier: float = 1.5
    edge_margin: float = 0.001
    confirmation_count: int = 2
    strong_reversal_threshold: float = 0.02
    liquidity_lookback_decisions: int = 96
    liquidity_lower_quantile: float = 0.10
    liquidity_safety_multiplier: float = 0.80
```

Implement `to_payload()` and `digest`; reject every value that differs from the
spec. In `CausalAlphaV6TargetPath.__post_init__`, copy arrays to canonical
readonly dtypes, require one reason and one slow state per decision, recompute
reason counts, validate non-negative caps/uncertainties, and bind all arrays with
`content_and_arrays_digest`.

- [ ] **Step 4: Run tests, Ruff, and Mypy**

Run:

```powershell
uv run pytest -q tests/learning/test_causal_alpha_v6.py
uv run ruff check trade_rl/learning/causal_alpha_v6.py tests/learning/test_causal_alpha_v6.py
uv run mypy --follow-imports=skip trade_rl/learning/causal_alpha_v6.py tests/learning/test_causal_alpha_v6.py
```

Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add trade_rl/learning/causal_alpha_v6.py tests/learning/test_causal_alpha_v6.py
git commit -m "feat: define causal alpha v6 target contracts"
```

---

### Task 2: Fast-first compiler and slow retention state machine

**Files:**
- Create: `trade_rl/learning/causal_alpha_v6_target.py`
- Create: `tests/learning/test_causal_alpha_v6_target.py`

**Interfaces:**
- Consumes: `CausalAlphaV4Forecast`, `CausalAlphaV6Candidate`, `CausalAlphaV6TargetConfig`.
- Produces:

```python
def causal_alpha_v6_target_path(
    forecast: CausalAlphaV4Forecast,
    *,
    uncertainty: Mapping[str, np.ndarray],
    one_way_cost_rates: object,
    liquidity_weight_caps: object,
    actionable_mask: object,
    candidate: CausalAlphaV6Candidate,
    config: CausalAlphaV6TargetConfig,
    initial_weight: float,
    risk_weight_caps: object | None = None,
) -> CausalAlphaV6TargetPath: ...
```

- [ ] **Step 1: Add RED tests for the complete transition oracle**

Write independent tests proving:

```python
assert _targets(fast=[0.03, 0.03], initial=0.0)[-1] > 0.0
assert _targets(fast=[-0.03, -0.03], initial=0.0)[-1] < 0.0
assert _targets(fast=[0.03], initial=0.0)[-1] == 0.0  # first confirmation
assert _retention(fast=[-0.005], slow=[0.02, 0.03], initial=0.10)[-1] == 0.10
assert abs(_retention(fast=[0.03], slow=[-0.02, -0.03], initial=0.10)[-1]) <= 0.10
assert _strong_reversal(fast=-0.03, initial=0.10) < 0.10
assert _zero_liquidity(initial=0.10)[-1] == 0.0
```

Also assert long/short mirror symmetry, no add under mixed/opposed slow state,
cost suppression, uncertainty suppression, maximum delta, non-zero initial
weight, cadence holds, missing actionability, reason precedence, and identical
paths between candidates while flat until a position exists.

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run pytest -q tests/learning/test_causal_alpha_v6_target.py`

Expected: import or symbol failure for `causal_alpha_v6_target_path`.

- [ ] **Step 3: Implement small domain functions**

Implement focused functions (each under 50 lines):

```python
def causal_alpha_v6_slow_state(previous: float, p24: float, p72: float) -> CausalAlphaV6SlowState: ...
def causal_alpha_v6_fast_candidates(previous: float, cap: float, config: CausalAlphaV6TargetConfig) -> tuple[float, ...]: ...
def causal_alpha_v6_fast_objective(previous: float, target: float, mu: float, sigma: float, cost: float, config: CausalAlphaV6TargetConfig) -> float: ...
def causal_alpha_v6_retention_allows(previous: float, proposed: float, state: CausalAlphaV6SlowState, confirmed_reversal: bool) -> bool: ...
```

The main compiler applies override precedence before economic proposal logic,
updates pending direction/count only on fast cadence, and emits every evidence
array required by `CausalAlphaV6TargetPath`. Do not call V5 calibration or V5
target functions.

- [ ] **Step 4: Run targeted regression and static checks**

```powershell
uv run pytest -q tests/learning/test_causal_alpha_v6.py tests/learning/test_causal_alpha_v6_target.py tests/learning/test_causal_alpha_v4_target.py tests/learning/test_causal_alpha_v5_target.py
uv run ruff check trade_rl/learning/causal_alpha_v6.py trade_rl/learning/causal_alpha_v6_target.py tests/learning/test_causal_alpha_v6*.py
uv run mypy --follow-imports=skip trade_rl/learning/causal_alpha_v6.py trade_rl/learning/causal_alpha_v6_target.py
```

Expected: all pass and V4/V5 results remain unchanged.

- [ ] **Step 5: Commit**

```powershell
git add trade_rl/learning/causal_alpha_v6*.py tests/learning/test_causal_alpha_v6*.py
git commit -m "feat: compile fast-first v6 target paths"
```

---

### Task 3: V4-fast-bound V6 Signal evidence

**Files:**
- Create: `trade_rl/workflows/universal_causal_alpha_v6_signal.py`
- Create: `tests/workflows/test_universal_causal_alpha_v6_signal.py`

**Interfaces:**
- Produces `CausalAlphaV6SignalScopeMetric`, `CausalAlphaV6CandidateSignalEvidence`, `CausalAlphaV6SignalEvidence`.
- Produces:

```python
def build_causal_alpha_v6_signal_scope_metric(...) -> CausalAlphaV6SignalScopeMetric: ...
def evaluate_causal_alpha_v6_signal_gate(
    metrics: tuple[CausalAlphaV6SignalScopeMetric, ...],
    *,
    expected_symbols: tuple[str, ...],
    v4_fast_lane: CausalAlphaV4LaneSignalEvidence,
) -> CausalAlphaV6SignalEvidence: ...
```

- [ ] **Step 1: Write RED scope/gate tests**

Build exactly 144 metrics (72 per candidate) and assert pass only when:

```python
assert evidence.raw_scope_count_per_candidate == 72
assert evidence.independent_episode_count == 8
assert evidence.symbol_count == 9
assert evidence.v4_fast_lane_passed
assert evidence.fast_only.non_flat_target_count > 0
```

Parametrize missing symbol, missing episode, duplicate identity, unaccounted
reason, drifted initial weight, target/contract mismatch, all-flat fast baseline,
and failed V4 fast lane. Each mutation must produce its named rejection reason.

- [ ] **Step 2: Verify RED**

Run: `uv run pytest -q tests/workflows/test_universal_causal_alpha_v6_signal.py`

- [ ] **Step 3: Implement immutable evidence and the fixed gate**

Keep scope metrics compact: persist counts and digests, not full target arrays.
Bind candidate, symbol, episode, contract interval/digest, fit/forecast/target
digests, initial weight, reason counts, actionability, non-flat count, change
count, and flip count. `CausalAlphaV6SignalEvidence.to_payload()` must expose
scalar values needed for later audit rather than only nested digests.

- [ ] **Step 4: Validate**

```powershell
uv run pytest -q tests/workflows/test_universal_causal_alpha_v6_signal.py tests/workflows/test_universal_causal_alpha_v4_signal.py
uv run ruff check trade_rl/workflows/universal_causal_alpha_v6_signal.py tests/workflows/test_universal_causal_alpha_v6_signal.py
uv run mypy --follow-imports=skip trade_rl/workflows/universal_causal_alpha_v6_signal.py
```

- [ ] **Step 5: Commit**

```powershell
git add trade_rl/workflows/universal_causal_alpha_v6_signal.py tests/workflows/test_universal_causal_alpha_v6_signal.py
git commit -m "feat: gate v6 on the admitted fast signal"
```

---

### Task 4: V6 economic replay evidence

**Files:**
- Create: `trade_rl/workflows/universal_causal_alpha_v6_replay.py`
- Create: `tests/workflows/test_universal_causal_alpha_v6_replay.py`

**Interfaces:**
- Consumes: `CausalAlphaV6TargetPath`, maintained `RolloutEvaluation`, runtime/context/contract identities.
- Produces: `CausalAlphaV6ReplayMetric` and `build_causal_alpha_v6_replay_metric(...)`.

- [ ] **Step 1: Write RED accounting tests**

Use hand-computable evaluations to assert:

```python
assert metric.gross_wealth == pytest.approx(math.exp(metric.gross_return))
assert metric.net_wealth == pytest.approx(math.exp(metric.net_return))
assert metric.reward_total == pytest.approx(metric.net_return * reward_scale)
assert metric.completed_holding_durations_hours == (12.0,)
assert metric.open_holding_duration_hours == 6.0
```

Cover all-flat, one open position, closed long, closed short, flip, rejected
order, hard-risk termination, target/evaluation length mismatch, and candidate
identity drift.

- [ ] **Step 2: Verify RED**

Run: `uv run pytest -q tests/workflows/test_universal_causal_alpha_v6_replay.py`

- [ ] **Step 3: Implement replay metric using maintained evaluation fields**

Port the proven V5 holding attribution without importing a V5 type. Add
`candidate`, `target_digest`, `fit_digest`, `forecast_digest`, and all upstream
identities. Recompute reason counts and reject reward/net-return mismatches.

- [ ] **Step 4: Validate**

```powershell
uv run pytest -q tests/workflows/test_universal_causal_alpha_v6_replay.py tests/workflows/test_universal_causal_alpha_v5_replay.py
uv run ruff check trade_rl/workflows/universal_causal_alpha_v6_replay.py tests/workflows/test_universal_causal_alpha_v6_replay.py
uv run mypy --follow-imports=skip trade_rl/workflows/universal_causal_alpha_v6_replay.py
```

- [ ] **Step 5: Commit**

```powershell
git add trade_rl/workflows/universal_causal_alpha_v6_replay.py tests/workflows/test_universal_causal_alpha_v6_replay.py
git commit -m "feat: record v6 after-cost replay evidence"
```

---

### Task 5: Paired train-only Selection

**Files:**
- Create: `trade_rl/workflows/universal_causal_alpha_v6_selection.py`
- Create: `tests/workflows/test_universal_causal_alpha_v6_selection.py`

**Interfaces:**
- Produces `CausalAlphaV6CandidateSelectionEvidence`, `CausalAlphaV6SelectionEvidence`.
- Produces:

```python
def evaluate_causal_alpha_v6_selection(
    metrics: tuple[CausalAlphaV6ReplayMetric, ...],
    *,
    expected_symbols: tuple[str, ...],
) -> CausalAlphaV6SelectionEvidence: ...
```

- [ ] **Step 1: Write RED economic and dominance tests**

Assert common eligibility rejects balanced gross/net wealth at or below one,
any symbol aggregate net wealth below one, median below one, positive scope
fraction below 0.5, turnover p95 above 1.0, no meaningful execution, risk, or
unexplained rejection.

Assert selection behavior:

```python
assert _selection(fast_pass=True, retention_pass=False).selected_candidate is FAST_ONLY
assert _selection(fast_pass=False, retention_pass=True).selected_candidate is FAST_SLOW_RETENTION
assert _dominating_retention().selected_candidate is FAST_SLOW_RETENTION
assert _higher_net_but_more_turnover().selected_candidate is FAST_ONLY
assert _both_failed().passed is False
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest -q tests/workflows/test_universal_causal_alpha_v6_selection.py`

- [ ] **Step 3: Implement balanced summaries and exact dominance**

Use sums of per-scope log returns per symbol, geometric balanced wealth across
symbols, median symbol wealth, scope positive fraction, CVaR10, and quantiles.
Require exact candidate/scope pairing by `(symbol, episode, contract_digest,
fit_digest, forecast_digest)` before comparing. Persist both complete candidate
summaries and the selected candidate/config digest.

- [ ] **Step 4: Validate**

```powershell
uv run pytest -q tests/workflows/test_universal_causal_alpha_v6_selection.py tests/workflows/test_universal_causal_alpha_v5_selection.py
uv run ruff check trade_rl/workflows/universal_causal_alpha_v6_selection.py tests/workflows/test_universal_causal_alpha_v6_selection.py
uv run mypy --follow-imports=skip trade_rl/workflows/universal_causal_alpha_v6_selection.py
```

- [ ] **Step 5: Commit**

```powershell
git add trade_rl/workflows/universal_causal_alpha_v6_selection.py tests/workflows/test_universal_causal_alpha_v6_selection.py
git commit -m "feat: select v6 by paired symbol economics"
```

---

### Task 6: Paired untouched Admission

**Files:**
- Create: `trade_rl/workflows/universal_causal_alpha_v6_admission.py`
- Create: `tests/workflows/test_universal_causal_alpha_v6_admission.py`

**Interfaces:**
- Produces `CausalAlphaV6AdmissionEvidence`.
- Produces:

```python
def evaluate_causal_alpha_v6_admission(
    selected_records: tuple[CausalAlphaV6ReplayMetric, ...],
    baseline_records: tuple[CausalAlphaV6ReplayMetric, ...],
    *,
    signal_evidence: CausalAlphaV6SignalEvidence,
    selection_evidence: CausalAlphaV6SelectionEvidence,
    fit_knowledge_cutoff: int,
    holdout_start: int,
) -> CausalAlphaV6AdmissionEvidence: ...
```

- [ ] **Step 1: Write RED fail-closed tests**

Cover upstream gate bypass, cutoff mismatch, non-unique/missing nine symbols,
selected candidate mismatch, non-paired baseline contracts, aggregate gross/net
at or below zero, fewer than six positive symbols, worst symbol below `-0.02`,
risk/rejection, and retention underperforming fast-only. Assert fast-only selected
records may share the baseline tuple and pass without artificial uplift.

- [ ] **Step 2: Verify RED**

Run: `uv run pytest -q tests/workflows/test_universal_causal_alpha_v6_admission.py`

- [ ] **Step 3: Implement exact holdout identity and summary validation**

Bind Signal/Selection digests, selected candidate/config, all fit/forecast/target/
contract identities, paired fast-only digest, cutoff, aggregate statistics,
positive-symbol count, worst symbol, execution/risk counts, and rejection reasons.
Set `promotion_eligible=False` even when research Admission passes.

- [ ] **Step 4: Validate**

```powershell
uv run pytest -q tests/workflows/test_universal_causal_alpha_v6_admission.py tests/workflows/test_universal_causal_alpha_v5_admission.py
uv run ruff check trade_rl/workflows/universal_causal_alpha_v6_admission.py tests/workflows/test_universal_causal_alpha_v6_admission.py
uv run mypy --follow-imports=skip trade_rl/workflows/universal_causal_alpha_v6_admission.py
```

- [ ] **Step 5: Commit**

```powershell
git add trade_rl/workflows/universal_causal_alpha_v6_admission.py tests/workflows/test_universal_causal_alpha_v6_admission.py
git commit -m "feat: add paired untouched v6 admission"
```

---

### Task 7: Immutable pipeline, store, and rejection semantics

**Files:**
- Create: `trade_rl/workflows/universal_causal_alpha_v6_artifact_store.py`
- Create: `trade_rl/workflows/universal_causal_alpha_v6_pipeline.py`
- Create: `tests/workflows/test_universal_causal_alpha_v6_pipeline.py`

**Interfaces:**
- Produces `CausalAlphaV6ArtifactStore`, `CausalAlphaV6RunLock`, `CausalAlphaV6StageRejected`, `CausalAlphaV6ResearchPackage`.
- Pipeline stages are exactly `prepare -> signal -> selection -> admission`.

- [ ] **Step 1: Write RED order/publication tests**

For each stage rejection, assert later callbacks are not called, the completed
stage evidence and terminal result exist, and `package.json` does not. Assert a
pass writes all evidence, diagnostics, result, and package exactly once. Assert
rerunning into the same immutable leaf fails.

- [ ] **Step 2: Verify RED**

Run: `uv run pytest -q tests/workflows/test_universal_causal_alpha_v6_pipeline.py`

- [ ] **Step 3: Implement by adapting proven V5 atomic primitives**

Use V6 schema names and no calibration stage. Stable rejection exit mapping is
Signal `2`, Selection `3`, Admission `4`, invalid preparation/config `5`.
`CausalAlphaV6ResearchPackage` binds selected candidate/config plus Signal,
Selection, Admission, run, context, and generator digests.

- [ ] **Step 4: Validate**

```powershell
uv run pytest -q tests/workflows/test_universal_causal_alpha_v6_pipeline.py tests/workflows/test_universal_causal_alpha_v5_pipeline.py
uv run ruff check trade_rl/workflows/universal_causal_alpha_v6_artifact_store.py trade_rl/workflows/universal_causal_alpha_v6_pipeline.py
uv run mypy --follow-imports=skip trade_rl/workflows/universal_causal_alpha_v6_artifact_store.py trade_rl/workflows/universal_causal_alpha_v6_pipeline.py
```

- [ ] **Step 5: Commit**

```powershell
git add trade_rl/workflows/universal_causal_alpha_v6_artifact_store.py trade_rl/workflows/universal_causal_alpha_v6_pipeline.py tests/workflows/test_universal_causal_alpha_v6_pipeline.py
git commit -m "feat: orchestrate fail-closed v6 research"
```

---

### Task 8: Artifact-bound concrete stage execution

**Files:**
- Create: `trade_rl/workflows/universal_causal_alpha_v6_stage_entry.py`
- Create: `trade_rl/workflows/universal_causal_alpha_v6_stage_execution.py`
- Create: `tests/workflows/test_universal_causal_alpha_v6_stage_entry.py`

**Interfaces:**
- Consumes the V4 runtime adapter and `prepare_causal_alpha_v4_stage_data`.
- Produces:

```python
def run_causal_alpha_v6_concrete_entry(
    *,
    config_path: Path,
    run_config_path: Path,
    runtime_manifest_path: Path,
    v4_context_manifest_path: Path,
    frozen_metadata_root: Path,
    output_root: Path,
) -> CausalAlphaV6ResearchPackage: ...
```

- [ ] **Step 1: Write RED tests for fit timing, paired inputs, and memory ownership**

Mock the runtime adapter and assert superseded `context`, `runtime`, and
`prepared_v3` weak references are dead before Signal. Record fit cutoffs and
assert Signal fits only Signal starts, Selection fits only economic starts, and
Admission first fits at holdout start. Assert each candidate shares the exact
same forecast, costs, caps, contract, environment initial state, and evaluation
range. Assert failed Signal invokes no economic replay.

- [ ] **Step 2: Verify RED**

Run: `uv run pytest -q tests/workflows/test_universal_causal_alpha_v6_stage_entry.py`

- [ ] **Step 3: Implement sequential stage assemblers**

Factor the file into focused private functions for preparation, one-cutoff fit,
one-symbol target bundle, one replay, Signal, Selection, Admission, and
diagnostic persistence. For every cutoff:

```python
fit = fit_causal_alpha_v4(..., knowledge_cutoff=cutoff)
try:
    # predict/replay one symbol at a time and retain only compact evidence
    ...
finally:
    del fit
    gc.collect()
```

Do not fit or load Admission before Selection passes. Print one sorted JSON
progress line per cutoff/candidate/stage. Release superseded preparation inputs
before entering the run lock.

- [ ] **Step 4: Validate focused and V4/V5 compatibility tests**

```powershell
uv run pytest -q tests/workflows/test_universal_causal_alpha_v6_stage_entry.py tests/workflows/test_universal_causal_alpha_v4_fitting.py tests/workflows/test_universal_causal_alpha_v5_stage_entry.py
uv run ruff check trade_rl/workflows/universal_causal_alpha_v6_stage_entry.py trade_rl/workflows/universal_causal_alpha_v6_stage_execution.py
uv run mypy --follow-imports=skip trade_rl/workflows/universal_causal_alpha_v6_stage_entry.py trade_rl/workflows/universal_causal_alpha_v6_stage_execution.py
```

- [ ] **Step 5: Commit**

```powershell
git add trade_rl/workflows/universal_causal_alpha_v6_stage_*.py tests/workflows/test_universal_causal_alpha_v6_stage_entry.py
git commit -m "feat: execute v6 stages on artifact-bound data"
```

---

### Task 9: Exact runner, CLI, configuration, and architecture boundaries

**Files:**
- Create: `trade_rl/workflows/universal_causal_alpha_v6_runner.py`
- Create: `scripts/run_universal_causal_alpha_v6_research.py`
- Create: `examples/binance/universal-causal-alpha-v6-research.json`
- Create: `tests/workflows/test_universal_causal_alpha_v6_runner.py`
- Create: `tests/architecture/test_causal_alpha_v6_boundaries.py`
- Modify: `pyproject.toml`
- Modify: `docs/ARCHITECTURE.md`

**Interfaces:**
- Config parser accepts only exact ordered fields and fixed values from the spec.
- CLI delegates through `run_causal_alpha_v6_stage_entry` and returns stable stage codes.

- [ ] **Step 1: Write RED parser, CLI, and import-boundary tests**

Parametrize missing, unknown, reordered, boolean-coerced, and changed config
fields. Assert V4/V5 cannot import `causal_alpha_v6`, learning cannot import
workflow/framework code, and serving cannot import V6 research code. Assert the
script works when invoked from outside repository cwd.

- [ ] **Step 2: Verify RED**

```powershell
uv run pytest -q tests/workflows/test_universal_causal_alpha_v6_runner.py tests/architecture/test_causal_alpha_v6_boundaries.py
```

- [ ] **Step 3: Implement exact JSON and narrow CLI**

Use this top-level JSON shape and reject any drift:

```json
{
  "schema_version": "universal_causal_alpha_v6_research_config_v1",
  "target": {
    "target_magnitudes": [0.0, 0.025, 0.05, 0.1, 0.25],
    "maximum_absolute_target": 0.25,
    "maximum_target_delta": 0.125,
    "fast_rebalance_decisions": 4,
    "slow_context_decisions": 16,
    "uncertainty_multiplier": 1.0,
    "execution_cost_multiplier": 1.5,
    "edge_margin": 0.001,
    "confirmation_count": 2,
    "strong_reversal_threshold": 0.02,
    "liquidity_lookback_decisions": 96,
    "liquidity_lower_quantile": 0.1,
    "liquidity_safety_multiplier": 0.8
  }
}
```

Add the V6 workflow layer to import-linter without weakening existing contracts.
Document V6 as research-only and non-serving.

- [ ] **Step 4: Validate**

```powershell
uv run pytest -q tests/workflows/test_universal_causal_alpha_v6_runner.py tests/architecture/test_causal_alpha_v6_boundaries.py
uv run ruff check scripts/run_universal_causal_alpha_v6_research.py trade_rl/workflows/universal_causal_alpha_v6_runner.py tests/workflows/test_universal_causal_alpha_v6_runner.py
uv run mypy --follow-imports=skip trade_rl/workflows/universal_causal_alpha_v6_runner.py scripts/run_universal_causal_alpha_v6_research.py
uv run lint-imports
```

- [ ] **Step 5: Commit**

```powershell
git add pyproject.toml docs/ARCHITECTURE.md examples/binance/universal-causal-alpha-v6-research.json scripts/run_universal_causal_alpha_v6_research.py trade_rl/workflows/universal_causal_alpha_v6_runner.py tests/workflows/test_universal_causal_alpha_v6_runner.py tests/architecture/test_causal_alpha_v6_boundaries.py
git commit -m "feat: expose the causal alpha v6 research runner"
```

---

### Task 10: Full verification, Docker real-data run, and terminal report

**Files:**
- Create after terminal: `report/causal-alpha-v6-<generation>-20260825.md`
- Modify only if a verified defect is found: the smallest V6 source/test files that own the defect.

**Interfaces:**
- Produces a clean exact-head commit, provenance-bound image, immutable run root, and evidence-backed terminal report.

- [ ] **Step 1: Run the complete V4/V5/V6 targeted suite**

```powershell
uv run pytest -q tests/learning/test_causal_alpha_v4_*.py tests/learning/test_causal_alpha_v5_*.py tests/learning/test_causal_alpha_v6*.py tests/workflows/test_universal_causal_alpha_v4_*.py tests/workflows/test_universal_causal_alpha_v5_*.py tests/workflows/test_universal_causal_alpha_v6_*.py tests/architecture/test_causal_alpha_v6_boundaries.py
```

Expected: all collected tests pass. If PowerShell wildcard expansion differs,
resolve the same file set with `Get-ChildItem` and pass the explicit paths.

- [ ] **Step 2: Run static and architecture validation**

```powershell
uv run ruff check trade_rl/learning/causal_alpha_v6*.py trade_rl/workflows/universal_causal_alpha_v6*.py scripts/run_universal_causal_alpha_v6_research.py tests/learning/test_causal_alpha_v6*.py tests/workflows/test_universal_causal_alpha_v6*.py tests/architecture/test_causal_alpha_v6_boundaries.py
uv run mypy --follow-imports=skip trade_rl/learning/causal_alpha_v6.py trade_rl/learning/causal_alpha_v6_target.py trade_rl/workflows/universal_causal_alpha_v6_signal.py trade_rl/workflows/universal_causal_alpha_v6_replay.py trade_rl/workflows/universal_causal_alpha_v6_selection.py trade_rl/workflows/universal_causal_alpha_v6_admission.py trade_rl/workflows/universal_causal_alpha_v6_pipeline.py trade_rl/workflows/universal_causal_alpha_v6_runner.py trade_rl/workflows/universal_causal_alpha_v6_stage_entry.py
uv run lint-imports
git diff --check
git status --short
```

- [ ] **Step 3: Commit the verified implementation and compute provenance**

```powershell
git commit -am "feat: complete causal alpha v6 research"
git rev-parse HEAD
uv run python -c "from trade_rl.artifacts.provenance import source_tree_digest; print(source_tree_digest('.'))"
```

Use `git add` first if new files remain. Require an empty `git status --short`.

- [ ] **Step 4: Build the provenance-bound training image**

Build `docker/Dockerfile.training` with exact `TRADE_RL_GIT_COMMIT`,
`TRADE_RL_GIT_DIRTY=false`, source-tree digest, lockfile digest, and frozen runtime
manifest digest. Record the resulting immutable image digest and labels. The
Dockerfile's torch compile, lock, source, provenance, and non-root probes must
pass.

- [ ] **Step 5: Launch the DB-backed V6 run**

Use the same `trade_rl_default` network, `trade-rl-training-data` volume,
read-only universal artifact bind, frozen metadata root, V4 context manifest,
thread limits, and non-root `trainer` user as V5 r7. Use a fresh output root and
container name. The command is:

```text
python scripts/run_universal_causal_alpha_v6_research.py
  --config examples/binance/universal-causal-alpha-v6-research.json
  --run-config examples/binance-multitimeframe/universal-u6-ppo.json
  --runtime-manifest /workspace/var/universal/runtime-manifest.json
  --v4-context-manifest /workspace/var/v4-context/causal-alpha-v4-prod-20260824-r10/manifest.json
  --frozen-metadata-root /workspace/var/cache/frozen-metadata/usds-m
  --output-root /workspace/var/runs/<fresh-generation>
```

- [ ] **Step 6: Monitor and repair only implementation defects**

Inspect per-cutoff/candidate progress, Docker memory, OOM state, stage evidence,
and scalar diagnostics. For crashes or invalid evidence, use systematic debugging,
write a RED regression, fix the owning code, rebuild a new exact-head image, and
run a fresh generation. Do not alter the fixed target, gate, reward, symbols,
episodes, or holdout based on observed outcomes.

- [ ] **Step 7: Classify the terminal honestly**

- Signal rejection: preserve diagnostics; do not replay Selection.
- Selection rejection: preserve both candidate economic evidence; do not open holdout.
- Admission rejection: preserve paired nine-symbol holdout; perform zero BC/RL.
- Admission pass: publish the research package and immediately write the separate
  admitted-teacher BC/RL implementation plan before training.

- [ ] **Step 8: Write and commit the report**

The report must include branch, exact run commit, report commit, source/lock/
runtime/context/image/run/config/evidence digests, container/output root,
stage counts, candidate and per-symbol economics, reward/net-return equality,
turnover/cost/holding evidence, gate reasons, OOM state, tests, and explicit
non-claims. Commit with:

```powershell
git add report/causal-alpha-v6-*-20260825.md
git commit -m "docs: report causal alpha v6 research"
```

---

## Self-review

- Spec coverage: every architecture, target, transition, Signal, paired
  Selection, Admission, durability, memory, artifact, and one-minute-data
  requirement maps to Tasks 1-10.
- Placeholder scan: no TBD/TODO or unspecified implementation/test step remains.
- Type consistency: candidate, config, target, Signal, replay, Selection,
  Admission, pipeline, runner, and entrypoint names are consistent across tasks.
- Scope split: this plan ends at a real Admission terminal. BC/RL integration is
  intentionally a second plan created only from an admitted package, so a failed
  research candidate cannot leak into learner training.

