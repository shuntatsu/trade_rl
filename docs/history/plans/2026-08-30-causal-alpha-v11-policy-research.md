# Causal Alpha V11 Policy Research Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute behavior-neutral V9 diagnostics plus independent V11 L1/E1/C1/S1 policy experiments, then preserve their evidence for one consolidated report.

**Architecture:** V11 owns immutable candidate/config/diagnostic evidence while reusing V9 fitting, V10 lifecycle-aware replay, V4 artifact storage, and unchanged V8 numerical selection gates. A sequential trace policy emits precomputed actions and model metadata so exact V9 control can be compared digest-for-digest with r21 before variants are judged.

**Tech Stack:** Python 3.12, NumPy, pytest, Ruff, MyPy, PostgreSQL-backed Binance artifacts, Docker training-runtime.

**Spec:** `docs/implementation-plans/specs/2026-08-30-causal-alpha-v11-policy-research-design.md`

## Global Constraints

- Do not modify V10 schema, r21 artifacts, reward, fees, runtime risk thresholds, existing gates, symbols, or 15-minute execution.
- Do not use symbol identity or symbol-specific calibration.
- Treat current train-symbol folds as development evidence and do not open untouched Admission unless one candidate is locked.
- Every artifact must carry schema, input identity, content digest, and restart-safe resume validation.
- Use tests-first for every production change and run focused pytest, Ruff, format check, MyPy, and `git diff --check` after each task.

---

### Task 1: V11 immutable policy contracts

**Files:**
- Create: `trade_rl/learning/causal_alpha_v11.py`
- Create: `tests/learning/test_causal_alpha_v11.py`

**Interfaces:**
- Produces: `CausalAlphaV11StudyArm`, `CausalAlphaV11Candidate`, `CausalAlphaV11Config`, `CausalAlphaV11TargetPath`, and `CausalAlphaV11SizingFeasibility`.
- Consumes: `CausalAlphaV9Config`, `CausalAlphaV6TargetPath`, SHA-256 artifact helpers.

- [ ] **Step 1: Write failing immutable-contract tests**

```python
def test_v11_candidates_are_fixed_and_v10_is_not_extended() -> None:
    assert tuple(item.value for item in CausalAlphaV11Candidate) == (
        "v8_cash_sanity", "v9_control", "treatment",
    )
    assert tuple(item.value for item in CausalAlphaV11StudyArm) == (
        "neutral_expiry_2", "after_cost_entry",
        "sign_calibrated_entry", "calibrated_edge_sizing",
    )

def test_v11_sizing_preflight_rejects_subthreshold_targets() -> None:
    result = evaluate_v11_sizing_feasibility(
        targets=np.asarray([0.0, 0.04, 0.099]),
        entry_threshold=0.1,
        no_trade_band=0.05,
    )
    assert not result.executable
    assert result.rejection_reasons == ("entry_threshold",)
```

- [ ] **Step 2: Run RED**

Run: `.venv/Scripts/python.exe -m pytest -q tests/learning/test_causal_alpha_v11.py`

Expected: import failure because V11 contracts do not exist.

- [ ] **Step 3: Implement the minimal immutable contracts**

```python
class CausalAlphaV11Candidate(str, Enum):
    V8_CASH_SANITY = "v8_cash_sanity"
    V9_CONTROL = "v9_control"
    TREATMENT = "treatment"

class CausalAlphaV11StudyArm(str, Enum):
    NEUTRAL_EXPIRY_2 = "neutral_expiry_2"
    AFTER_COST_ENTRY = "after_cost_entry"
    SIGN_CALIBRATED_ENTRY = "sign_calibrated_entry"
    CALIBRATED_EDGE_SIZING = "calibrated_edge_sizing"

@dataclass(frozen=True, slots=True)
class CausalAlphaV11Config:
    neutral_expiry_count: int = 2
    calibration_hours: int = 168
    calibration_ridge_strength: float = 1.0
    sizing_epsilon: float = 1e-12
```

Require fixed constants and digest all payloads. Bind target artifacts to V9 fit, forecast, V11 config, policy input, and optional calibration digest.

- [ ] **Step 4: Run GREEN and validation**

Run: `.venv/Scripts/python.exe -m pytest -q tests/learning/test_causal_alpha_v11.py`

Run: `.venv/Scripts/python.exe -m ruff check trade_rl/learning/causal_alpha_v11.py tests/learning/test_causal_alpha_v11.py`

Run: `.venv/Scripts/python.exe -m mypy trade_rl/learning/causal_alpha_v11.py`

- [ ] **Step 5: Commit**

```powershell
git add trade_rl/learning/causal_alpha_v11.py tests/learning/test_causal_alpha_v11.py
git commit -m "feat: define causal alpha v11 policy contracts"
```

### Task 2: Exact V9 trace policy and independent compilers

**Files:**
- Create: `trade_rl/learning/causal_alpha_v11_policy.py`
- Create: `tests/learning/test_causal_alpha_v11_policy.py`

**Interfaces:**
- Consumes: V9 head predictions, costs, caps, actionable mask, initial weight, optional sign calibration.
- Produces: `compile_causal_alpha_v11_target` returning `CausalAlphaV11CompiledTarget`, and `CausalAlphaV11TracePolicy` with `predict()` and `last_step_trace_metadata`.

- [ ] **Step 1: Write failing behavior and independence tests**

```python
def test_v11_control_actions_equal_v9_exactly() -> None:
    v9 = causal_alpha_v9_wave_target_path(**inputs)
    v11 = compile_causal_alpha_v11_target(
        candidate=CausalAlphaV11Candidate.V9_CONTROL, **inputs
    )
    np.testing.assert_array_equal(v11.target.v6_target_path.targets, v9.targets)

def test_v11_l1_exits_after_two_neutral_cadences() -> None:
    target = compile_causal_alpha_v11_target(
        study_arm=CausalAlphaV11StudyArm.NEUTRAL_EXPIRY_2,
        head_predictions=heads_with_entry_then_two_neutral_observations(),
        **fixed_inputs,
    )
    assert target.target.v6_target_path.targets[48] == 0.0
    assert "neutral_expiry_2" in target.target.v6_target_path.reasons

def test_v11_e1_filters_entry_below_round_trip_cost() -> None:
    target = compile_causal_alpha_v11_target(
        study_arm=CausalAlphaV11StudyArm.AFTER_COST_ENTRY,
        head_predictions=heads_with_raw_edge(0.0012),
        one_way_cost_rates=np.full(rows, 0.0007),
        **fixed_inputs_without_costs,
    )
    assert np.all(target.target.v6_target_path.targets == 0.0)
```

- [ ] **Step 2: Run RED**

Run: `.venv/Scripts/python.exe -m pytest -q tests/learning/test_causal_alpha_v11_policy.py`

- [ ] **Step 3: Implement one shared state machine with candidate-specific qualification/exit hooks**

The compiler accepts an optional study arm, aligned decision indices, head
predictions, one-way costs, liquidity/risk caps, actionable mask, source
forecast and wave-fit digests, immutable V11 config, initial weight, and an
optional sign calibration. It returns one immutable compiled target whose
arrays all have the decision count and whose digest binds every input.

The control path must call the existing V9 compiler and compare arrays/reasons before returning. Metadata arrays must be immutable and aligned to every decision.

- [ ] **Step 4: Add trace-policy tests**

```python
action, _ = policy.predict({"current_weights": np.asarray([0.0])})
assert action.shape == (1,)
assert policy.last_step_trace_metadata["fast_qualified_direction"] in (-1, 0, 1)
assert "after_cost_entry_objective" in policy.last_step_trace_metadata
```

- [ ] **Step 5: Run GREEN and validation**

Run: `.venv/Scripts/python.exe -m pytest -q tests/learning/test_causal_alpha_v11_policy.py tests/learning/test_causal_alpha_v9_wave.py`

Run Ruff and MyPy on both V11 files and tests.

- [ ] **Step 6: Commit**

```powershell
git add trade_rl/learning/causal_alpha_v11_policy.py tests/learning/test_causal_alpha_v11_policy.py
git commit -m "feat: add causal alpha v11 policy experiments"
```

### Task 3: Leak-free pooled sign calibration

**Files:**
- Create: `trade_rl/learning/causal_alpha_v11_calibration.py`
- Create: `tests/learning/test_causal_alpha_v11_calibration.py`

**Interfaces:**
- Produces: `CausalAlphaV11SignCalibration` and `fit_causal_alpha_v11_sign_calibration`.
- Consumes: pooled V9 training rows, a source fit whose cutoff equals calibration start, current outer cutoff, fixed V11 config.

- [ ] **Step 1: Write failing causality, pooling, and prediction tests**

```python
def test_v11_sign_calibration_is_pooled_causal_and_symbol_free() -> None:
    fit = fit_causal_alpha_v11_sign_calibration(
        rows={"A": rows_a, "B": rows_b}, source_fit=prior_fit,
        outer_cutoff=1000, config=CausalAlphaV11Config(),
    )
    assert fit.maximum_label_end_index < 1000
    assert fit.long_support > 0 and fit.short_support > 0
    assert not hasattr(fit, "symbol_coefficients")

def test_v11_sign_calibration_digest_changes_with_coefficients() -> None:
    assert first.digest != replace(first, long_coefficients=(9.0, 9.0), digest="").digest
```

- [ ] **Step 2: Run RED**

Run: `.venv/Scripts/python.exe -m pytest -q tests/learning/test_causal_alpha_v11_calibration.py`

- [ ] **Step 3: Implement fixed one-week out-of-sample calibration**

Select rows in `[outer_cutoff - 168*4, outer_cutoff)`, require label end below `outer_cutoff`, predict with the source fit trained at calibration start, calculate `raw_edge=abs(mean)-std-edge_margin`, and fit `[intercept, slope]` separately for positive and negative agreed directions with ridge `1.0`.

- [ ] **Step 4: Run GREEN and validation**

Run focused pytest, Ruff, and MyPy.

- [ ] **Step 5: Commit**

```powershell
git add trade_rl/learning/causal_alpha_v11_calibration.py tests/learning/test_causal_alpha_v11_calibration.py
git commit -m "feat: add pooled v11 sign calibration"
```

### Task 4: D1 and entry-quality evidence

**Files:**
- Create: `trade_rl/learning/causal_alpha_v11_diagnostics.py`
- Create: `tests/learning/test_causal_alpha_v11_diagnostics.py`

**Interfaces:**
- Produces: `CausalAlphaV11TradeDecomposition`, `CausalAlphaV11DiagnosticEvidence`, and `build_causal_alpha_v11_diagnostics`.
- Consumes: exact-control step/lifecycle traces, aligned 4h labels, costs, actionable mask, symbol/episode identity.

- [ ] **Step 1: Write failing segment and entry-oracle tests**

```python
def test_d1_splits_trade_at_first_neutral_cadence() -> None:
    evidence = build_causal_alpha_v11_diagnostics(trace=trace_with_one_trade, **identity)
    trade = evidence.trades[0]
    assert trade.entry_index == 16
    assert trade.first_neutral_index == 48
    assert trade.exit_index == 80
    assert trade.entry_to_neutral_net_log_return == expected_before
    assert trade.neutral_to_exit_net_log_return == expected_after

def test_entry_edge_uses_directional_4h_label_minus_round_trip_cost() -> None:
    assert evidence.entries[0].entry_edge == pytest.approx(0.02 - 2 * 0.0007)
```

- [ ] **Step 2: Run RED**

Run: `.venv/Scripts/python.exe -m pytest -q tests/learning/test_causal_alpha_v11_diagnostics.py`

- [ ] **Step 3: Implement immutable per-scope and aggregate evidence**

Use lifecycle `entry`/`exit`, cadence-aligned qualified direction, `np.log1p` step economics, realized exposure-hours, and trace turnover. Persist pooled/long/short and symbol summaries with mean, median, positive fraction, CVaR10, MAE, MFE, net/exposure-hour, and net/turnover.

- [ ] **Step 4: Run GREEN and validation**

Run focused pytest, Ruff, and MyPy.

- [ ] **Step 5: Commit**

```powershell
git add trade_rl/learning/causal_alpha_v11_diagnostics.py tests/learning/test_causal_alpha_v11_diagnostics.py
git commit -m "feat: add v11 trade diagnostics"
```

### Task 5: V11 experiment gates and artifacts

**Files:**
- Create: `trade_rl/workflows/universal_causal_alpha_v11_gates.py`
- Create: `tests/workflows/test_universal_causal_alpha_v11_gates.py`

**Interfaces:**
- Produces: `CausalAlphaV11SelectionEvidence` and `evaluate_causal_alpha_v11_selection` for exactly one study arm.
- Consumes: one cash metric set, one exact-control metric set, one treatment metric set, diagnostics, and sizing feasibility.

- [ ] **Step 1: Write failing gate tests**

```python
def test_v11_keeps_one_treatment_in_an_independent_three_way_gate() -> None:
    evidence = evaluate_causal_alpha_v11_selection(
        study_arm=CausalAlphaV11StudyArm.NEUTRAL_EXPIRY_2,
        cash_metrics=cash_metrics,
        control_metrics=control_metrics,
        treatment_metrics=treatment_metrics,
        diagnostics=diagnostics,
        sizing_feasibility=None,
    )
    assert tuple(item.candidate for item in evidence.candidates) == (
        CausalAlphaV11Candidate.V8_CASH_SANITY,
        CausalAlphaV11Candidate.V9_CONTROL,
        CausalAlphaV11Candidate.TREATMENT,
    )

def test_v11_binds_the_study_arm_to_terminal_evidence() -> None:
    assert evidence.study_arm is CausalAlphaV11StudyArm.NEUTRAL_EXPIRY_2
```

- [ ] **Step 2: Run RED**

Run: `.venv/Scripts/python.exe -m pytest -q tests/workflows/test_universal_causal_alpha_v11_gates.py`

- [ ] **Step 3: Implement V11-owned wrapper evidence over unchanged V8 gates**

Map cash/control/treatment metrics to the three existing gate identities inside one study-arm run. Restore V11 names in the outer payload, bind every source digest and study-arm digest, and persist S1 as preflight-rejected when non-executable.

- [ ] **Step 4: Run GREEN and validation**

Run focused pytest, Ruff, and MyPy.

- [ ] **Step 5: Commit**

```powershell
git add trade_rl/workflows/universal_causal_alpha_v11_gates.py tests/workflows/test_universal_causal_alpha_v11_gates.py
git commit -m "feat: add v11 independent selection gates"
```

### Task 6: Restart-safe DB-backed V11 runner

**Files:**
- Create: `trade_rl/workflows/universal_causal_alpha_v11_stage_entry.py`
- Create: `scripts/run_universal_causal_alpha_v11_research.py`
- Create: `tests/workflows/test_universal_causal_alpha_v11_stage_entry.py`
- Create: `tests/scripts/test_run_universal_causal_alpha_v11_research.py`

**Interfaces:**
- Produces: `run_causal_alpha_v11_selection` with an explicit `study_arm`, returning `CausalAlphaV11SelectionEvidence`; CLI exit `0` means eligible development Selection, `3` numerical rejection, `4` preregistered skip/preflight stop, and `5` execution failure.
- Consumes: V10 split-run provenance, V9 training/fit helpers, V11 compilers/policies/diagnostics/gates, V4 artifact store.

- [ ] **Step 1: Write failing stage identity and resume tests**

```python
def test_v11_leaf_schema_requires_candidate_policy_and_trace_digests() -> None:
    assert stage._REPLAY_LEAF_SCHEMA == "causal_alpha_v11_replay_leaf_v1"

def test_v11_control_rejects_any_r21_economic_drift() -> None:
    with pytest.raises(ValueError, match="behavior-neutral control drifted"):
        stage._require_control_equivalence(v11_metric, changed_r21_metric)
```

- [ ] **Step 2: Run RED**

Run focused workflow and script tests.

- [ ] **Step 3: Implement shared fit/control and variant replay orchestration**

Prepare Signal and Selection with the same dual-run binding rules as r21. For
D1, regenerate the exact V9 target and signal arrays from the bound fit and
forecast, require the regenerated target digest to equal the r21 leaf target
digest, and then join the signal arrays to the authoritative r21 step economics
by decision index. For a study-arm run, rebuild exact V9 control with metadata,
compare it with r21 by symbol/episode, write one treatment leaf for the requested
study arm, and evaluate the three-way gate only after every expected leaf exists.

- [ ] **Step 4: Implement CLI**

Use the existing V10 arguments plus required `--r21-output-root` and `--study-arm`. Emit a single JSON terminal status and never catch `KeyboardInterrupt` or bypass failed artifact validation.

- [ ] **Step 5: Run GREEN and validation**

Run all V11 tests, related V9/V10 tests, Ruff, format check, targeted MyPy, and `git diff --check`.

- [ ] **Step 6: Commit**

```powershell
git add trade_rl/workflows/universal_causal_alpha_v11_stage_entry.py scripts/run_universal_causal_alpha_v11_research.py tests/workflows/test_universal_causal_alpha_v11_stage_entry.py tests/scripts/test_run_universal_causal_alpha_v11_research.py
git commit -m "feat: add db-backed causal alpha v11 research"
```

### Task 7: Docker execution and artifact audit

**Files:**
- Modify only if validation finds a tested defect; otherwise no source change.

- [ ] **Step 1: Build a provenance-bound training image**

Build from a clean commit with revision, source-tree, lockfile, and runtime-manifest digests. Record the image manifest.

- [ ] **Step 2: Launch one new V11 output root**

Use DB network `trade_rl_default`, volume `trade-rl-training-data`, frozen runtime/V4 manifests, split Signal/Selection configs, and `flat_start_activation`. Use a distinct output root for D1/L1/E1/C1/S1 and never resume one arm into another.

- [ ] **Step 3: Monitor and validate intermediate evidence**

At each persisted scope, parse the leaf, verify outer/internal digests, exact-control equivalence, trace availability, return/cost reconciliation, hard-risk evidence, and resume identity.

- [ ] **Step 4: Apply stop conditions**

If no candidate passes, do not open Admission. Mark S1 preflight outcome. Start a separately specified V12 only after V11 terminal evidence exists. Do not launch V13 without both a passed 15-minute candidate and real one-minute data.

### Task 8: Single consolidated report

**Files:**
- Create: `report/gpt-causal-alpha-r8-v13-consolidated.md`

- [ ] **Step 1: Consolidate prior and new evidence**

Include r8-r21 history, V11 design identity, D1 pooled/long/short/symbol results, entry-quality metrics, each candidate gate, S1 feasibility, any V12 evidence, V13 prerequisites, commits, image digests, artifact paths, test output, and explicit non-claims.

- [ ] **Step 2: Verify report facts against artifacts**

Recompute all displayed totals from JSON leaves and evidence. Search for stale r20/r21 paths, placeholder text, contradictory gate claims, and any statement that confuses development evidence with untouched validation.

- [ ] **Step 3: Commit the report**

```powershell
git add report/gpt-causal-alpha-r8-v13-consolidated.md
git commit -m "docs: consolidate causal alpha policy research"
```
