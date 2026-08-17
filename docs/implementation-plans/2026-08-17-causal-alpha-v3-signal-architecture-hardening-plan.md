# Causal Alpha V3 Signal Architecture Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the Causal Alpha V3 Signal sidecar architecture so the paired producer is statically typed, the canonical metric-only API is independent from diagnostic construction, and persisted forecast observations are cryptographically rebound to `forecast_digest` without changing canonical Signal numerics.

**Architecture:** Keep the current per-scope sidecar schema and file layout. Introduce one private shared Signal-computation object/helper in `universal_causal_alpha_v3_teacher.py`, materialize the canonical metric and diagnostic as sibling consumers of that shared state, type the pipeline's Signal producer seam with a Protocol, and reconstruct `CausalAlphaV3Forecast` during strict diagnostic validation.

**Tech Stack:** Python 3.12, dataclasses, typing `Protocol`, NumPy, pytest, Mypy, Ruff, Import Linter, GitHub Actions.

## Global Constraints

- `causal_alpha_v3_signal_scope_v2` remains the sole Signal Gate evidence leaf.
- The diagnostic sidecar schema/path remains `causal_alpha_v3_signal_diagnostic_scope_v1` under `signal/diagnostics/<fit>/<symbol>/<episode>.json`.
- No Signal Gate thresholds, bootstrap semantics, independent-episode semantics, ridge fitting, label formulas/timing, horizon blend, target compiler, selection, Teacher admission, BC/RL, reward, risk, execution, or promotion logic changes.
- Historical Signal V2 runs are not migrated or post-hoc refitted.
- The maintained runner continues to require paired metric+diagnostic evidence for newly built scopes.
- Cross-host bitwise ridge reproducibility is not claimed; exact no-op comparison is under one numerical environment.
- Do not add fit-level diagnostic artifacts, per-row availability bitsets, raw feature values, or PR #411 consumer changes.

---

### Task 1: Type the paired Signal producer seam

**Files:**
- Modify: `trade_rl/workflows/universal_causal_alpha_v3_teacher.py`
- Modify: `trade_rl/workflows/universal_causal_alpha_v3_pipeline.py`
- Test: `tests/workflows/test_universal_causal_alpha_v3_runner_orchestration.py`
- Test: `tests/workflows/test_universal_causal_alpha_v3_signal_resume.py`

**Interfaces:**
- Consumes: existing `CausalAlphaV3SignalScopeBuild`, `CausalAlphaV3FitCache`, `CausalAlphaV3Candidate`, `CausalAlphaSymbolSamples`, and `OracleEpisodeContract`.
- Produces: `CausalAlphaV3SignalScopeBuilder` Protocol whose `__call__(...)` returns `CausalAlphaV3SignalScopeBuild`; `run_universal_causal_alpha_v3_research_pipeline(..., signal_scope_builder: CausalAlphaV3SignalScopeBuilder = build_causal_alpha_v3_signal_scope, ...)`.

- [ ] **Step 1: Write the static-contract RED fixture**

Add a Mypy-visible assignment or helper annotation in an existing workflow test module so the paired builder satisfies the Protocol, and add a negative type fixture using the legacy `build_causal_alpha_v3_signal_scope_metric` guarded by the repository's established Mypy test-fixture pattern if one exists. If the repository has no negative-fixture harness, make the production signature precise and rely on exact-head Mypy plus source-shape assertion that the annotation is not `Callable[..., Any]`.

The positive type contract must be equivalent to:

```python
builder: CausalAlphaV3SignalScopeBuilder = build_causal_alpha_v3_signal_scope
```

and the legacy metric-only builder must not be assignable to that Protocol.

- [ ] **Step 2: Run the smallest static/test oracle and record RED**

Run exact-head Mypy and the affected runner/resume tests. Expected RED before production typing: the source-shape/static contract proving the old `Callable[..., Any]` seam is rejected or absent should fail.

- [ ] **Step 3: Add the explicit Protocol**

Place the Protocol next to `CausalAlphaV3SignalScopeBuild` unless that creates a circular import. Its exact callable signature is:

```python
class CausalAlphaV3SignalScopeBuilder(Protocol):
    def __call__(
        self,
        *,
        run_manifest_digest: str,
        symbol: str,
        train_symbols: tuple[str, ...],
        samples: Mapping[str, CausalAlphaSymbolSamples],
        contract: OracleEpisodeContract,
        candidate: CausalAlphaV3Candidate,
        fit_cache: CausalAlphaV3FitCache | None = None,
    ) -> CausalAlphaV3SignalScopeBuild: ...
```

Annotate the pipeline parameter with this Protocol. Keep the existing runtime `isinstance(built, CausalAlphaV3SignalScopeBuild)` check as defense in depth.

- [ ] **Step 4: Re-run Mypy and focused tests for GREEN**

Expected: Mypy accepts the maintained paired builder wiring; runner and resume tests remain Green.

- [ ] **Step 5: Review import direction**

Confirm the Protocol placement does not add a workflow circular import and Import Linter still sees no new macro-layer violation.

---

### Task 2: Split shared Signal computation from metric/diagnostic materialization

**Files:**
- Modify: `trade_rl/workflows/universal_causal_alpha_v3_teacher.py`
- Test: `tests/workflows/test_universal_causal_alpha_v3_signal_diagnostic.py`
- Test: `tests/workflows/test_universal_causal_alpha_v3_signal_diagnostic_metric_oracle.py`

**Interfaces:**
- Consumes: `_prediction_scope(...)`, `_labels_for_decisions(...)`, `non_overlapping_causal_alpha_v3_rows(...)` and existing diagnostic builder.
- Produces: private immutable shared computation object/helper plus a private canonical metric materializer; public signatures of `build_causal_alpha_v3_signal_scope(...)` and `build_causal_alpha_v3_signal_scope_metric(...)` remain unchanged.

- [ ] **Step 1: Add RED proving metric-only diagnostic independence**

Monkeypatch or spy on `build_causal_alpha_v3_signal_diagnostic_scope` so it raises a diagnostic-only sentinel exception. Call `build_causal_alpha_v3_signal_scope_metric(...)` with the existing deterministic paired fixture and assert the canonical metric still succeeds. Before the refactor this test must fail because the metric-only wrapper calls the paired producer.

A representative oracle is:

```python
def explode(**_: object) -> NoReturn:
    raise AssertionError("diagnostic builder must not run")

monkeypatch.setattr(teacher_module, "build_causal_alpha_v3_signal_diagnostic_scope", explode)
metric = build_causal_alpha_v3_signal_scope_metric(**kwargs)
assert metric.cohort_indices == (10, 13)
```

- [ ] **Step 2: Add RED proving paired/metric exact equality remains required**

Keep or strengthen the current exact payload equality assertion:

```python
metric_only = build_causal_alpha_v3_signal_scope_metric(**kwargs)
paired = build_causal_alpha_v3_signal_scope(**kwargs)
assert paired.metric.to_payload() == metric_only.to_payload()
```

Add call-count instrumentation around fit/prediction where practical to prevent a second fit/forecast in one paired call.

- [ ] **Step 3: Run the two focused tests and record RED**

Expected: diagnostic-independence test fails on the current implementation; existing exact canonical equality remains Green.

- [ ] **Step 4: Implement one shared private computation path**

Create a private frozen dataclass or equivalent named `_CausalAlphaV3SignalScopeComputation` containing only ephemeral canonical inputs already available in memory:

```python
@dataclass(frozen=True, slots=True)
class _CausalAlphaV3SignalScopeComputation:
    fitted: CausalAlphaV3Fit
    forecast: CausalAlphaV3Forecast
    block: CausalAlphaSymbolSamples
    decisions: np.ndarray
    actionable: np.ndarray
    feature_available: np.ndarray
    matched: np.ndarray
    labels_24h: np.ndarray
    labels_72h: np.ndarray
    ends_24h: np.ndarray
    ends_72h: np.ndarray
    cohort_rows: np.ndarray
```

Use one helper to build this state and one helper to materialize `CausalAlphaV3SignalScopeMetric`. Preserve the current formulas and exception conditions byte-for-byte where possible:

```python
prediction = forecast.expected_return_24h_equivalent[cohort_rows]
realized = 0.5 * (labels_24h[cohort_rows] + labels_72h[cohort_rows] / 3.0)
```

`build_causal_alpha_v3_signal_scope_metric(...)` becomes `compute -> metric -> return` and does not call the diagnostic builder.

`build_causal_alpha_v3_signal_scope(...)` becomes `compute -> metric -> diagnostic from same state -> CausalAlphaV3SignalScopeBuild`.

- [ ] **Step 5: Re-run focused Signal tests for GREEN**

Expected: diagnostic-independence, exact paired/metric equality, horizon/availability extraction and canonical oracle all pass.

- [ ] **Step 6: Falsification review of the split**

Verify that a diagnostic-only error still fails the paired producer, while the metric-only API succeeds; verify the paired producer still builds diagnostic evidence and the pipeline still never invokes the metric-only compatibility wrapper.

---

### Task 3: Rebind persisted forecast observations to `forecast_digest`

**Files:**
- Modify: `trade_rl/workflows/universal_causal_alpha_v3_signal_diagnostic.py` or `trade_rl/workflows/universal_causal_alpha_v3_signal_diagnostic_codec.py` according to the smallest responsibility-preserving boundary.
- Test: `tests/workflows/test_universal_causal_alpha_v3_signal_diagnostic_store.py`
- Test: `tests/workflows/test_universal_causal_alpha_v3_signal_diagnostic.py`

**Interfaces:**
- Consumes: persisted ordered `prediction_rows`, `model_24h.weighted_residual_rmse`, `model_72h.weighted_residual_rmse`, and canonical `CausalAlphaV3Forecast` constructor/contract.
- Produces: fail-closed sidecar construction/decoding when persisted forecast observations do not reproduce `forecast_digest`.

- [ ] **Step 1: Write forged-forecast RED**

Start from a valid sidecar payload produced by the real paired builder, not the synthetic `_diagnostic()` fixture with arbitrary digests. Mutate one persisted forecast field, preserve all row-level algebra needed by the dataclass if applicable, recompute the sidecar's outer `artifact_digest`, keep the old `forecast_digest`, and assert strict decoding rejects it specifically for forecast identity drift.

At minimum falsify one prediction and one horizon RMSE path so the oracle would catch wrong-array and wrong-RMSE reconstruction.

- [ ] **Step 2: Run the forged-forecast test and record RED**

Expected before implementation: the payload is accepted when its outer sidecar digest is recomputed, demonstrating that `forecast_digest` is currently only a referenced identity rather than independently rebound evidence.

- [ ] **Step 3: Implement exact forecast reconstruction**

After basic row/model validation and before final sidecar acceptance, reconstruct:

```python
reconstructed = CausalAlphaV3Forecast(
    prediction_24h=np.asarray([row.prediction_24h for row in prediction_rows], dtype=np.float64),
    prediction_72h=np.asarray([row.prediction_72h for row in prediction_rows], dtype=np.float64),
    expected_return_24h_equivalent=np.asarray(
        [row.expected_return_24h_equivalent for row in prediction_rows], dtype=np.float64
    ),
    uncertainty_24h_equivalent=np.asarray(
        [row.uncertainty_24h_equivalent for row in prediction_rows], dtype=np.float64
    ),
    signal_to_uncertainty=np.asarray(
        [row.signal_to_uncertainty for row in prediction_rows], dtype=np.float64
    ),
    residual_rmse_24h=model_24h.weighted_residual_rmse,
    residual_rmse_72h=model_72h.weighted_residual_rmse,
)
if reconstructed.digest != forecast_digest:
    raise ValueError("V3 diagnostic forecast identity drifted")
```

Prefer putting this invariant in `CausalAlphaV3SignalDiagnosticScope.__post_init__` so both in-memory construction and JSON decoding enforce it. Do not refit, read raw features, or add schema fields.

- [ ] **Step 4: Re-run diagnostic contract/store tests for GREEN**

Expected: valid real sidecars round-trip; self-consistent outer forgeries with stale forecast identity fail closed.

- [ ] **Step 5: Check synthetic test fixtures**

Any synthetic `_scope()` / `_diagnostic()` helper that currently invents arbitrary `forecast_digest` must be updated to construct a digest from exactly the same persisted prediction/RMSE fields. Do not weaken the new invariant to preserve invalid fixtures.

---

### Task 4: Integration regression and exact canonical no-op verification

**Files:**
- Modify only if failures expose a real regression: runner/pipeline/tests listed above.
- Verify: full PR diff against `main`.

**Interfaces:**
- Consumes: hardened paired builder, typed port, strict sidecar contract.
- Produces: exact evidence that maintained behavior is unchanged except for the intended validation/type boundary.

- [ ] **Step 1: Run focused workflow suite**

Run at least:

```text
tests/workflows/test_universal_causal_alpha_v3_signal_diagnostic.py
tests/workflows/test_universal_causal_alpha_v3_signal_diagnostic_metric_oracle.py
tests/workflows/test_universal_causal_alpha_v3_signal_diagnostic_store.py
tests/workflows/test_universal_causal_alpha_v3_signal_resume.py
tests/workflows/test_universal_causal_alpha_v3_runner_orchestration.py
```

Expected: all Green.

- [ ] **Step 2: Run static architecture gates**

Run Ruff, format check, Mypy, Import Linter and dead-code checks on the exact candidate head. Expected: all Green and no circular import.

- [ ] **Step 3: Run same-runner pre-sidecar vs final-tree oracle**

On one numerical environment with fixed numerical thread settings, execute the pre-sidecar base `7b02d92b0234a84c7c4e240d5422be276efdfe73` and the exact final candidate tree against the same controlled Signal input and require byte-identical full canonical metric payload including artifact/fit/forecast digests.

- [ ] **Step 4: Run full repository quality matrix**

Require exact-final-HEAD success for full pytest/branch coverage, critical coverage ratchet, Ruff, format, Mypy, import architecture, dead-code, recovery/serving smoke, frontend tests/typecheck/build/layout, multiprocessing regressions, Ubuntu/Windows compatibility, Training image, packaged non-root runtime probe, package/module/CLI/uv identity, and applicable PostgreSQL Catalog workflow.

- [ ] **Step 5: Review final diff and PR relationship**

Confirm no changes outside the hardening scope except specs/plans/tests; no temporary workflow remains on the feature branch; `main` is protected and unmodified; PR #413 remains unmerged.

---

### Task 5: Independent-style falsification and completion gate

**Files:**
- Update: PR #413 body with final hardening evidence after exact-final verification.
- No production file change unless review finds a defect.

**Interfaces:**
- Consumes: original sidecar spec, architecture-hardening spec, final diff, actual tests/assertions and exact-head workflow evidence.
- Produces: final Ready-for-review state only if all gates pass.

- [ ] **Step 1: Reconstruct review criteria from the specs**

Review without assuming implementation conclusions. Attempt to falsify:

```text
legacy metric-only builder accepted by paired seam
metric-only path executes diagnostic code
paired path executes two fits/forecasts
forged forecast observations survive with recomputed outer digest
diagnostic object reaches Gate evaluator
partial resume overwrites valid persisted member
future label or raw feature leakage
```

- [ ] **Step 2: Inspect final assertions**

Confirm tests assert observable state/digests/call counts rather than merely checking that functions execute.

- [ ] **Step 3: Fix any substantive finding and restart affected verification layers**

Do not weaken assertions, skip failing tests, or alter acceptance criteria to obtain Green.

- [ ] **Step 4: Verify exact final HEAD / CI HEAD equality**

Record final branch SHA and ensure required GitHub Actions runs are for that exact SHA.

- [ ] **Step 5: Update PR body and mark Ready**

Only after all required checks pass, update PR #413 with the hardening rationale, TDD RED/GREEN evidence, exact final HEAD, tests, falsification findings, limitations and deferred P2 items, then transition Draft -> Ready. Do not merge.
