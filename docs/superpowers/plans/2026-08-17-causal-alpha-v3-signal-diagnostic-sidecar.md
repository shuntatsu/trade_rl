# Causal Alpha V3 Signal Diagnostic Sidecar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist run-bound, research-only Causal Alpha V3 Signal diagnostics during the original Signal computation so fresh runs expose 24h/72h/fused predictions, realized labels, feature availability, model state, and weighted effective sample size without changing canonical Signal Gate numerics.

**Architecture:** Keep `CausalAlphaV3SignalScopeMetric` and `causal_alpha_v3_signal_scope_v2` unchanged as the sole Signal Gate evidence. Add a separate immutable `causal_alpha_v3_signal_diagnostic_scope_v1` sidecar under `signal/diagnostics/<fit>/<symbol>/<episode>.json`, produced from the same in-memory fit/forecast/labels as the canonical metric. Pair-aware persistence/resume reuses two valid artifacts, builds both when absent, repairs only a one-file interrupted write after recomputation agrees with the valid member, and fails closed on corrupt or identity-drifted evidence.

**Tech Stack:** Python 3.12, NumPy, dataclasses, existing content-addressed artifact utilities, pytest, Ruff, Mypy, Import Linter, vulture, GitHub Actions.

## Global Constraints

- Canonical Signal leaf schema remains `causal_alpha_v3_signal_scope_v2`.
- Signal Gate receives only `CausalAlphaV3SignalScopeMetric`; diagnostic values never affect pass/fail, thresholds, bootstrap, candidate filtering, or independent episode counts.
- Ridge fitting, labels, horizon blend, target controller, economic selection, Teacher admission, BC, critic warm start, PPO/Lagrangian, reward, risk, and execution numerical semantics do not change.
- Sidecars are always `research_only=true` and `promotion_eligible=false`.
- Old V2 runs without diagnostics are not migrated or post-hoc refitted.
- Diagnostic realized rows must never include a label whose endpoint is outside the Signal contract.
- No raw feature matrix or Teacher-admission holdout evidence is serialized.
- Partial-write recovery may rebuild a scope only when exactly one valid paired member exists; corruption/staleness/wrong-path/wrong-run/wrong-contract/wrong-fit evidence fails closed.
- Existing canonical metric payload/digest and fit-level Gate evidence/pass state must be numerically identical for identical inputs.

---

### Task 1: Diagnostic contract and deterministic weight evidence

**Files:**
- Create: `trade_rl/workflows/universal_causal_alpha_v3_signal_diagnostic.py`
- Modify: `trade_rl/workflows/universal_causal_alpha_v3.py`
- Test: `tests/workflows/test_universal_causal_alpha_v3_signal_diagnostic.py`

**Interfaces:**
- Consumes: `CausalAlphaV3Fit`, `CausalAlphaSymbolSamples`, `CausalAlphaRidgeModel`, `build_causal_alpha_v3_symbol_balanced_weights(...)`.
- Produces: `CausalAlphaV3SignalDiagnosticScope`, `CausalAlphaV3SignalDiagnosticModel`, `CausalAlphaV3SignalDiagnosticPredictionRow`, `CausalAlphaV3SignalDiagnosticRealizedRow`, `build_causal_alpha_v3_signal_diagnostic_scope(...)`, `causal_alpha_v3_weight_digest(...)`, and `weighted_effective_sample_size(...)`.

- [ ] **Step 1: Write failing contract tests**

Add synthetic tests that construct a diagnostic scope and assert:

```python
assert diagnostic.schema_version == "causal_alpha_v3_signal_diagnostic_scope_v1"
assert diagnostic.research_only is True
assert diagnostic.promotion_eligible is False
assert diagnostic.model_24h.model_digest == fitted.model_24h.digest
assert diagnostic.model_72h.model_digest == fitted.model_72h.digest
assert diagnostic.canonical_cohort_indices == metric.cohort_indices
assert "features" not in diagnostic.to_payload()
assert "targets" not in diagnostic.to_payload()
```

Add strict constructor/parser failures for non-finite predictions, mismatched row lengths, duplicated/non-monotone decision indices, wrong SHA-256 identities, feature-order drift between models, and `promotion_eligible=True`.

- [ ] **Step 2: Write failing ESS/digest tests**

For exact weights `w`, independently assert:

```python
expected = float(np.square(w.sum()) / np.square(w).sum())
assert weighted_effective_sample_size(w) == pytest.approx(expected)
```

For each horizon, construct symbol-balanced weights, compute the public `causal_alpha_v3_weight_digest(...)`, and require equality with `CausalAlphaV3Fit.weight_digest_24h` / `weight_digest_72h`. Mutating one weight must change the digest and be rejected by diagnostic construction.

- [ ] **Step 3: Run RED**

Run:

```bash
uv run pytest tests/workflows/test_universal_causal_alpha_v3_signal_diagnostic.py -q
```

Expected: collection/import failures because the diagnostic module and public weight-digest helper do not yet exist.

- [ ] **Step 4: Expose the deterministic weight digest without changing fit numerics**

In `universal_causal_alpha_v3.py`, rename/private-wrap `_weight_digest` as a public pure helper:

```python
def causal_alpha_v3_weight_digest(
    symbols: tuple[str, ...],
    weights: Mapping[str, np.ndarray],
    *,
    horizon: str,
    knowledge_cutoff: int,
) -> str:
    ...
```

Make `fit_causal_alpha_v3(...)` call this helper so diagnostic recomputation and the fit share one digest definition. Do not change the digest payload fields or ordering.

- [ ] **Step 5: Implement immutable diagnostic value objects**

Use frozen/slots dataclasses. `CausalAlphaV3SignalDiagnosticScope` must serialize only summaries/row evidence required by the design and calculate its own `artifact_digest` via existing `content_digest`. Arrays/tuples must be finite, aligned, immutable, and deterministic.

Model evidence must include:

```python
model_digest
feature_names
intercept
coefficients
location
scale
constant_mask
fitted_row_count
weighted_residual_rmse
pooled_weighted_ess
per_symbol_weighted_ess
overlap_weight_digest
```

Prediction rows must include decision/actionable state, availability count/fraction, 24h prediction, 72h prediction, 72h-equivalent prediction, fused expected return, uncertainty, and signal-to-uncertainty.

Realized evidence must identify horizon (`"24h"`, `"72h"`, or `"fused"`), decision index, label end index, prediction, realized return, and availability count/fraction. For 72h evidence store 24h-equivalent prediction/realized values as the canonical values while also retaining raw 72h values in explicit fields.

- [ ] **Step 6: Implement `weighted_effective_sample_size` and diagnostic model builder**

Reject empty, non-finite, negative, or all-zero weights. Compute:

```python
positive = weights[weights > 0.0]
ess = float(np.square(positive.sum()) / np.square(positive).sum())
```

Recompute exact horizon weights with the fit `knowledge_cutoff`, verify `causal_alpha_v3_weight_digest(...) == fitted.weight_digest_*`, then compute pooled ESS from concatenated weights and per-symbol ESS from each symbol vector.

- [ ] **Step 7: Run GREEN and regression fit tests**

Run:

```bash
uv run pytest \
  tests/workflows/test_universal_causal_alpha_v3_signal_diagnostic.py \
  tests/workflows/test_universal_causal_alpha_v3.py -q
```

Expected: PASS, including existing symbol-balance and causal-cutoff tests.

- [ ] **Step 8: Commit**

```bash
git add trade_rl/workflows/universal_causal_alpha_v3.py \
        trade_rl/workflows/universal_causal_alpha_v3_signal_diagnostic.py \
        tests/workflows/test_universal_causal_alpha_v3_signal_diagnostic.py
git commit -m "feat: add V3 signal diagnostic contract"
```

---

### Task 2: Paired Signal computation with canonical metric invariance

**Files:**
- Modify: `trade_rl/workflows/universal_causal_alpha_v3_teacher.py`
- Test: `tests/workflows/test_universal_causal_alpha_v3_signal_diagnostic.py`
- Test: `tests/workflows/test_universal_causal_alpha_v3_runner_signal.py`

**Interfaces:**
- Consumes: Task 1 diagnostic builder and existing `_prediction_scope(...)`, `_labels_for_decisions(...)`, `non_overlapping_causal_alpha_v3_rows(...)`.
- Produces: `CausalAlphaV3SignalScopeBuild(metric, diagnostic)` and `build_causal_alpha_v3_signal_scope(...)`; keeps `build_causal_alpha_v3_signal_scope_metric(...)` as a compatibility wrapper.

- [ ] **Step 1: Freeze the old metric oracle before refactor**

Add a regression fixture whose expected canonical payload is assembled from controlled synthetic inputs. Capture these exact fields from the pre-refactor path:

```python
sample_count
rank_correlation
direction_accuracy
top_bottom_realized_spread
cohort_indices
fit_digest
forecast_digest
digest
```

The test must compare the compatibility wrapper result with the paired-build metric using exact equality for tuples/digests and `pytest.approx(..., rel=0, abs=0)` or exact float equality where current deterministic NumPy operations permit it.

- [ ] **Step 2: Write RED for horizon/availability extraction**

Assert the paired result includes:

```python
assert build.diagnostic.prediction_rows[0].decision_index == contract.start
assert len(build.diagnostic.prediction_rows) == contract.stop - contract.start - 1
assert tuple(r.decision_index for r in build.diagnostic.realized_24h_rows) == expected_24h
assert tuple(r.decision_index for r in build.diagnostic.realized_72h_rows) == expected_72h
assert tuple(r.decision_index for r in build.diagnostic.realized_fused_rows) == expected_fused
```

Construct one row with partial `feature_available=False` and one missing decision. Verify availability count/fraction and `actionable=False` are persisted without persisting raw feature values.

- [ ] **Step 3: Write RED for future-label exclusion and phase evidence**

Create labels whose `label_end_indices_24h`/`72h` equal or exceed `contract.stop`; assert they are absent from realized sidecar rows. Assert all eligible realized rows are retained, while `canonical_cohort_indices` remains only the existing greedy non-overlap cohort.

- [ ] **Step 4: Run RED**

Run:

```bash
uv run pytest \
  tests/workflows/test_universal_causal_alpha_v3_signal_diagnostic.py \
  tests/workflows/test_universal_causal_alpha_v3_runner_signal.py -q
```

Expected: failures for the missing paired-build API and diagnostic extraction.

- [ ] **Step 5: Refactor to one internal computation**

Add:

```python
@dataclass(frozen=True, slots=True)
class CausalAlphaV3SignalScopeBuild:
    metric: CausalAlphaV3SignalScopeMetric
    diagnostic: CausalAlphaV3SignalDiagnosticScope


def build_causal_alpha_v3_signal_scope(...) -> CausalAlphaV3SignalScopeBuild:
    ...


def build_causal_alpha_v3_signal_scope_metric(...) -> CausalAlphaV3SignalScopeMetric:
    return build_causal_alpha_v3_signal_scope(...).metric
```

The paired function must call `_prediction_scope(...)` exactly once, derive labels exactly once, construct the canonical metric with the unchanged existing formulas, then construct the diagnostic from those same objects.

- [ ] **Step 6: Preserve canonical eligibility and metric formulas verbatim**

Do not alter:

```python
eligible = actionable & matched & ... & (ends_72h < contract.stop)
cohort_rows = non_overlapping_causal_alpha_v3_rows(... ends_72h ...)
prediction = forecast.expected_return_24h_equivalent[cohort_rows]
realized = 0.5 * (labels_24h[cohort_rows] + labels_72h[cohort_rows] / 3.0)
bucket = max(1, prediction.size // 5)
```

The sidecar may have horizon-specific realized eligibility masks, but those masks must not feed the canonical metric.

- [ ] **Step 7: Run GREEN and canonical regression**

Run the Task 2 tests plus:

```bash
uv run pytest tests/workflows/test_universal_causal_alpha_v3_signal_resume.py -q
```

Expected: all existing canonical Signal tests remain green and the exact metric oracle is unchanged.

- [ ] **Step 8: Commit**

```bash
git add trade_rl/workflows/universal_causal_alpha_v3_teacher.py \
        tests/workflows/test_universal_causal_alpha_v3_signal_diagnostic.py \
        tests/workflows/test_universal_causal_alpha_v3_runner_signal.py
git commit -m "refactor: pair V3 signal metric with diagnostics"
```

---

### Task 3: Strict diagnostic artifact store and pair-aware resume

**Files:**
- Modify: `trade_rl/workflows/universal_causal_alpha_v3_artifact_store.py`
- Test: `tests/workflows/test_universal_causal_alpha_v3_signal_diagnostic_store.py`
- Modify: `tests/workflows/test_universal_causal_alpha_v3_signal_resume.py`

**Interfaces:**
- Consumes: `CausalAlphaV3SignalDiagnosticScope` parser/serializer from Task 1.
- Produces: `SignalDiagnosticIdentity`, `write_signal_diagnostic_scope(...)`, `load_signal_diagnostic_scopes(...)` and strict paired-state information for the pipeline.

- [ ] **Step 1: Write RED for exact diagnostic path and loader validation**

Expected path:

```text
signal/diagnostics/<fit_config_digest>/<symbol>/<episode_index>.json
```

Tests must reject wrong run manifest, fit config, symbol, contract digest, canonical metric digest, fit digest, forecast digest, path identity, duplicate identity, malformed schema, and artifact digest mismatch.

- [ ] **Step 2: Write RED for four paired resume states**

Extend `test_universal_causal_alpha_v3_signal_resume.py` with builder call counts/file state for:

```text
A. metric + diagnostic valid -> 0 build calls
B. neither present -> 1 build call, 2 files written
C. metric only -> 1 build call, metric unchanged, diagnostic added
D. diagnostic only -> 1 build call, diagnostic unchanged, metric added
```

Add corrupt-member cases proving the builder is never invoked before the loader fails closed.

- [ ] **Step 3: Run RED**

Run:

```bash
uv run pytest \
  tests/workflows/test_universal_causal_alpha_v3_signal_diagnostic_store.py \
  tests/workflows/test_universal_causal_alpha_v3_signal_resume.py -q
```

Expected: failures for missing store APIs and unchanged pipeline semantics.

- [ ] **Step 4: Implement strict diagnostic store**

Mirror canonical Signal leaf path safety and identity checks; do not loosen canonical loader behavior. Loading a diagnostics directory that exists must reject any unknown or malformed record rather than skip it.

- [ ] **Step 5: Define persisted-member agreement checks**

When a scope is recomputed because exactly one paired member is missing:

```python
if persisted_metric is not None:
    require recomputed.metric.digest == persisted_metric.digest
if persisted_diagnostic is not None:
    require recomputed.diagnostic.digest == persisted_diagnostic.digest
```

Also require both recomputed objects bind to each other through `signal_metric_digest`, `fit_digest`, `forecast_digest`, contract identity, and run identity before writing the missing file.

- [ ] **Step 6: Run GREEN**

Run Task 3 tests. Expected: all four states and corrupt-state fail-closed tests pass.

- [ ] **Step 7: Commit**

```bash
git add trade_rl/workflows/universal_causal_alpha_v3_artifact_store.py \
        tests/workflows/test_universal_causal_alpha_v3_signal_diagnostic_store.py \
        tests/workflows/test_universal_causal_alpha_v3_signal_resume.py
git commit -m "feat: persist V3 signal diagnostic sidecars"
```

---

### Task 4: Pipeline wiring with zero diagnostic-to-Gate leakage

**Files:**
- Modify: `trade_rl/workflows/universal_causal_alpha_v3_pipeline.py`
- Test: `tests/workflows/test_universal_causal_alpha_v3_signal_diagnostic_integration.py`
- Modify: `tests/workflows/test_universal_causal_alpha_v3_runner_signal.py`

**Interfaces:**
- Consumes: `build_causal_alpha_v3_signal_scope(...)` and store APIs from Tasks 2-3.
- Produces: maintained pipeline behavior with paired persistence; public pipeline injection point `signal_scope_builder` accepts a paired build result for the maintained default path while test adapters may continue to construct canonical metrics through an explicit compatibility adapter if needed.

- [ ] **Step 1: Write RED for Gate argument purity**

Inject a spy evaluator:

```python
def gate_spy(metrics, **kwargs):
    assert all(isinstance(item, CausalAlphaV3SignalScopeMetric) for item in metrics)
    assert not any(isinstance(item, CausalAlphaV3SignalDiagnosticScope) for item in metrics)
    ...
```

Assert fit-level evidence/pass state matches a baseline run whose canonical metrics are prebuilt.

- [ ] **Step 2: Write RED for normal fresh persistence**

A synthetic rejected run must produce, for every expected scope, exactly one canonical leaf and one diagnostic leaf with matching identities. `signal/rejection.json` must remain schema/digest compatible with the existing V2 rejection contract and contain no diagnostic digest fields.

- [ ] **Step 3: Write RED for target-only candidate deduplication**

For candidates sharing a fit config, assert only one paired Signal scope is built per `(fit_config_digest, symbol, episode)` exactly as today. Sidecars must not multiply by target-controller variants.

- [ ] **Step 4: Run RED**

Run:

```bash
uv run pytest \
  tests/workflows/test_universal_causal_alpha_v3_signal_diagnostic_integration.py \
  tests/workflows/test_universal_causal_alpha_v3_runner_signal.py \
  tests/workflows/test_universal_causal_alpha_v3_signal_resume.py -q
```

- [ ] **Step 5: Wire paired builder/store into the Signal stage**

At run start, load canonical metrics and diagnostics independently against the same expected scope map. For each scope:

```python
metric = persisted_metrics.get(identity)
diagnostic = persisted_diagnostics.get(identity)
if metric is not None and diagnostic is not None:
    validate_pair(metric, diagnostic)
else:
    build = signal_scope_builder(...)
    validate build.metric identity
    validate build.diagnostic identity and pair binding
    compare against whichever member already exists
    write only missing members
metrics.append(metric_or_build_metric)
```

Do not pass diagnostics to `signal_gate_evaluator(...)` or fit result/rejection payloads.

- [ ] **Step 6: Preserve unavailable-scope semantics**

`CausalAlphaV3SignalScopeUnavailable` must still mark the canonical contract unavailable exactly as before. Do not create a diagnostic sidecar for a scope that cannot produce the canonical metric.

- [ ] **Step 7: Run GREEN and pipeline regressions**

Run Task 4 tests plus:

```bash
uv run pytest \
  tests/workflows/test_universal_causal_alpha_v3_runner_orchestration.py \
  tests/workflows/test_universal_causal_alpha_v3_runner_engine.py \
  tests/workflows/test_universal_causal_alpha_v3_falsification.py -q
```

Expected: existing stage ordering, rejection, admission blocking, and fail-closed identity tests remain green.

- [ ] **Step 8: Commit**

```bash
git add trade_rl/workflows/universal_causal_alpha_v3_pipeline.py \
        tests/workflows/test_universal_causal_alpha_v3_signal_diagnostic_integration.py \
        tests/workflows/test_universal_causal_alpha_v3_runner_signal.py \
        tests/workflows/test_universal_causal_alpha_v3_signal_resume.py
git commit -m "feat: wire V3 signal diagnostic sidecars"
```

---

### Task 5: Falsification, architecture review, documentation, and exact-head verification

**Files:**
- Modify: `docs/UNIVERSAL_TRAINING.md`
- Create: `tests/workflows/test_universal_causal_alpha_v3_signal_diagnostic_falsification.py`
- Review only: all files changed in Tasks 1-4

**Interfaces:**
- Consumes: final producer/store/pipeline behavior.
- Produces: documented research contract and verification evidence; no new runtime numerical behavior.

- [ ] **Step 1: Add falsification tests from the original spec**

Explicitly attempt to make invalid evidence pass:

```text
- sidecar metric digest points at another valid scope
- model digest belongs to another fit
- forecast digest belongs to another symbol
- 24h/72h arrays are swapped but lengths remain valid
- realized row with label_end == contract.stop is injected
- availability count disagrees with availability fraction
- ESS is recomputed from mutated weights
- copied valid sidecar is placed under another run/path
- sidecar is passed into a Gate-spy input
- one paired member is corrupt while the other is missing
```

Every invalid case must fail before silent reuse/write or Gate evaluation.

- [ ] **Step 2: Add deterministic/non-secret serialization tests**

Assert repeated construction produces identical payload/digest and payload contains neither raw `features` matrices nor any Teacher holdout/admission payload fields.

- [ ] **Step 3: Document operational semantics**

In `docs/UNIVERSAL_TRAINING.md`, state:

```text
signal/records/* remains canonical Gate evidence.
signal/diagnostics/* is research-only sidecar evidence generated during fresh Signal computation.
Old V2 runs without sidecars remain valid historical Gate artifacts but cannot expose per-horizon/model diagnostics.
Sidecars never change Gate pass/fail and must not be used as promotion evidence.
```

- [ ] **Step 4: Run targeted falsification suite**

```bash
uv run pytest \
  tests/workflows/test_universal_causal_alpha_v3_signal_diagnostic.py \
  tests/workflows/test_universal_causal_alpha_v3_signal_diagnostic_store.py \
  tests/workflows/test_universal_causal_alpha_v3_signal_diagnostic_integration.py \
  tests/workflows/test_universal_causal_alpha_v3_signal_diagnostic_falsification.py \
  tests/workflows/test_universal_causal_alpha_v3_signal_resume.py \
  tests/workflows/test_universal_causal_alpha_v3_runner_signal.py -q
```

- [ ] **Step 5: Run static and architecture checks**

Use the repository's maintained commands/workflows for:

```text
Ruff
Ruff format --check
Mypy
Import Architecture
100%-confidence vulture/dead-code check
```

No check may be skipped or weakened to obtain Green.

- [ ] **Step 6: Run broader related tests then full suite**

Run all `test_universal_causal_alpha_v3*.py` tests, then the full pytest + branch-coverage workflow used by repository CI. Confirm changed code executes through both happy and error paths.

- [ ] **Step 7: Architecture/self-review loop**

Review final diff for responsibility boundaries, duplicate validation, canonical/diagnostic contract coupling, accidental Signal numerical changes, large-file growth, hidden future-label access, artifact-size risk, and open-PR overlap. Fix substantive findings and rerun targeted tests after each fix.

- [ ] **Step 8: Independent/falsification review**

Reconstruct review criteria from the design rather than the implementation. Verify the public contract, actual diff, assertions, pair-state transitions, and exact persisted payloads. Specifically search for a wrong implementation that existing tests would still accept.

- [ ] **Step 9: Final git/CI quality gate**

Before completion claim, inspect:

```text
final diff
git status / untracked files
final HEAD
targeted tests
related V3 tests
full pytest + coverage
Ruff / format
Mypy
Import Architecture
dead-code/static analysis
build/package identity
Ubuntu and Windows compatibility
training image/runtime probe
PostgreSQL/catalog workflow if path-applicable
GitHub Actions/required checks on the exact final HEAD
```

If a check is unavailable or path-filtered, report it as unverified/not applicable; do not call it successful.

- [ ] **Step 10: Commit**

```bash
git add docs/UNIVERSAL_TRAINING.md \
        tests/workflows/test_universal_causal_alpha_v3_signal_diagnostic_falsification.py
git commit -m "test: falsify V3 signal diagnostic sidecars"
```

## Completion Contract

The task is complete only when the final exact HEAD demonstrates that canonical Signal metric/Gate outputs are unchanged, fresh runs persist complete paired diagnostic evidence, partial writes recover without masking corruption, diagnostics cannot leak into Gate/promotion paths, required tests/static/build checks succeed, and remaining limitations are explicitly reported. Passing tests alone are not sufficient.