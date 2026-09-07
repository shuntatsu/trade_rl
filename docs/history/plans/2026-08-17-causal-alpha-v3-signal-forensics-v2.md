# Causal Alpha V3 Signal Forensics V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic, read-only `causal_alpha_v3_signal_forensics_v2` report that preserves the verified V1 historical analysis unchanged and, when a complete fresh diagnostic-sidecar set exists, diagnoses 24h/72h/fused forecast quality, fitted-model stability, prediction geometry, feature availability, weighted ESS, and chronological sensitivity.

**Architecture:** Import the already-verified V1 analyzer implementation from PR #411 without changing its public output contract. Add a separate V2 contracts module, a strict V2 sidecar binder that reuses the #413 codec and canonical V1 validation helpers, a pure V2 analysis module, and a thin V2 orchestration module. The existing CLI keeps V1 as the default and selects V2 only through an explicit `--schema v2` argument.

**Tech Stack:** Python 3.12, frozen dataclasses, NumPy, pytest, existing `trade_rl.learning.causal_alpha_diagnostics`, existing strict Causal Alpha V3 Signal metric/diagnostic codecs, Ruff, Mypy, Import Linter, GitHub Actions.

## Global Constraints

- `causal_alpha_v3_signal_scope_v2` remains the sole canonical Signal Gate evidence leaf.
- `causal_alpha_v3_signal_diagnostic_scope_v1` remains research-only and `promotion_eligible=false`.
- The existing V1 API `load_causal_alpha_v3_signal_forensics(root)` and schema `causal_alpha_v3_signal_forensics_v1` remain payload/digest compatible for the same source artifacts.
- V2 schema is exactly `causal_alpha_v3_signal_forensics_v2` and binds the complete V1 base report plus `base_forensics_digest`.
- V2 sidecar modes are exactly `historical_unavailable` and `sidecar_complete`.
- If `signal/diagnostics` is absent, V2 returns historical mode without refitting or reconstruction.
- If `signal/diagnostics` exists, even empty, the diagnostic identity set must exactly equal the canonical metric identity set; partial, extra, duplicate, corrupt, stale, wrong-path, wrong-run, or drifted evidence fails closed.
- 24h/72h/fused realized-forecast diagnostics reuse `evaluate_causal_alpha_signal_diagnostics`; no parallel correlation/rank/bin semantics are introduced.
- Direct 24h-vs-72h comparisons use identical realized decision indices and 72h values in 24h-equivalent units.
- Raw overlapping realized rows are descriptive only and never become independent Signal Gate samples.
- Pooled fit/model evidence is deduplicated to one snapshot per `(fit_config_digest, contract_start, contract_stop)` only after exact cross-symbol equality validation.
- No Signal Gate thresholds/bootstrap semantics, ridge fitting, labels, selection, Teacher admission, Teacher package, BC, critic warm start, PPO/Lagrangian, reward, risk, execution, or promotion behavior changes.
- Source run artifacts are byte-for-byte read-only; report output may only be written outside the source run root.
- No historical sidecar backfill, model refit, data rebuild, environment replay, regime classifier, row-by-feature causal attribution, profitability claim, alpha claim, RL-uplift claim, or Production GO claim.

---

### Task 1: Restore the verified V1 analyzer on the current main baseline

**Files:**
- Create from exact PR #411 blobs: `trade_rl/workflows/universal_causal_alpha_v3_signal_forensics.py`
- Create from exact PR #411 blobs: `scripts/analyze_universal_causal_alpha_v3_signal.py`
- Create from exact PR #411 blobs: `tests/workflows/test_universal_causal_alpha_v3_signal_forensics.py`
- Create from exact PR #411 blobs: `tests/workflows/test_universal_causal_alpha_v3_signal_forensics_falsification.py`
- Create from exact PR #411 blobs: `tests/scripts/test_analyze_universal_causal_alpha_v3_signal.py`

**Interfaces:**
- Produces unchanged V1 `CausalAlphaV3SignalForensicsReport` and `load_causal_alpha_v3_signal_forensics(root: Path)`.
- Preserves the existing CLI behavior with no `--schema` argument: V1 JSON to stdout or external output path.

- [ ] **Step 1: Reuse exact verified PR #411 blobs rather than retyping V1**

Copy the five code/test blobs from exact PR #411 head `32803a50678374f3eef2bf111dc2dff8e18d3d57`. Do not import the old `docs/UNIVERSAL_TRAINING.md` patch or mutate current main documentation.

- [ ] **Step 2: Run the V1 focused compatibility tests on current main**

Run:

```bash
uv run pytest -q \
  tests/workflows/test_universal_causal_alpha_v3_signal_forensics.py \
  tests/workflows/test_universal_causal_alpha_v3_signal_forensics_falsification.py \
  tests/scripts/test_analyze_universal_causal_alpha_v3_signal.py
```

Expected: all V1 tests pass on the post-#412/#413 baseline. Any failure is an integration defect to fix before V2 work.

- [ ] **Step 3: Record a V1 no-op oracle fixture**

In the later V2 test module, keep one deterministic assertion that a V1 report generated before and after V2 additions has the exact same `to_payload()` and digest when the same canonical Signal records are used.

- [ ] **Step 4: Commit the baseline integration**

Commit only the exact V1 code/test files. The V1 payload contract is now the compatibility oracle for all later tasks.

---

### Task 2: Add V2 report contracts and historical-unavailable mode

**Files:**
- Create: `trade_rl/workflows/universal_causal_alpha_v3_signal_forensics_v2_contracts.py`
- Create: `trade_rl/workflows/universal_causal_alpha_v3_signal_forensics_v2.py`
- Create: `tests/workflows/test_universal_causal_alpha_v3_signal_forensics_v2.py`

**Interfaces:**
- Produces `CausalAlphaV3SignalForensicsReportV2` and `load_causal_alpha_v3_signal_forensics_v2(root: Path)`.
- `CausalAlphaV3SignalForensicsReportV2.to_payload()` contains exactly:

```text
schema_version
base_forensics_digest
base_forensics
sidecar_mode
sidecar_analysis
unavailable_analyses
research_only
promotion_eligible
artifact_digest
```

- [ ] **Step 1: Write RED tests for V2 historical mode and V1 compatibility**

Use the existing V1 `_build_run(tmp_path)` fixture with no `signal/diagnostics` directory.

Assert:

```python
base = load_causal_alpha_v3_signal_forensics(tmp_path)
v2 = load_causal_alpha_v3_signal_forensics_v2(tmp_path)
assert v2.schema_version == "causal_alpha_v3_signal_forensics_v2"
assert v2.base_forensics_digest == base.digest
assert v2.base_forensics.to_payload() == base.to_payload()
assert v2.sidecar_mode == "historical_unavailable"
assert v2.sidecar_analysis is None
assert v2.research_only is True
assert v2.promotion_eligible is False
assert v2.digest == load_causal_alpha_v3_signal_forensics_v2(tmp_path).digest
```

Also snapshot the entire source tree before/after and assert byte equality.

- [ ] **Step 2: Run the historical-mode tests and verify RED**

Expected: import/module failure because the V2 implementation does not exist yet.

- [ ] **Step 3: Implement strict top-level V2 contract**

`CausalAlphaV3SignalForensicsReportV2.__post_init__` must:

```python
if schema_version != "causal_alpha_v3_signal_forensics_v2":
    raise ValueError(...)
if research_only is not True or promotion_eligible is not False:
    raise ValueError(...)
if base_forensics_digest != base_forensics.digest:
    raise ValueError(...)
if sidecar_mode == "historical_unavailable" and sidecar_analysis is not None:
    raise ValueError(...)
if sidecar_mode == "sidecar_complete" and sidecar_analysis is None:
    raise ValueError(...)
expected = content_digest(to_payload(include_digest=False))
```

Reject any sidecar mode other than the two allowed literals.

- [ ] **Step 4: Implement historical-mode orchestration**

`load_causal_alpha_v3_signal_forensics_v2` first calls V1. It then checks only path existence:

```python
diagnostics_root = Path(root) / "signal" / "diagnostics"
if not diagnostics_root.exists():
    return CausalAlphaV3SignalForensicsReportV2(
        base_forensics=base,
        base_forensics_digest=base.digest,
        sidecar_mode="historical_unavailable",
        sidecar_analysis=None,
        unavailable_analyses=base.unavailable_analyses,
    )
```

Do not create directories and do not refit/replay.

- [ ] **Step 5: Run focused V2 historical-mode tests for GREEN**

Expected: deterministic V2 wrapper, exact V1 base payload, no source mutation.

---

### Task 3: Bind complete diagnostic sidecars fail-closed

**Files:**
- Create: `trade_rl/workflows/universal_causal_alpha_v3_signal_forensics_v2_loader.py`
- Modify: `trade_rl/workflows/universal_causal_alpha_v3_signal_forensics_v2.py`
- Extend: `tests/workflows/test_universal_causal_alpha_v3_signal_forensics_v2.py`
- Create: `tests/workflows/test_universal_causal_alpha_v3_signal_forensics_v2_falsification.py`

**Interfaces:**
- Produces typed `CausalAlphaV3SignalForensicsV2BoundScope(metric, diagnostic)` pairs.
- Reuses `signal_diagnostic_scope_from_payload` as diagnostic parser authority.
- Reuses V1 canonical validation/helpers instead of creating permissive parallel Signal metric validation.

- [ ] **Step 1: Build a deterministic complete-sidecar test fixture**

Starting from the existing V1 synthetic run, create valid `CausalAlphaV3SignalDiagnosticScope` objects for every canonical identity. For each scope:

- prediction decisions are inside the canonical contract;
- `realized_24h_rows`, `realized_72h_rows`, and `realized_fused_rows` each contain at least two rows;
- 72h `raw_prediction` / `raw_realized_return` are exactly three times their 24h-equivalent stored values;
- models are exactly equal across symbols for the same fit/chronological episode;
- model coefficients and scaler state vary deterministically across chronological episodes;
- availability contains both complete and incomplete rows;
- forecast digest is calculated through `CausalAlphaV3Forecast` from the persisted prediction rows and horizon residual RMSE;
- the corresponding canonical metric is rewritten with that exact `forecast_digest`, then its digest and the rejection evidence metric-digest set are regenerated through the existing canonical Signal gate helper.

Persist each sidecar only at:

```text
signal/diagnostics/<fit_config_digest>/<symbol>/<episode_index>.json
```

- [ ] **Step 2: Write RED binding tests**

Assert complete mode succeeds, then independently falsify:

```text
empty diagnostics directory
one missing sidecar
one extra sidecar
wrong sidecar path
wrong run_manifest_digest
wrong fit_config_digest
wrong symbol
wrong episode_index
wrong contract interval/digest
wrong signal_metric_digest
wrong fit_digest
wrong forecast_digest
wrong canonical_cohort_indices
corrupt outer artifact digest
```

Every case must raise instead of falling back to historical mode.

- [ ] **Step 3: Run the binder tests and verify RED**

Expected: sidecar-complete cases fail because the V2 loader/binder does not exist.

- [ ] **Step 4: Implement exact canonical/diagnostic bijection**

The loader must:

1. obtain already-V1-validated canonical metrics through the V1 module's existing strict helper path;
2. require `diagnostics_root.is_dir()` once the path exists;
3. recursively discover only JSON leaves and decode each with `signal_diagnostic_scope_from_payload`;
4. verify the persisted path equals the path implied by decoded identity;
5. reject duplicate identities;
6. require `set(diagnostic_identities) == set(metric_identities)`;
7. bind each pair and check exact equality for run, fit config, symbol, episode, interval, contract digest, signal metric digest, fit digest, forecast digest, and canonical cohort.

- [ ] **Step 5: Run binder and V1 regression tests for GREEN**

Expected: valid complete fixture enters `sidecar_complete`; every malformed/partial fixture fails closed; V1 remains unchanged.

---

### Task 4: Add pure horizon and prediction-distribution diagnostics

**Files:**
- Create: `trade_rl/workflows/universal_causal_alpha_v3_signal_forensics_v2_analysis.py`
- Modify: `trade_rl/workflows/universal_causal_alpha_v3_signal_forensics_v2_contracts.py`
- Modify: `trade_rl/workflows/universal_causal_alpha_v3_signal_forensics_v2.py`
- Extend: `tests/workflows/test_universal_causal_alpha_v3_signal_forensics_v2.py`

**Interfaces:**
- Reuses `evaluate_causal_alpha_signal_diagnostics(predicted, realized)` and `CAUSAL_ALPHA_SIGNAL_QUANTILES`.
- Produces per-scope horizon diagnostics plus deterministic prediction-distribution summaries.

- [ ] **Step 1: Write RED horizon tests**

For one known scope, assert exact expected input units by independently evaluating:

```python
expected_24h = evaluate_causal_alpha_signal_diagnostics(
    [row.prediction for row in diagnostic.realized_24h_rows],
    [row.realized_return for row in diagnostic.realized_24h_rows],
)
expected_72h = evaluate_causal_alpha_signal_diagnostics(
    [row.prediction for row in diagnostic.realized_72h_rows],
    [row.realized_return for row in diagnostic.realized_72h_rows],
)
expected_fused = evaluate_causal_alpha_signal_diagnostics(
    [row.prediction for row in diagnostic.realized_fused_rows],
    [row.realized_return for row in diagnostic.realized_fused_rows],
)
```

Assert V2 payloads equal these exact existing diagnostic payloads.

- [ ] **Step 2: Write RED paired 24h-vs-72h test**

Intersect 24h and 72h realized rows by `decision_index`, preserve chronological order, and assert only matched decisions are used. If fewer than two matched rows remain, assert an explicit unavailable reason instead of widening the sample set.

- [ ] **Step 3: Write RED distribution tests**

For each scope, verify deterministic summaries for:

```text
prediction_24h
prediction_72h_24h_equivalent
expected_return_24h_equivalent
uncertainty_24h_equivalent
signal_to_uncertainty
```

Each summary must include count, mean, std, min, max, and quantiles at exactly `(0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)`.

- [ ] **Step 4: Run analysis tests and verify RED**

Expected: missing V2 analysis contracts/builders.

- [ ] **Step 5: Implement pure horizon/distribution builders**

No filesystem reads are permitted in the analysis module. Convert only already-bound typed rows into NumPy arrays and existing diagnostic contracts. Never feed raw sidecar rows into Signal Gate/bootstrap functions.

- [ ] **Step 6: Run focused tests for GREEN**

Expected: existing diagnostic semantics are reproduced exactly and distribution quantiles are deterministic.

---

### Task 5: Add model stability, weighted ESS, and availability diagnostics

**Files:**
- Modify: `trade_rl/workflows/universal_causal_alpha_v3_signal_forensics_v2_analysis.py`
- Modify: `trade_rl/workflows/universal_causal_alpha_v3_signal_forensics_v2_contracts.py`
- Extend: `tests/workflows/test_universal_causal_alpha_v3_signal_forensics_v2.py`
- Extend: `tests/workflows/test_universal_causal_alpha_v3_signal_forensics_v2_falsification.py`

**Interfaces:**
- Produces deduplicated episode model snapshots and consecutive transition summaries per fit/horizon.

- [ ] **Step 1: Write RED cross-symbol deduplication test**

For each `(fit_config_digest, contract_start, contract_stop)`, assert one retained model snapshot despite multiple symbols. Mutate one symbol's model coefficient, residual RMSE, ESS, scaler field, or overlap-weight digest and recompute the sidecar's outer digest; V2 must reject the cluster before summarization.

- [ ] **Step 2: Write RED coefficient/scaler transition tests**

For consecutive snapshots with coefficient vectors `a` and `b`, assert:

```python
cosine = dot(a, b) / (norm(a) * norm(b))
active = [(x, y) for x, y in zip(a, b) if x != 0.0 and y != 0.0]
sign_flip_rate = sum(x * y < 0.0 for x, y in active) / len(active)
location_shift_rms = sqrt(mean(((mu1 - mu0) / s0) ** 2))
log_scale_ratio_rms = sqrt(mean(log(s1 / s0) ** 2))
```

If either coefficient norm is `<= 1e-15`, cosine is unavailable with an explicit reason. If `active` is empty, sign-flip rate is unavailable.

- [ ] **Step 3: Write RED ESS/residual tests**

Assert each deduplicated snapshot exposes fitted row count, weighted residual RMSE, pooled weighted ESS, per-symbol weighted ESS, model digest, and overlap-weight digest. Assert chronological min/mean/max plus early/late/slope summaries use episode order only.

- [ ] **Step 4: Write RED availability tests**

Partition realized rows by `available_feature_fraction == 1.0` vs `< 1.0`. For partitions with at least two rows, call the existing signal diagnostic evaluator. For fewer than two rows, return an explicit unavailable reason. Report per-feature availability fractions and row-level availability-fraction distributions without claiming a missing feature caused a row error.

- [ ] **Step 5: Run model/availability tests and verify RED**

Expected: missing builders/contracts or acceptance of intentionally drifted duplicate pooled model evidence.

- [ ] **Step 6: Implement model dedupe and pure summaries**

Compare exact pooled model payloads across symbols before selecting the first snapshot. Preserve feature order and fail if it drifts between consecutive snapshots. Interpret weight/model digest changes only as identity transitions; do not label them quality degradation.

- [ ] **Step 7: Run focused tests for GREEN**

Expected: one fit-level snapshot per episode, correct transition formulas, strict duplicate consistency, and truthful availability limits.

---

### Task 6: Add chronological V2 aggregation and explicit limitations

**Files:**
- Modify: `trade_rl/workflows/universal_causal_alpha_v3_signal_forensics_v2_analysis.py`
- Modify: `trade_rl/workflows/universal_causal_alpha_v3_signal_forensics_v2_contracts.py`
- Extend: `tests/workflows/test_universal_causal_alpha_v3_signal_forensics_v2.py`

**Interfaces:**
- Produces fit-level and chronological summaries from per-scope diagnostics without inventing market regimes.

- [ ] **Step 1: Write RED early/late/slope tests**

Use authored `(contract_start, contract_stop)` ordering only. For each fit/horizon/metric series, assert deterministic early mean, late mean, and ordinary least-squares slope over ordinal episode index. A single episode has slope `0.0` and identical early/late values.

- [ ] **Step 2: Write RED limitation tests**

Assert V2 does not expose or claim:

```text
row-level missing-feature causality
reconstructable canonical ridge-model digest
market regime labels
independent-sample confidence from overlapping raw rows
promotion eligibility
```

These remain explicit in `unavailable_analyses` / limitations.

- [ ] **Step 3: Implement fit/chronological aggregation**

Aggregate scope diagnostics only after complete sidecar binding and model dedupe. Use chronological contract order and train-symbol scope inherited from V1. Never reinterpret raw prediction rows as independent episodes.

- [ ] **Step 4: Run focused tests for GREEN**

Expected: deterministic fit/episode aggregation and no unsupported regime/causal claims.

---

### Task 7: Add explicit CLI schema selection without breaking V1 default

**Files:**
- Modify: `scripts/analyze_universal_causal_alpha_v3_signal.py`
- Modify: `tests/scripts/test_analyze_universal_causal_alpha_v3_signal.py`

**Interfaces:**
- CLI adds exactly:

```text
--schema {v1,v2}
```

with default `v1`.

- [ ] **Step 1: Write CLI RED tests**

Assert no `--schema` calls only `load_causal_alpha_v3_signal_forensics`. Assert `--schema v1` does the same. Assert `--schema v2` calls only `load_causal_alpha_v3_signal_forensics_v2`. Both use the same canonical JSON and external-output safety path.

- [ ] **Step 2: Run CLI tests and verify RED**

Expected: parser rejects `--schema` before implementation.

- [ ] **Step 3: Implement explicit selector**

Add:

```python
parser.add_argument("--schema", choices=("v1", "v2"), default="v1")
```

Dispatch explicitly. Do not auto-detect V2 based on sidecar presence.

- [ ] **Step 4: Run CLI and V1 compatibility tests for GREEN**

Expected: legacy invocation remains byte-for-byte V1 JSON; explicit V2 returns V2 schema.

---

### Task 8: Falsification, architecture review, and full quality gate

**Files:**
- Review all new/modified V1/V2 analyzer, CLI, tests, spec, and plan files.
- Update `docs/UNIVERSAL_TRAINING.md` only with a concise V2 consumer paragraph after software behavior is verified.

**Interfaces:**
- No new runtime/learner/promotion dependency from the V2 analyzer.

- [ ] **Step 1: Run targeted V1+V2 tests**

```bash
uv run pytest -q \
  tests/workflows/test_universal_causal_alpha_v3_signal_forensics.py \
  tests/workflows/test_universal_causal_alpha_v3_signal_forensics_falsification.py \
  tests/workflows/test_universal_causal_alpha_v3_signal_forensics_v2.py \
  tests/workflows/test_universal_causal_alpha_v3_signal_forensics_v2_falsification.py \
  tests/scripts/test_analyze_universal_causal_alpha_v3_signal.py
```

- [ ] **Step 2: Run targeted static/architecture checks**

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run lint-imports
```

- [ ] **Step 3: Perform falsification review against the original specification**

Attempt to prove acceptance of:

```text
partial sidecar graph
extra sidecar
identity drift with recomputed outer digest
cross-symbol pooled-model disagreement
24h/72h unmatched-row leakage
72h raw-unit mixing
V1 default-output drift
source-run mutation
absolute-path-dependent report digest
sidecar evidence entering Gate/selection/Teacher/learner code
```

Fix any defect found and rerun the nearest tests before broad verification.

- [ ] **Step 4: Perform architecture/self review**

Review responsibility boundaries, imports, public APIs, serialization determinism, naming/units, side effects, error propagation, duplicate logic, dead code, and documentation. Confirm V2 remains a consumer-only research path.

- [ ] **Step 5: Run full repository verification**

Use the repository CI-equivalent full pytest+branch coverage, workflow security, critical coverage, frontend, compatibility, training-image, package identity, and PostgreSQL jobs on the same exact final HEAD.

Quality gate requires all exact-head checks Green. A prior commit's CI does not count.

- [ ] **Step 6: Independent-style re-review**

Because no independent subagent runtime is available, reconstruct the review from the original V2 acceptance criteria and actual final diff/test assertions rather than from implementation intent. Record unresolved limitations and residual risks in the PR body.

- [ ] **Step 7: Do not merge automatically**

Leave the implementation PR Draft/Ready according to the quality gate and report exact final HEAD, CI runs, guarantees, non-guarantees, and remaining empirical dependency. Merge requires a separate explicit user decision.
