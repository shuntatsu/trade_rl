# Causal Alpha V3 Market/Residual Counterfactual Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Falsify or admit one predeclared train-only market/residual ridge hypothesis with the unchanged canonical Signal V2 evaluator on the immutable real Binance Universal runtime.

**Architecture:** Build a disposable ignored experiment library that constructs one cross-symbol market feature/label row per decision, fits 24h/72h market ridge heads plus 24h/72h per-symbol residual ridge heads at each expanding cutoff, and sends only their summed `CausalAlphaV3Forecast` into the existing canonical metric and clustered Gate. Persist the exact script, tests, identities, per-episode evidence, final Gate evidence, and logs in one retained generation. Stop after Signal rejection; write a separate tracked production plan only after a complete Signal pass.

**Tech Stack:** Python 3.12, NumPy, existing `fit_causal_alpha_ridge`, Causal Alpha V3 Signal contracts, pytest, Docker, immutable Universal runtime artifacts.

**Spec:** `docs/implementation-plans/specs/2026-08-22-causal-alpha-v3-market-residual-decomposition-design.md`

## Global Constraints

- Use runtime content digest `6726b3737df9fbacf6787f3d02894e846c512a840bec4dd037538a02af1480b0` and the existing nine train symbols only.
- Use exactly eight Signal contracts, market ridge `1.0`, and residual ridge `0.1`; do not expand the grid after seeing results.
- Preserve canonical 24h/72h fusion, realized-return fusion, cohort construction, moving-block bootstrap, seed, thresholds, coverage, and independent-episode semantics.
- Read no Teacher holdout, validation, or test result for fitting, tuning, or acceptance.
- Do not modify tracked production Python, reward, target, execution, risk, Gate, selection, Teacher, BC, or RL code in this counterfactual.
- A pass requires every maintained Signal requirement; positive means or two passing lower bounds are not sufficient.
- A failure must persist evidence and stop before production implementation, selection, Teacher admission, BC, or RL.

---

## File Structure

- Create ignored `var/market_residual_counterfactual_lib.py`: pure aggregate construction, label decomposition, fitting, prediction, metric assembly, artifact payload, and CLI entrypoint.
- Create ignored `var/test_market_residual_counterfactual.py`: focused TDD tests for aggregation, label decomposition, identity, cutoff, and component-sum behavior.
- Create ignored `var/run_market_residual_counterfactual.py`: minimal executable wrapper calling the library `main()`.
- Create retained `var/retained-causal-alpha-v3/market-residual-cf-20260822-r1/`: copied source/tests, launch manifest, log, per-scope JSONL, per-episode JSONL, and final result.
- Do not edit tracked `trade_rl/` or `tests/` files during this plan.

### Task 1: Pure market aggregation and label decomposition

**Files:**
- Create: `var/test_market_residual_counterfactual.py`
- Create: `var/market_residual_counterfactual_lib.py`

**Interfaces:**
- Consumes: aligned `Mapping[str, CausalAlphaSymbolSamples]`, ordered `tuple[str, ...]`, and decision indices.
- Produces: `MarketAggregateInputs(features, feature_available, matched, digest)` and `MarketResidualLabels(market_labels, residual_labels, label_end_indices, eligible, digest)`.

- [ ] **Step 1: Write failing aggregation tests**

```python
def test_market_aggregate_uses_available_mean_and_fraction():
    # Two symbols, two decisions, two source features.
    features = np.asarray(
        [
            [[2.0, 10.0], [4.0, 20.0]],
            [[6.0, 30.0], [8.0, 40.0]],
        ]
    )
    available = np.asarray(
        [
            [[True, False], [True, True]],
            [[True, True], [False, True]],
        ]
    )
    result = aggregate_market_feature_arrays(
        features=features,
        feature_available=available,
        feature_names=("a", "b"),
        train_symbols=("A", "B"),
    )
    np.testing.assert_allclose(result.features, [[4.0, 30.0, 1.0, 0.5], [4.0, 30.0, 0.5, 1.0]])
    assert result.feature_available.tolist() == [[True, True, True, True], [True, True, True, True]]
    assert result.feature_names == (
        "market_mean::a",
        "market_mean::b",
        "market_available_fraction::a",
        "market_available_fraction::b",
    )


def test_market_residual_labels_sum_back_to_symbol_returns():
    labels = np.asarray([[0.03, -0.01], [0.01, 0.05]])
    ends = np.asarray([[10, 20], [10, 20]])
    result = decompose_market_residual_labels(
        labels=labels,
        label_end_indices=ends,
        train_symbols=("A", "B"),
    )
    expected_market = np.asarray([0.02, 0.02])
    np.testing.assert_allclose(result.market_labels, expected_market)
    np.testing.assert_allclose(
        result.residual_labels + expected_market[None, :], labels
    )
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
python -m pytest var/test_market_residual_counterfactual.py -q
```

Expected: collection/import failure because `market_residual_counterfactual_lib` and its contracts do not exist.

- [ ] **Step 3: Implement immutable pure contracts and constructors**

Implement frozen dataclasses with copied read-only NumPy arrays and `content_and_arrays_digest` identities. `aggregate_market_feature_arrays` must validate rank-3 `(symbol, row, feature)` arrays, exact ordered symbol count, finite source features, aligned availability, and unique feature names. Compute available sums with `np.where(available, features, 0.0)`, divide only where counts are positive, append availability fractions, and mark a mean channel unavailable only when its count is zero.

`decompose_market_residual_labels` must require rank-2 `(symbol, row)` labels, exact identical label-end arrays across symbols, and either all-finite or all-unavailable labels per row. Compute the equal-weight market label only for complete rows; keep unavailable market/residual labels as `NaN`; require exact finite reconstruction within `1e-15`; and bind symbol order, labels, ends, and schema in the digest.

Use these exact public signatures:

```python
def aggregate_market_feature_arrays(
    *,
    features: object,
    feature_available: object,
    feature_names: tuple[str, ...],
    train_symbols: tuple[str, ...],
) -> MarketAggregateInputs: ...


def decompose_market_residual_labels(
    *,
    labels: object,
    label_end_indices: object,
    train_symbols: tuple[str, ...],
) -> MarketResidualLabels: ...
```

- [ ] **Step 4: Add falsification tests for incomplete universe and identity drift**

```python
def test_market_labels_reject_partial_symbol_realization():
    labels = np.asarray([[0.03], [np.nan]])
    ends = np.asarray([[10], [-1]])
    with pytest.raises(ValueError, match="complete fixed universe"):
        decompose_market_residual_labels(
            labels=labels,
            label_end_indices=ends,
            train_symbols=("A", "B"),
        )


def test_market_aggregate_digest_binds_symbol_order():
    left = aggregate_market_feature_arrays(
        features=np.asarray([[[1.0]], [[2.0]]]),
        feature_available=np.ones((2, 1, 1), dtype=np.bool_),
        feature_names=("x",),
        train_symbols=("A", "B"),
    )
    right = aggregate_market_feature_arrays(
        features=np.asarray([[[2.0]], [[1.0]]]),
        feature_available=np.ones((2, 1, 1), dtype=np.bool_),
        feature_names=("x",),
        train_symbols=("B", "A"),
    )
    assert left.digest != right.digest
```

- [ ] **Step 5: Run focused tests and verify GREEN**

Run:

```powershell
python -m pytest var/test_market_residual_counterfactual.py -q
```

Expected: all Task 1 tests pass.

### Task 2: Expanding two-head fit and exact composite forecast

**Files:**
- Modify: `var/test_market_residual_counterfactual.py`
- Modify: `var/market_residual_counterfactual_lib.py`

**Interfaces:**
- Consumes: `MarketAggregateInputs`, horizon-specific `MarketResidualLabels`, original target-local sample arrays, knowledge cutoff, market ridge `1.0`, residual ridge `0.1`.
- Produces: `MarketResidualFit` with two market models, ordered per-symbol residual models, directly measured composite RMSE values, component digests, and `predict(...) -> MarketResidualForecast`.

- [ ] **Step 1: Write failing synthetic fit test**

Create deterministic synthetic aligned samples with at least 40 decisions and two symbols. Make the common component depend on the cross-symbol mean feature and each residual depend on its local feature. Assert:

```python
fit = fit_market_residual(
    train_symbols=("A", "B"),
    samples=samples,
    knowledge_cutoff=30,
    market_ridge_strength=1.0,
    residual_ridge_strength=0.1,
)
forecast = fit.predict(samples=samples, symbol="A", decision_indices=np.arange(30, 36))
np.testing.assert_array_equal(
    forecast.forecast.prediction_24h,
    forecast.market_prediction_24h + forecast.residual_prediction_24h,
)
assert fit.market_models["24h"].knowledge_cutoff == 30
assert fit.residual_models["A"]["72h"].knowledge_cutoff == 30
```

- [ ] **Step 2: Run the synthetic fit test and verify RED**

Run the named pytest node and expect failure because `fit_market_residual` is undefined.

- [ ] **Step 3: Implement fit assembly**

Stack the nine symbol samples only after requiring identical feature names, feature-schema digests, decision-index arrays, and horizon label-end arrays. Build market inputs once. Fit each market horizon with `fit_causal_alpha_ridge(..., normalize_objective=True)` and temporal uniqueness weights from `causal_alpha_overlap_uniqueness_weights`. Fit each residual horizon from the target-local features and residual label using the same cutoff and uniqueness rule.

Compute final in-prefix predictions as market plus residual. Compute composite horizon RMSE from the summed forecast against the original symbol label using per-symbol normalized uniqueness weights, then pooled equal symbol mass. Persist both component RMSEs and composite RMSEs. Construct the public final bundle only through:

```python
causal_alpha_v3_forecast(
    prediction_24h,
    prediction_72h,
    residual_rmse_24h=fit.composite_rmse_24h,
    residual_rmse_72h=fit.composite_rmse_72h,
)
```

- [ ] **Step 4: Add cutoff and wrong-symbol falsification tests**

Assert that modifying any label ending at or after the cutoff cannot change model digests, that a requested symbol outside the ordered residual-model mapping raises `ValueError`, and that a deliberately permuted source decision array fails before fitting.

- [ ] **Step 5: Run all ignored counterfactual tests and verify GREEN**

Run:

```powershell
python -m pytest var/test_market_residual_counterfactual.py -q
```

Expected: all aggregation, label, fit, prediction, cutoff, and identity tests pass.

### Task 3: Canonical Signal metric adapter and retained evidence

**Files:**
- Modify: `var/test_market_residual_counterfactual.py`
- Modify: `var/market_residual_counterfactual_lib.py`
- Create: `var/run_market_residual_counterfactual.py`

**Interfaces:**
- Consumes: prepared immutable runtime, `split_causal_alpha_v3_partitions`, `MarketResidualFit`, current `CausalAlphaV3SignalScopeMetric`, and `evaluate_causal_alpha_v3_signal_gate_clustered`.
- Produces: exactly 72 scope metrics, eight episode summaries, final Gate evidence, and retained immutable experiment files.

- [ ] **Step 1: Write failing canonical metric adapter test**

Use a controlled contract and prediction arrays. Require the adapter to use:

```python
prediction = forecast.expected_return_24h_equivalent[cohort_rows]
realized = 0.5 * (labels_24h[cohort_rows] + labels_72h[cohort_rows] / 3.0)
```

Assert cohort rows come from `non_overlapping_causal_alpha_v3_rows` with 72-hour label ends, rank uses `evaluate_causal_alpha_signal_diagnostics`, direction accuracy is unchanged, and spread uses stable `mergesort` top/bottom fifth buckets.

- [ ] **Step 2: Run the adapter test and verify RED**

Run the named pytest node and expect failure because `build_counterfactual_scope_metric` is undefined.

- [ ] **Step 3: Implement the canonical adapter and experiment CLI**

The CLI must load:

```python
UniversalRuntimeFactoryContext(
    runtime_manifest_path=Path("/workspace/var/universal/runtime-manifest.json"),
    frozen_metadata_root=Path("/workspace/var/cache/frozen-metadata/usds-m"),
)
```

Build the runtime from `examples/binance-multitimeframe/universal-u6-ppo.json`, load the current research Gate config, prepare train data, and require the exact expected runtime digest and nine-symbol tuple. Split with `signal_contract_count=8` and the maintained minimum economic count. At each contract cutoff, fit exactly once, generate all nine scope metrics, append one JSONL record per scope plus one per episode with fsync, and print the episode summary with `flush=True`.

The final call must be exactly:

```python
evidence = evaluate_causal_alpha_v3_signal_gate_clustered(
    tuple(metrics),
    expected_raw_scope_count=72,
    expected_independent_episode_count=8,
    gate=research_config.signal_gate,
)
```

- [ ] **Step 4: Add retained artifact validation tests**

Create a temporary completed result with 72 scopes and verify that the reader rejects a missing scope, duplicate scope, wrong runtime digest, changed fit config, absent script digest, non-finite component evidence, or a declared `passed=True` inconsistent with the canonical Gate payload.

- [ ] **Step 5: Run the full ignored test file and compile check**

Run:

```powershell
python -m pytest var/test_market_residual_counterfactual.py -q
python -m py_compile var/market_residual_counterfactual_lib.py var/run_market_residual_counterfactual.py
```

Expected: all tests and compilation pass.

### Task 4: Execute the immutable train-only counterfactual

**Files:**
- Read: `var/market_residual_counterfactual_lib.py`
- Read: `var/run_market_residual_counterfactual.py`
- Create: `var/retained-causal-alpha-v3/market-residual-cf-20260822-r1/*`

**Interfaces:**
- Consumes: exact scripts from Tasks 1–3, image `trade-rl-causal-alpha-v3:c6303dd12ff3-6726b3737df9`, volume `trade-rl-training-data`, and the read-only Universal artifact bind.
- Produces: an immutable retained experiment with one authoritative `result.json` whose `passed` field is the existing Gate result.

- [ ] **Step 1: Freeze launch identity before execution**

Compute SHA-256 digests for both scripts and the ignored test, record source HEAD, source-tree status, image ID, runtime digest, research-config digest, authored model parameters, expected scope counts, output path, and launch timestamp in `launch-manifest.json`. Copy the exact scripts and test into `source/` under the retained directory before launch.

- [ ] **Step 2: Launch one named Docker container**

Use a new container name and read-only source/script/artifact mounts. Mount only the retained output path writable. Set numerical thread variables consistently with the canonical runner. Do not reuse or alter `sidecar-fresh-20260822-r1`.

- [ ] **Step 3: Monitor every completed episode**

After each of eight episode records, validate record count, cutoff order, nine-symbol coverage, component finiteness, fit/prediction identities, rank/spread/direction means, CPU/RAM, OOM state, and log progress. Do not infer Gate passage before the final clustered evidence exists.

- [ ] **Step 4: Validate completion artifacts independently**

Require container exit `0`, OOM false, 72 unique scope identities, eight unique chronological episodes, full raw coverage, matching source/runtime/config/script digests, and a recomputed canonical Gate payload equal to `result.json`.

- [ ] **Step 5: Apply the scientific stop/go rule**

If `evidence.passed` is false, preserve the generation, record every rejection reason and lower bound, leave production code untouched, and return to architecture diagnosis. If true, preserve the same evidence, update the working plan, and write a separate production TDD implementation plan covering tracked config, fit, forecast, sidecar V2, store/pipeline, canonical runner, and new immutable training generation.

### Task 5: Counterfactual review and handoff

**Files:**
- Read: retained generation files from Task 4
- Modify: this plan only to mark executed checkboxes after evidence exists
- Create conditionally after a pass: `docs/superpowers/plans/2026-08-22-causal-alpha-v3-market-residual-production.md`

**Interfaces:**
- Consumes: independently validated Task 4 evidence.
- Produces: either a documented rejected hypothesis with no tracked production implementation, or a production implementation plan grounded in a passed real-data Signal result.

- [ ] **Step 1: Re-run ignored tests after the experiment**

Run the full ignored pytest file and verify copied retained sources have the same digests as executed sources.

- [ ] **Step 2: Self-review for leakage and evaluator drift**

Inspect every experiment reference to `holdout`, `validation`, `test`, Gate thresholds, bootstrap, cohort, fusion, and reward. The only permitted holdout/validation/test references are negative assertions or unchanged metadata; no such data may affect results.

- [ ] **Step 3: Report the exact decision**

Report source HEAD, image ID, runtime/config/script digests, artifact directory, scope/episode counts, three means and confidence bounds, Gate status/reasons, OOM/exit state, and the next gated action. Never describe a partial metric pass as model admission.
