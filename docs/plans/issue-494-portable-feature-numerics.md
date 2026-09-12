# Portable Feature Numerics Implementation Plan

Status: Active

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove CPU-dispatched floating-point drift from identity-bound `MarketDataset` feature construction and prove byte-identical canonical Dataset identity across independent hosted CPU families.

**Architecture:** Add one internal portable-numerics module using scalar `math.log` and fixed-order `math.fsum`, then route only Dataset feature/global-feature construction through it. Version the build semantics as `market_build_v3` + `portable_feature_numerics_v1`; keep Dataset artifact/identity container readers backward compatible and keep Study 004 immutable.

**Tech Stack:** Python 3.12, NumPy 1.26.4, pytest, Ruff, Mypy, GitHub Actions.

**Spec:** `docs/specs/issue-494-portable-feature-numerics.md`

## Global Constraints

- Do not round or quantize stored research features as a hash workaround.
- Do not modify `risk/`, `simulation/`, forecast/RL fitting numerics, execution economics, PPO Observation v2, or Study 004.
- Stored Dataset feature/global-feature dtype remains `float32`.
- `MARKET_DATASET_IDENTITY_SCHEMA` remains `market_dataset_identity_v6`.
- New build semantics are `market_build_v3` with `feature_numerics_schema="portable_feature_numerics_v1"`.
- Unsupported explicit build schema versions fail closed.
- Final oracle is byte equality across independent hosted CPU families using one exact implementation and one sealed source universe.

---

### Task 1: Portable numerical primitives and schema RED/GREEN

**Files:**
- Create: `tests/data/test_portable_feature_numerics.py`
- Create: `trade_rl/data/features/numerics.py`
- Modify: `trade_rl/data/contracts.py`

**Interfaces:**
- Produces `PORTABLE_FEATURE_NUMERICS_SCHEMA = "portable_feature_numerics_v1"`.
- Produces `portable_log_scalar`, `portable_log`, `portable_sum`, `portable_mean`, `portable_variance`, `portable_std`, `portable_dot`, `portable_covariance`, `portable_correlation`.
- `MarketBuildConfig.canonical_payload()` emits `schema_version="market_build_v3"` and `feature_numerics_schema`.

- [ ] **Step 1: Write failing primitive/schema tests**

```python
values = np.asarray([1e16, 1.0, -1e16, 3.0], dtype=np.float64)
assert portable_sum(values) == 4.0
assert portable_mean(values) == 1.0
assert portable_dot(values, np.ones(4, dtype=np.float64)) == 4.0
assert portable_variance(np.asarray([1.0, 2.0, 3.0])) == pytest.approx(2.0 / 3.0)
assert portable_std(np.asarray([1.0, 2.0, 3.0])) == pytest.approx(math.sqrt(2.0 / 3.0))
assert portable_correlation(
    np.asarray([1.0, 2.0, 3.0]),
    np.asarray([2.0, 4.0, 6.0]),
) == 1.0
with pytest.raises(ValueError):
    portable_log(np.asarray([1.0, 0.0]))

config = MarketBuildConfig(
    base_timeframe="1h",
    features=(FeatureSpec(name="ret", kind=FeatureKind.LOG_RETURN),),
)
assert config.canonical_payload()["schema_version"] == "market_build_v3"
assert config.canonical_payload()["feature_numerics_schema"] == PORTABLE_FEATURE_NUMERICS_SCHEMA
with pytest.raises(ValueError, match="unsupported market build schema"):
    MarketBuildConfig(
        base_timeframe="1h",
        features=(FeatureSpec(name="ret", kind=FeatureKind.LOG_RETURN),),
        schema_version="market_build_v2",
    )
```

- [ ] **Step 2: Run targeted tests and verify RED**

Run: `uv run pytest -q tests/data/test_portable_feature_numerics.py`
Expected: collection/import fails because `trade_rl.data.features.numerics` does not exist.

- [ ] **Step 3: Implement minimal deterministic primitives**

`portable_sum` uses `math.fsum` in C-order and returns `0.0` for an empty sequence. Mean/variance/std require a non-empty 1-D finite sample. Dot/covariance/correlation require equal-size finite 1-D inputs. `portable_log[_scalar]` requires finite positive values and evaluates each element with `math.log`.

- [ ] **Step 4: Bump/validate build semantics**

Set `MarketBuildConfig.schema_version` default to `market_build_v3`, reject every other explicit value in `__post_init__`, and add `feature_numerics_schema` to `canonical_payload()`.

- [ ] **Step 5: Run targeted tests GREEN, Ruff, Mypy**

Run:
`uv run pytest -q tests/data/test_portable_feature_numerics.py`
`uv run ruff check trade_rl/data/features/numerics.py trade_rl/data/contracts.py tests/data/test_portable_feature_numerics.py`
`uv run mypy trade_rl/data/features/numerics.py trade_rl/data/contracts.py`

### Task 2: Real mismatch-window RED and local/core feature routing

**Files:**
- Modify: `tests/data/test_portable_feature_numerics.py`
- Modify: `trade_rl/data/features/core.py`

**Interfaces:**
- Consumes portable primitives from Task 1.
- `calculate_feature_events()` retains its signature and causal/availability contract.

- [ ] **Step 1: Add the exact real mismatch fixtures**

Use these sealed-source cases from Issue #494:

```python
TREND_CLOSE = np.asarray([
    29015.0, 29448.4, 29237.06, 29302.11, 29237.07, 29213.8,
    29197.3, 29107.71, 29025.89, 29229.6, 29259.29, 29341.99,
    29257.82, 29493.66, 29354.58, 29210.84, 29324.21, 29099.0,
    29086.62, 29048.47, 29214.14, 29177.77, 29270.89, 29337.16,
], dtype=np.float64)

CORR_CLOSE = np.asarray([
    2843.24, 2856.95, 2850.99, 2933.6, 2954.63, 2944.99, 2935.22,
    2925.0, 2939.76, 2942.98, 2939.95, 2946.1, 2941.37, 2944.51,
    2931.12, 2927.95, 2933.63, 2924.87, 2923.3, 2928.07, 2925.4,
    2929.04, 2893.79, 2829.29, 2740.65,
], dtype=np.float64)
CORR_VOLUME = np.asarray([
    178371479.99161, 217700240.79219, 170218251.66315, 985906310.00321,
    499528188.98515, 123546521.23818, 90631407.76651, 115347509.62008,
    124767163.22539, 183858682.93128, 72922996.68414, 83538939.41229,
    48627626.81203, 42328264.51671, 102484093.06324, 69669619.92084,
    73489528.86574, 178626929.63437, 100539933.68714, 96449855.22685,
    68576812.92353, 81692701.41202, 292732021.44597, 871571360.35913,
    1265830807.72211,
], dtype=np.float64)

REALIZED_VOL_CLOSE = np.asarray(
    [63283.1, 63146.6, 63192.1, 63255.1, 63237.4], dtype=np.float64
)
```

The test-only oracle uses scalar `math.log` and `math.fsum`; it must not import the production portable helpers. Assert exact final `np.float32` bit patterns.

- [ ] **Step 2: Add a static architecture RED guard**

AST-scan `trade_rl/data/features/core.py`, `trade_rl/data/features/cross_asset.py`, and `trade_rl/data/build/builder.py`. Reject calls to `np.log`, `np.mean`, `np.std`, `np.sum`, `np.dot`, `np.var`, `np.cov`, and `np.corrcoef`, plus ndarray `.mean(...)` in the builder global-feature path. This test fails on current code even on a CPU whose numeric result happens to match the portable oracle.

- [ ] **Step 3: Verify RED against current implementation**

Run the mismatch/static tests and require a failure caused by a forbidden NumPy identity-path call.

- [ ] **Step 4: Route local/core calculations**

Use `portable_log`, `portable_mean`, `portable_std`, `portable_sum`, `portable_dot`, and `portable_correlation` for every matching identity-affecting call in `core.py`: rolling normalization; Wilder/RSI initialization; funding z-score; Parkinson/Garman-Klass; downside/upside/vol-of-vol; range expansion; trend regression; MFI; CMF; VWAP; price-volume correlation; OBV denominators; relative volume; realized volatility; volume z-score; Bollinger; stochastic-D; CCI. Keep `np.min`, `np.max`, `np.diff`, masks, stable ordering, clipping, and exact indexing operations.

- [ ] **Step 5: Run focused feature suites GREEN**

Run:
`uv run pytest -q tests/data/test_portable_feature_numerics.py tests/data/test_extended_indicator_features_v2.py tests/data/test_extended_prefix_causality.py tests/data/test_multitimeframe_builder.py`

### Task 3: Cross-asset and global-feature portable semantics

**Files:**
- Modify: `trade_rl/data/features/cross_asset.py`
- Modify: `trade_rl/data/build/builder.py`
- Modify: `tests/data/test_cross_asset_features.py`
- Modify: `tests/data/test_one_bar_returns.py`
- Modify: `tests/data/test_complete_96_feature_dataset.py`

**Interfaces:**
- Cross-asset feature function signature remains unchanged.
- `_calculate_one_bar_returns()` remains unchanged externally.

- [ ] **Step 1: Add failing portable integration assertions**

Add an independent `math.fsum` oracle for rolling cross-asset variance/covariance/correlation and for global market-return mean/dispersion. Add one-bar-return expectations from scalar `math.log` rather than `np.log`.

- [ ] **Step 2: Verify RED through the static guard/integration expectations**

Run the three focused suites before production edits and retain the failure output.

- [ ] **Step 3: Route cross-asset calculations**

Replace stored-value `np.std`, `np.var`, `np.cov`, `np.corrcoef`, `np.mean`, and Python `sum` with portable primitives. Keep stable rank sorting and exact max-age operations unchanged.

- [ ] **Step 4: Route builder calculations**

Compute one-bar logs with `portable_log`; compute global market return mean/dispersion with portable mean/std; compute active/tradable fractions as `np.count_nonzero(row) / float(n_symbols)` for each row.

- [ ] **Step 5: Run focused suites GREEN**

Run:
`uv run pytest -q tests/data/test_cross_asset_features.py tests/data/test_one_bar_returns.py tests/data/test_market_builder.py tests/data/test_complete_96_feature_dataset.py`

### Task 4: Identity and compatibility closure

**Files:**
- Modify: `tests/data/test_market_builder.py`
- Modify: `tests/data/test_market_dataset_identity_v2.py`
- Modify: `tests/data/test_market_artifact.py`
- Inspect and update only if RED proves intentional v3 identity movement: `tests/evaluation/experiments/bootstrap/test_execution_economics_config.py`, `tests/evaluation/experiments/bootstrap/test_execution_economics_fullpath.py`, `tests/evaluation/experiments/bootstrap/test_execution_economics_workflow.py`, `tests/integrations/test_binance_execution_economics.py`

- [ ] **Step 1: Run full data/bootstrap/integration suites to enumerate exact failures**

Run: `uv run pytest -q tests/data tests/evaluation/experiments/bootstrap tests/integrations`

- [ ] **Step 2: Replace the obsolete pre-#494 fixed identity oracle**

Rename `test_builder_preserves_pre481_identity_without_execution_profile` to describe the v3 portable identity. Recompute its `dataset_id`, `feature_config_digest`, and `normalization_digest` from the fixed binary-exact source, then independently confirm the config payload contains only the intended v3 numerical-semantics change before updating expected constants.

- [ ] **Step 3: Update only additional fixed digests demonstrated by RED**

For each failing bootstrap/integration constant, inspect the payload/arrays and record why v3 changes it. Do not weaken equality or replace exact checks with tolerances.

- [ ] **Step 4: Add historical artifact-reader coverage**

Use an existing persisted v2 identity fixture/artifact, not a rebuild, and assert the current artifact reader verifies and loads it unchanged.

- [ ] **Step 5: Re-run data/bootstrap/integration suites GREEN**

### Task 5: Full quality gate before cross-runner evidence

**Files:** no production changes unless a reproduced defect requires returning to RED/GREEN.

- [ ] Run Ruff and format check.
- [ ] Run production Mypy and architecture-tooling Mypy.
- [ ] Run full `pytest -q tests`.
- [ ] Run build, sdist/direct-wheel/rebuilt-wheel source closure, clean non-editable install/public imports, Candidate CLI, bootstrap CLI, and package identity exactly as permanent CI does.
- [ ] Review current-main-relative diff for scope, dead/debug code, accidental contract changes, and performance hot paths.

### Task 6: Cross-runner full Dataset oracle

**Files:**
- Temporary branch-only workflow `.github/workflows/issue494-portable-dataset-cross-runner.yml`; delete it before PR Ready.

- [ ] **Step 1: Freeze exact implementation HEAD and source Artifact `10293251579` from run `34679031604`**

Verify outer digest `sha256:1bc5bfffc314a0aee951dbba2b5fb8a05e0ea7754bd5ee4faa4dc0f035145ce2`, raw-source roster digest `0293e7575533bb3bd8876f5b130495491b2496144bff2ac9a5d0ba78553fe6d0`, and Vision-resolution digest `ba0a247f5b32d82493e6ad67d106ccb60b1dedafb7948870179afbc904d15fcf` before each rebuild.

- [ ] **Step 2: Rebuild the complete canonical priced Dataset on at least eight independent `ubuntu-24.04` runners**

Each replica records CPU model, NumPy runtime, feature/global-feature SHA-256, feature config digest, normalization digest, Dataset ID, economic-array checks, and wall-clock build duration.

- [ ] **Step 3: Aggregate immutable replica artifacts without recomputation**

Require exact equality for `features`, `global_features`, `feature_config_digest`, `normalization_digest`, and Dataset ID across every replica. Require at least two observed CPU model families; if the first eight are homogeneous, run another eight rather than weakening the oracle.

- [ ] **Step 4: Falsification control**

In the same workflow, run the three known v2 mismatch windows through the old NumPy path and the new portable path. Require the portable path to have one fingerprint across every replica; retain the old-path CPU split when heterogeneous CPUs are present.

### Task 7: New canonical lineage and independent verification

**Files:** temporary research workflows only until evidence is sealed; durable status docs after success.

- [ ] Bootstrap a new canonical Dataset/Study from the same sealed source, execution-economics profile, and Observation-v2 assumptions under the portable implementation.
- [ ] Record new feature-config, normalization, Dataset/artifact, Study, and EvidenceSet identities.
- [ ] Verify `research/m2-canonical-study-004`, its head/tree, and immutable artifacts are unchanged before and after the new run.
- [ ] Run a fresh post-Artifact verifier on an independent runner that downloads immutable inputs and reconstructs Dataset/Study/EvidenceSet/Candidate Runs without using the publication index as its oracle.
- [ ] Require every trading observation to have `total_cost > 0`, cash observations to remain zero-trade/zero-cost, and the same strategy/seed roster; do not infer profitability or a winning strategy.

### Task 8: Durable docs, cleanup, final exact-head verification

**Files:**
- Modify: `docs/architecture/lean-core.md`
- Modify: `docs/architecture/package-boundaries.md`
- Modify: `docs/research/current-status.md`
- Modify: `docs/README.md`
- Delete: `docs/specs/issue-494-portable-feature-numerics.md`
- Delete: `docs/plans/issue-494-portable-feature-numerics.md`
- Delete every temporary Issue #494 workflow and remove temporary Issue #494 branches after evidence is preserved.

- [ ] Document portable Dataset numerics as a durable identity invariant and label Study 004 as pre-portable immutable evidence.
- [ ] Remove completed active spec/plan and restore `docs/README.md` to no active work.
- [ ] Re-run permanent CI on the exact final PR HEAD and inspect its logs.
- [ ] Perform final current-main-relative diff review plus adversarial review: identify any wrong implementation that could still satisfy current tests; add a targeted RED/GREEN regression if one exists.
- [ ] Update Issue #494 and the PR with exact run/artifact/digest evidence; merge only after exact-head gates are Green, then require post-merge main CI Green before closing #494.
