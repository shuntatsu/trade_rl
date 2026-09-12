# Portable Feature Numerics Implementation Plan

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
import math
import numpy as np
import pytest

from trade_rl.data.contracts import FeatureKind, FeatureSpec, MarketBuildConfig
from trade_rl.data.features.numerics import (
    PORTABLE_FEATURE_NUMERICS_SCHEMA,
    portable_correlation,
    portable_dot,
    portable_log,
    portable_mean,
    portable_std,
    portable_sum,
    portable_variance,
)


def test_portable_reductions_have_fixed_scalar_contract() -> None:
    values = np.asarray([1e16, 1.0, -1e16, 3.0], dtype=np.float64)
    assert portable_sum(values) == 4.0
    assert portable_mean(values) == 1.0
    assert portable_dot(values, np.ones(4, dtype=np.float64)) == 4.0
    assert portable_variance(np.asarray([1.0, 2.0, 3.0])) == pytest.approx(2.0 / 3.0)
    assert portable_std(np.asarray([1.0, 2.0, 3.0])) == pytest.approx(math.sqrt(2.0 / 3.0))
    assert portable_correlation(np.asarray([1.0, 2.0, 3.0]), np.asarray([2.0, 4.0, 6.0])) == 1.0


def test_market_build_v3_binds_portable_numerics_schema() -> None:
    config = MarketBuildConfig(
        base_timeframe="1h",
        features=(FeatureSpec(name="ret", kind=FeatureKind.LOG_RETURN),),
    )
    payload = config.canonical_payload()
    assert payload["schema_version"] == "market_build_v3"
    assert payload["feature_numerics_schema"] == PORTABLE_FEATURE_NUMERICS_SCHEMA
    with pytest.raises(ValueError, match="unsupported market build schema"):
        MarketBuildConfig(
            base_timeframe="1h",
            features=(FeatureSpec(name="ret", kind=FeatureKind.LOG_RETURN),),
            schema_version="market_build_v2",
        )
```

- [ ] **Step 2: Run targeted tests and verify RED**

Run: `uv run pytest -q tests/data/test_portable_feature_numerics.py`
Expected: import/schema failures because the portable module and v3 contract do not exist.

- [ ] **Step 3: Implement minimal deterministic primitives**

```python
PORTABLE_FEATURE_NUMERICS_SCHEMA = "portable_feature_numerics_v1"


def _flat(values):
    return np.asarray(values, dtype=np.float64).reshape(-1, order="C")


def portable_sum(values) -> float:
    return math.fsum(float(value) for value in _flat(values))


def portable_mean(values) -> float:
    flat = _flat(values)
    if flat.size == 0:
        raise ValueError("portable mean requires a non-empty sample")
    return portable_sum(flat) / float(flat.size)


def portable_log(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    flat = np.fromiter(
        (portable_log_scalar(float(value)) for value in array.reshape(-1, order="C")),
        dtype=np.float64,
        count=array.size,
    )
    return flat.reshape(array.shape, order="C")
```

Variance/std/dot/covariance/correlation use the same fixed sequence order and `math.fsum`; shape mismatch and empty-sample cases fail explicitly.

- [ ] **Step 4: Bump/validate build semantics**

Set `MarketBuildConfig.schema_version` default to `market_build_v3`, require that exact value in `__post_init__`, and add `feature_numerics_schema` to `canonical_payload()`.

- [ ] **Step 5: Run targeted tests GREEN, Ruff, Mypy**

Run:
`uv run pytest -q tests/data/test_portable_feature_numerics.py`
`uv run ruff check trade_rl/data/features/numerics.py trade_rl/data/contracts.py tests/data/test_portable_feature_numerics.py`
`uv run mypy trade_rl/data/features/numerics.py trade_rl/data/contracts.py`

### Task 2: RED real mismatch windows, then route local/core features

**Files:**
- Modify: `tests/data/test_portable_feature_numerics.py`
- Modify: `trade_rl/data/features/core.py`

**Interfaces:**
- Consumes portable primitives from Task 1.
- `calculate_feature_events()` retains its signature and causal/availability contract.

- [ ] **Step 1: Add real Issue #494 regression windows**

Embed the exact small close/volume windows extracted from the sealed source for the demonstrated BTC realized-volatility, ETH trend-R2, and ETH price-volume-correlation mismatches. Independently compute expected values with scalar `math.log` + `math.fsum` in test-only oracle functions; assert production feature output matches exact float64 oracle and the final `np.float32` bit pattern.

- [ ] **Step 2: Verify RED against current NumPy-dispatched core**

Run only the three regression tests. At least the known CPU-sensitive paths must differ from the portable oracle before production changes.

- [ ] **Step 3: Replace identity-affecting `np.log` calls in `core.py`**

Use `portable_log` for Parkinson/Garman-Klass, downside/upside/vol-of-vol, trend regression, price-volume correlation, realized volatility, and any other vector-log path. Existing scalar `math.log` paths stay scalar.

- [ ] **Step 4: Replace identity-affecting reductions in `core.py`**

Use portable mean/std/sum/dot/correlation helpers for rolling normalization, funding z-score, initial Wilder/RSI means, volatility features, range expansion, trend regression, MFI, CMF, VWAP, price-volume correlation, OBV denominators, relative volume, Bollinger, stochastic-D, CCI, and other scalar reductions that feed stored feature values. `np.min`/`np.max` and exact boolean/index operations remain NumPy.

- [ ] **Step 5: Run focused feature suites GREEN**

Run:
`uv run pytest -q tests/data/test_portable_feature_numerics.py tests/data/test_extended_indicator_features_v2.py tests/data/test_extended_prefix_causality.py tests/data/test_multitimeframe_builder.py`

### Task 3: Cross-asset and global-feature portable semantics

**Files:**
- Modify: `trade_rl/data/features/cross_asset.py`
- Modify: `trade_rl/data/build/builder.py`
- Modify: `tests/data/test_cross_asset_features.py`
- Modify: `tests/data/test_one_bar_returns.py`
- Modify or add focused builder tests as needed.

**Interfaces:**
- Cross-asset feature function signature remains unchanged.
- `_calculate_one_bar_returns()` remains unchanged externally.

- [ ] **Step 1: Add failing seed-order/portable correlation and global-return tests**

Use samples whose naïve floating reduction differs from `math.fsum`; assert cross-asset dispersion/correlation/beta and global market mean/dispersion match independent portable oracles.

- [ ] **Step 2: Verify RED**

Run the focused cross-asset/builder tests before production edits.

- [ ] **Step 3: Route cross-asset reductions**

Replace `np.std`, `np.var`, `np.cov`, `np.corrcoef`, and arithmetic `sum` used in stored cross-asset feature values with portable primitives. Keep stable sort/rank ordering unchanged.

- [ ] **Step 4: Route builder one-bar/global calculations**

Compute available one-bar log returns with scalar portable log evaluation. Compute global market return mean/dispersion with portable mean/std. Compute boolean fractions from deterministic integer counts divided by symbol count.

- [ ] **Step 5: Run focused suites GREEN**

Run:
`uv run pytest -q tests/data/test_cross_asset_features.py tests/data/test_one_bar_returns.py tests/data/test_market_builder.py tests/data/test_complete_96_feature_dataset.py`

### Task 4: Identity and compatibility closure

**Files:**
- Modify: `tests/data/test_market_builder.py`
- Modify: `tests/data/test_market_dataset_identity_v2.py`
- Modify: `tests/data/test_market_artifact.py`
- Modify: economics/bootstrap tests whose fixed digests intentionally change.

- [ ] **Step 1: Run full data/bootstrap tests to enumerate intentional fixed-ID failures**

Run: `uv run pytest -q tests/data tests/evaluation/experiments/bootstrap tests/integrations`

- [ ] **Step 2: Update only fixed expected digests caused by v3 numerical semantics**

For each changed digest, independently verify the payload/array cause. Do not change expectations merely to make tests pass.

- [ ] **Step 3: Add explicit historical artifact-reader coverage**

Load an existing v2-identity fixture/artifact through the current reader and assert identity verification still succeeds without rebuilding it.

- [ ] **Step 4: Re-run data/bootstrap/integration suites GREEN**

### Task 5: Full quality gate before cross-runner evidence

**Files:** no production changes unless a real defect is found.

- [ ] Run Ruff and format check.
- [ ] Run production Mypy and architecture-tooling Mypy.
- [ ] Run full `pytest -q tests`.
- [ ] Run build, sdist/direct-wheel/rebuilt-wheel source closure, clean non-editable install/public imports, Candidate CLI, bootstrap CLI, and package identity exactly as permanent CI does.
- [ ] Review current-main-relative diff for scope, dead/debug code, accidental contract changes, and performance hot paths.

### Task 6: Cross-runner full Dataset oracle

**Files:**
- Temporary branch-only workflow under `.github/workflows/` for Issue #494; remove before PR Ready.

- [ ] **Step 1: Freeze exact implementation HEAD and one existing sealed source Artifact**

Use the same source universe already used by Issue #494; do not fetch market data again.

- [ ] **Step 2: Rebuild the complete canonical priced Dataset on at least four independent `ubuntu-24.04` runners**

Each replica records CPU model, NumPy runtime, feature/global-feature SHA-256, feature config digest, normalization digest, Dataset ID, and economic-array checks.

- [ ] **Step 3: Aggregate artifacts without recomputation**

Require exact equality for `features`, `global_features`, `normalization_digest`, and Dataset ID across every replica. Require at least two observed CPU model families when the hosted pool supplies them; otherwise repeat replicas until the diagnostic has heterogeneous CPU evidence.

- [ ] **Step 4: Falsification comparison**

Demonstrate that the old Issue #494 v2 implementation still separates AMD/Intel on the known windows while the v3 implementation does not, so a homogeneous runner allocation cannot create a false Green.

### Task 7: New canonical lineage and independent verification

**Files:** temporary research workflow only; durable status docs after success.

- [ ] Bootstrap a new canonical Dataset/Study from the same sealed source, execution-economics profile, and Observation-v2 assumptions under the portable build implementation.
- [ ] Record new feature-config, normalization, Dataset/artifact, Study, and EvidenceSet identities.
- [ ] Verify Study 004 branch/head/tree/artifacts are unchanged.
- [ ] Run a fresh independent post-Artifact verifier that downloads immutable inputs and reconstructs the new Dataset/Study/EvidenceSet without trusting the publication index as oracle.
- [ ] Require trading observations with `total_cost > 0` as in the repaired real-cost lineage; do not infer profitability or a winning strategy.

### Task 8: Durable docs, cleanup, final exact-head verification

**Files:**
- Modify: `docs/architecture/lean-core.md`
- Modify: `docs/architecture/package-boundaries.md`
- Modify: `docs/research/current-status.md`
- Modify: `docs/README.md`
- Delete: `docs/specs/issue-494-portable-feature-numerics.md`
- Delete: `docs/plans/issue-494-portable-feature-numerics.md`
- Remove temporary Issue #494 workflows/branches after evidence is preserved.

- [ ] Document portable Dataset numerics as a durable identity invariant and label Study 004 as pre-portable immutable evidence.
- [ ] Remove completed active spec/plan and restore `docs/README.md` to no active work.
- [ ] Re-run permanent CI on the exact final PR HEAD and inspect logs, not only check status.
- [ ] Perform final diff review plus adversarial review: ask which wrong implementation could still pass current tests, then add/repair tests if needed.
- [ ] Update Issue #494 and PR with exact run/artifact/digest evidence; close #494 only after merged-main CI is Green.
