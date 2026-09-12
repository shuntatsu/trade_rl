# Portable feature numerics for canonical MarketDataset

Status: Active

## Objective

Make canonical market-feature and normalization bytes reproducible across supported CPU families/runners when code, configuration, and sealed source inputs are identical.

## Verified root cause

Cross-runner diagnostics using identical code and sealed source data reproduced CPU-dependent drift in identity-bound feature values. The drift is not caused by execution-economics plumbing.

A second isolation diagnostic ran the same known failing windows on eight independent hosted runners. It separated vector transcendental dispatch from reduction dispatch:

- NumPy vector `log` plus NumPy reductions drifted across CPU families.
- Replacing only vector `log` with scalar `math.log` did not eliminate all drift.
- Replacing only reductions did not eliminate vector-log drift.
- Scalar `math.log` plus fixed-order `math.fsum` reductions produced one identical result fingerprint across all eight replicas.

Therefore canonical Dataset construction must not depend on hardware-dispatched NumPy transcendental/reduction implementations for identity-bound values.

## Design

### Portable numerical boundary

Create `trade_rl/data/features/numerics.py` as the single internal authority for scalar/reduction operations that contribute to canonical Dataset feature bytes.

The maintained primitives are:

- `portable_log_scalar(value: float) -> float`
- `portable_log(values: np.ndarray) -> np.ndarray`
- `portable_sum(values: np.ndarray | Sequence[float]) -> float`
- `portable_mean(values: np.ndarray | Sequence[float]) -> float`
- `portable_variance(values: np.ndarray | Sequence[float]) -> float`
- `portable_std(values: np.ndarray | Sequence[float]) -> float`
- `portable_dot(left: np.ndarray, right: np.ndarray) -> float`
- `portable_covariance(left: np.ndarray, right: np.ndarray) -> float`
- `portable_correlation(left: np.ndarray, right: np.ndarray) -> float`

Iteration order is C-order / sequence order. Summation uses `math.fsum`. Logarithms use Python `math.log` scalar evaluation in deterministic element order. Variance/covariance/correlation are defined from these primitives rather than delegated to NumPy/BLAS reductions.

`portable_log_scalar` and `portable_log` accept only finite, strictly-positive inputs and fail closed otherwise. This matches the existing Dataset source contract: OHLC prices are already required to be finite and strictly positive, while zero volume is converted to the existing positive epsilon before any logarithm.

`portable_sum` permits an empty sample and returns `0.0`, matching the feature call sites that use an empty masked sum as zero flow. Mean/variance/std/covariance/correlation require non-empty compatible samples; correlation additionally requires non-zero variance. Call-site guards remain responsible for selecting economically valid samples.

The portable layer must not round, quantize, or alter the stored dtype contract merely to make hashes match.

### Scope

Replace identity-affecting hardware-dispatched numerics in:

- `trade_rl/data/features/core.py`
- `trade_rl/data/features/cross_asset.py`
- `trade_rl/data/build/builder.py` for one-bar log returns and global market mean/dispersion

`trade_rl/data/features/multitimeframe.py` inherits the same semantics through `calculate_feature_events()`.

Do not change `risk/`, `simulation/`, forecast/RL fitting, or generic evaluation numerics in this issue. Those outputs have different contracts and are not part of the demonstrated MarketDataset feature-identity defect.

### Build semantics and identity

Bump `MarketBuildConfig.schema_version` from `market_build_v2` to `market_build_v3` and bind `feature_numerics_schema = "portable_feature_numerics_v1"` into `MarketBuildConfig.canonical_payload()`.

`market_build_v2` is not silently reinterpreted. Current production construction emits v3 semantics; an explicitly supplied unsupported schema version fails closed rather than claiming v2 while running v3 numerics.

Keep `MARKET_DATASET_IDENTITY_SCHEMA = "market_dataset_identity_v6"`: the Dataset identity container/field set does not change. Feature/config bytes change naturally, so `feature_config_digest`, `normalization_digest`, Dataset ID, artifact digest, Study digest, and EvidenceSet identity for a rebuilt lineage are expected to change.

Historical Dataset artifacts remain readable because artifact and Dataset identity readers continue to accept the existing persisted identity container/schema. Old immutable research artifacts are never rewritten.

## Research lineage

Study 004 remains immutable evidence produced under the former `market_build_v2` numerical semantics. It is not rewritten or relabeled as portable.

After the portable build implementation passes cross-runner reconstruction, bootstrap a new canonical lineage from the same sealed source universe and the same economic/Observation-v2 research assumptions. Use a new Study identity and preregistration; do not append new evidence to Study 004.

## Invariants

- Same code + same build config + same sealed source bytes must yield byte-identical identity-bound Dataset feature arrays across supported hosted CPU families.
- No hash-only rounding or tolerance-based equality is allowed.
- Stored feature dtype remains `float32`; computations may use deterministic `float64` intermediates before the existing cast.
- Causality, availability, staleness, feature roster, symbol roster, economics arrays, raw/aligned market arrays, and source provenance semantics do not change.
- Economics-only comparisons remain exact within one build semantics generation.
- Historical artifacts and Study 004 remain immutable and inspectable.

## Failure modes to falsify

- replacing `np.log` but leaving reduction/BLAS drift;
- replacing reductions but leaving vector transcendental drift;
- portable helpers accidentally depending on unordered iteration;
- `market_build_v2` being accepted while executing portable v3 semantics;
- hash-time quantization hiding unequal stored feature arrays;
- global feature numerics remaining CPU-dependent after local feature repair;
- cross-asset rolling correlation/beta retaining NumPy covariance/correlation reductions;
- multi-timeframe features bypassing portable core semantics;
- new portable build accidentally mutating Study 004 or its artifacts;
- material performance regression that makes canonical bootstrap operationally impractical.

## Test oracle

Observe all of the following:

1. Unit vectors reproduce fixed expected portable sum/mean/variance/std/dot/covariance/correlation results.
2. Regression windows extracted from the real Issue #494 mismatch produce fixed expected float32 outputs independent of NumPy dispatch.
3. Build config canonical payload explicitly records `market_build_v3` and `portable_feature_numerics_v1`; unsupported explicit schema values fail closed.
4. A static architecture guard rejects direct `np.log`, `np.mean`, `np.std`, `np.sum`, `np.dot`, `np.var`, `np.cov`, or `np.corrcoef` use in the three identity-bound implementation files.
5. Existing causality/feature tests remain Green after expected-value updates required by intentional semantics change.
6. A same-run economics-only pair remains exact for features and normalization while Dataset identity differs through economics/content.
7. Independent hosted runners on different observed CPU families rebuild the same sealed source and produce byte-identical `features`, `global_features`, `normalization_digest`, and Dataset ID.
8. New canonical Study lineage is generated and independently verified without modifying Study 004.

## Non-goals

- No portability guarantee for model-training floating point, simulation P&L reductions, or arbitrary NumPy operations outside Dataset construction.
- No claim that all IEEE-754 implementations/platforms in existence are supported; the contract is validated against the repository's supported Python/runner environment and multiple hosted CPU families.
- No change to trading strategy logic, execution economics, PPO Observation v2, or experiment decision thresholds.
- No rounding/quantization of research features as a shortcut.
