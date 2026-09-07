# Causal Alpha V4 Hierarchical Teacher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an opt-in, research-only Causal Alpha V4 lane that adds reproducible Spot/perpetual/global context, a BTC-proxy plus shared-residual hierarchical teacher with 4h/24h/72h responsibilities, state-conditioned uncertainty, and correctly defined economic replay through Teacher admission while leaving reward, risk, execution, and every V3 historical meaning unchanged.

**Architecture:** Keep the maintained Universal runtime and V3 lane intact. Materialize V4 auxiliary context into separately digested artifacts, expose exactly the same current-time context and missingness to the Universal student observation surface, fit deterministic V4 market-proxy/residual/direction heads from train-only data, compile a 24h/72h slow anchor plus bounded 4h fast impulse, and evaluate it through a V4-only Signal/selection/admission evidence chain.

**Tech Stack:** Python 3.12, NumPy `<2.0`, Gymnasium 0.29.1, existing Binance Vision/Public REST infrastructure, existing weighted ridge primitives, pytest, Hypothesis, Ruff, MyPy, Import Linter, Docker training image.

**Spec:** `docs/implementation-plans/specs/2026-08-23-causal-alpha-v4-hierarchical-teacher-design.md`

## Global Constraints

- Reward remains the maintained cost-inclusive pure net-equity log-growth objective. Do not modify reward configuration or `trade_rl/rl/rewards.py`.
- Risk, execution, latency, partial-fill, liquidation, target-weight, and accounting semantics remain unchanged.
- Existing V3 source files and retained V3 artifacts keep their historical meaning; V4 gets new schemas and artifact roots.
- Do not reinterpret V3 `trade_count`; V4 replay evidence persists `executed_change_count` and `closed_trade_count` separately from inception.
- Keep the maintained 206 target-local Universal market features and nine instrument descriptors semantically unchanged.
- `cross_market_core_v1` has exactly 24 local value channels; `cross_market_derivatives_v1` has exactly 31.
- `global_market_core_v1` has exactly 38 global value channels; `global_market_derivatives_v1` has exactly 44.
- Value channels, availability masks, and staleness arrays are separate. Missing context is unavailable, never an informative numeric zero.
- The first V4 market proxy is `BTCUSDT` USD-M perpetual.
- Causal beta uses 4h returns, a 720h lookback, at least 90 complete samples, and clipping to `[-3.0, 3.0]`; BTC beta is exactly `1.0`.
- A required local/global/beta availability failure makes that V4 decision non-actionable for alpha changes; risk-reducing environment actions remain available.
- On-chain `pit_flow_v1` is disabled unless a provider-specific artifact proves point-in-time or revision-frozen history for the complete authored interval. No scraped reconstructed history is allowed.
- The first V4 deterministic model hypothesis uses weighted/objective-normalized ridge only: market-proxy ridge `1.0`, residual ridge `0.1`, direction ridge `0.1`.
- The first target hypothesis uses slow target magnitudes `(0.0, 0.025, 0.05, 0.10, 0.25)`, fast deviations `(0.0, 0.025, 0.05)`, slow cadence `16` decisions, fast cadence `4` decisions, maximum final target delta `0.125`, maximum fast absolute deviation `0.05`, execution-cost multiplier `1.5`, and edge margin `0.001`.
- Direction evidence is a signed score, not a calibrated probability. Increasing exposure or reversing sign requires return forecast and direction score to agree in sign. Flattening or reducing absolute exposure is never blocked by direction disagreement.
- Uncertainty state precedence is `basis_positioning_stress > low_liquidity > high_realized_volatility > normal`; thresholds are derived from eligible train-prefix quantiles and included in fit identity.
- A state whose effective sample size is below `30.0` uses the horizon-global weighted RMSE and records the fallback.
- Signal liveness is required evidence but is not a post-hoc tunable hard threshold in the first generation. Exact-zero dynamic prediction variance with varying supported inputs is an integrity failure; otherwise liveness remains descriptive until a separate authored gate is justified.
- No validation symbol, test symbol, Teacher-admission holdout outcome, BC result, RL result, or sealed evaluation may tune V4 features, thresholds, ridge strengths, target parameters, or state definitions.
- No BC/PPO training starts in this plan. This plan ends at a student-compatible, Teacher-admitted V4 package or a preserved research rejection. A learner implementation plan is authored only after Teacher admission.

## Frozen Feature Names

`CROSS_MARKET_CORE_NAMES` is exactly:

```python
(
    "spot_log_return_1h",
    "spot_log_return_4h",
    "spot_log_return_24h",
    "spot_log_quote_volume_robust_z_4h",
    "spot_log_quote_volume_robust_z_24h",
    "spot_perp_log_basis",
    "spot_perp_basis_change_1h",
    "spot_perp_basis_change_4h",
    "spot_perp_basis_robust_z_7d",
    "spot_minus_perp_log_return_1h",
    "spot_minus_perp_log_return_4h",
    "spot_to_perp_log_quote_volume_ratio_1h",
    "spot_to_perp_log_quote_volume_ratio_4h",
    "spot_to_perp_log_quote_volume_ratio_24h",
    "spot_taker_quote_imbalance_1h",
    "spot_taker_quote_imbalance_4h",
    "perp_taker_quote_imbalance_1h",
    "perp_taker_quote_imbalance_4h",
    "spot_minus_perp_taker_imbalance_1h",
    "spot_minus_perp_taker_imbalance_4h",
    "funding_rate",
    "funding_rate_change",
    "funding_rate_robust_z_7d",
    "basis_z_x_flow_divergence_4h",
)
```

`CROSS_MARKET_DERIVATIVE_NAMES` is exactly:

```python
(
    "open_interest_log_change_1h",
    "open_interest_log_change_4h",
    "open_interest_log_change_24h",
    "global_long_short_ratio_robust_z_4h",
    "top_position_long_short_ratio_robust_z_4h",
    "basis_z_x_open_interest_change_4h",
    "funding_z_x_open_interest_change_4h",
)
```

`GLOBAL_MARKET_CORE_NAMES` is exactly:

```python
(
    "btc_spot_log_return_1h",
    "btc_spot_log_return_4h",
    "btc_spot_log_return_24h",
    "btc_perp_log_return_1h",
    "btc_perp_log_return_4h",
    "btc_perp_log_return_24h",
    "btc_spot_perp_log_basis",
    "btc_spot_perp_basis_change_4h",
    "btc_spot_perp_basis_robust_z_7d",
    "btc_spot_taker_quote_imbalance_1h",
    "btc_spot_taker_quote_imbalance_4h",
    "btc_perp_taker_quote_imbalance_1h",
    "btc_perp_taker_quote_imbalance_4h",
    "btc_spot_to_perp_log_quote_volume_ratio_4h",
    "btc_spot_to_perp_log_quote_volume_ratio_24h",
    "btc_funding_rate",
    "btc_funding_rate_robust_z_7d",
    "eth_spot_log_return_1h",
    "eth_spot_log_return_4h",
    "eth_spot_log_return_24h",
    "eth_perp_log_return_1h",
    "eth_perp_log_return_4h",
    "eth_perp_log_return_24h",
    "eth_spot_perp_log_basis",
    "eth_spot_perp_basis_change_4h",
    "eth_spot_perp_basis_robust_z_7d",
    "eth_spot_taker_quote_imbalance_1h",
    "eth_spot_taker_quote_imbalance_4h",
    "eth_perp_taker_quote_imbalance_1h",
    "eth_perp_taker_quote_imbalance_4h",
    "eth_spot_to_perp_log_quote_volume_ratio_4h",
    "eth_spot_to_perp_log_quote_volume_ratio_24h",
    "eth_funding_rate",
    "eth_funding_rate_robust_z_7d",
    "btc_minus_eth_perp_return_4h",
    "btc_minus_eth_perp_return_24h",
    "btc_minus_eth_basis",
    "btc_eth_perp_return_dispersion_4h",
)
```

`GLOBAL_MARKET_DERIVATIVE_NAMES` is exactly:

```python
(
    "btc_open_interest_log_change_4h",
    "btc_open_interest_log_change_24h",
    "btc_global_long_short_ratio_robust_z_4h",
    "eth_open_interest_log_change_4h",
    "eth_open_interest_log_change_24h",
    "eth_global_long_short_ratio_robust_z_4h",
)
```

---

## File Structure

Create these focused V4 files:

- `trade_rl/data/v4_context.py` — context schemas, aligned arrays, formulas, availability/staleness rules, deterministic digests.
- `trade_rl/data/v4_context_artifact.py` — filesystem artifact writer/loader for per-symbol V4 context arrays.
- `trade_rl/integrations/binance_v4_context.py` — Binance Spot/perpetual/funding/metrics adapters and context materialization inputs.
- `trade_rl/workflows/universal_causal_alpha_v4_manifest.py` — V4 auxiliary manifest referencing one immutable base `UniversalRuntimeManifest`.
- `trade_rl/rl/universal_v4_context.py` — policy observation provider and V4 context schema metadata.
- `trade_rl/learning/causal_alpha_v4.py` — beta, forecast, direction, uncertainty, liveness, and fast/slow target primitives.
- `trade_rl/workflows/universal_causal_alpha_v4_runtime.py` — load/validate base runtime plus V4 context and prepare train-only samples.
- `trade_rl/workflows/universal_causal_alpha_v4_fitting.py` — fit/cache market-proxy, shared residual, and shared direction heads.
- `trade_rl/workflows/universal_causal_alpha_v4_signal.py` — canonical 4h and slow-fused Signal evidence plus liveness sidecars.
- `trade_rl/workflows/universal_causal_alpha_v4_replay.py` — production-environment economic replay with correct activity accounting.
- `trade_rl/workflows/universal_causal_alpha_v4_selection.py` — V4 candidate admission/rejection.
- `trade_rl/workflows/universal_causal_alpha_v4_admission.py` — untouched selected-teacher holdout admission.
- `trade_rl/workflows/universal_causal_alpha_v4_artifact_store.py` — immutable/restart-safe V4 evidence storage.
- `trade_rl/workflows/universal_causal_alpha_v4_pipeline.py` — ordered Gate orchestration.
- `trade_rl/workflows/universal_causal_alpha_v4_runner.py` — thin facade matching V3 responsibility boundaries.
- `scripts/materialize_universal_causal_alpha_v4_context.py` — deterministic auxiliary context materializer.
- `scripts/run_universal_causal_alpha_v4_research.py` — research-only CLI.
- `examples/binance/universal-causal-alpha-v4-research.json` — frozen first V4 hypothesis.

Modify only these existing integration seams unless a failing maintained contract proves another edit necessary:

- `trade_rl/integrations/binance_cache.py`
- `trade_rl/rl/universal_single_instrument_env.py`
- `trade_rl/workflows/binance_universal_runtime.py`
- `trade_rl/workflows/universal_full_research_entrypoint.py`

---

### Task 1: Immutable V4 context contracts and formulas

**Files:**
- Create: `trade_rl/data/v4_context.py`
- Test: `tests/data/test_v4_context.py`

**Interfaces:**
- Consumes: `V4CrossMarketInputs` and `V4GlobalMarketInputs` on the maintained 15m decision clock.
- Produces: `V4ContextBlock`, `V4TargetContext`, `build_cross_market_context`, `build_global_market_context`, `robust_trailing_zscore`, `taker_quote_imbalance`, `spot_perp_log_basis`.

- [ ] **Step 1: Write failing schema/formula tests**

```python
from trade_rl.data.v4_context import (
    CROSS_MARKET_CORE_NAMES,
    GLOBAL_MARKET_CORE_NAMES,
    spot_perp_log_basis,
    taker_quote_imbalance,
)


def test_v4_core_channel_counts_are_frozen():
    assert len(CROSS_MARKET_CORE_NAMES) == 24
    assert len(GLOBAL_MARKET_CORE_NAMES) == 38
    assert len(set(CROSS_MARKET_CORE_NAMES)) == 24
    assert len(set(GLOBAL_MARKET_CORE_NAMES)) == 38


def test_taker_quote_imbalance_is_signed_and_bounded():
    assert taker_quote_imbalance(75.0, 100.0) == 0.5
    assert taker_quote_imbalance(25.0, 100.0) == -0.5


def test_spot_perp_log_basis_uses_perp_over_spot():
    assert spot_perp_log_basis(spot=100.0, perp=101.0) > 0.0
```

Also assert non-positive prices, non-positive quote volume, mismatched array shapes, duplicate feature names, and non-finite available values fail closed.

- [ ] **Step 2: Run tests and verify RED**

```bash
uv run pytest -q tests/data/test_v4_context.py
```

Expected: collection/import failure because `trade_rl.data.v4_context` does not exist.

- [ ] **Step 3: Implement input and output contracts**

```python
@dataclass(frozen=True, slots=True)
class V4CrossMarketInputs:
    decision_indices: np.ndarray
    spot_close: np.ndarray
    spot_quote_volume: np.ndarray
    spot_taker_buy_quote_volume: np.ndarray
    perp_close: np.ndarray
    perp_mark_price: np.ndarray
    perp_quote_volume: np.ndarray
    perp_taker_buy_quote_volume: np.ndarray
    funding_rate: np.ndarray
    funding_available: np.ndarray
    open_interest_value: np.ndarray | None
    open_interest_available: np.ndarray | None
    global_long_short_ratio: np.ndarray | None
    top_position_long_short_ratio: np.ndarray | None
    derivatives_available: np.ndarray | None
    source_digest: str


@dataclass(frozen=True, slots=True)
class V4ContextBlock:
    feature_names: tuple[str, ...]
    decision_indices: np.ndarray
    values: np.ndarray
    available: np.ndarray
    staleness_hours: np.ndarray
    source_digest: str
    digest: str = ""


@dataclass(frozen=True, slots=True)
class V4TargetContext:
    symbol: str
    local: V4ContextBlock
    global_market: V4ContextBlock
    profile_name: str
    digest: str = ""
```

`V4GlobalMarketInputs` contains `btc: V4CrossMarketInputs`, `eth: V4CrossMarketInputs`, and one `source_digest`.

Every array is copied C-contiguous, validated, made read-only, and included in `content_and_arrays_digest`.

- [ ] **Step 4: Implement primitive formulas**

```python
def taker_quote_imbalance(taker_buy_quote: float, total_quote: float) -> float:
    if not math.isfinite(taker_buy_quote) or taker_buy_quote < 0.0:
        raise ValueError("taker_buy_quote must be finite and non-negative")
    if not math.isfinite(total_quote) or total_quote <= 0.0:
        raise ValueError("total_quote must be finite and positive")
    return float(np.clip(2.0 * taker_buy_quote / total_quote - 1.0, -1.0, 1.0))


def spot_perp_log_basis(*, spot: float, perp: float) -> float:
    if not math.isfinite(spot) or not math.isfinite(perp) or spot <= 0.0 or perp <= 0.0:
        raise ValueError("basis prices must be finite and positive")
    return math.log(perp / spot)
```

`robust_trailing_zscore` uses only the current and earlier rows, a causal trailing window, median, `1.4826 * MAD`, and minimum support `32`. Zero MAD returns `0.0` with `available=True` only when support exists.

- [ ] **Step 5: Implement context builders using frozen names**

`build_cross_market_context(inputs, include_derivatives)` returns 24 or 31 columns in the frozen order. `build_global_market_context(inputs, include_derivatives)` returns 38 or 44 columns in the frozen order. Windowed returns and ratios use only complete trailing windows. Missing source values set `available=False`; stored numeric value is zero only as inert storage.

- [ ] **Step 6: Add prefix-causality/property tests**

For every feature family, mutate all source rows after a selected decision index and assert all context rows at or before that decision are bitwise unchanged.

- [ ] **Step 7: Verify and commit**

```bash
uv run pytest -q tests/data/test_v4_context.py
uv run ruff check trade_rl/data/v4_context.py tests/data/test_v4_context.py
uv run mypy trade_rl/data/v4_context.py
```

```bash
git add trade_rl/data/v4_context.py tests/data/test_v4_context.py
git commit -m "feat: define causal alpha v4 context contracts"
```

---

### Task 2: V4 context artifact storage and identity

**Files:**
- Create: `trade_rl/data/v4_context_artifact.py`
- Test: `tests/data/test_v4_context_artifact.py`

**Interfaces:**
- Consumes: `V4TargetContext`.
- Produces: `write_v4_target_context_artifact(path: Path, context: V4TargetContext) -> Path` and `load_v4_target_context_artifact(path: Path) -> V4TargetContext`.

- [ ] **Step 1: Write failing round-trip/corruption tests**

```python
def test_v4_context_artifact_round_trip(tmp_path, sample_v4_context):
    path = tmp_path / "BTCUSDT"
    write_v4_target_context_artifact(path, sample_v4_context)
    loaded = load_v4_target_context_artifact(path)
    assert loaded.digest == sample_v4_context.digest
    np.testing.assert_array_equal(loaded.local.values, sample_v4_context.local.values)
```

Corruption cases: changed array bytes, changed feature order, changed source digest, missing `manifest.json`, missing `arrays.npz`, unexpected array member, and manifest digest drift.

- [ ] **Step 2: Verify RED**

```bash
uv run pytest -q tests/data/test_v4_context_artifact.py
```

- [ ] **Step 3: Implement artifact layout**

Each context directory contains exactly:

```text
manifest.json
arrays.npz
```

Schema is `causal_alpha_v4_target_context_artifact_v1`. Manifest binds symbol, profile, ordered names, source digests, row count, first/last decision index, array digest, and context digest. Existing identical bytes are idempotent; different content at the same path raises `FileExistsError`.

- [ ] **Step 4: Verify and commit**

```bash
uv run pytest -q tests/data/test_v4_context.py tests/data/test_v4_context_artifact.py
uv run ruff check trade_rl/data/v4_context.py trade_rl/data/v4_context_artifact.py tests/data/test_v4_context.py tests/data/test_v4_context_artifact.py
uv run mypy trade_rl/data/v4_context.py trade_rl/data/v4_context_artifact.py
```

```bash
git add trade_rl/data/v4_context_artifact.py tests/data/test_v4_context_artifact.py
git commit -m "feat: persist causal alpha v4 context artifacts"
```

---

### Task 3: Binance Spot/perpetual and futures-metrics adapter

**Files:**
- Create: `trade_rl/integrations/binance_v4_context.py`
- Modify: `trade_rl/integrations/binance_cache.py`
- Create: `tests/integrations/test_binance_v4_context.py`
- Modify: `tests/examples/test_market_data_sync.py`

**Interfaces:**
- Consumes: `BinancePublicTransport`, official Binance Vision cache, target symbol, anchors `BTCUSDT`/`ETHUSDT`, research start/end.
- Produces: `BinanceFuturesMetricsSeries`, `BinanceV4ProfileCapability`, and `build_binance_v4_context`.

- [ ] **Step 1: Write failing metrics URL and parser tests**

```python
def vision_futures_metrics_url(symbol: str, day: datetime) -> str:
    date = day.astimezone(UTC).strftime("%Y-%m-%d")
    return (
        "https://data.binance.vision/data/futures/um/daily/metrics/"
        f"{symbol}/{symbol}-metrics-{date}.zip"
    )
```

Require these exact CSV headers:

```python
BINANCE_FUTURES_METRICS_COLUMNS = (
    "create_time",
    "symbol",
    "sum_open_interest",
    "sum_open_interest_value",
    "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
)
```

Map `count_long_short_ratio` to `global_long_short_ratio` and `sum_toptrader_long_short_ratio` to `top_position_long_short_ratio`. Do not use `sum_taker_long_short_vol_ratio` as a substitute for kline-derived taker quote imbalance.

- [ ] **Step 2: Verify RED**

```bash
uv run pytest -q tests/integrations/test_binance_v4_context.py tests/examples/test_market_data_sync.py
```

- [ ] **Step 3: Add public generic Vision URL cache helpers**

```python
@dataclass(frozen=True, slots=True)
class BinanceVisionUrlPlan:
    urls: tuple[str, ...]


def inspect_binance_vision_urls(
    urls: Sequence[str], *, cache_root: str | Path
) -> BinanceVisionCacheReport:
    return _inspect_urls(tuple(urls), cache_root=Path(cache_root))


def sync_binance_vision_urls(
    urls: Sequence[str], *, transport: _VisionArchiveTransport | BinancePublicTransport
) -> BinanceVisionCacheReport:
    return _sync_urls(tuple(urls), transport=transport)
```

Implement `_inspect_urls` and `_sync_urls` by reusing `vision_cache_path` and `validate_cached_vision_payload`; reject any URL outside `https://data.binance.vision/data/`.

- [ ] **Step 4: Implement source series parsing and capability**

```python
@dataclass(frozen=True, slots=True)
class BinanceFuturesMetricsSeries:
    timestamps_ms: np.ndarray
    open_interest_value: np.ndarray
    global_long_short_ratio: np.ndarray
    top_position_long_short_ratio: np.ndarray
    source_digest: str


@dataclass(frozen=True, slots=True)
class BinanceV4ProfileCapability:
    profile_name: str
    derivative_metrics_complete: bool
    missing_url_count: int
    invalid_url_count: int
    source_digest: str
```

Derivative profile is enabled only if every required daily metrics archive for every required target/anchor day is present and valid. Do not backfill old OI/ratio history from short-retention REST endpoints.

- [ ] **Step 5: Implement Spot/perpetual/global context assembly**

`build_binance_v4_context` uses existing public Binance dataset construction for Spot and USD-M kline/funding data, then aligns source rows to the maintained decision clock and calls Task 1 builders. The source digest binds every input dataset/metrics archive identity.

- [ ] **Step 6: Add timing and missing-data falsification tests**

A metrics row with `create_time` after a decision is unavailable at that decision. Missing metrics reject derivative profile instead of filling OI/ratio values with zero. Future source mutations cannot change earlier context.

- [ ] **Step 7: Verify and commit**

```bash
uv run pytest -q tests/integrations/test_binance_v4_context.py tests/examples/test_market_data_sync.py
uv run ruff check trade_rl/integrations/binance_v4_context.py trade_rl/integrations/binance_cache.py tests/integrations/test_binance_v4_context.py tests/examples/test_market_data_sync.py
uv run mypy trade_rl/integrations/binance_v4_context.py trade_rl/integrations/binance_cache.py
```

```bash
git add trade_rl/integrations/binance_v4_context.py trade_rl/integrations/binance_cache.py tests/integrations/test_binance_v4_context.py tests/examples/test_market_data_sync.py
git commit -m "feat: add binance v4 cross-market sources"
```

---

### Task 4: V4 context manifest and deterministic materialization

**Files:**
- Create: `trade_rl/workflows/universal_causal_alpha_v4_manifest.py`
- Create: `scripts/materialize_universal_causal_alpha_v4_context.py`
- Create: `tests/workflows/test_universal_causal_alpha_v4_manifest.py`
- Create: `tests/scripts/test_materialize_universal_causal_alpha_v4_context.py`

**Interfaces:**
- Consumes: base `UniversalRuntimeManifest`, all maintained target symbols, Task 3 capability.
- Produces: `CausalAlphaV4ContextManifest` and one V4 target context artifact per train/validation/test symbol.

- [ ] **Step 1: Write failing strict-manifest tests**

```python
@dataclass(frozen=True, slots=True)
class CausalAlphaV4ContextManifest:
    base_runtime_manifest_digest: str
    profile_name: str
    context_artifact_relpath: Path
    context_digests: tuple[tuple[str, str], ...]
    local_schema_digest: str
    global_schema_digest: str
    pit_flow_profile: str | None
    source_capability_digest: str
    schema_version: str = "causal_alpha_v4_context_manifest_v1"
    manifest_digest: str = ""
```

`context_digests` follows exact `train + validation + test` order from the base runtime. Reject unknown fields, missing symbols, reordered symbols, profile drift, and base-manifest digest drift.

- [ ] **Step 2: Verify RED**

```bash
uv run pytest -q tests/workflows/test_universal_causal_alpha_v4_manifest.py tests/scripts/test_materialize_universal_causal_alpha_v4_context.py
```

- [ ] **Step 3: Implement CLI and coverage-only profile resolution**

Arguments are:

```text
--runtime-manifest
--frozen-metadata-root
--market-data-cache-root
--output-root
--profile core|derivatives-auto
```

`derivatives-auto` inspects source coverage only, before model labels or outcomes, and freezes exactly one resolved profile into the manifest.

- [ ] **Step 4: Implement idempotency and partial recovery**

If 14 valid target artifacts exist and one is missing, generate the missing artifact and then write the manifest. Any existing digest mismatch aborts without overwrite. Manifest is created only when every expected target artifact validates.

- [ ] **Step 5: Verify and commit**

```bash
uv run pytest -q tests/workflows/test_universal_causal_alpha_v4_manifest.py tests/scripts/test_materialize_universal_causal_alpha_v4_context.py
uv run python -m py_compile scripts/materialize_universal_causal_alpha_v4_context.py
```

```bash
git add trade_rl/workflows/universal_causal_alpha_v4_manifest.py scripts/materialize_universal_causal_alpha_v4_context.py tests/workflows/test_universal_causal_alpha_v4_manifest.py tests/scripts/test_materialize_universal_causal_alpha_v4_context.py
git commit -m "feat: materialize causal alpha v4 context generation"
```

---

### Task 5: Student-visible V4 observation capability

**Files:**
- Create: `trade_rl/rl/universal_v4_context.py`
- Modify: `trade_rl/rl/universal_single_instrument_env.py`
- Modify: `trade_rl/workflows/universal_full_research_entrypoint.py`
- Modify: `trade_rl/workflows/binance_universal_runtime.py`
- Create: `tests/rl/test_universal_v4_context.py`
- Modify: `tests/rl/test_universal_research_u3_u6.py`
- Modify: `tests/workflows/test_binance_universal_runtime.py`

**Interfaces:**
- Consumes: `CausalAlphaV4ContextManifest` and loaded `V4TargetContext` artifacts.
- Produces: `V4ContextProvider`, `V4PolicyContext`, and opt-in Dict observation keys for values, availability, staleness, and beta.

- [ ] **Step 1: Write failing observation-space parity tests**

Non-V4 runtime must preserve existing keys. V4 runtime adds exactly:

```text
local_cross_market_context          float32 shape (1, 24) or (1, 31)
local_cross_market_available        float32 shape (1, 24) or (1, 31)
local_cross_market_staleness_hours  float32 shape (1, 24) or (1, 31)
global_market_context               float32 shape (1, 38) or (1, 44)
global_market_available             float32 shape (1, 38) or (1, 44)
global_market_staleness_hours       float32 shape (1, 38) or (1, 44)
causal_beta                         float32 shape (1, 1)
causal_beta_available               float32 shape (1, 1)
```

- [ ] **Step 2: Verify RED**

```bash
uv run pytest -q tests/rl/test_universal_v4_context.py tests/workflows/test_binance_universal_runtime.py tests/rl/test_universal_research_u3_u6.py
```

- [ ] **Step 3: Implement provider contract**

```python
@dataclass(frozen=True, slots=True)
class V4PolicyContext:
    local_values: np.ndarray
    local_available: np.ndarray
    local_staleness_hours: np.ndarray
    global_values: np.ndarray
    global_available: np.ndarray
    global_staleness_hours: np.ndarray
    beta: np.ndarray
    beta_available: np.ndarray
    digest: str


class V4ContextProvider:
    local_width: int
    global_width: int
    schema_digest: str

    def resolve(
        self, *, symbol: str, decision_index: int, beta: float, beta_available: bool
    ) -> V4PolicyContext:
        context = self._contexts[symbol]
        return _resolve_exact_row(context, decision_index, beta, beta_available)
```

Context values and masks are visible to the student. Provider rejects missing target artifact, duplicate decision index, and schema drift.

- [ ] **Step 4: Extend routed environment only when provider is present**

Add `v4_context_provider: V4ContextProvider | None = None` to `EpisodeRoutedSingleInstrumentEnv`. `_policy_observation_space` adds the eight keys only for V4. `_policy_observation` resolves the exact current decision row. Bind provider schema digest into `observation_contract_digest` and sequence layout metadata.

- [ ] **Step 5: Extend runtime factory context**

Add `v4_context_manifest_path: Path | None = None` to `UniversalRuntimeFactoryContext`. Non-null path loads the V4 manifest and validates `base_runtime_manifest_digest == context.manifest.manifest_digest`. Null preserves current runtime behavior.

- [ ] **Step 6: Add falsification tests**

Reject stale/missing global context, wrong symbol artifact, wrong decision index, wrong feature order, base-manifest mismatch, and any teacher action input that is not exposed through the V4 observation contract.

- [ ] **Step 7: Verify and commit**

```bash
uv run pytest -q tests/rl/test_universal_v4_context.py tests/rl/test_universal_research_u3_u6.py tests/workflows/test_binance_universal_runtime.py
uv run mypy trade_rl/rl/universal_v4_context.py trade_rl/rl/universal_single_instrument_env.py trade_rl/workflows/binance_universal_runtime.py trade_rl/workflows/universal_full_research_entrypoint.py
uv run lint-imports
```

```bash
git add trade_rl/rl/universal_v4_context.py trade_rl/rl/universal_single_instrument_env.py trade_rl/workflows/universal_full_research_entrypoint.py trade_rl/workflows/binance_universal_runtime.py tests/rl/test_universal_v4_context.py tests/rl/test_universal_research_u3_u6.py tests/workflows/test_binance_universal_runtime.py
git commit -m "feat: expose v4 market context to universal policy"
```

---

### Task 6: 4h labels, causal beta, and residual reconstruction

**Files:**
- Create: `trade_rl/learning/causal_alpha_v4.py`
- Create: `tests/learning/test_causal_alpha_v4_beta.py`
- Create: `trade_rl/workflows/universal_causal_alpha_v4_runtime.py`
- Create: `tests/workflows/test_universal_causal_alpha_v4_runtime.py`

**Interfaces:**
- Consumes: existing train-symbol datasets and V4 context artifacts.
- Produces: `CausalBetaConfig`, `CausalBetaSeries`, `CausalAlphaV4SymbolSamples`, `causal_alpha_v4_beta_series`.

- [ ] **Step 1: Write failing beta tests**

```python
@dataclass(frozen=True, slots=True)
class CausalBetaConfig:
    return_horizon_hours: float = 4.0
    lookback_hours: float = 720.0
    minimum_complete_samples: int = 90
    minimum_market_variance: float = 1e-12
    minimum_beta: float = -3.0
    maximum_beta: float = 3.0
```

Assert BTC beta is `1.0`; future target/BTC returns cannot change earlier beta; insufficient support returns `available=False`; clipping is exact.

- [ ] **Step 2: Verify RED**

```bash
uv run pytest -q tests/learning/test_causal_alpha_v4_beta.py tests/workflows/test_universal_causal_alpha_v4_runtime.py
```

- [ ] **Step 3: Implement 4h labels through the existing timing primitive**

Call `forward_log_return_label` with `horizon_hours=4.0`. Do not duplicate execution-start/label-end timing math.

```python
@dataclass(frozen=True, slots=True)
class CausalAlphaV4SymbolSamples:
    symbol: str
    decision_indices: np.ndarray
    target_local_features: np.ndarray
    target_local_available: np.ndarray
    local_context: V4ContextBlock
    global_context: V4ContextBlock
    beta: np.ndarray
    beta_available: np.ndarray
    labels_4h: np.ndarray
    label_end_indices_4h: np.ndarray
    labels_24h: np.ndarray
    label_end_indices_24h: np.ndarray
    labels_72h: np.ndarray
    label_end_indices_72h: np.ndarray
    digest: str = ""
```

- [ ] **Step 4: Implement residual decomposition**

For each horizon:

```python
residual = symbol_label - beta * btc_market_proxy_label
reconstructed = beta * btc_market_proxy_label + residual
```

For finite eligible rows require `abs(reconstructed - symbol_label) <= 1e-15`.

- [ ] **Step 5: Prove fit scope uses train labels only**

Prepared V4 fit arrays include only authored train-symbol labels. Validation/test current public context may be read during later prediction, but validation/test future labels never enter fit, beta calibration, state thresholds, or uncertainty calibration.

- [ ] **Step 6: Verify and commit**

```bash
uv run pytest -q tests/learning/test_causal_alpha_v4_beta.py tests/workflows/test_universal_causal_alpha_v4_runtime.py
uv run ruff check trade_rl/learning/causal_alpha_v4.py trade_rl/workflows/universal_causal_alpha_v4_runtime.py tests/learning/test_causal_alpha_v4_beta.py tests/workflows/test_universal_causal_alpha_v4_runtime.py
uv run mypy trade_rl/learning/causal_alpha_v4.py trade_rl/workflows/universal_causal_alpha_v4_runtime.py
```

```bash
git add trade_rl/learning/causal_alpha_v4.py trade_rl/workflows/universal_causal_alpha_v4_runtime.py tests/learning/test_causal_alpha_v4_beta.py tests/workflows/test_universal_causal_alpha_v4_runtime.py
git commit -m "feat: add v4 causal beta and residual labels"
```

---

### Task 7: Market-proxy, shared residual, and shared direction fits

**Files:**
- Create: `trade_rl/workflows/universal_causal_alpha_v4_fitting.py`
- Create: `tests/workflows/test_universal_causal_alpha_v4_fitting.py`
- Modify: `trade_rl/learning/causal_alpha_v4.py`
- Create: `tests/learning/test_causal_alpha_v4_forecast.py`

**Interfaces:**
- Consumes: Task 6 samples, `fit_causal_alpha_ridge`, `causal_alpha_overlap_uniqueness_weights`.
- Produces: `CausalAlphaV4FitConfig`, `CausalAlphaV4Fit`, `CausalAlphaV4Forecast`, `fit_causal_alpha_v4`.

- [ ] **Step 1: Write failing synthetic fit test**

```python
fit = fit_causal_alpha_v4(
    train_symbols=("BTCUSDT", "ETHUSDT"),
    samples=samples,
    knowledge_cutoff=300,
    config=CausalAlphaV4FitConfig(
        market_ridge_strength=1.0,
        residual_ridge_strength=0.1,
        direction_ridge_strength=0.1,
    ),
)
assert set(fit.market_models) == {"4h", "24h", "72h"}
assert set(fit.residual_models) == {"4h", "24h", "72h"}
assert set(fit.direction_models) == {"4h", "24h", "72h"}
```

The synthetic data makes common return depend on global context and residual return depend on local context, proving decomposition and shared-model behavior.

- [ ] **Step 2: Verify RED**

```bash
uv run pytest -q tests/workflows/test_universal_causal_alpha_v4_fitting.py tests/learning/test_causal_alpha_v4_forecast.py
```

- [ ] **Step 3: Implement market-proxy and shared residual heads**

Market head fits BTC forward return on global context. Residual head pools train symbols with equal eligible weight mass and fits residual return on concatenated target-local features, local context, global context, nine instrument descriptors, beta, and their explicit availability masks. There is one residual model per horizon, never a `symbol -> model` dispatch.

- [ ] **Step 4: Implement shared direction heads**

Fit labels `-1.0` and `+1.0` from original symbol forward-return sign. Exact-zero return labels are excluded and counted. Direction model consumes the same current-time feature surface as residual prediction.

- [ ] **Step 5: Implement forecast composition**

```python
final_prediction = beta * market_prediction + residual_prediction
```

Persist per horizon: market prediction, beta, beta-scaled market contribution, residual prediction, final prediction, direction score, component model digests, and forecast digest.

- [ ] **Step 6: Add cutoff/order falsification tests**

Changing a label ending at or after cutoff cannot change fit digest. Train-symbol order is part of sample-scope identity. No validation/test label can affect a fit. There is no symbol-specific residual model mapping.

- [ ] **Step 7: Verify and commit**

```bash
uv run pytest -q tests/workflows/test_universal_causal_alpha_v4_fitting.py tests/learning/test_causal_alpha_v4_forecast.py
uv run mypy trade_rl/workflows/universal_causal_alpha_v4_fitting.py trade_rl/learning/causal_alpha_v4.py
```

```bash
git add trade_rl/workflows/universal_causal_alpha_v4_fitting.py trade_rl/learning/causal_alpha_v4.py tests/workflows/test_universal_causal_alpha_v4_fitting.py tests/learning/test_causal_alpha_v4_forecast.py
git commit -m "feat: fit hierarchical causal alpha v4 heads"
```

---

### Task 8: State-conditioned uncertainty and liveness evidence

**Files:**
- Modify: `trade_rl/learning/causal_alpha_v4.py`
- Create: `trade_rl/workflows/universal_causal_alpha_v4_signal.py`
- Create: `tests/learning/test_causal_alpha_v4_uncertainty.py`
- Create: `tests/workflows/test_universal_causal_alpha_v4_liveness.py`

**Interfaces:**
- Consumes: final reconstructed train-prefix forecasts and state variables.
- Produces: `CausalAlphaV4UncertaintyModel`, `CausalAlphaV4LivenessEvidence`.

- [ ] **Step 1: Write failing state precedence/fallback tests**

```python
class V4ForecastState(str, Enum):
    NORMAL = "normal"
    HIGH_REALIZED_VOLATILITY = "high_realized_volatility"
    LOW_LIQUIDITY = "low_liquidity"
    BASIS_POSITIONING_STRESS = "basis_positioning_stress"
```

Thresholds: high realized volatility at or above the eligible train-prefix 80th percentile; low liquidity at or below the 20th percentile; basis/positioning stress at or above the 80th percentile of absolute stress score. Precedence is stress, low liquidity, high volatility, normal.

- [ ] **Step 2: Verify RED**

```bash
uv run pytest -q tests/learning/test_causal_alpha_v4_uncertainty.py tests/workflows/test_universal_causal_alpha_v4_liveness.py
```

- [ ] **Step 3: Implement final-residual RMSE by state**

Use residuals of final `beta * market + residual` prediction against original symbol labels. Effective sample size is `sum(w)^2 / sum(w^2)`. ESS below `30.0` selects the global horizon RMSE and records `fallback_reason="insufficient_state_ess"`.

- [ ] **Step 4: Implement liveness sidecar**

Persist per fit/symbol/horizon:

```text
prediction_mean
prediction_std
prediction_min
prediction_max
prediction_quantiles
unique_count_at_tolerance_1e-12
median_near_identical_run_length
maximum_near_identical_run_length
intercept
dynamic_prediction_std
weighted_final_rmse
dynamic_to_rmse_ratio
constant_feature_count
available_feature_count
contribution_variance_existing_15m
contribution_variance_existing_1h
contribution_variance_existing_4h
contribution_variance_existing_1d
contribution_variance_local_cross_market
contribution_variance_global_market
contribution_variance_beta_scaled_proxy
contribution_variance_shared_residual
direction_score_mean
direction_score_std
direction_positive_fraction
direction_negative_fraction
```

If `dynamic_prediction_std == 0.0` while at least two fitted features are available and non-constant, evidence construction fails. Otherwise sidecar is descriptive and always `promotion_eligible=False`.

- [ ] **Step 5: Add intercept-dominated falsification test**

A non-zero intercept with zero dynamic coefficients must fail liveness integrity when source features vary.

- [ ] **Step 6: Verify and commit**

```bash
uv run pytest -q tests/learning/test_causal_alpha_v4_uncertainty.py tests/workflows/test_universal_causal_alpha_v4_liveness.py
```

```bash
git add trade_rl/learning/causal_alpha_v4.py trade_rl/workflows/universal_causal_alpha_v4_signal.py tests/learning/test_causal_alpha_v4_uncertainty.py tests/workflows/test_universal_causal_alpha_v4_liveness.py
git commit -m "feat: calibrate v4 uncertainty and liveness"
```

---

### Task 9: Slow anchor plus fast 4h target compiler

**Files:**
- Modify: `trade_rl/learning/causal_alpha_v4.py`
- Create: `tests/learning/test_causal_alpha_v4_target.py`

**Interfaces:**
- Consumes: 4h/24h/72h forecast, state uncertainty, one-way cost, liquidity cap, current weight.
- Produces: `CausalAlphaV4TargetConfig`, `CausalAlphaV4TargetPath`, `causal_alpha_v4_target_path`.

- [ ] **Step 1: Write failing target invariants**

```python
@dataclass(frozen=True, slots=True)
class CausalAlphaV4TargetConfig:
    slow_target_magnitudes: tuple[float, ...] = (0.0, 0.025, 0.05, 0.10, 0.25)
    fast_deviation_magnitudes: tuple[float, ...] = (0.0, 0.025, 0.05)
    uncertainty_multiplier: float = 1.0
    execution_cost_multiplier: float = 1.5
    edge_margin: float = 0.001
    slow_rebalance_decisions: int = 16
    fast_rebalance_decisions: int = 4
    maximum_final_target_delta: float = 0.125
    maximum_fast_absolute_deviation: float = 0.05
```

Tests require direction disagreement to block exposure increase but never block movement toward zero.

- [ ] **Step 2: Verify RED**

```bash
uv run pytest -q tests/learning/test_causal_alpha_v4_target.py
```

- [ ] **Step 3: Implement slow anchor**

Use:

```python
slow_mu = 0.5 * (prediction_24h + prediction_72h / 3.0)
```

Slow uncertainty combines state-conditioned 24h/72h RMSE and horizon disagreement. Score candidate anchor turnover from current actual weight using the V3 incremental-edge semantics.

- [ ] **Step 4: Implement bounded fast impulse**

The 4h lane chooses a deviation around the selected slow anchor. Final target is clipped by liquidity/absolute cap, `maximum_fast_absolute_deviation`, and `maximum_final_target_delta`. Exposure-increasing fast changes require 4h return forecast and direction score sign agreement.

- [ ] **Step 5: Prove cost is charged exactly once**

For current `w0`, anchor `a`, final `f`, tests assert direct final turnover cost equals the total cost charged by the staged objective. Implement fast improvement as `direct_objective(w0, f) - direct_objective(w0, a)`, not as a second independent `a -> f` fee if that would double count the authored turnover model.

- [ ] **Step 6: Add no-edge and risk-reduction falsification tests**

A positive 4h forecast below uncertainty plus execution hurdle plus edge margin cannot create a target change. Direction disagreement cannot block flattening or liquidity deleveraging.

- [ ] **Step 7: Verify and commit**

```bash
uv run pytest -q tests/learning/test_causal_alpha_v4_target.py
uv run mypy trade_rl/learning/causal_alpha_v4.py
```

```bash
git add trade_rl/learning/causal_alpha_v4.py tests/learning/test_causal_alpha_v4_target.py
git commit -m "feat: compile v4 slow anchor and fast impulse"
```

---

### Task 10: Canonical fast and slow Signal gates

**Files:**
- Modify: `trade_rl/workflows/universal_causal_alpha_v4_signal.py`
- Create: `tests/workflows/test_universal_causal_alpha_v4_signal.py`
- Create: `examples/binance/universal-causal-alpha-v4-research.json`

**Interfaces:**
- Consumes: V4 forecasts, uncertainty, liveness, chronological partitions.
- Produces: `CausalAlphaV4SignalScopeMetric`, `CausalAlphaV4SignalEvidence`.

- [ ] **Step 1: Write failing independent-cohort tests**

4h Signal uses non-overlapping rows based on 4h label ends. Slow Signal uses non-overlapping rows based on 72h label ends for fused 24h/72h evidence. Adjacent 15m overlapping labels are not independent observations.

- [ ] **Step 2: Freeze exact first V4 research config**

```json
{
  "schema_version": "universal_causal_alpha_v4_research_config_v1",
  "fit": {
    "market_ridge_strength": 1.0,
    "residual_ridge_strength": 0.1,
    "direction_ridge_strength": 0.1
  },
  "target": {
    "slow_target_magnitudes": [0.0, 0.025, 0.05, 0.1, 0.25],
    "fast_deviation_magnitudes": [0.0, 0.025, 0.05],
    "uncertainty_multiplier": 1.0,
    "execution_cost_multiplier": 1.5,
    "edge_margin": 0.001,
    "slow_rebalance_decisions": 16,
    "fast_rebalance_decisions": 4,
    "maximum_final_target_delta": 0.125,
    "maximum_fast_absolute_deviation": 0.05
  },
  "signal_gate": {
    "independent_episode_count": 8,
    "minimum_rank_ic_lower_ci": 0.0,
    "minimum_top_bottom_spread_lower_ci": 0.0,
    "minimum_direction_accuracy_excess_lower_ci": 0.0,
    "bootstrap_resamples": 10000,
    "bootstrap_seed": 20260823,
    "bootstrap_block_size": 2
  }
}
```

Apply the same lower-bound requirements independently to `fast_4h` and `slow_fused`; both must pass before economic replay.

- [ ] **Step 3: Verify RED**

```bash
uv run pytest -q tests/workflows/test_universal_causal_alpha_v4_signal.py
```

- [ ] **Step 4: Implement canonical metrics**

Reuse V3 rank/spread/direction/bootstrap primitives only where their mathematical semantics are identical. V4 writes V4 schema identities and keeps fast/slow evidence separate. Liveness digest is bound to scope evidence but cannot relax Signal thresholds.

- [ ] **Step 5: Verify and commit**

```bash
uv run pytest -q tests/workflows/test_universal_causal_alpha_v4_signal.py
```

```bash
git add trade_rl/workflows/universal_causal_alpha_v4_signal.py tests/workflows/test_universal_causal_alpha_v4_signal.py examples/binance/universal-causal-alpha-v4-research.json
git commit -m "feat: add causal alpha v4 signal gates"
```

---

### Task 11: V4 replay metric with correct activity accounting

**Files:**
- Create: `trade_rl/workflows/universal_causal_alpha_v4_replay.py`
- Create: `tests/workflows/test_universal_causal_alpha_v4_replay.py`

**Interfaces:**
- Consumes: V4 target path and existing `ActionPathEvaluation`.
- Produces: `CausalAlphaV4ReplayMetric`.

- [ ] **Step 1: Write failing positive-execution/zero-closed-trade test**

Construct an `ActionPathEvaluation` with positive filled turnover and `executed_change_count > 0` but `trade_count == 0`; assert V4 evidence calls it meaningful execution.

- [ ] **Step 2: Define complete metric schema**

```python
@dataclass(frozen=True, slots=True)
class CausalAlphaV4ReplayMetric:
    run_manifest_digest: str
    v4_context_manifest_digest: str
    config_digest: str
    symbol: str
    episode_index: int
    contract_digest: str
    fit_digest: str
    forecast_digest: str
    target_path_digest: str
    gross_return: float
    net_return: float
    turnover_per_day: float
    total_execution_cost: float
    submitted_change_count: int
    executed_change_count: int
    closed_trade_count: int
    sign_flip_count: int
    maximum_drawdown: float
    execution_rejection_reason_counts: tuple[tuple[str, int], ...]
    risk_projection_reason_counts: tuple[tuple[str, int], ...]
    target_reason_counts: tuple[tuple[str, int], ...]
    hard_risk_violation: bool
    schema_version: str = "causal_alpha_v4_replay_metric_v1"
    digest: str = ""
```

`has_meaningful_execution` is true when `executed_change_count > 0` or total filled turnover exceeds the existing evaluation action-change tolerance. `closed_trade_count` alone is never a liveness condition.

- [ ] **Step 3: Verify RED then implement adapter**

```bash
uv run pytest -q tests/workflows/test_universal_causal_alpha_v4_replay.py
```

Read `evaluation.collapse_evidence.executed_change_count`, `evaluation.performance.trade_count`, `evaluation.performance.turnover_total`, actual simulator cost, and drawdown. Do not infer execution from target changes.

- [ ] **Step 4: Add attribution tests**

Separate no submitted change, submitted-but-suppressed, execution rejection, positive fill without close, and hard-risk violation.

- [ ] **Step 5: Verify and commit**

```bash
uv run pytest -q tests/workflows/test_universal_causal_alpha_v4_replay.py
```

```bash
git add trade_rl/workflows/universal_causal_alpha_v4_replay.py tests/workflows/test_universal_causal_alpha_v4_replay.py
git commit -m "feat: add v4 economic replay accounting"
```

---

### Task 12: V4 economic selection and untouched Teacher admission

**Files:**
- Create: `trade_rl/workflows/universal_causal_alpha_v4_selection.py`
- Create: `trade_rl/workflows/universal_causal_alpha_v4_admission.py`
- Create: `tests/workflows/test_universal_causal_alpha_v4_selection.py`
- Create: `tests/workflows/test_universal_causal_alpha_v4_admission.py`

**Interfaces:**
- Consumes: complete V4 replay metrics; selected single authored hypothesis for holdout.
- Produces: `CausalAlphaV4SelectionEvidence`, `CausalAlphaV4AdmissionEvidence`.

- [ ] **Step 1: Write failing selection tests**

The first generation has one authored candidate. Require all of:

```text
mean gross return >= 0
mean net return >= 0
worst symbol/episode net return >= -0.05
positive gross episode fraction >= 0.5
unexplained execution rejection count == 0
hard risk violation count == 0
meaningful executed activity exists
```

Do not reject solely because all `closed_trade_count` values are zero.

- [ ] **Step 2: Verify RED and implement selection**

```bash
uv run pytest -q tests/workflows/test_universal_causal_alpha_v4_selection.py
```

Do not add a parameter-ranking grid in V1. Failure preserves evidence and stops.

- [ ] **Step 3: Write untouched holdout tests**

Teacher admission opens only after both Signal gates and economic selection pass. The selected fit cutoff is at holdout start. Holdout labels cannot enter fit, beta history, state thresholds, uncertainty calibration, liveness thresholding, or target configuration.

- [ ] **Step 4: Implement admission**

Require aggregate gross/net non-negative, no more than half negative-gross symbol holdouts, worst net at least `-0.05`, zero hard-risk violations, zero unexplained rejections, and meaningful execution across holdout records. Persist executed and closed counts separately.

- [ ] **Step 5: Verify and commit**

```bash
uv run pytest -q tests/workflows/test_universal_causal_alpha_v4_selection.py tests/workflows/test_universal_causal_alpha_v4_admission.py
```

```bash
git add trade_rl/workflows/universal_causal_alpha_v4_selection.py trade_rl/workflows/universal_causal_alpha_v4_admission.py tests/workflows/test_universal_causal_alpha_v4_selection.py tests/workflows/test_universal_causal_alpha_v4_admission.py
git commit -m "feat: gate causal alpha v4 economics"
```

---

### Task 13: Artifact store, pipeline, runner, and CLI

**Files:**
- Create: `trade_rl/workflows/universal_causal_alpha_v4_artifact_store.py`
- Create: `trade_rl/workflows/universal_causal_alpha_v4_pipeline.py`
- Create: `trade_rl/workflows/universal_causal_alpha_v4_runner.py`
- Create: `scripts/run_universal_causal_alpha_v4_research.py`
- Create: `tests/workflows/test_universal_causal_alpha_v4_artifact_store.py`
- Create: `tests/workflows/test_universal_causal_alpha_v4_pipeline.py`
- Create: `tests/scripts/test_run_universal_causal_alpha_v4_research.py`

**Interfaces:**
- Consumes: V4 JSON config, base runtime context, V4 context manifest.
- Produces: terminal `signal_rejected`, `selection_rejected`, `admission_rejected`, or research-only admitted V4 teacher package.

- [ ] **Step 1: Write failing resume/corruption tests**

Store identity binds base runtime manifest digest, V4 context manifest digest, V4 config digest, source-tree/generator digest, contract digest, fit digest, forecast digest, and target-path digest. Missing scopes can resume; corrupt/stale/wrong-run evidence fails closed and is not overwritten as valid.

- [ ] **Step 2: Implement exact artifact layout**

```text
signal/fast/BTCUSDT/0.json
signal/slow/BTCUSDT/0.json
signal/liveness/BTCUSDT/0.json
selection/BTCUSDT/0.json
admission/BTCUSDT.json
result.json
```

The same directory pattern applies to each symbol/episode. Every leaf is atomic and content-digested.

- [ ] **Step 3: Implement pipeline order**

```text
strict V4 config
base runtime + V4 context identity closure
train-only V4 preparation
fast Signal gate
slow Signal gate
economic replay and selection
untouched Teacher admission
research-only V4 teacher package
```

Any failure prevents all later stages.

- [ ] **Step 4: Implement CLI exit codes**

```text
0 admitted research teacher
2 signal_rejected
3 selection_rejected
4 admission_rejected
other execution or configuration failure
```

CLI requires `--v4-context-manifest` in addition to the V3-equivalent runtime inputs.

- [ ] **Step 5: Verify and commit**

```bash
uv run pytest -q tests/workflows/test_universal_causal_alpha_v4_artifact_store.py tests/workflows/test_universal_causal_alpha_v4_pipeline.py tests/scripts/test_run_universal_causal_alpha_v4_research.py
uv run python -m py_compile scripts/run_universal_causal_alpha_v4_research.py
```

```bash
git add trade_rl/workflows/universal_causal_alpha_v4_artifact_store.py trade_rl/workflows/universal_causal_alpha_v4_pipeline.py trade_rl/workflows/universal_causal_alpha_v4_runner.py scripts/run_universal_causal_alpha_v4_research.py tests/workflows/test_universal_causal_alpha_v4_artifact_store.py tests/workflows/test_universal_causal_alpha_v4_pipeline.py tests/scripts/test_run_universal_causal_alpha_v4_research.py
git commit -m "feat: add causal alpha v4 research runner"
```

---

### Task 14: Cross-layer regression, static checks, and falsification review

**Files:**
- Modify only files required to correct failures found by this task; keep each correction in a separate focused commit.

**Interfaces:**
- Consumes: Tasks 1-13 implementation.
- Produces: verification evidence for V4 plus proof that V3/non-V4 behavior did not drift.

- [ ] **Step 1: Run focused V4 suite**

```bash
uv run pytest -q \
  tests/data/test_v4_context.py \
  tests/data/test_v4_context_artifact.py \
  tests/integrations/test_binance_v4_context.py \
  tests/rl/test_universal_v4_context.py \
  tests/learning/test_causal_alpha_v4_beta.py \
  tests/learning/test_causal_alpha_v4_forecast.py \
  tests/learning/test_causal_alpha_v4_uncertainty.py \
  tests/learning/test_causal_alpha_v4_target.py \
  tests/workflows/test_universal_causal_alpha_v4_manifest.py \
  tests/workflows/test_universal_causal_alpha_v4_runtime.py \
  tests/workflows/test_universal_causal_alpha_v4_fitting.py \
  tests/workflows/test_universal_causal_alpha_v4_liveness.py \
  tests/workflows/test_universal_causal_alpha_v4_signal.py \
  tests/workflows/test_universal_causal_alpha_v4_replay.py \
  tests/workflows/test_universal_causal_alpha_v4_selection.py \
  tests/workflows/test_universal_causal_alpha_v4_admission.py \
  tests/workflows/test_universal_causal_alpha_v4_artifact_store.py \
  tests/workflows/test_universal_causal_alpha_v4_pipeline.py \
  tests/scripts/test_materialize_universal_causal_alpha_v4_context.py \
  tests/scripts/test_run_universal_causal_alpha_v4_research.py
```

- [ ] **Step 2: Run maintained V3 regression set**

```bash
uv run pytest -q \
  tests/workflows/test_universal_causal_alpha_v3_runner_engine.py \
  tests/workflows/test_universal_causal_alpha_v3_runner_orchestration.py \
  tests/workflows/test_universal_causal_alpha_v3_runner_failure_semantics.py \
  tests/workflows/test_universal_causal_alpha_v3_runner_store.py \
  tests/workflows/test_universal_causal_alpha_v3_selection_diagnostics.py \
  tests/learning/test_causal_alpha_weighted_ridge.py
```

Expected: no V3 fixture or schema migration is needed for V4.

- [ ] **Step 3: Run static and architecture checks**

```bash
uv run ruff check --diff .
uv run ruff format --check --diff .
uv run mypy .
uv run lint-imports
uv run vulture trade_rl tests --min-confidence 100
```

- [ ] **Step 4: Run full tests and branch coverage**

```bash
uv run pytest -q --cov=trade_rl --cov-branch --cov-report=term-missing --cov-report=json:coverage.json
uv run python .github/check_critical_coverage.py coverage.json pyproject.toml
```

- [ ] **Step 5: Run the repository compatibility scope on Linux and Windows CI**

```bash
uv run pytest -q \
  tests/architecture \
  tests/data \
  tests/simulation \
  tests/evaluation \
  tests/serving \
  tests/examples/test_docker_training_assets.py::test_provenance_validation_target_fails_fast_without_valid_arguments \
  tests/operations/test_training_capability_audit.py::test_run_training_capability_audit_preserves_report_contract \
  tests/operations/test_training_capability_audit.py::test_sequence_training_exercises_real_hierarchical_behavior_cloning
```

- [ ] **Step 6: Build and probe the training image**

Use the exact final HEAD/source/lock digests and the same `docker/Dockerfile.training` build procedure as `.github/workflows/ci.yml`. Inside the non-root image, import every new V4 module and run `python scripts/run_universal_causal_alpha_v4_research.py --help`.

- [ ] **Step 7: Perform independent falsification review**

Attempt to construct a counterexample for every item below. Any successful counterexample blocks empirical execution until corrected and reverified.

```text
Teacher sees a current input Student cannot reproduce
future Spot or OI changes earlier context
future target or BTC return changes earlier beta
validation or test label enters fit
symbol order silently changes model semantics
non-zero intercept hides zero dynamic signal
4h lane trades below cost and uncertainty hurdle
direction disagreement blocks flattening
missing context becomes informative zero
provider revision preserves stale digest
slow and fast stages charge two costs for one final delta
closed-trade count is treated as executed activity
```

- [ ] **Step 8: Re-run Steps 1-7 after every corrective code change**

Do not weaken a test or acceptance rule to obtain green status.

---

### Task 15: Execute one immutable train-only V4 generation

**Files:**
- No scientific source change is permitted after generation identity is frozen.
- Retained output root: `var/retained-causal-alpha-v4/hierarchical-v1-20260823-r1`.

**Interfaces:**
- Consumes: clean final HEAD, complete V4 context manifest, exact authored V4 config, maintained immutable Universal runtime.
- Produces: retained identities, Signal/liveness/replay/admission evidence, terminal result.

- [ ] **Step 1: Freeze generation identity**

Persist source HEAD, source-tree digest, `uv.lock` digest, training image identity, base runtime manifest digest, V4 context manifest digest, V4 config digest, V4 generator code digest, resolved profile, and PIT-flow state. Refuse dirty source.

- [ ] **Step 2: Materialize and validate V4 context**

Use these environment variables already supported by the project/operator contract:

```bash
RUNTIME_ROOT="$TRADE_RL_UNIVERSAL_ARTIFACT_ROOT"
CACHE_ROOT="${TRADE_RL_MARKET_DATA_CACHE_ROOT:-/workspace/market-data/binance-vision}"
V4_CONTEXT_ROOT="$RUNTIME_ROOT/v4-context/hierarchical-v1-20260823-r1"

uv run python scripts/materialize_universal_causal_alpha_v4_context.py \
  --runtime-manifest "$RUNTIME_ROOT/runtime-manifest.json" \
  --frozen-metadata-root "$RUNTIME_ROOT/frozen-metadata/usds-m" \
  --market-data-cache-root "$CACHE_ROOT" \
  --output-root "$V4_CONTEXT_ROOT" \
  --profile derivatives-auto
```

The resolved profile is frozen by source coverage only.

- [ ] **Step 3: Run authored V4 research path once**

```bash
OUTPUT_ROOT="var/retained-causal-alpha-v4/hierarchical-v1-20260823-r1"

uv run python scripts/run_universal_causal_alpha_v4_research.py \
  --config examples/binance/universal-causal-alpha-v4-research.json \
  --run-config examples/binance-multitimeframe/universal-u6-ppo.json \
  --runtime-manifest "$RUNTIME_ROOT/runtime-manifest.json" \
  --v4-context-manifest "$V4_CONTEXT_ROOT/manifest.json" \
  --frozen-metadata-root "$RUNTIME_ROOT/frozen-metadata/usds-m" \
  --output-root "$OUTPUT_ROOT"
```

- [ ] **Step 4: Validate retained evidence before interpreting outcomes**

Re-load every retained file through strict readers. Require every expected fast/slow Signal scope and liveness sidecar, plus complete replay/admission records for the terminal stage reached. Recompute and compare all identity digests.

- [ ] **Step 5: Apply terminal semantics without tuning**

- Signal failure: retain evidence and stop; no parameter or feature change in this generation.
- Selection failure: retain evidence and stop before holdout admission.
- Admission failure: retain evidence and do not start BC/PPO.
- Admission pass: create a research-only V4 teacher package with `promotion_eligible=false`; then author the learner plan as a new task/design cycle.

- [ ] **Step 6: Produce final evidence report**

Report fast/slow Signal lower bounds, liveness distributions, uncertainty-state support/fallback, gross/net/worst returns, submitted/executed/closed counts, turnover/cost, target-reason attribution, hard-risk/rejection counts, and untouched admission result. State separately what is proven and what remains unverified.

---

## Final Quality Gate

Do not report the implementation complete unless all items below hold on the same final HEAD:

1. Reward/risk/execution/action semantics have no V4-driven change.
2. Existing V3 tests and historical schema readers remain valid without migration.
3. Every V4 current-time teacher input, availability mask, staleness value, and beta needed for actions is exposed through the V4 student observation capability.
4. Core/derivative feature profile names and ordered counts match this plan and the spec exactly.
5. Future source values cannot alter earlier context, beta, fit, or prediction.
6. Market-proxy/residual reconstruction is exact within `1e-15` on finite eligible rows.
7. Direction disagreement cannot block flattening or liquidity deleveraging.
8. State uncertainty uses train-prefix final residuals only and records fallback support.
9. Slow/fast objective tests prove execution cost is charged once.
10. V4 meaningful execution uses executed changes/filled turnover, with closed trades separate.
11. Focused tests, V3 regressions, full pytest with branch coverage, Ruff, format, MyPy, Import Linter, Vulture, compatibility checks, and training-image build/probe succeed.
12. Independent falsification review finds no uncorrected counterexample to the listed invariants.
13. CI/required checks are green for the exact final commit used for the retained generation.
14. The immutable V4 generation is validly rejected at its first failed gate or reaches Teacher admission without post-outcome scientific parameter changes.
15. Remaining unverified items are explicit; Teacher admission is not a profitability claim and is not Production GO.
