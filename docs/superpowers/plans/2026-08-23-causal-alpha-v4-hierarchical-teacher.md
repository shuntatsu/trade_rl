# Causal Alpha V4 Hierarchical Teacher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an opt-in, research-only Causal Alpha V4 lane that adds reproducible Spot/perpetual/global market context, a BTC-proxy + shared-residual hierarchical teacher with 4h/24h/72h responsibilities, state-conditioned uncertainty, and correctly defined economic replay through Teacher admission while leaving reward, risk, execution, and all V3 historical meanings unchanged.

**Architecture:** Keep the maintained Universal runtime and V3 lane intact. Materialize V4 auxiliary context into separately digested artifacts, expose the same context to the Universal student observation surface, fit deterministic V4 market-proxy/residual/direction heads from train-only data, compile a 24h/72h slow anchor plus bounded 4h fast impulse, and evaluate it through a V4-only Signal/selection/admission evidence chain.

**Tech Stack:** Python 3.12, NumPy `<2.0`, Gymnasium 0.29.1, existing Binance Vision/Public REST infrastructure, existing weighted ridge primitives, pytest, Hypothesis, Ruff, MyPy, Import Linter, Docker training image.

**Spec:** `docs/implementation-plans/specs/2026-08-23-causal-alpha-v4-hierarchical-teacher-design.md`

## Global Constraints

- Reward remains the maintained cost-inclusive pure net-equity log-growth objective. Do not modify reward configuration or `trade_rl/rl/rewards.py`.
- Risk, execution, latency, partial-fill, liquidation, target-weight, and accounting semantics remain unchanged.
- Existing V3 source files and retained V3 artifacts keep their historical meaning; V4 gets new schemas and artifact roots.
- Do not reinterpret V3 `trade_count`; V4 replay evidence must persist `executed_change_count` and `closed_trade_count` separately from inception.
- Keep the maintained 206 target-local Universal market features and nine instrument descriptors semantically unchanged.
- `cross_market_core_v1` has exactly 24 local channels; `cross_market_derivatives_v1` has exactly 31.
- `global_market_core_v1` has exactly 38 channels; `global_market_derivatives_v1` has exactly 44.
- The first V4 market proxy is `BTCUSDT` USD-M perpetual.
- Causal beta uses 4h returns, a 720h lookback, at least 90 complete samples, and clipping to `[-3.0, 3.0]`; BTC beta is exactly `1.0`.
- Missing required local/global context is unavailable, not informative zero. A required-context miss makes the V4 decision non-actionable.
- On-chain `pit_flow_v1` is disabled unless a provider-specific artifact proves point-in-time/revision-frozen history for the complete authored interval. No scraped reconstructed history is allowed.
- The first V4 deterministic model hypothesis uses weighted/objective-normalized ridge only: market-proxy ridge `1.0`, residual ridge `0.1`, direction ridge `0.1`.
- The first target hypothesis uses slow target magnitudes `(0.0, 0.025, 0.05, 0.10, 0.25)`, fast deviations `(0.0, 0.025, 0.05)`, slow cadence `16` decisions, fast cadence `4` decisions, maximum final target delta `0.125`, maximum fast absolute deviation `0.05`, execution-cost multiplier `1.5`, and edge margin `0.001`.
- Direction evidence is a signed score, not a calibrated probability. Increasing exposure or reversing sign requires return forecast and direction score to agree in sign. Flattening/reducing absolute exposure is never blocked by direction disagreement.
- Uncertainty state precedence is `basis_positioning_stress > low_liquidity > high_realized_volatility > normal`; state thresholds are derived from eligible train-prefix quantiles and bound into fit identity.
- A state whose effective sample size is below `30.0` uses the horizon-global weighted RMSE and records the fallback.
- Signal liveness is required evidence but is not a tunable post-hoc hard threshold in the first generation. Exact-zero dynamic prediction variance is an integrity failure; otherwise liveness metrics remain descriptive until a separate authored gate is justified.
- No validation symbol, test symbol, Teacher-admission holdout outcome, BC result, RL result, or sealed evaluation may tune V4 features, thresholds, ridge strengths, target parameters, or state definitions.
- No BC/PPO training starts in this plan. This plan ends at a student-compatible, Teacher-admitted V4 package or a preserved research rejection. A downstream learner plan is authored only after Teacher admission.

---

## File Structure

Create focused V4 files instead of expanding the large V3 and Binance modules:

- `trade_rl/data/v4_context.py` — immutable local/global context schemas, aligned arrays, feature formulas, availability/staleness rules, and deterministic digests.
- `trade_rl/data/v4_context_artifact.py` — filesystem artifact writer/loader for per-symbol V4 context arrays.
- `trade_rl/integrations/binance_v4_context.py` — Binance Spot/perpetual/funding/metrics source adapters and context materialization inputs.
- `trade_rl/workflows/universal_causal_alpha_v4_manifest.py` — V4 auxiliary manifest referencing one immutable base `UniversalRuntimeManifest`.
- `trade_rl/rl/universal_v4_context.py` — policy observation provider and schema metadata for local/global V4 context.
- `trade_rl/learning/causal_alpha_v4.py` — beta, forecast contracts, direction score, state uncertainty, liveness primitives, and fast/slow target compiler.
- `trade_rl/workflows/universal_causal_alpha_v4_runtime.py` — load/validate base runtime plus V4 context artifacts and prepare train-only V4 samples.
- `trade_rl/workflows/universal_causal_alpha_v4_fitting.py` — fit/cache market-proxy, shared residual, and shared direction heads.
- `trade_rl/workflows/universal_causal_alpha_v4_signal.py` — canonical 4h and slow fused Signal evidence plus liveness sidecars.
- `trade_rl/workflows/universal_causal_alpha_v4_replay.py` — production-environment economic replay with correct activity accounting.
- `trade_rl/workflows/universal_causal_alpha_v4_selection.py` — V4 candidate evidence/ranking.
- `trade_rl/workflows/universal_causal_alpha_v4_admission.py` — untouched selected-teacher holdout admission.
- `trade_rl/workflows/universal_causal_alpha_v4_artifact_store.py` — immutable/restart-safe V4 evidence storage.
- `trade_rl/workflows/universal_causal_alpha_v4_pipeline.py` — ordered Gate orchestration.
- `trade_rl/workflows/universal_causal_alpha_v4_runner.py` — thin facade matching V3 responsibility boundaries.
- `scripts/materialize_universal_causal_alpha_v4_context.py` — deterministic auxiliary context materializer.
- `scripts/run_universal_causal_alpha_v4_research.py` — research-only CLI.
- `examples/binance/universal-causal-alpha-v4-research.json` — frozen first V4 hypothesis.

Only modify existing files at integration seams:

- `trade_rl/integrations/binance_cache.py` — public generic Vision URL planning/sync helper required by the metrics archive adapter.
- `trade_rl/rl/universal_single_instrument_env.py` — opt-in auxiliary context provider insertion into Dict observation space/observations.
- `trade_rl/workflows/binance_universal_runtime.py` — opt-in construction of the V4 observation provider when a V4 manifest is supplied; canonical V3 behavior remains byte-for-byte configuration-equivalent when absent.
- `trade_rl/workflows/universal_full_research_entrypoint.py` — optional V4 context manifest path in the runtime factory context, default `None`.

Tests are created under matching `tests/data`, `tests/integrations`, `tests/rl`, `tests/learning`, `tests/workflows`, and `tests/scripts` paths.

---

### Task 1: Immutable V4 context contracts and core formulas

**Files:**
- Create: `trade_rl/data/v4_context.py`
- Test: `tests/data/test_v4_context.py`

**Interfaces:**
- Consumes: finite aligned arrays already placed on the maintained 15m decision clock.
- Produces: `V4ContextProfile`, `V4ContextBlock`, `V4TargetContext`, `build_cross_market_core(...)`, `build_global_market_core(...)`, `robust_trailing_zscore(...)`.

- [ ] **Step 1: Write failing schema and formula tests**

Add tests that assert exact channel order and basic formulas:

```python
from trade_rl.data.v4_context import (
    CROSS_MARKET_CORE_NAMES,
    GLOBAL_MARKET_CORE_NAMES,
    taker_quote_imbalance,
    spot_perp_log_basis,
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

Also assert malformed volume, non-positive prices, duplicate names, mismatched availability arrays, and non-finite values fail closed.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
uv run pytest -q tests/data/test_v4_context.py
```

Expected: import/collection failure because `trade_rl.data.v4_context` does not exist.

- [ ] **Step 3: Implement frozen names and immutable blocks**

Define these public contracts:

```python
CROSS_MARKET_CORE_NAMES: tuple[str, ...] = (...24 exact spec names...)
CROSS_MARKET_DERIVATIVE_NAMES: tuple[str, ...] = (...7 exact spec names...)
GLOBAL_MARKET_CORE_NAMES: tuple[str, ...] = (...38 exact spec names...)
GLOBAL_MARKET_DERIVATIVE_NAMES: tuple[str, ...] = (...6 exact spec names...)

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

Every NumPy array is copied C-contiguous, validated, made read-only, and included in `content_and_arrays_digest`. `values` remains finite even where unavailable; availability controls semantic use. Unavailable values must be numerically zero only as storage representation and must never be interpreted without the Boolean mask.

- [ ] **Step 4: Implement causal formula primitives**

Implement:

```python
def taker_quote_imbalance(taker_buy_quote: float, total_quote: float) -> float:
    if not math.isfinite(total_quote) or total_quote <= 0.0:
        raise ValueError("total_quote must be finite and positive")
    value = 2.0 * taker_buy_quote / total_quote - 1.0
    return float(np.clip(value, -1.0, 1.0))


def spot_perp_log_basis(*, spot: float, perp: float) -> float:
    if spot <= 0.0 or perp <= 0.0:
        raise ValueError("basis prices must be positive")
    return math.log(perp / spot)
```

`robust_trailing_zscore` uses only rows `<= current row`, median and `1.4826 * MAD`, a fixed minimum support `32`, and returns `(value, available)`; zero MAD returns `0.0, True` only when support exists.

- [ ] **Step 5: Add future-mutation causality/property tests**

Use Hypothesis or deterministic prefix copies to prove changing rows after decision `t` cannot alter any context value at or before `t`.

- [ ] **Step 6: Run tests and commit**

Run:

```bash
uv run pytest -q tests/data/test_v4_context.py
uv run ruff check trade_rl/data/v4_context.py tests/data/test_v4_context.py
uv run mypy trade_rl/data/v4_context.py
```

Commit:

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
- Consumes: `V4TargetContext` from Task 1.
- Produces: `write_v4_target_context_artifact(path, context) -> Path`, `load_v4_target_context_artifact(path) -> V4TargetContext`.

- [ ] **Step 1: Write failing round-trip and corruption tests**

Tests must assert exact round-trip equality of digests and rejection of changed array bytes, changed feature order, changed source digest, duplicate symbol, missing manifest, and unexpected extra arrays.

```python
def test_v4_context_artifact_round_trip(tmp_path, sample_v4_context):
    output = write_v4_target_context_artifact(tmp_path / "BTCUSDT", sample_v4_context)
    loaded = load_v4_target_context_artifact(output)
    assert loaded.digest == sample_v4_context.digest
    np.testing.assert_array_equal(loaded.local.values, sample_v4_context.local.values)
```

- [ ] **Step 2: Verify RED**

```bash
uv run pytest -q tests/data/test_v4_context_artifact.py
```

Expected: import failure.

- [ ] **Step 3: Implement immutable artifact format**

Use:

```text
<artifact>/manifest.json
<artifact>/arrays.npz
```

Manifest schema: `causal_alpha_v4_target_context_artifact_v1`. Bind symbol, profile, ordered local/global feature names, source digests, array SHA-256/content digest, row count, first/last decision index, and context digest. Existing content at the same path may be reused only when bytes/digest match exactly; otherwise fail with `FileExistsError`.

- [ ] **Step 4: Verify GREEN and commit**

```bash
uv run pytest -q tests/data/test_v4_context.py tests/data/test_v4_context_artifact.py
uv run ruff check trade_rl/data/v4_context*.py tests/data/test_v4_context*.py
uv run mypy trade_rl/data/v4_context.py trade_rl/data/v4_context_artifact.py
```

Commit:

```bash
git add trade_rl/data/v4_context_artifact.py tests/data/test_v4_context_artifact.py
git commit -m "feat: persist causal alpha v4 context artifacts"
```

---

### Task 3: Binance Spot/perpetual and futures-metrics source adapter

**Files:**
- Create: `trade_rl/integrations/binance_v4_context.py`
- Modify: `trade_rl/integrations/binance_cache.py`
- Test: `tests/integrations/test_binance_v4_context.py`
- Modify/Test: `tests/examples/test_market_data_sync.py`

**Interfaces:**
- Consumes: `BinancePublicTransport`, official Binance Vision cache, target symbol, BTC/ETH anchor symbols, research start/end.
- Produces: `BinanceV4SourceBundle`, `build_binance_v4_context(...) -> V4TargetContext`, optional `BinanceFuturesMetricsSeries`.

- [ ] **Step 1: Write failing official-URL and parser tests**

Freeze the Vision futures metrics URL form:

```python
def vision_futures_metrics_url(symbol: str, day: datetime) -> str:
    date = day.astimezone(UTC).strftime("%Y-%m-%d")
    return (
        "https://data.binance.vision/data/futures/um/daily/metrics/"
        f"{symbol}/{symbol}-metrics-{date}.zip"
    )
```

Parser tests use small ZIP/CSV fixtures and require fields to map by header name, not column position. Required derivative fields are `sum_open_interest`, `sum_open_interest_value`, and the exact long/short/taker columns selected by the adapter. Missing required columns reject the derivative profile.

- [ ] **Step 2: Verify RED**

```bash
uv run pytest -q tests/integrations/test_binance_v4_context.py tests/examples/test_market_data_sync.py
```

- [ ] **Step 3: Add generic immutable Vision URL plan/sync helper**

In `binance_cache.py`, add public helpers without changing existing plan semantics:

```python
@dataclass(frozen=True, slots=True)
class BinanceVisionUrlPlan:
    urls: tuple[str, ...]


def inspect_binance_vision_urls(
    urls: Sequence[str], *, cache_root: str | Path
) -> BinanceVisionCacheReport: ...


def sync_binance_vision_urls(
    urls: Sequence[str], *, transport: _VisionArchiveTransport | BinancePublicTransport
) -> BinanceVisionCacheReport: ...
```

Reuse `vision_cache_path` and `validate_cached_vision_payload`; reject non-Binance-Vision URLs.

- [ ] **Step 4: Implement source bundle assembly**

`build_binance_v4_context` loads/aligned target Spot and target USD-M datasets plus BTC/ETH Spot/perpetual sources. Core profile is available from klines/funding. Derivative profile is enabled only when the complete required metrics URL plan validates for the full interval.

Do not fetch short-retention REST OI history to backfill missing old metrics. Return a capability decision object:

```python
@dataclass(frozen=True, slots=True)
class BinanceV4ProfileCapability:
    profile_name: str
    derivative_metrics_complete: bool
    missing_url_count: int
    source_digest: str
```

- [ ] **Step 5: Add timestamp/as-of causality tests**

Construct a metrics row whose event timestamp lies after a decision close and prove it is unavailable at that decision. Prove modifying future Spot/perp rows cannot alter earlier context.

- [ ] **Step 6: Verify and commit**

```bash
uv run pytest -q tests/integrations/test_binance_v4_context.py tests/examples/test_market_data_sync.py
uv run ruff check trade_rl/integrations/binance_v4_context.py trade_rl/integrations/binance_cache.py tests/integrations/test_binance_v4_context.py
uv run mypy trade_rl/integrations/binance_v4_context.py trade_rl/integrations/binance_cache.py
```

Commit:

```bash
git add trade_rl/integrations/binance_v4_context.py trade_rl/integrations/binance_cache.py tests/integrations/test_binance_v4_context.py tests/examples/test_market_data_sync.py
git commit -m "feat: add binance v4 cross-market sources"
```

---

### Task 4: V4 auxiliary manifest and deterministic context materialization

**Files:**
- Create: `trade_rl/workflows/universal_causal_alpha_v4_manifest.py`
- Create: `scripts/materialize_universal_causal_alpha_v4_context.py`
- Create: `tests/workflows/test_universal_causal_alpha_v4_manifest.py`
- Create: `tests/scripts/test_materialize_universal_causal_alpha_v4_context.py`

**Interfaces:**
- Consumes: base `UniversalRuntimeManifest`, all 15 maintained target symbols, Task 3 source capability.
- Produces: `CausalAlphaV4ContextManifest` plus one V4 context artifact per train/validation/test target.

- [ ] **Step 1: Write failing strict-manifest tests**

Define:

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

The ordered `context_digests` must exactly follow `train + validation + test` order from the base manifest. Unknown fields, missing symbols, reordering, profile drift, or base-manifest digest drift fail closed.

- [ ] **Step 2: Verify RED**

```bash
uv run pytest -q tests/workflows/test_universal_causal_alpha_v4_manifest.py tests/scripts/test_materialize_universal_causal_alpha_v4_context.py
```

- [ ] **Step 3: Implement materializer**

CLI arguments:

```text
--runtime-manifest
--frozen-metadata-root
--market-data-cache-root
--output-root
--profile {core,derivatives-auto}
```

`derivatives-auto` decides only from complete source coverage before reading model labels or outcomes. It resolves once to `cross_market_core_v1/global_market_core_v1` or `cross_market_derivatives_v1/global_market_derivatives_v1` and records the decision.

- [ ] **Step 4: Test idempotency and partial failure**

If 14 artifacts exist and one is missing, regenerate only the missing artifact; any existing digest mismatch aborts without overwrite. Manifest is written only after all 15 target artifacts validate.

- [ ] **Step 5: Verify and commit**

```bash
uv run pytest -q tests/workflows/test_universal_causal_alpha_v4_manifest.py tests/scripts/test_materialize_universal_causal_alpha_v4_context.py
uv run py_compile scripts/materialize_universal_causal_alpha_v4_context.py
```

Commit:

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
- Consumes: `CausalAlphaV4ContextManifest` + loaded `V4TargetContext` artifacts.
- Produces: `V4ContextProvider` and opt-in Dict observation keys `local_cross_market_context`, `global_market_context`, `causal_beta`.

- [ ] **Step 1: Write failing observation-space parity tests**

Canonical runtime without a V4 manifest must expose exactly its previous observation keys/digest behavior. V4 runtime must add exactly:

```text
local_cross_market_context: shape (1, 24 or 31)
global_market_context: shape (1, 38 or 44)
causal_beta: shape (1, 1)
```

Provider test:

```python
def test_v4_provider_returns_context_for_exact_decision(provider, binding):
    result = provider(environment_at(index=100), binding)
    assert result.local.shape == (1, 24)
    assert result.global_market.shape == (1, 38)
```

- [ ] **Step 2: Verify RED**

```bash
uv run pytest -q tests/rl/test_universal_v4_context.py tests/workflows/test_binance_universal_runtime.py tests/rl/test_universal_research_u3_u6.py
```

- [ ] **Step 3: Implement provider contract**

```python
@dataclass(frozen=True, slots=True)
class V4PolicyContext:
    local: np.ndarray
    global_market: np.ndarray
    beta: np.ndarray
    digest: str

class V4ContextProvider:
    local_width: int
    global_width: int
    schema_digest: str
    def __call__(self, environment: object, binding: InstrumentDatasetBinding) -> V4PolicyContext: ...
```

The provider finds the exact `current_index` row. If local/global/beta availability required for the current action is false, it raises `V4ContextUnavailable`; the environment catches this only to mark the V4 action decision non-actionable in the V4 path. Canonical non-V4 environments never instantiate this provider.

- [ ] **Step 4: Extend routed environment opt-in only**

Add `v4_context_provider: V4ContextProvider | None = None` to `EpisodeRoutedSingleInstrumentEnv`. `_policy_observation_space` adds the three keys only when provider is present. `_policy_observation` appends the values using the same provider instance. Bind its schema digest into `observation_contract_digest` and sequence layout metadata.

- [ ] **Step 5: Extend runtime context without mutating base manifest**

Add `v4_context_manifest_path: Path | None = None` to `UniversalRuntimeFactoryContext`. If non-null, load and validate the V4 manifest against `context.manifest.manifest_digest`; otherwise current behavior is unchanged.

- [ ] **Step 6: Add falsification tests**

Reject stale/missing global context, wrong symbol artifact, wrong decision index, wrong feature order, base-manifest mismatch, and V4 teacher context that cannot be surfaced to policy observation.

- [ ] **Step 7: Verify and commit**

```bash
uv run pytest -q tests/rl/test_universal_v4_context.py tests/rl/test_universal_research_u3_u6.py tests/workflows/test_binance_universal_runtime.py
uv run mypy trade_rl/rl/universal_v4_context.py trade_rl/rl/universal_single_instrument_env.py trade_rl/workflows/binance_universal_runtime.py trade_rl/workflows/universal_full_research_entrypoint.py
uv run lint-imports
```

Commit:

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
- Consumes: existing `CausalAlphaSymbolSamples`, V4 context artifacts, base datasets.
- Produces: `CausalAlphaV4SymbolSamples`, `CausalBetaSeries`, `causal_alpha_v4_beta(...)`.

- [ ] **Step 1: Write failing beta tests**

Freeze exact contract:

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

Tests assert BTC beta equals `1.0`, future target/BTC returns do not alter earlier beta, insufficient support produces `available=False`, and clipping works.

- [ ] **Step 2: Verify RED**

```bash
uv run pytest -q tests/learning/test_causal_alpha_v4_beta.py tests/workflows/test_universal_causal_alpha_v4_runtime.py
```

- [ ] **Step 3: Implement 4h forward labels with existing timing primitive**

Use `forward_log_return_label(..., horizon_hours=4.0, signal_delay_decisions=..., decision_bars=...)`; do not duplicate timing math. `CausalAlphaV4SymbolSamples` contains original 24h/72h arrays plus 4h label/end arrays, local/global context arrays, context availability, beta and beta availability.

- [ ] **Step 4: Implement residual decomposition**

For each horizon:

```python
residual = symbol_label - beta * btc_market_proxy_label
reconstructed = beta * btc_market_proxy_label + residual
```

Require `abs(reconstructed - symbol_label) <= 1e-15` for finite eligible rows.

- [ ] **Step 5: Reject validation/test BTC labels in fit preparation**

Prepared V4 fitting scope must contain only authored train-symbol label blocks. Current-time BTC/ETH context for validation/test evaluation is permitted; future validation/test labels cannot enter fit arrays.

- [ ] **Step 6: Verify and commit**

```bash
uv run pytest -q tests/learning/test_causal_alpha_v4_beta.py tests/workflows/test_universal_causal_alpha_v4_runtime.py
uv run ruff check trade_rl/learning/causal_alpha_v4.py trade_rl/workflows/universal_causal_alpha_v4_runtime.py tests/learning/test_causal_alpha_v4_beta.py tests/workflows/test_universal_causal_alpha_v4_runtime.py
uv run mypy trade_rl/learning/causal_alpha_v4.py trade_rl/workflows/universal_causal_alpha_v4_runtime.py
```

Commit:

```bash
git add trade_rl/learning/causal_alpha_v4.py trade_rl/workflows/universal_causal_alpha_v4_runtime.py tests/learning/test_causal_alpha_v4_beta.py tests/workflows/test_universal_causal_alpha_v4_runtime.py
git commit -m "feat: add v4 causal beta and residual labels"
```

---

### Task 7: Market-proxy, shared residual, and direction fits

**Files:**
- Create: `trade_rl/workflows/universal_causal_alpha_v4_fitting.py`
- Create: `tests/workflows/test_universal_causal_alpha_v4_fitting.py`
- Modify: `trade_rl/learning/causal_alpha_v4.py`
- Create: `tests/learning/test_causal_alpha_v4_forecast.py`

**Interfaces:**
- Consumes: Task 6 V4 samples, existing `fit_causal_alpha_ridge`, existing overlap uniqueness weights.
- Produces: `CausalAlphaV4FitConfig`, `CausalAlphaV4Fit`, `CausalAlphaV4Forecast`.

- [ ] **Step 1: Write failing synthetic decomposition fit test**

Generate two symbols where common return depends on a global feature and residual return depends on a local feature. Assert the shared residual fit is one model per horizon, not one per symbol.

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

- [ ] **Step 2: Verify RED**

```bash
uv run pytest -q tests/workflows/test_universal_causal_alpha_v4_fitting.py tests/learning/test_causal_alpha_v4_forecast.py
```

- [ ] **Step 3: Implement symbol-balanced overlap-aware shared fits**

Market-proxy head fits BTC forward return on global context. Residual head pools train symbols with equal eligible weight mass and fits target residual on concatenated target-local + local cross-market + global context + instrument descriptors + beta. Direction head fits `sign(original symbol return)` on the same current-time input matrix using labels `-1.0` and `+1.0`; exact-zero labels are excluded from direction fit and recorded.

- [ ] **Step 4: Implement forecast composition**

```python
final_h = beta * market_prediction_h + residual_prediction_h
```

Persist `market_prediction`, `beta`, `beta_scaled_market`, `residual_prediction`, `final_prediction`, and `direction_score` for each horizon. The forecast digest binds all component arrays and model digests.

- [ ] **Step 5: Add symbol-order and future-cutoff falsification tests**

Changing a label whose end is at/after cutoff cannot change fit digest. Permuting train-symbol order changes scope digest but, after re-aligning equal semantic rows, cannot silently produce the same scope identity. No hidden symbol-specific model mapping exists.

- [ ] **Step 6: Verify and commit**

```bash
uv run pytest -q tests/workflows/test_universal_causal_alpha_v4_fitting.py tests/learning/test_causal_alpha_v4_forecast.py
uv run mypy trade_rl/workflows/universal_causal_alpha_v4_fitting.py trade_rl/learning/causal_alpha_v4.py
```

Commit:

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
- Consumes: Task 7 final reconstructed in-prefix forecasts and context state variables.
- Produces: `CausalAlphaV4UncertaintyModel`, `CausalAlphaV4LivenessEvidence`.

- [ ] **Step 1: Write failing state precedence/fallback tests**

Freeze state order:

```python
class V4ForecastState(str, Enum):
    NORMAL = "normal"
    HIGH_REALIZED_VOLATILITY = "high_realized_volatility"
    LOW_LIQUIDITY = "low_liquidity"
    BASIS_POSITIONING_STRESS = "basis_positioning_stress"
```

State thresholds are train-prefix weighted quantiles: high volatility >= 80th percentile, low liquidity <= 20th percentile, basis/positioning stress >= 80th percentile of absolute authored stress score. Precedence is stress, low-liquidity, high-volatility, normal.

- [ ] **Step 2: Verify RED**

```bash
uv run pytest -q tests/learning/test_causal_alpha_v4_uncertainty.py tests/workflows/test_universal_causal_alpha_v4_liveness.py
```

- [ ] **Step 3: Implement weighted final-residual RMSE by state**

Each horizon/state uses residuals of the final `beta * market + residual` prediction against original symbol labels. Effective sample size is `sum(w)^2 / sum(w^2)`. If ESS `< 30.0`, select the global horizon RMSE and record `fallback_reason="insufficient_state_ess"`.

- [ ] **Step 4: Implement liveness sidecar**

`CausalAlphaV4LivenessEvidence` persists, per fit/symbol/horizon:

```text
prediction_mean/std/min/max/quantiles
unique_count_at_tolerance_1e-12
median_near_identical_run_length
maximum_near_identical_run_length
intercept
dynamic_prediction_std = std(prediction - intercept)
weighted_final_rmse
dynamic_to_rmse_ratio
constant_feature_count
available_feature_count
contribution_variance_by_family
contribution_variance_by_existing_timeframe
direction_score_mean/std/positive_fraction/negative_fraction
```

If `dynamic_prediction_std == 0.0` for a scope with at least two available varying source features, artifact construction fails because the forecast is not state-dependent. Otherwise the evidence is descriptive and `promotion_eligible=False`.

- [ ] **Step 5: Add intercept-dominated falsification test**

Construct a model with non-zero intercept and all dynamic coefficients zero. Assert non-zero mean prediction does not satisfy liveness construction when varying features exist.

- [ ] **Step 6: Verify and commit**

```bash
uv run pytest -q tests/learning/test_causal_alpha_v4_uncertainty.py tests/workflows/test_universal_causal_alpha_v4_liveness.py
```

Commit:

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
- Consumes: 4h/24h/72h forecast, state uncertainty, one-way execution cost, liquidity cap, current weight.
- Produces: `CausalAlphaV4TargetConfig`, `CausalAlphaV4TargetPath`.

- [ ] **Step 1: Write failing target invariants**

Freeze config:

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

Reuse V3 24h-equivalent slow fusion:

```python
slow_mu = 0.5 * (prediction_24h + prediction_72h / 3.0)
```

Slow uncertainty uses state-conditioned 24h/72h RMSE plus horizon disagreement. Score candidate anchor turnover from the current actual weight with the same incremental edge formula used by V3.

- [ ] **Step 4: Implement fast impulse as bounded deviation**

4h chooses a deviation around the already chosen slow anchor. Final candidate is clipped by liquidity/absolute cap and `maximum_final_target_delta` from current weight. Require 4h return forecast and 4h direction score sign agreement for exposure-increasing fast deviation.

- [ ] **Step 5: Prove execution cost is charged exactly once**

Test direct and staged objective equality:

```python
assert total_cost_hurdle(current, final) == (
    slow_stage_cost(current, anchor) + fast_marginal_cost(current, anchor, final)
)
```

The implementation may compute the direct total cost once and express fast marginal improvement relative to the selected anchor; it must never charge `current -> anchor` and `anchor -> final` if that exceeds the authored absolute one-way final turnover model.

- [ ] **Step 6: Add no-edge fast-lane falsification test**

A positive 4h forecast below `uncertainty + cost + margin` must not create a target change. Strong fast direction does not bypass the economic hurdle.

- [ ] **Step 7: Verify and commit**

```bash
uv run pytest -q tests/learning/test_causal_alpha_v4_target.py
uv run mypy trade_rl/learning/causal_alpha_v4.py
```

Commit:

```bash
git add trade_rl/learning/causal_alpha_v4.py tests/learning/test_causal_alpha_v4_target.py
git commit -m "feat: compile v4 slow anchor and fast impulse"
```

---

### Task 10: Canonical V4 Signal gates

**Files:**
- Modify: `trade_rl/workflows/universal_causal_alpha_v4_signal.py`
- Create: `tests/workflows/test_universal_causal_alpha_v4_signal.py`
- Create: `examples/binance/universal-causal-alpha-v4-research.json`

**Interfaces:**
- Consumes: Task 7 forecasts, Task 8 uncertainty/liveness, chronological V3-compatible partitions.
- Produces: `CausalAlphaV4SignalScopeMetric`, `CausalAlphaV4SignalEvidence`.

- [ ] **Step 1: Write failing 4h/slow cohort tests**

Use non-overlapping cohort selection based on each metric horizon's label ends. The 4h cohort must not count 15m-overlapping 4h labels as independent. Slow cohort continues to use 72h label-end spacing for the fused 24h/72h metric.

- [ ] **Step 2: Freeze first V4 authored config**

Create JSON with exactly one model hypothesis and target config:

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

The gate is applied independently to `fast_4h` and `slow_fused`; both must pass every lower-bound requirement before economic replay.

- [ ] **Step 3: Verify RED**

```bash
uv run pytest -q tests/workflows/test_universal_causal_alpha_v4_signal.py
```

- [ ] **Step 4: Implement canonical metrics and clustered episode bootstrap**

Reuse V3 signal diagnostic math where semantics match, but write V4 schemas and preserve separate fast/slow evidence. Liveness sidecar digests are bound to scope evidence but not used to relax Signal thresholds.

- [ ] **Step 5: Verify and commit**

```bash
uv run pytest -q tests/workflows/test_universal_causal_alpha_v4_signal.py
```

Commit:

```bash
git add trade_rl/workflows/universal_causal_alpha_v4_signal.py tests/workflows/test_universal_causal_alpha_v4_signal.py examples/binance/universal-causal-alpha-v4-research.json
git commit -m "feat: add causal alpha v4 signal gates"
```

---

### Task 11: V4 replay metrics with correct activity accounting

**Files:**
- Create: `trade_rl/workflows/universal_causal_alpha_v4_replay.py`
- Create: `tests/workflows/test_universal_causal_alpha_v4_replay.py`

**Interfaces:**
- Consumes: V4 target path and existing `evaluate_episode_action_path_on_environment` result.
- Produces: `CausalAlphaV4ReplayMetric`.

- [ ] **Step 1: Write failing zero-closed-trade/positive-execution test**

Construct an evaluation with positive turnover and `executed_change_count > 0` but `closed_trade_count == 0`. Assert it is meaningful activity.

- [ ] **Step 2: Define metric schema**

```python
@dataclass(frozen=True, slots=True)
class CausalAlphaV4ReplayMetric:
    symbol: str
    episode_index: int
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
    ...identity digests...
```

`has_meaningful_execution` is true when `executed_change_count > 0` or total turnover exceeds the existing action-change tolerance. `closed_trade_count` never participates alone in liveness rejection.

- [ ] **Step 3: Verify RED then implement replay adapter**

```bash
uv run pytest -q tests/workflows/test_universal_causal_alpha_v4_replay.py
```

Use `evaluation.collapse_evidence.executed_change_count`, `evaluation.performance.trade_count`, and actual simulator turnover/cost. Do not reconstruct execution from target changes.

- [ ] **Step 4: Add rejection attribution tests**

Differentiate no submitted change, submitted-but-suppressed, execution rejection, positive filled turnover with no close, and hard-risk failure.

- [ ] **Step 5: Verify and commit**

```bash
uv run pytest -q tests/workflows/test_universal_causal_alpha_v4_replay.py
```

Commit:

```bash
git add trade_rl/workflows/universal_causal_alpha_v4_replay.py tests/workflows/test_universal_causal_alpha_v4_replay.py
git commit -m "feat: add v4 economic replay accounting"
```

---

### Task 12: V4 selection and untouched Teacher admission

**Files:**
- Create: `trade_rl/workflows/universal_causal_alpha_v4_selection.py`
- Create: `trade_rl/workflows/universal_causal_alpha_v4_admission.py`
- Create: `tests/workflows/test_universal_causal_alpha_v4_selection.py`
- Create: `tests/workflows/test_universal_causal_alpha_v4_admission.py`

**Interfaces:**
- Consumes: complete V4 replay metrics; selected candidate only for holdout.
- Produces: `CausalAlphaV4SelectionEvidence`, `CausalAlphaV4AdmissionEvidence`.

- [ ] **Step 1: Write failing selection tests**

First V4 has one authored candidate, so selection is admission/rejection rather than grid tuning. Require:

```text
mean gross return >= 0
mean net return >= 0
worst symbol/episode net >= -0.05
positive gross episode fraction >= 0.5
no unexplained execution rejection
no hard-risk violation
meaningful executed activity exists
```

Do not reject solely because `closed_trade_count == 0`.

- [ ] **Step 2: Verify RED and implement selection**

```bash
uv run pytest -q tests/workflows/test_universal_causal_alpha_v4_selection.py
```

Ranking code must not exist for an un-authored parameter sweep in V1. A failed candidate preserves evidence and ends the run.

- [ ] **Step 3: Write untouched holdout admission tests**

Admission opens only after Signal and selection pass. Fit cutoff is the holdout start and no holdout label is available to fitting, state thresholds, liveness thresholds, or target tuning.

- [ ] **Step 4: Implement admission evidence**

Use the same V4 metric semantics including executed/closed count separation. Require aggregate gross/net non-negative, at most half negative-gross symbol holdouts, worst net >= `-0.05`, no hard risk, no unexplained rejection, and meaningful execution across the holdout set.

- [ ] **Step 5: Verify and commit**

```bash
uv run pytest -q tests/workflows/test_universal_causal_alpha_v4_selection.py tests/workflows/test_universal_causal_alpha_v4_admission.py
```

Commit:

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
- Consumes: V4 config, base runtime context, V4 context manifest.
- Produces: terminal `signal_rejected`, `selection_rejected`, `admission_rejected`, or research-only admitted V4 teacher package.

- [ ] **Step 1: Write failing resume/corruption tests**

Store identities include base runtime manifest digest, V4 context manifest digest, V4 config digest, source-tree/generator digest, contract digest, fit digest, forecast digest, and target-path digest. Missing scope can resume; corrupt/stale/wrong-run evidence fails closed and is never overwritten as if valid.

- [ ] **Step 2: Implement artifact paths**

Use:

```text
signal/fast/<symbol>/<episode>.json
signal/slow/<symbol>/<episode>.json
signal/liveness/<symbol>/<episode>.json
selection/<symbol>/<episode>.json
admission/<symbol>.json
result.json
```

Every leaf is atomic and content-digested.

- [ ] **Step 3: Implement pipeline order**

```text
strict config
-> base runtime + V4 context identity closure
-> train-only V4 preparation
-> fast Signal gate
-> slow Signal gate
-> economic replay/selection
-> untouched Teacher admission
-> research-only V4 teacher package
```

Any failure prevents all later stages.

- [ ] **Step 4: Implement CLI exit codes**

Match V3 operator semantics without sharing artifact schema:

```text
0 = admitted research teacher
2 = signal_rejected
3 = selection_rejected
4 = admission_rejected
other = execution/configuration failure
```

CLI requires `--v4-context-manifest` in addition to V3-equivalent runtime inputs.

- [ ] **Step 5: Verify and commit**

```bash
uv run pytest -q tests/workflows/test_universal_causal_alpha_v4_artifact_store.py tests/workflows/test_universal_causal_alpha_v4_pipeline.py tests/scripts/test_run_universal_causal_alpha_v4_research.py
uv run py_compile scripts/run_universal_causal_alpha_v4_research.py
```

Commit:

```bash
git add trade_rl/workflows/universal_causal_alpha_v4_artifact_store.py trade_rl/workflows/universal_causal_alpha_v4_pipeline.py trade_rl/workflows/universal_causal_alpha_v4_runner.py scripts/run_universal_causal_alpha_v4_research.py tests/workflows/test_universal_causal_alpha_v4_artifact_store.py tests/workflows/test_universal_causal_alpha_v4_pipeline.py tests/scripts/test_run_universal_causal_alpha_v4_research.py
git commit -m "feat: add causal alpha v4 research runner"
```

---

### Task 14: Cross-layer regression and architecture verification

**Files:**
- Modify only tests/docs required by failing maintained contracts discovered in this task.
- Test: existing architecture, data, RL, learning, workflow, and serving suites.

**Interfaces:**
- Consumes: complete implementation from Tasks 1-13.
- Produces: objective evidence that V3/non-V4 behavior did not drift and V4 context is student-reproducible.

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

Expected: no V3 fixture/schema changes are required for V4.

- [ ] **Step 3: Run static and architecture checks**

```bash
uv run ruff check --diff .
uv run ruff format --check --diff .
uv run mypy .
uv run lint-imports
uv run vulture trade_rl tests --min-confidence 100
```

- [ ] **Step 4: Run full tests with branch coverage**

```bash
uv run pytest -q --cov=trade_rl --cov-branch --cov-report=term-missing --cov-report=json:coverage.json
uv run python .github/check_critical_coverage.py coverage.json pyproject.toml
```

- [ ] **Step 5: Run cross-platform compatibility scope**

On Linux and Windows runners, execute the same compatibility set used by CI:

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

- [ ] **Step 6: Build/probe training image**

Use the exact final HEAD/source/lock digests and the same `docker/Dockerfile.training` build procedure as `.github/workflows/ci.yml`. Verify packaged import of every new V4 module and the V4 CLI `--help` path under the non-root runtime.

- [ ] **Step 7: Falsification review before empirical run**

Independently inspect the final diff against the spec and attempt to prove each of the following failure modes still exists:

```text
Teacher-only current input hidden from Student
future Spot/OI row changes earlier context
future target/BTC return changes earlier beta
validation/test label enters fit
symbol order silently changes semantics
non-zero intercept hides zero dynamic signal
4h fast lane trades below cost/uncertainty hurdle
direction disagreement blocks flattening
missing context becomes informative zero
provider revision preserves stale digest
slow+fast target double-charges one turnover
closed-trade count is treated as executed activity
```

Any successful counterexample blocks the next task.

- [ ] **Step 8: Commit verification-only fixes, if any, as separate commits**

Do not bundle unrelated refactors. Re-run Steps 1-7 after any code correction.

---

### Task 15: Execute one immutable train-only V4 generation

**Files:**
- No scientific source changes after generation identity is frozen.
- Persisted output only under a new retained V4 generation root.

**Interfaces:**
- Consumes: clean final HEAD, complete V4 context manifest, exact authored V4 JSON config, maintained immutable Universal runtime.
- Produces: retained source/config/context identities, per-scope Signal/liveness/replay/admission evidence, terminal result.

- [ ] **Step 1: Freeze generation identity before reading outcomes**

Persist:

```text
source HEAD
source tree digest
uv.lock digest
training image identity
base runtime manifest digest
V4 context manifest digest
V4 research config digest
V4 generator code digest
profile name and PIT-flow state
```

Refuse dirty source.

- [ ] **Step 2: Run V4 context materialization/validation**

```bash
uv run python scripts/materialize_universal_causal_alpha_v4_context.py \
  --runtime-manifest <immutable-runtime>/runtime-manifest.json \
  --frozen-metadata-root <immutable-runtime>/frozen-metadata/usds-m \
  --market-data-cache-root <cache>/binance-vision \
  --output-root <immutable-runtime>/v4-context \
  --profile derivatives-auto
```

The resolved profile is frozen by data coverage only.

- [ ] **Step 3: Run the authored V4 research path once**

```bash
uv run python scripts/run_universal_causal_alpha_v4_research.py \
  --config examples/binance/universal-causal-alpha-v4-research.json \
  --run-config examples/binance-multitimeframe/universal-u6-ppo.json \
  --runtime-manifest <immutable-runtime>/runtime-manifest.json \
  --v4-context-manifest <immutable-runtime>/v4-context/manifest.json \
  --frozen-metadata-root <immutable-runtime>/frozen-metadata/usds-m \
  --output-root <retained-v4-generation>
```

- [ ] **Step 4: Validate retained evidence before interpretation**

Re-load every artifact through strict readers, require all expected Signal scopes, liveness sidecars, replay records, and admission records for the terminal stage reached, and verify all digests against the frozen generation identity.

- [ ] **Step 5: Apply terminal semantics without tuning**

- Signal failure: retain evidence and stop. Do not alter feature pack, ridge strengths, direction rule, bootstrap, or target parameters in this generation.
- Selection failure: retain evidence and stop before holdout admission.
- Admission failure: retain evidence and do not create a BC/PPO learner plan from this generation.
- Admission pass: create a research-only V4 teacher package with `promotion_eligible=false`; only then author a separate learner implementation plan.

- [ ] **Step 6: Final report evidence**

Report exact fast/slow Signal lower bounds, liveness distributions, state uncertainty support/fallback, gross/net/worst returns, submitted/executed/closed counts, turnover/cost, target reason attribution, hard-risk/rejection counts, and untouched admission result. Explicitly separate what the run proves from what remains unverified.

---

## Final Quality Gate

Do not call the implementation complete unless all of the following hold on the same final HEAD:

1. Reward/risk/execution/action semantics have no V4-driven change.
2. Existing V3 tests and historical schema readers remain valid.
3. All V4 current-time teacher inputs are exposed through the V4 student observation capability.
4. Core/derivative feature profile names and ordered counts match the spec exactly.
5. Future source values cannot alter earlier context, beta, fit, or prediction.
6. Market-proxy/residual reconstruction is exact within the authored tolerance.
7. Direction disagreement cannot block risk-reducing flatten/deleveraging.
8. State uncertainty uses train-prefix residuals only and records fallback support.
9. Slow/fast objective tests prove execution cost is charged once.
10. V4 meaningful execution uses executed changes/filled turnover, with closed trades separate.
11. Targeted tests, V3 regressions, full pytest/branch coverage, Ruff, format, MyPy, Import Linter, Vulture, compatibility checks, and training-image build/probe pass.
12. Independent falsification review finds no uncorrected counterexample to the listed invariants.
13. CI/required checks are green for the exact final commit used for the retained generation.
14. The immutable V4 generation is either validly rejected at its first failed gate or reaches Teacher admission without post-outcome parameter changes.
15. Remaining unverified items are explicitly recorded; Teacher admission alone is not a profitability or Production claim.
