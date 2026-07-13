# Causal Market Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a causal, content-addressed real-market-data path and one shared training/serving observation contract.

**Architecture:** Add explicit source, instrument and feature contracts under `trade_rl.data`; build a regular point-in-time dataset through a deterministic NumPy pipeline; expand `MarketDataset` with active, staleness and volume semantics; move observation construction behind `ObservationBuilder`; use the same builder from the environment and serving runtime.

**Tech Stack:** Python 3.12, NumPy, standard-library CSV, Gymnasium, pytest.

## Global Constraints

- Do not restore `mars_lite`.
- Do not add pandas, psycopg or another mandatory runtime dependency.
- A policy observation at `t` must depend only on information available at or before `t`.
- Execution may consult the actual next execution bar, but observation construction may not.
- Dataset identity must be content-addressed and order-sensitive.
- Production status remains `NO-GO`.

---

### Task 1: Data and instrument contracts

**Files:**
- Create: `trade_rl/data/contracts.py`
- Create: `tests/data/test_market_contracts.py`

**Interfaces:**
- Produces `VolumeUnit`, `InstrumentContract`, `FeatureKind`, `NormalizationMode`, `FeatureSpec`, `MarketBuildConfig`.

- [ ] Write tests for invalid lifetimes, units, multipliers, lookbacks, windows and staleness limits.
- [ ] Verify the tests fail because the contracts do not exist.
- [ ] Implement immutable validated contracts and canonical payload methods.
- [ ] Verify the contract tests pass.

### Task 2: Raw source and CSV adapter

**Files:**
- Create: `trade_rl/data/source.py`
- Create: `tests/data/test_market_source.py`

**Interfaces:**
- Consumes: `InstrumentContract`.
- Produces: `RawMarketSeries`, `MarketDataSource`, `InMemoryMarketDataSource`, `CsvMarketDataSource`.

- [ ] Write tests for UTC timestamp parsing, optional funding/tradable columns, duplicate rejection and immutable arrays.
- [ ] Verify expected failures.
- [ ] Implement source contracts and the standard-library CSV adapter.
- [ ] Verify source tests pass.

### Task 3: Expand MarketDataset v3

**Files:**
- Modify: `trade_rl/data/market.py`
- Modify: `tests/data/test_market_dataset_v2.py`
- Modify dataset factories in `tests/rl`, `tests/strategies` and `tests/simulation`.

**Interfaces:**
- Adds `symbol_active`, `feature_staleness`, `volume_units`, `contract_multipliers`, `feature_config_digest`, `normalization_digest`.
- Adds `market_notional(index, prices)`.

- [ ] Add failing tests for masks, staleness range, metadata digests and volume-unit conversion.
- [ ] Verify expected failures.
- [ ] Implement v3 validation and market-notional conversion.
- [ ] Update explicit test fixtures.
- [ ] Verify existing dataset and consumer tests pass.

### Task 4: Causal MarketDatasetBuilder

**Files:**
- Create: `trade_rl/data/builder.py`
- Modify: `trade_rl/data/__init__.py`
- Create: `tests/data/test_market_builder.py`

**Interfaces:**
- Consumes: `MarketDataSource`, ordered `InstrumentContract` values, `MarketBuildConfig`.
- Produces: `MarketDatasetBuilder.build(source, instruments) -> MarketDataset`.

- [ ] Write failing tests for real CSV construction, point-in-time active/tradable masks and deterministic identity.
- [ ] Add the prefix-invariance test: build prefix and full datasets and compare every prefix feature-related array.
- [ ] Add identity tests for symbol order, feature configuration, normalization and contract metadata.
- [ ] Implement regular union-clock alignment, causal feature calculations, availability/staleness, global features and array hashing.
- [ ] Verify builder tests pass.

### Task 5: Shared ObservationBuilder and causality

**Files:**
- Modify: `trade_rl/rl/observations.py`
- Modify: `trade_rl/rl/environment.py`
- Modify: `tests/rl/test_observation_v2.py`
- Create: `tests/rl/test_observation_causality.py`

**Interfaces:**
- Produces: `ObservationInput`, `ObservationBuilder.layout`, `ObservationBuilder.build`.
- Keeps `build_observation` as a compatibility wrapper.

- [ ] Write failing tests that assert per-feature availability/staleness and current active/tradable values are present.
- [ ] Write a regression test showing that changing only `tradable[t + 1]` leaves the observation at `t` unchanged.
- [ ] Write a regression test showing that mutating all rows after `t` leaves the observation at `t` unchanged.
- [ ] Implement the shared builder and switch the environment to it.
- [ ] Verify observation and environment tests pass.

### Task 6: Point-in-time trend and explicit execution volume

**Files:**
- Modify: `trade_rl/strategies/trend.py`
- Modify: `trade_rl/simulation/execution.py`
- Modify: `tests/strategies/test_trend_time_config.py`
- Modify: `tests/simulation/test_execution_v2.py`

**Interfaces:**
- Trend excludes symbols inactive at either lookback endpoint.
- Execution calls `MarketDataset.market_notional`.

- [ ] Write failing tests for pre-listing/post-delisting zero targets and all volume-unit capacity calculations.
- [ ] Implement active-universe trend centering.
- [ ] Replace implicit `price * volume` capacity with explicit notional conversion.
- [ ] Verify strategy and execution tests pass.

### Task 7: Shared serving observation path

**Files:**
- Modify: `trade_rl/serving/runtime.py`
- Create: `tests/serving/test_shared_observation_builder.py`

**Interfaces:**
- `ServingRuntime` owns an `ObservationBuilder`.
- Adds `build_observation(input: ObservationInput)` and `predict_state(input: ObservationInput)`.

- [ ] Write a failing test comparing environment and serving observation bytes for identical state.
- [ ] Implement the structured serving path while preserving raw `predict` compatibility.
- [ ] Verify serving tests pass.

### Task 8: Dataset manifest v3 and documentation

**Files:**
- Modify: `trade_rl/domain/datasets.py`
- Modify: `tests/artifacts/test_codec.py`
- Modify: `tests/rl/test_ensemble_training.py`
- Modify: `tests/domain/test_artifact_invariants.py`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `README.md`
- Modify: `README.ja.md`

**Interfaces:**
- `DatasetManifest` records global feature names, feature/normalization digests, volume units and multipliers.

- [ ] Write failing manifest validation and codec tests.
- [ ] Implement the v3 manifest.
- [ ] Document the maintained CSV builder, point-in-time universe and observation contract.
- [ ] Verify artifact and documentation tests pass.

### Task 9: Full verification

- [ ] Run `uv run ruff check .`.
- [ ] Run `uv run ruff format --check .`.
- [ ] Run `uv run mypy trade_rl`.
- [ ] Run `uv run lint-imports`.
- [ ] Run `uv run pytest --cov=trade_rl --cov-branch`.
- [ ] Inspect the branch diff against `main` and confirm every approved requirement has a corresponding implementation and regression test.
