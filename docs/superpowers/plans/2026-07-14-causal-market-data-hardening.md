# Causal Market Data Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the remaining causal-data, eligibility, serving-schema, artifact, and CLI gaps in PR #24.

**Architecture:** Add one point-in-time eligibility contract to `MarketDataset` and make Trend, Alpha, and pre-trade targeting consume it. Bind the observation schema into serving bundles and fail closed on dataset/schema/size mismatches. Extend raw sources with information availability timestamps and provide a maintained JSON-configured CSV-to-dataset-artifact CLI path.

**Tech Stack:** Python 3.12, NumPy, argparse, pytest, GitHub Actions.

## Global Constraints

- Policy observations and target construction must not read rows after the decision index.
- Future execution-state tradability remains the responsibility of `MarketExecutor`.
- Existing public defaults remain backward compatible where practical.
- Dataset and serving identities must be deterministic and content-addressed.
- Production status remains `NO-GO` until fresh real-data Walk-Forward evaluation passes.

---

### Task 1: Point-in-time eligibility

**Files:**
- Modify: `trade_rl/data/market.py`
- Modify: `trade_rl/strategies/trend.py`
- Modify: `trade_rl/rl/environment.py`
- Test: `tests/rl/test_decision_eligibility.py`

- [ ] Write failing tests proving next-bar tradability cannot alter current pre-trade targets, suspended lookback bars remove Trend exposure, and Alpha is zeroed outside the current eligible universe.
- [ ] Add `MarketDataset.eligibility_mask(index, lookback=0, require_features=False)`.
- [ ] Use the mask in Trend, Alpha, and `_constrain_target()`.
- [ ] Run focused tests and commit.

### Task 2: Information availability contract

**Files:**
- Modify: `trade_rl/data/source.py`
- Modify: `trade_rl/data/builder.py`
- Modify: `trade_rl/data/market.py`
- Test: `tests/data/test_information_availability.py`

- [ ] Write failing tests for `available_at`, delayed information masking, identity changes, and CSV parsing.
- [ ] Add validated `RawMarketSeries.available_at` with CSV support.
- [ ] Build and hash `MarketDataset.information_available`; use it for causal feature/global-feature availability.
- [ ] Run focused tests and commit.

### Task 3: Serving fail-closed schema binding

**Files:**
- Modify: `trade_rl/rl/observations.py`
- Modify: `trade_rl/serving/bundle.py`
- Modify: `trade_rl/serving/runtime.py`
- Modify: `tests/serving/helpers.py`
- Test: `tests/serving/test_runtime_schema_guard.py`
- Modify: `tests/serving/test_bundle.py`

- [ ] Write failing tests for dataset ID, observation schema digest, and vector-size mismatches.
- [ ] Add a deterministic observation schema digest.
- [ ] Add schema digest and size to serving manifest/snapshot and validate them before inference.
- [ ] Run focused tests and commit.

### Task 4: Maintained dataset artifact and CLI path

**Files:**
- Create: `trade_rl/data/artifact.py`
- Create: `trade_rl/data/config.py`
- Modify: `trade_rl/cli/app.py`
- Test: `tests/cli/test_data_build_cli.py`

- [ ] Write a failing end-to-end test for JSON config plus per-symbol CSV input.
- [ ] Implement deterministic artifact write/load with JSON metadata and compressed NumPy arrays.
- [ ] Implement strict build-config parsing and `trade-rl data build --config ... --output ...`.
- [ ] Run focused tests and commit.

### Task 5: Expanded future-mutation regressions and documentation

**Files:**
- Modify: `tests/data/test_market_builder.py`
- Modify: `tests/rl/test_observation_causality.py`
- Modify: `README.md`
- Modify: `README.ja.md`
- Modify: `docs/ARCHITECTURE.md`

- [ ] Add mutations for future volume, funding, availability, missing rows, and universe metadata.
- [ ] Document availability semantics, artifact format, CLI usage, eligibility, and serving guards.
- [ ] Run Ruff, Ruff format, Mypy, Import Linter, Vulture advisory, full pytest/coverage, and CLI smoke.
- [ ] Update PR #24 summary and verification results.
