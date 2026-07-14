# Multi-Timeframe Three-Asset Full Training Implementation Plan

> Execute this plan with TDD. Do not add production behavior before its focused test has failed for the expected reason.

**Goal:** Make native multi-timeframe features part of the authoritative dataset/training path and complete a reproducible three-asset Binance PPO and nested walk-forward research run.

**Architecture:** Feature specifications carry their native timeframe. The generic market builder computes each feature on its native closed-bar series and causally as-of aligns it onto the 1-hour decision clock. Binance supplies cached native series through REST or monthly/daily Vision archives. Base OHLCV and execution arrays remain 1-hour; auxiliary timeframes affect observations only.

**Stack:** Python 3.12, NumPy, Stable-Baselines3 PPO, existing Trade RL artifact and walk-forward workflows, GitHub Actions for isolated Linux execution.

---

## Task 1: Lock multi-timeframe contracts in tests

**Files:**
- Modify: `tests/data/test_market_contracts.py`
- Create: `tests/data/test_multitimeframe_builder.py`
- Modify: `tests/data/test_information_availability.py`

**Steps:**
1. Add a failing contract test showing `FeatureSpec(timeframe="4h")` serializes the timeframe and changes canonical identity.
2. Add validation tests for empty, unsupported, and malformed timeframes.
3. Add an in-memory multi-timeframe source used by real builder tests.
4. Add a failing test showing a 4-hour close is unavailable to earlier hourly decisions and appears exactly at its close.
5. Add a failing test showing a delayed native `available_at` prevents visibility until the delay has elapsed.
6. Add a failing staleness-expiry test measured on the base clock.
7. Run only these tests and confirm failure is due to missing timeframe contract/protocol/alignment behavior.

## Task 2: Implement the generic causal multi-timeframe builder

**Files:**
- Modify: `trade_rl/data/contracts.py`
- Modify: `trade_rl/data/source.py`
- Modify: `trade_rl/data/builder.py`
- Modify: `trade_rl/data/__init__.py` if exports require it

**Steps:**
1. Add validated optional `FeatureSpec.timeframe` and a `resolved_timeframe(base_timeframe)` helper.
2. Include resolved/native timeframe data in canonical configuration payloads.
3. Add runtime-checkable `MultiTimeframeMarketDataSource` with `load_timeframe`.
4. Refactor native feature calculation to retain event timestamps and maximum window availability.
5. Add causal as-of alignment from native events to base timestamps.
6. Recompute age, staleness, and availability on the base clock with no backward fill.
7. Preserve the existing single-timeframe result byte-for-byte where no auxiliary timeframe is requested.
8. Run focused tests until green, then run all `tests/data`.

## Task 3: Lock Binance multi-timeframe and monthly archive behavior in tests

**Files:**
- Modify: `tests/integrations/test_binance.py`
- Modify: `tests/cli/test_binance_data_command.py`
- Create: `tests/integrations/test_binance_multitimeframe.py`

**Steps:**
1. Add failing tests for official Spot and USDⓈ-M monthly kline URLs.
2. Add a failing test that complete calendar months use monthly archives and partial months use daily archives.
3. Add a failing cache test proving one `(symbol, timeframe)` load does not redownload.
4. Add a failing test for the maintained 15m/1h/4h/1d feature preset and feature names.
5. Add CLI tests for repeatable `--feature-timeframe` and machine-readable result fields.
6. Add a failure test for duplicate/base-only invalid timeframe requests and COIN-M.
7. Run focused tests and confirm the expected RED failures.

## Task 4: Implement Binance multi-timeframe ingestion

**Files:**
- Modify: `trade_rl/integrations/binance.py`
- Modify: `trade_rl/cli/app.py`

**Steps:**
1. Add monthly Vision URL construction and bounded monthly/daily range planning.
2. Add `load_timeframe`, per-symbol/timeframe cache, and shared funding cache.
3. Add the maintained multi-timeframe feature preset.
4. Extend `build_binance_market_dataset` with ordered feature timeframes.
5. Extend `trade-rl data binance` with repeatable `--feature-timeframe` and report the resolved set.
6. Ensure every source URL and transport used is recorded deterministically.
7. Run focused Binance and CLI tests, Ruff, and Mypy.

## Task 5: Add reproducible full-run assets

**Files:**
- Create: `examples/binance-multitimeframe/training-full.json`
- Create: `examples/binance-multitimeframe/walk-forward-full.json`
- Create: `examples/binance-multitimeframe/run_full_research.py`
- Create: `tests/examples/test_binance_multitimeframe_full_assets.py`
- Modify: `docs/BINANCE.md`
- Modify: `README.md`

**Steps:**
1. Add failing tests that the full training config has three seeds and at least 131,072 timesteps per seed.
2. Add failing tests that walk-forward has two folds and 32,768 candidate timesteps per seed.
3. Add a runner that obtains or loads an exchange-info metadata snapshot, builds the fixed dataset twice, verifies identity, runs full training, runs walk-forward, validates artifacts, and writes `summary.json`.
4. Fix range to `2024-12-01T00:00:00Z` through `2026-06-01T00:00:00Z` and symbols to BTCUSDT, ETHUSDT, BNBUSDT.
5. Persist the exact current tick, lot, minimum-notional, listing-time, and exchange-info snapshot used.
6. Add documentation with explicit `NO-GO` and no profitability claim.
7. Run asset tests, Ruff, format, and Mypy.

## Task 6: Verify RED then commit production implementation

**Files:**
- Temporary: `.github/workflows/multitimeframe-development.yml`

**Steps:**
1. Create a temporary focused workflow on the feature branch.
2. Run the new contract/builder/Binance/CLI/asset tests before production implementation and capture the expected failures.
3. Add production implementation in the smallest TDD increments.
4. Re-run focused tests after every increment.
5. Run Ruff, format check, Mypy, import-linter, and all data/integration/CLI tests.
6. Delete diagnostic-only patch scripts or markers immediately after use.

## Task 7: Build the live three-asset multi-timeframe dataset twice

**Files:**
- Temporary workflow: `.github/workflows/multitimeframe-live-run.yml`

**Steps:**
1. Install the maintained training dependencies on an isolated Ubuntu runner.
2. Fetch official Binance public data for all three symbols and four timeframes.
3. Build two independent dataset artifacts from the exact same fixed range and metadata snapshot.
4. Fail if either dataset ID or artifact digest differs.
5. Validate expected hourly bar count, symbol order, feature names, finite values, availability masks, and artifact closure.
6. Upload the metadata snapshot, manifests, and build summary as workflow evidence.

## Task 8: Run full three-seed PPO training

**Steps:**
1. Run `trade-rl train run` using `training-full.json` and the verified dataset.
2. Require all three member policies, checkpoints, ensemble manifest, environment manifest, run manifest, and latest pointer.
3. Record wall-clock details, selected checkpoints, terminal metrics, and any warnings.
4. Fail on non-finite outputs, incomplete artifacts, or silent timestep reduction.
5. If the bounded runner fails solely from runtime limits, preserve evidence and reduce walk-forward fold count before changing the full training contract.

## Task 9: Run nested walk-forward evaluation

**Steps:**
1. Run the configured two-fold nested walk-forward using the same dataset identity.
2. Require sealed-test evidence and complete fold artifacts.
3. Record candidate selection, baseline comparison, per-fold return, drawdown, turnover, costs, and aggregate statistics.
4. Report negative results or baseline selection without reinterpretation.
5. If runtime limits block two folds, rerun one fold and explicitly record the deviation; do not reduce full-run training timesteps silently.

## Task 10: Fix live-run issues systematically

**Steps:**
1. For each failure, capture full logs and identify the exact failing component boundary.
2. Add the smallest failing regression test.
3. Fix the root cause in production code, not in the test or workflow wrapper.
4. Re-run the focused test, repository verification, and the interrupted live stage.
5. Stop after three failed fix attempts on the same issue and reconsider the architecture before continuing.

## Task 11: Final verification and integration

**Files:**
- Keep: a manual/weekly non-required multi-timeframe live workflow if stable
- Delete: all diagnostic-only workflows and generated patch scripts

**Steps:**
1. Run the complete repository CI: Ruff, format, Mypy, import architecture, dead-code, full tests/coverage, critical coverage, CLI smoke, Ubuntu, Windows.
2. Re-run the final official Binance data → repeated dataset → full PPO → walk-forward path from only committed files.
3. Verify the PR diff contains only production code, tests, maintained examples/workflow, design, plan, and documentation.
4. Update the PR body with exact dataset IDs, artifact digests, training evidence, walk-forward metrics, deviations, and `NO-GO` status.
5. Mark ready only after clean CI and live evidence.
6. Merge using the repository-supported method and verify the resulting `main` commit.
