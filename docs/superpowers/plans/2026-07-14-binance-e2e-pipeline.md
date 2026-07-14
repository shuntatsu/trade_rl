# Binance End-to-End Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible public-Binance-to-dataset-to-training-to-walk-forward pipeline while failing closed for unsupported inverse COIN-M accounting.

**Architecture:** Add an exchange-specific adapter in `trade_rl.integrations`, keep `MarketDatasetBuilder` exchange-agnostic, and expose one authoritative `trade-rl data binance` command. Use deterministic parser tests plus a fixed historical Binance Vision live smoke.

**Tech Stack:** Python 3.12 standard library networking/ZIP/CSV, NumPy, argparse, pytest, Stable-Baselines3, GitHub Actions.

## Global Constraints

- No authenticated Binance endpoint and no order placement.
- Runtime dependencies remain NumPy and Gymnasium only; do not add requests or pandas.
- Spot and USDⓈ-M are supported linear products.
- COIN-M inverse products fail before artifact publication.
- Timestamps are exact bar-close boundaries and incomplete bars are excluded.
- Fixed start/end inputs must produce deterministic dataset identities.
- Direct exchange routing remains `NO-GO`.

---

### Task 1: Broadcast static execution metadata

**Files:**
- Modify: `trade_rl/data/contracts.py`
- Modify: `trade_rl/data/builder.py`
- Test: `tests/data/test_binance_contract_metadata.py`

**Interfaces:**
- Consumes: existing `InstrumentContract` and `MarketDatasetBuilder.build`.
- Produces: `InstrumentContract.tick_size`, `lot_size`, and `minimum_notional`; dataset arrays with the same names.

- [ ] **Step 1: Write failing tests**

```python
def test_builder_broadcasts_static_execution_metadata() -> None:
    contract = InstrumentContract(
        symbol="BTCUSDT",
        tick_size=0.1,
        lot_size=0.001,
        minimum_notional=5.0,
        volume_unit=VolumeUnit.QUOTE_NOTIONAL,
    )
    dataset = MarketDatasetBuilder(config).build(source, (contract,))
    np.testing.assert_allclose(dataset.resolved_array("tick_size"), 0.1)
    np.testing.assert_allclose(dataset.resolved_array("lot_size"), 0.001)
    np.testing.assert_allclose(dataset.resolved_array("minimum_notional"), 5.0)
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest -q tests/data/test_binance_contract_metadata.py`
Expected: failure because `InstrumentContract` does not accept the metadata fields.

- [ ] **Step 3: Implement validation and broadcasting**

Add three float fields defaulting to `0.0`, reject non-finite or negative values, include them in `canonical_payload`, and pass constant `(bars, symbols)` arrays into `MarketDataset` from the builder.

- [ ] **Step 4: Verify GREEN**

Run: `uv run pytest -q tests/data/test_binance_contract_metadata.py tests/data/test_market_builder.py`
Expected: all pass.

### Task 2: Implement Binance public adapter

**Files:**
- Create: `trade_rl/integrations/binance.py`
- Modify: `trade_rl/integrations/__init__.py`
- Test: `tests/integrations/test_binance.py`

**Interfaces:**
- Produces: `BinanceMarket`, `BinanceTransportMode`, `BinancePublicTransport`, `BinanceMarketDataSource`, `BinanceBuildRequest`, `build_binance_dataset`.
- `BinanceMarketDataSource.load(symbol) -> RawMarketSeries`.

- [ ] **Step 1: Write parser and contract tests**

Tests must demonstrate:

```python
assert series.timestamps[0] == np.datetime64("2026-06-01T01:00:00", "ns")
assert series.volume[0] == 6_250_000.0  # quote-asset volume, kline index 7
assert not series.funding_available[0]
assert series.funding_available[7]
with pytest.raises(BinanceUnsupportedContractError, match="inverse"):
    build_binance_dataset(request_for_coin_m)
```

Also test malformed rows, duplicate timestamps, missing Vision days, retry limits, and REST-to-Vision fallback.

- [ ] **Step 2: Verify RED**

Run: `uv run pytest -q tests/integrations/test_binance.py`
Expected: import failure because the integration module does not exist.

- [ ] **Step 3: Implement transport and parsing**

Use `urllib.request`, `urllib.parse`, `zipfile`, `csv`, and bounded exponential backoff. Parse only closed bars. Convert each open timestamp to `open + interval`. Use kline quote volume at index 7. Parse funding CSV variants containing either `calc_time`/`last_funding_rate` or `fundingTime`/`fundingRate`.

- [ ] **Step 4: Implement deterministic request validation**

Require timezone-aware start/end values, supported intervals, unique symbols, exact metadata cardinality, and at least three bars. Reject COIN-M before network publication.

- [ ] **Step 5: Verify GREEN**

Run: `uv run pytest -q tests/integrations/test_binance.py`
Expected: all pass.

### Task 3: Add authoritative CLI command

**Files:**
- Modify: `trade_rl/cli/app.py`
- Test: `tests/cli/test_binance_data_command.py`

**Interfaces:**
- Produces: `trade-rl data binance`.

- [ ] **Step 1: Write failing CLI tests**

```python
exit_code = main([
    "data", "binance", "--market", "usds-m", "--symbol", "BTCUSDT",
    "--interval", "1h", "--start-time", "2026-06-01T00:00:00Z",
    "--end-time", "2026-06-08T00:00:00Z", "--transport", "vision",
    "--tick-size", "0.1", "--lot-size", "0.001",
    "--minimum-notional", "5", "--output", str(output),
], stdout=stdout)
assert exit_code == 0
assert json.loads(stdout.getvalue())["schema"] == "binance_dataset_build_result_v1"
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest -q tests/cli/test_binance_data_command.py`
Expected: argparse rejects the unknown `binance` subcommand.

- [ ] **Step 3: Implement command parsing and JSON output**

Support repeated `--symbol`, `--tick-size`, `--lot-size`, `--minimum-notional`, and `--listed-at`. Emit market, interval, start/end, symbols, bars, dataset ID, artifact digest, transport, and `production_status: NO-GO`.

- [ ] **Step 4: Verify GREEN**

Run: `uv run pytest -q tests/cli/test_binance_data_command.py tests/cli`
Expected: all pass.

### Task 4: Add reproducible Binance E2E fixtures

**Files:**
- Create: `examples/binance/training-smoke.json`
- Create: `examples/binance/walk-forward-smoke.json`
- Create: `scripts/run_binance_e2e_smoke.py`
- Test: `tests/examples/test_binance_smoke_assets.py`

**Interfaces:**
- The script runs the public CLI three times and validates the resulting manifests.

- [ ] **Step 1: Write failing asset validation tests**

Verify both JSON configs load, reward baseline underperformance is disabled for the short smoke, the training run uses CPU PPO with one seed, and the walk-forward workflow resolves at least one complete fold for 672 hourly bars.

- [ ] **Step 2: Verify RED**

Run: `uv run pytest -q tests/examples/test_binance_smoke_assets.py`
Expected: missing files.

- [ ] **Step 3: Implement the smoke runner**

The script must execute:

```text
trade-rl data binance -> load dataset artifact -> trade-rl train run -> trade-rl walk-forward run
```

It must fail on any non-zero exit, verify artifact files and digests, rerun the dataset build in a second directory, and assert the two dataset IDs match.

- [ ] **Step 4: Verify GREEN with deterministic fakes**

Run: `uv run pytest -q tests/examples/test_binance_smoke_assets.py`
Expected: all pass.

### Task 5: Run live Binance Vision E2E

**Files:**
- Create temporarily: `.github/workflows/binance-e2e-development.yml`
- Retain: `.github/workflows/binance-live-smoke.yml`

- [ ] **Step 1: Run fixed-range live smoke**

Run in GitHub Actions:

```bash
uv sync --extra dev --extra train-sb3
uv run python scripts/run_binance_e2e_smoke.py \
  --start-time 2026-06-01T00:00:00Z \
  --end-time 2026-06-29T00:00:00Z \
  --work-root var/binance-live-smoke
```

Expected: dataset publication twice with identical IDs, PPO run published, one walk-forward run published.

- [ ] **Step 2: Diagnose failures systematically**

For each failure, capture the requested URL, HTTP status, archive member names, parsed row counts, resolved time range, CLI stderr JSON, and failing artifact path before changing code.

- [ ] **Step 3: Retain non-required scheduled smoke**

Keep a weekly/manual workflow using the fixed historical range. Do not make external-network availability a required PR check.

### Task 6: Full verification and publication

**Files:**
- Modify: `START.md`
- Modify: `README.md`
- Remove: temporary development scripts/workflows.

- [ ] **Step 1: Run all verification gates**

```bash
uv run ruff check trade_rl tests scripts
uv run ruff format --check trade_rl tests scripts
uv run mypy trade_rl
uv run lint-imports
uv run pytest --cov=trade_rl --cov-branch --cov-report=term-missing
uv run python scripts/check_critical_coverage.py coverage.json
uv run trade-rl --version
```

Expected: zero failures and coverage floors satisfied.

- [ ] **Step 2: Run compatibility jobs**

Expected: Ubuntu and Windows data/simulation/evaluation/serving suites pass.

- [ ] **Step 3: Open a PR and merge only after fresh final-head evidence**

The PR body must distinguish deterministic tests, live Binance Vision evidence, supported linear products, rejected inverse products, and the continuing `NO-GO` production status.
