# Binance End-to-End Pipeline Design

## Status

Approved by the user through the request to run the Binance path through dataset construction, training, and walk-forward evaluation, fixing any discovered problems before completion.

## Goal

Provide one maintained, reproducible path from public Binance market data to a validated `MarketDataset` artifact, a short Stable-Baselines3 training run, and a nested walk-forward run.

## Supported market models

The first maintained release supports:

- Binance Spot, represented as a linear asset position.
- Binance USDⓈ-M futures, represented as a linear base-quantity position.

Binance COIN-M inverse futures are rejected before dataset publication. Their contract value and PnL use inverse-price accounting, while the current `BookState` is linear (`quantity * price * multiplier`). Treating `contractSize` as the existing multiplier would materially misstate value and PnL. The adapter therefore fails closed instead of publishing a misleading dataset.

## Architecture

`trade_rl.integrations.binance` owns all exchange-specific transport and parsing. It depends only on the Python standard library, NumPy, and the lower-level data contracts. The data builder remains exchange-agnostic.

The adapter has three boundaries:

1. `BinancePublicTransport` retrieves JSON or ZIP bytes with bounded retries, explicit timeouts, and a stable user agent.
2. `BinanceMarketDataSource` converts closed Binance klines and funding events into `RawMarketSeries` values.
3. `build_binance_dataset` resolves instrument metadata, constructs `InstrumentContract` values, invokes `MarketDatasetBuilder`, and publishes through the canonical data artifact writer.

The authoritative CLI adds `trade-rl data binance` and emits one JSON result containing the dataset identity, artifact digest, market, symbols, bar range, and transport used.

## Deterministic time contract

Binance kline open timestamps are converted to exact bar-close boundaries using `open_time + interval_duration`. The incomplete current bar is never included. A supplied `--end-time` is rounded down to the last fully closed interval.

For reproducible research, callers can provide explicit `--start-time` and `--end-time`. The same payload and configuration must produce the same dataset identity.

## Transport strategy

`rest` uses the official public REST endpoints. `vision` uses official Binance Vision ZIP archives. `auto` tries REST and falls back to Vision only for historical bar and funding retrieval when REST is unavailable or geographically blocked.

Vision-only runs require explicit static execution metadata when exchange information cannot be queried: tick size, lot size, minimum notional, and listing time. These values are incorporated into the dataset identity.

## Volume and funding semantics

Spot and USDⓈ-M klines use Binance quote-asset volume and publish `VolumeUnit.QUOTE_NOTIONAL`. This avoids converting volume with the wrong price and gives portfolio-liquidity constraints a direct market-notional series.

Funding is represented as a sparse event series. Bars without a funding event have `funding_rate=0` and `funding_available=False`. Funding values are not forward-filled in raw data; the causal feature builder controls staleness and carry behavior.

## Execution metadata

`InstrumentContract` gains optional, non-negative `tick_size`, `lot_size`, and `minimum_notional` fields. `MarketDatasetBuilder` broadcasts them across bars into the existing dataset arrays. Defaults remain zero for backward compatibility.

## CLI

The maintained command is:

```bash
trade-rl data binance \
  --market usds-m \
  --symbol BTCUSDT \
  --interval 1h \
  --start-time 2026-06-01T00:00:00Z \
  --end-time 2026-06-30T00:00:00Z \
  --transport vision \
  --tick-size 0.1 \
  --lot-size 0.001 \
  --minimum-notional 5 \
  --output artifacts/datasets/binance-btcusdt
```

Multiple `--symbol` values are allowed. Static metadata options may be repeated in symbol order or obtained from REST exchange information.

## Error handling

- HTTP 404 from a required Vision day is a hard error; silently skipping a missing day would create an irregular or incomplete research range.
- Duplicate, unordered, malformed, or incomplete kline rows are rejected.
- Funding records outside the requested range are discarded.
- Symbol status, metadata shape, interval, and market mismatches fail before artifact publication.
- COIN-M requests raise a specific unsupported inverse-contract error.
- Partial downloads and staging artifacts never become published datasets.

## Testing

Deterministic unit tests cover URL construction, timestamp conversion, quote-volume selection, funding alignment, metadata parsing, retry/fallback behavior, CLI parsing, static execution metadata broadcasting, and COIN-M rejection.

A live Binance smoke workflow downloads a fixed historical USDⓈ-M BTCUSDT range from Binance Vision, builds and reloads the dataset artifact, performs a short PPO training run, and executes one walk-forward fold. The workflow records CLI JSON and artifact manifests as a downloadable Actions artifact.

## Acceptance criteria

- `trade-rl data binance` produces a reloadable, content-addressed dataset from fixed Binance public data.
- Repeating the same fixed-range command produces the same dataset ID.
- Dataset timestamps are exact, regular close boundaries and contain no incomplete bar.
- USDⓈ-M volume is quote notional and funding events remain causal.
- Tick size, lot size, and minimum notional reach the final dataset arrays.
- COIN-M publication is rejected with an explanation of the inverse-accounting limitation.
- Short training and one complete walk-forward fold finish using the generated dataset.
- Ruff, formatting, Mypy, Import Linter, full branch-coverage tests, critical coverage, CLI smoke, Ubuntu compatibility, and Windows compatibility pass.
- Direct exchange order routing remains `NO-GO`.
