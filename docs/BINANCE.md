# Binance Public Data Workflow

Trade RL supports deterministic research-dataset construction from Binance public market data.

> Production status: **NO-GO**. This integration does not authenticate, read an account, place orders, or authorize live capital deployment.

## Supported markets

- Binance Spot: linear asset quantities.
- Binance USDⓈ-M futures: linear base-asset quantities, public funding events, and quote-notional volume.
- Binance COIN-M futures: intentionally rejected. COIN-M uses inverse contract value and inverse PnL, while the current `BookState` uses linear `quantity × price × multiplier` accounting. Publishing a COIN-M dataset into the linear simulator would produce misleading PnL and risk calculations.

## Build a fixed-range dataset

Install the maintained training dependencies:

```bash
uv sync --extra dev --extra train-sb3
```

Build one reproducible BTCUSDT USDⓈ-M hourly dataset from official Binance Vision archives:

```bash
uv run trade-rl data binance \
  --market usds-m \
  --symbol BTCUSDT \
  --interval 1h \
  --start-time 2026-06-01T00:00:00Z \
  --end-time 2026-06-29T00:00:00Z \
  --transport vision \
  --tick-size 0.1 \
  --lot-size 0.001 \
  --minimum-notional 5 \
  --listed-at 2019-09-08T00:00:00Z \
  --output var/binance/dataset
```

The command publishes canonical `manifest.json` and `arrays.npz` files and prints one JSON result. Timestamps are exact bar-close boundaries, incomplete bars are excluded, kline quote-asset volume is stored as `quote_notional`, and funding remains a sparse event series.

`--transport` accepts:

- `rest`: official public REST only;
- `vision`: official Binance Vision ZIP archives only;
- `auto`: REST first, with Vision fallback for historical bars and funding.

Binance Vision does not publish complete exchange metadata. A Vision-only run must therefore provide `--tick-size`, `--lot-size`, `--minimum-notional`, and `--listed-at` once per symbol. These values participate in the dataset identity.

## Run the verified end-to-end smoke

The repository contains a fixed smoke runner that performs:

1. the same dataset build twice and verifies identical dataset and artifact digests;
2. short CPU PPO training and atomic run publication;
3. one complete nested walk-forward fold and atomic evaluation publication.

```bash
uv run python scripts/run_binance_e2e_smoke.py \
  --start-time 2026-06-01T00:00:00Z \
  --end-time 2026-06-29T00:00:00Z \
  --work-root var/binance-live-smoke
```

The smoke is a pipeline and integrity check, not a profitability benchmark. The maintained smoke intentionally uses only 64 PPO timesteps, so its selected policy and return must not be used to judge deployability.

## Multi-symbol usage

Repeat symbol and static-metadata options in the same order:

```bash
uv run trade-rl data binance \
  --market usds-m \
  --symbol BTCUSDT --symbol ETHUSDT \
  --tick-size 0.1 --tick-size 0.01 \
  --lot-size 0.001 --lot-size 0.001 \
  --minimum-notional 5 --minimum-notional 5 \
  --listed-at 2019-09-08T00:00:00Z \
  --listed-at 2019-11-27T00:00:00Z \
  --interval 1h \
  --start-time 2026-06-01T00:00:00Z \
  --end-time 2026-06-29T00:00:00Z \
  --transport vision \
  --output var/binance/btc-eth
```

Metadata cardinality, duplicate symbols, irregular timestamps, missing daily archives, malformed rows, unsupported intervals, and incomplete ranges fail before artifact publication.

## Live-smoke automation

`.github/workflows/binance-live-smoke.yml` is manual and weekly. It uses the fixed historical range so upstream data changes, parser regressions, and training-path regressions are detectable without making current market state part of the test identity. External-network availability is not a required pull-request check.
