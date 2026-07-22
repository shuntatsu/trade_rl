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

The command publishes canonical `manifest.json` and `arrays.npz` files and prints one JSON result. Timestamps are exact bar-close boundaries, incomplete bars are excluded, kline quote-asset volume is stored as `quote_notional`, and every funding event is aggregated into its containing completed native bar with an explicit event count. Volume semantics are identity-bound per instrument: base-asset quantity is converted with price, contract quantity is converted with its contract multiplier and price, while quote-notional volume is already denominated in quote currency and must not be multiplied by price again.

`--transport` accepts:

- `rest`: official public REST only;
- `vision`: official Binance Vision ZIP archives only;
- `auto`: REST first, with Vision fallback for historical bars and funding.

Binance Vision does not publish complete exchange metadata. A general Vision-only dataset may therefore provide `--tick-size`, `--lot-size`, `--minimum-notional`, and `--listed-at` once per symbol; those static values participate in the dataset identity. The maintained multi-timeframe runner never silently upgrades static values into historical facts. It resolves one explicit execution-metadata mode before either repeated dataset build and binds the canonical evidence payload into `dataset_id`.

## Execution-metadata modes

The full runner accepts `--metadata-mode` and the equivalent `TRADE_RL_METADATA_MODE` environment variable. The modes are deliberately not interchangeable:

- `historical_signed` is the highest-integrity mode. It verifies an Ed25519 envelope using a read-only purpose-bound public-key store, requires explicit signed symbol order and complete effective-dated rule coverage for every selected symbol, retains the original signed document, and reports authenticated point-in-time evidence.
- `frozen_snapshot` is the maintained Docker default. It fetches the official USDⓈ-M `exchangeInfo` response exactly once, preserves the received bytes, source URI, aware UTC retrieval time and raw SHA-256, then applies those current rules statically across the research interval. It is explicitly unauthenticated and non-point-in-time.
- `conservative_static` requires an explicit versioned JSON payload supplied with `--conservative-static-path` or `TRADE_RL_CONSERVATIVE_STATIC_PATH`. It is a declared approximation and is never described as Binance historical evidence.

`frozen_snapshot` writes `exchange-info.raw.json` byte-for-byte and every mode writes canonical `exchange-info.json`. Mode, source, evidence digest, as-of time, coverage, authentication state, point-in-time state, policy version and limitations are repeated in the dataset result and final summary. Dataset A and B reuse the same in-memory resolution, so a live metadata change cannot occur between the reproducibility builds.

## Conservative closed-loop execution sensitivity

When the walk-forward configuration declares execution sensitivity, replay occurs only after model and seed-ensemble selection. The selected policy and the shadow baseline are rerun on the same sealed OOS folds with unchanged normalizers under nominal, each-rule 2x, joint 2x and joint 5x conditions. This is a stateful closed-loop replay: changed tick, lot, minimum-notional, spread, fee, impact, and participation rules change order admission, latency/eligibility outcomes, shared processing-bar capacity, partial fills, carried residual orders, positions, costs, and later observations. Returns are never adjusted after evaluation.

The immutable `execution-sensitivity.json` artifact is bound to the dataset identity, experiment-plan digest, scenario-pack digest and base sealed-test access evidence. Joint 2x must retain positive selected return, nonnegative uplift over baseline and maximum independently reset fold drawdown no greater than 20%. Joint 5x is report-only until calibrated. Promotion evidence must use the conservative OHLC path and a matching execution-policy digest; neutral and optimistic paths are sensitivity diagnostics only. Sensitivity never participates in recipe selection, and all results remain research-only with production status `NO-GO`.

## Run the verified end-to-end smoke

The repository contains a fixed smoke runner that performs:

1. the same dataset build twice and verifies identical dataset and artifact digests;
2. short CPU PPO training and atomic run publication;
3. one complete nested walk-forward fold and atomic evaluation publication.

```bash
uv run python examples/binance/run_e2e_smoke.py \
  --start-time 2026-06-01T00:00:00Z \
  --end-time 2026-06-29T00:00:00Z \
  --work-root var/binance-live-smoke
```

The smoke is a pipeline, data-integrity, and artifact-integrity check. It is not model-selection evidence, conservative execution-promotion evidence, profitability evidence, release approval, or a deployability benchmark. The maintained smoke intentionally uses only 64 PPO timesteps.

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

## Point-in-time execution-rule history

The strict full-research runner reads `TRADE_RL_BINANCE_RULE_HISTORY` and `TRADE_RL_METADATA_PUBLIC_KEYS`. The rule-history file uses schema `binance_instrument_rule_history_v4`; its signed payload binds USDⓈ-M market, explicit ordered `symbol_order`, exact coverage start/end, issue time, source URI, policy version, authoritative listing times, and every effective execution rule. The Ed25519 envelope is verified before semantic parsing. Unknown or expired keys, altered payloads, reordered symbols, future issue times, missing symbols, duplicate effective times, inconsistent final metadata, and uncovered research timestamps fail before dataset publication. Only public keys are mounted into the trainer. Live `exchangeInfo` is never projected backward as historical truth.

Funding alignment follows the same point-in-time rule: events are assigned to `[bar_open, bar_close]` completed native bars, all events in the interval are summed, and the count is bound into dataset identity v6. This prevents daily features from discarding the non-midnight funding events that occur inside a day.
