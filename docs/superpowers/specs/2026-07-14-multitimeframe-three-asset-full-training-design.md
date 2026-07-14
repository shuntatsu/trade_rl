# Multi-Timeframe Three-Asset Full Training Design

## Objective

Make multi-timeframe market context a first-class dataset and training contract, then run a non-smoke research training and nested walk-forward evaluation on BTCUSDT, ETHUSDT, and BNBUSDT using official Binance public data.

This work does not claim profitability and does not add authenticated account access or order routing. Production status remains `NO-GO`.

## Current gap

The current market builder has one `base_timeframe`, and every feature is calculated from that same bar series. Long lookbacks on 1-hour bars are useful, but they are not genuine multi-timeframe features. The Binance adapter likewise loads one interval per dataset.

A correct multi-timeframe implementation must calculate each feature on its native closed-bar clock and expose it to a base-timeframe decision only after that native bar is available. A 4-hour or daily bar must never appear at an earlier 1-hour decision timestamp.

## Chosen architecture

### Feature contract

`FeatureSpec` gains an optional `timeframe` field. `None` means the dataset base timeframe. The resolved timeframe is included in the canonical feature payload and therefore in dataset identity.

Feature names remain globally unique and carry explicit prefixes in maintained Binance presets, for example:

- `15m__log_return_1bar`
- `15m__realized_volatility_4bar`
- `1h__log_return_1bar`
- `1h__log_return_1d`
- `4h__log_return_1bar`
- `4h__realized_volatility_6bar`
- `1d__log_return_1bar`
- `1d__log_return_7bar`
- `1h__funding_bps`

### Source contract

The existing `MarketDataSource.load(symbol)` remains the base-timeframe path. A new runtime-checkable `MultiTimeframeMarketDataSource` protocol adds `load_timeframe(symbol, timeframe)`.

The generic builder uses the protocol only when at least one feature resolves to a non-base timeframe. Single-timeframe sources and datasets continue to work unchanged.

### Native calculation and causal as-of alignment

For each symbol and each requested timeframe:

1. Load the native `RawMarketSeries`.
2. Calculate feature events on the native clock with the existing causal feature implementations.
3. Apply native carry and availability rules.
4. As-of align to each base timestamp using the latest native observation whose `available_at` is less than or equal to the base decision timestamp.
5. Recompute feature age and normalized staleness on the base clock.
6. Mark unavailable values explicitly rather than filling future or expired values.

No backward fill is allowed. Exact-close equality is allowed because the environment acts no earlier than the next base bar open.

The base OHLCV, funding, tradability, execution constraints, and accounting arrays remain on the base timeframe. Auxiliary timeframes affect observations only.

### Binance adapter

`BinanceMarketDataSource` supports `load_timeframe`. It caches kline and funding responses by symbol and timeframe to avoid duplicate downloads.

The dataset builder API accepts auxiliary feature timeframes. The CLI adds repeatable `--feature-timeframe` options while `--interval` remains the base timeframe.

Official Binance Vision monthly kline archives are used for complete historical months, with daily archives used for partial months or bounded fallback. This keeps a multi-year, four-timeframe, three-symbol run practical while preserving deterministic source URLs and fixed-range filtering.

Supported maintained research preset:

- base: `1h`
- auxiliary: `15m`, `4h`, `1d`
- symbols: `BTCUSDT`, `ETHUSDT`, `BNBUSDT`
- market: USDⓈ-M linear futures

COIN-M remains fail-closed because inverse PnL is not representable by the current linear book.

## Dataset identity and reproducibility

Dataset identity includes:

- resolved feature timeframe for every feature;
- ordered requested timeframe set;
- native feature lookbacks and staleness limits;
- three-symbol instrument metadata;
- all existing execution, availability, and accounting arrays.

A fixed-range build is performed twice into independent directories. Dataset IDs and artifact digests must match exactly before training begins.

## Full research run

The maintained full-run configuration uses:

- BTCUSDT, ETHUSDT, BNBUSDT;
- fixed closed range `2025-01-01T00:00:00Z` through `2026-06-29T00:00:00Z`;
- 1-hour decisions;
- native 15-minute, 1-hour, 4-hour, and 1-day features;
- PPO with three independent seeds `0`, `1`, and `2`;
- 131,072 requested timesteps per seed;
- 2,048-step rollouts, 64 batch size, 10 epochs;
- shared per-asset encoder with 128x128 policy network;
- realistic fees, spread, impact, participation, funding, and portfolio concentration limits;
- checkpointing and atomic artifact publication.

The nested walk-forward run uses two chronological folds with purge gaps. Candidate selection uses only checkpoint and selection ranges; outer test ranges remain sealed until selection. Fold-local candidate training uses 32,768 timesteps per seed so the evaluation is materially larger than smoke while remaining bounded on a standard CPU runner.

If runtime constraints make the configured run exceed the bounded CI job, the failure must be explicit. The implementation may reduce fold count from two to one before reducing per-seed full-run training, but it must not silently convert the full run back into the 64-step smoke.

## Failure handling

The pipeline fails before publication on:

- unsupported or duplicate timeframes;
- a source that lacks multi-timeframe loading when required;
- native bars that are irregular, duplicated, or incomplete;
- auxiliary data that cannot causally as-of align;
- unavailable static instrument metadata;
- inconsistent dataset identities across repeated builds;
- non-finite observations or training outputs;
- incomplete training or walk-forward artifact trees.

Temporary and failed runs remain isolated under failed artifact directories and never replace `latest.json`.

## Verification

### Unit and contract tests

- timeframe validation and identity changes;
- single-timeframe backward compatibility;
- 15-minute latest-closed-bar alignment at hourly decisions;
- 4-hour and daily bars unavailable before close;
- `available_at` delays prevent leakage;
- staleness expiry on the base clock;
- Binance monthly URL and partial-month fallback;
- cache behavior and multi-symbol cardinality;
- CLI parsing and machine-readable output.

### Repository verification

- Ruff and formatting;
- Mypy;
- import architecture;
- full tests and branch coverage;
- critical branch coverage;
- Ubuntu and Windows compatibility.

### Live evidence

- fixed official Binance Vision range downloaded for all three symbols and four timeframes;
- independent repeated dataset build has identical identity;
- full three-seed PPO run publishes all member policies and checkpoints;
- nested walk-forward publishes fold evidence and aggregate report;
- results are reported honestly, including baseline selection or negative returns.

## Non-goals

- authenticated Binance account access;
- live or paper order routing;
- inverse COIN-M accounting;
- profitability guarantees;
- tick-level order-book modelling;
- point-in-time historical exchange-filter reconstruction when Binance does not publish such snapshots.
