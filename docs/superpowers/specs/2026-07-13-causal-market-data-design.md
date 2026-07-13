# Causal Market Data and Observation Design

## Status

Approved for implementation from the 2026-07-13 input-data audit. This design replaces implicit, externally constructed feature tensors with one reproducible in-repository path.

## Goals

1. Remove future knowledge from the policy observation. In particular, a decision at bar `t` must not receive `tradable[t + 1]` or any value derived from bars after `t`.
2. Build `MarketDataset` from real timestamped market records through a maintained `trade_rl.data` builder.
3. Represent feature availability and staleness per feature and per symbol.
4. Bind feature-generation configuration, ordered symbols, instrument contracts, normalization configuration, normalization output and resolved arrays into `dataset_id`.
5. Use one `ObservationBuilder` in training and serving.
6. Enforce prefix invariance and observation causality with regression tests.
7. Represent a point-in-time universe with listing, delisting and bar-level trading status.
8. Make volume semantics explicit with a unit and base-asset contract multiplier for every symbol.

## Non-goals

- Reintroducing the removed `mars_lite` package.
- Recreating the former 92-feature library.
- Adding a production database dependency to the core package.
- Claiming production readiness or profitability.

## Data source boundary

`MarketDataSource` returns immutable `RawMarketSeries` values. A maintained `CsvMarketDataSource` reads one CSV per symbol using only the standard library and NumPy. The CSV schema is:

```text
timestamp,open,high,low,close,volume[,funding_rate][,tradable]
```

Timestamps are UTC ISO-8601 values or integer Unix milliseconds. `tradable` defaults to true for present rows. Missing rows remain missing; they are not removed from the shared market clock.

## Point-in-time instruments

Each symbol has an `InstrumentContract` containing:

- symbol;
- listing timestamp;
- optional delisting timestamp;
- volume unit: `base_asset`, `quote_notional` or `contracts`;
- positive contract multiplier, interpreted as base-asset units per contract.

The builder constructs a regular union clock. `symbol_active[t, s]` is true only inside the instrument lifetime. `tradable[t, s]` is true only when the instrument is active, the source row exists and the row is marked tradable. Missing, pre-listing and post-delisting prices are represented by finite sentinel or last-known values but are always masked from features, trend targets and execution.

## Feature pipeline

The first maintained feature set is intentionally small and causal:

- one-bar log return;
- configurable multi-bar log return;
- trailing realized volatility;
- trailing volume z-score;
- funding rate in basis points.

Every feature is declared by `FeatureSpec`, which contains its kind, lookback, normalization mode, normalization window and maximum accepted staleness in hours. Rolling calculations use only rows at or before `t`. Warm-up values are unavailable, not silently treated as genuine zero observations.

`feature_available[t, s, f]` records whether the value is valid. `feature_staleness[t, s, f]` is a normalized value in `[0, 1]`, where zero is fresh and one means the feature has reached or exceeded its configured maximum staleness. Unavailable values have availability false and staleness one.

Global features are causal aggregates over active and available symbols:

- active-symbol fraction;
- tradable-symbol fraction;
- mean one-bar market return;
- cross-sectional return dispersion.

## Dataset identity

`dataset_id` is SHA-256 over canonical metadata plus typed array payloads. It includes:

- ordered symbols;
- instrument contracts;
- base timeframe and resolved cadence;
- complete feature specifications;
- normalization configuration and normalization-state digest;
- feature and global-feature names;
- timestamps;
- OHLCV and funding arrays;
- active and tradable masks;
- feature values, availability and staleness.

Changing symbol order, a lookback, a normalization parameter, a contract multiplier, volume unit or any resolved input value must change the identity.

## Observation contract

One `ObservationBuilder` owns the exact flattened policy input. Per symbol it emits:

1. feature values;
2. per-feature availability flags;
3. per-feature normalized staleness;
4. current-bar tradable flag;
5. current-bar active flag;
6. fast, base and slow trend targets;
7. alpha target;
8. hybrid weight;
9. shadow weight;
10. hybrid-minus-shadow weight.

It never reads the next row. Global state retains the current global features, relative book state, drawdowns, exposures and risk scales. Environment and serving both call this same builder. The raw-vector serving method remains for compatibility, while the structured serving path is the maintained path.

## Trend and execution behavior

Trend targets are computed only for symbols active at both ends of the lookback. Ineligible symbols receive zero target weight and are excluded from cross-sectional centering.

Execution may use `tradable[t + 1]` internally because it simulates what actually happened at the execution bar, but that value is never placed in the decision observation. Market capacity is calculated from explicit volume semantics:

- `base_asset`: `volume * price`;
- `quote_notional`: `volume`;
- `contracts`: `volume * contract_multiplier * price`.

## Testing

The implementation must include tests proving:

- extending a source with future rows does not change any previously built feature, availability, staleness or global feature;
- mutating rows after `t` does not change the observation at `t`;
- changing only `tradable[t + 1]` does not change the observation at `t`;
- listing and delisting masks prevent pre-listing or post-delisting trend exposure;
- dataset identity changes for symbol-order, feature-config, normalization and contract changes;
- all three volume units resolve to the expected market notional;
- environment and serving produce byte-identical observations from the same structured input.

## Compatibility and migration

This is a schema change. `MarketDataset` and `DatasetManifest` advance to v3 contracts. Existing test fixtures are updated explicitly rather than hiding new fields behind permissive defaults. Production status remains `NO-GO` until fresh real-data evaluation is completed.
