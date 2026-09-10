# Canonical M2 Study 001

Status: Pre-registered

## Purpose

This Study asks whether one symbol-ID-free universal strategy family can beat simple controls robustly across a fixed Binance USD-M development universe under one frozen protocol.

This record was written before bootstrap, baseline execution, or any candidate return/Sharpe inspection.

## Frozen source universe

Ordered symbols:

1. `BTCUSDT`
2. `ETHUSDT`
3. `BNBUSDT`
4. `XRPUSDT`
5. `ADAUSDT`

These are long-lived, actively tradable USD-M perpetual markets selected for a compact multi-asset development universe. Selection is not based on Strategy P&L.

Source-availability preflight used the maintained `plan_binance_vision_cache()` URL authority and checked only HTTP status/content length; it did not parse market values or returns.

- preflight workflow run: `34495043958`
- source range: `2021-01-01T00:00:00Z` to `2025-01-01T00:00:00Z`
- native clocks: `1h`, `4h`, `1d`
- planned Vision archives including funding: `960`
- available: `960`
- missing: `0`
- each symbol: `192/192`

## Time split

- source start: `2021-01-01T00:00:00Z`
- fit cutoff: `2023-01-01T00:00:00Z`
- development start: `2023-01-01T00:00:00Z`
- development stop: `2025-01-01T00:00:00Z`

The split gives two calendar years of fit history and two calendar years of development evidence while leaving data from 2025 onward outside this M2 development bootstrap for later explicitly separated evaluation.

## Baseline degrees of freedom

- base timeframe: `1h`
- feature timeframes: `4h`, `1d`
- rule signal: `1h__log_return_24bar`
- baseline feature set: compact price-return, volatility, volume, funding, RSI/MACD, and slower-clock return/volatility features listed exactly in `bootstrap.json`
- fit symbols: all five Study symbols
- PPO seeds: `0,1,2,3,4`
- PPO budget: `100000` timesteps per seed run
- gross budget: `0.5`
- initial capital: `100000`
- paired/bootstrap analysis count: `2000`
- bootstrap seed: `1729`
- Experiment budget: `12`

Initial thresholds are round-number preregistered values, not data-fitted values:

- rule entry/exit: `0.01 / 0.0025`
- forecast entry/exit: `0.0025 / 0.0005`

Allowed controlled factors are deliberately narrower than the implementation supports:

- `FEATURE_SET`
- `RULE_SIGNAL`
- `RULE_THRESHOLDS`
- `FORECAST_THRESHOLDS`
- `PPO_TRAINING_BUDGET`

`FIT_SYMBOL_SCOPE` and `GROSS_BUDGET` are not allowed to vary inside this Study.

## Research boundary

The bootstrap must stop with an immutable dataset and StudyPlan before baseline execution. Baseline execution is a separate explicit step. Controlled Experiments may begin only after baseline evidence exists. Final unused-future access remains outside the Controlled Experiment Loop.

No profitability, winner, or Production claim is implied by this preregistration.
