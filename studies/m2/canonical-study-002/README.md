# Canonical M2 Study 002

Status: Pre-registered replacement lineage

## Purpose

This Study repeats the Canonical M2 Study 001 research question under an explicit, identity-bound execution-economics contract. It asks whether one symbol-ID-free universal strategy family can beat simple controls robustly across five long-lived Binance USD-M perpetual markets under one frozen development protocol.

This record is written before Study 002 bootstrap, baseline execution, candidate return inspection, or controlled Experiment execution.

## Why Study 002 replaces Study 001 for canonical advancement

Study 001 remains immutable historical evidence, but its Dataset encoded zero fee, zero maker/taker fee, zero spread, zero borrow rate, and participation rate 1.0. Funding was present. Because the maintained replay uses `zero_overlay_dataset_fields_authoritative`, Study 001 therefore does not satisfy the intended execution-economics boundary for canonical cost-aware advancement.

Study 001 is not deleted, rewritten, or retroactively reinterpreted. It is retained as a legacy/diagnostic lineage. Study 002 is a new lineage with a new bootstrap config, Dataset identity, artifact identity, Study identity, and baseline evidence.

## Frozen research conditions

All research degrees of freedom below are copied unchanged from the frozen Study 001 preregistration.

Ordered symbols:

1. `BTCUSDT`
2. `ETHUSDT`
3. `BNBUSDT`
4. `XRPUSDT`
5. `ADAUSDT`

Time contract:

- source start: `2021-01-01T00:00:00Z`
- fit cutoff: `2023-01-01T00:00:00Z`
- development start: `2023-01-01T00:00:00Z`
- development stop: `2025-01-01T00:00:00Z`
- base timeframe: `1h`
- feature timeframes: `4h`, `1d`

Baseline degrees of freedom:

- rule signal: `1h__log_return_24bar`
- feature set: exactly the 12 names in `bootstrap.json`
- fit symbols: all five Study symbols
- PPO seeds: `0,1,2,3,4`
- PPO budget: `100000` timesteps per seed run
- gross budget: `0.5`
- initial capital: `100000`
- paired/bootstrap analysis count: `2000`
- bootstrap seed: `1729`
- Experiment budget: `12`
- rule entry/exit: `0.01 / 0.0025`
- forecast entry/exit: `0.0025 / 0.0005`

Allowed controlled factors remain:

- `FEATURE_SET`
- `RULE_SIGNAL`
- `RULE_THRESHOLDS`
- `FORECAST_THRESHOLDS`
- `PPO_TRAINING_BUDGET`

`FIT_SYMBOL_SCOPE` and `GROSS_BUDGET` remain excluded from within-Study variation.

## Execution-economics contract

Study 002 adds exactly one protocol-level correction: `canonical_m2_research_assumption_v1`.

- generic fee: `0.0005`
- maker fee add-on: `0.0`
- taker fee add-on: `0.0`
- spread: `0.0002`
- maximum participation: `0.05`
- borrow available: `true`
- borrow rate: `0.0`

Impact and stochastic slippage remain zero in the maintained zero-overlay replay.

These values are explicit reproducible research assumptions. They are not asserted to be the historical fee tier or account-specific Binance trading costs of any person or account.

## Source comparability boundary

The original Study 001 GitHub Actions bootstrap artifact is no longer available through the repository Actions API, and the Study 001 Git branch contains only the preregistration files rather than the frozen raw source bytes. Therefore Study 002 cannot truthfully claim byte-identical reuse of Study 001 frozen source evidence.

Study 002 will re-freeze the same historical market, symbols, clocks, and date range. After bootstrap, source/economic/non-economic identities must be audited before cross-Study comparisons. Any source or exchange-metadata drift will be reported explicitly and will prevent attributing Study 001 versus Study 002 result differences solely to execution economics.

## Research boundary

Bootstrap must stop with an immutable Dataset and StudyPlan before baseline execution. Baseline execution is a separate explicit step. Controlled Experiments may begin only after baseline evidence exists and the replacement lineage passes bootstrap/dataset/economics verification.

No profitability, winner, Production, or Experiment conclusion is implied by this preregistration.
