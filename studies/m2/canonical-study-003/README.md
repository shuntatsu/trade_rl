# Canonical M2 Study 003

Status: Pre-registered replacement lineage

## Purpose

This Study repeats the Canonical M2 Study 001 research question under an explicit, identity-bound execution-economics contract. It asks whether one symbol-ID-free universal strategy family can beat simple controls robustly across five long-lived Binance USD-M perpetual markets under one frozen development protocol.

This record is written before Study 003 bootstrap, baseline execution, candidate return inspection, or controlled Experiment execution.

## Lineage status

Study 001 remains immutable historical evidence, but its Dataset encoded zero fee, zero maker/taker fee, zero spread, zero borrow rate, and participation rate 1.0. Funding was present. Because maintained replay uses `zero_overlay_dataset_fields_authoritative`, Study 001 is retained as a legacy/diagnostic lineage and is not eligible for canonical cost-aware advancement.

Study 002 was an invalid preregistration attempt. Its execution helper failed before bootstrap or baseline because the nested execution-economics payload omitted the required `execution_economics_profile_v1` schema marker. Study 002 is not repaired in place and produces no research result.

Study 003 is the replacement canonical lineage. Its preregistration bytes were independently parsed and compared with frozen Study 001 before this branch was created.

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

Study 003 adds exactly one protocol-level correction: `canonical_m2_research_assumption_v1`, schema `execution_economics_profile_v1`.

- generic fee: `0.0005`
- maker fee add-on: `0.0`
- taker fee add-on: `0.0`
- spread: `0.0002`
- maximum participation: `0.05`
- borrow available: `true`
- borrow rate: `0.0`

Impact and stochastic slippage remain zero in the maintained zero-overlay replay.

These values are explicit reproducible research assumptions. They are not asserted to be the historical fee tier or account-specific Binance trading costs of any person or account.

## Preregistration validation evidence

Before Study 003 preregistration, the exact candidate JSON was validated on an isolated temporary branch against the frozen Study 001 JSON and the maintained v2 parser.

- candidate raw SHA-256: `1aa2e378b14c731bbb259fa79f5494884dd64d528d0f3493f188dcdade218791`
- canonical config digest: `8d39d389d8a6b3e57439b45862063d14ccf1b3fdfd06135681804814202db425`
- allowed delta versus Study 001: top-level bootstrap schema v1 → v2 plus `execution_economics` only
- parser and `to_payload()` round-trip: verified

## Source comparability boundary

The original Study 001 GitHub Actions bootstrap artifact is no longer available through the repository Actions API, and the Study 001 Git branch contains the preregistration files rather than the frozen raw source bytes. Study 003 therefore cannot claim byte-identical reuse of Study 001 frozen source evidence.

Study 003 will re-freeze the same historical market, symbols, clocks, and date range. After bootstrap, source/economic/non-economic identities must be audited before cross-Study comparisons. Any source or exchange-metadata drift will be reported explicitly and will prevent attributing Study 001 versus Study 003 result differences solely to execution economics.

## Research boundary

Bootstrap must stop with an immutable Dataset and StudyPlan before baseline execution. Baseline execution is a separate explicit step. Controlled Experiments may begin only after baseline evidence exists and the replacement lineage passes bootstrap/dataset/economics verification.

No profitability, winner, Production, or Experiment conclusion is implied by this preregistration.
