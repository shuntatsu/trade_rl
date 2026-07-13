# Trade RL Architecture

## Status

`trade_rl` is a research-grade baseline-anchored residual reinforcement-learning core. It is not authorized for production trading. Software correctness, market-model realism and empirical profitability are separate gates.

## Authoritative package

Only `trade_rl` is maintained. The former `mars_lite`, direct-action PPO, legacy scripts and legacy tests remain deleted. Git history is the archive; no compatibility execution path was restored.

## Responsibility map

```text
trade_rl/
  domain/        immutable identities and state invariants
  artifacts/     canonical serialization, hashing, staging, publication
  data/          regular-time market and exchange contracts
  strategies/    deterministic baseline strategies
  risk/          operational guardrails and pre-trade constraints
  simulation/    execution, margin, funding and accounting
  evaluation/    metrics, comparisons, bootstrap, gates and folds
  rl/            residual decisions, observations, rewards, environment, training
  workflows/     typed application orchestration
  serving/       immutable bundles, registry activation and live decision adapter
  cli/           argument parsing and typed configuration conversion
```

Import Linter enforces the allowed dependency direction. `domain` remains standard-library only, serving does not import training, and training does not import serving.

## Market contract

`MarketDataset` represents an exact regular grid of completed bars. Timestamps are bar-close times and must agree with `periods_per_year`. The contract contains:

- open, high, low, close, mark and index prices;
- volume and per-bar funding;
- symbol tradability;
- full feature-availability and feature-age arrays;
- explicit warm-up completion and market-data age;
- optional time-varying taker fee and spread arrays;
- quantity steps, minimum notionals and maintenance-margin rates;
- episode-start sampling weights.

Missing information is never silently equivalent to a neutral feature value. Unavailable features are zeroed only after the availability and age channels have been retained in the observation.

## Self-financing account

`BookState` stores signed quantities, cash and mark prices. Portfolio value and weights are derived rather than assigned. Price movement therefore causes natural weight drift; no free target-weight maintenance occurs between decisions.

Every fill reconciles quantity, cash, fees and turnover. Funding changes cash. Mark-to-market records one base-bar return. Fill count, rebalance-event count and liquidation count are distinct. A deterministic state digest supports verified account handoff between deployment segments.

## Execution timing and exchange constraints

A policy observes information through close `t`. Existing positions experience the gap to open `t+1`; orders first execute at that open. Filled quantities are held for the decision interval. Only unfilled residual quantity may continue filling on later bars.

Execution supports:

- volume-participation caps and partial fills;
- taker fees and spread costs from either the dataset or a fallback configuration;
- nonlinear participation impact;
- seeded ordinary and tail slippage;
- per-episode fee, spread and impact randomization;
- quantity-step rounding and minimum-notional filtering;
- non-tradable-symbol blocking;
- funding at the explicit per-bar rate;
- mark-price maintenance margin, forced liquidation and liquidation fees.

The model is deliberately a taker-oriented low-frequency simulator. It does not pretend to reproduce maker queue position or a full limit-order book. Such a claim would require exchange-specific L2 event replay rather than fabricated precision.

## Shared decision path

`ResidualDecisionEngine` is the single proposal-to-target path for the environment and serving runtime. It performs:

1. residual composition around the trend baseline;
2. operational guardrails for stale data, unavailable features and daily loss;
3. next-bar tradability masking;
4. concentration, gross, turnover and drawdown constraints.

Zero residual action remains exact baseline identity after the same guardrails and risk constraints are applied to both hybrid and shadow books.

## Observation and reward

Per-symbol observations contain masked features, full availability masks, feature age, next-bar tradability, fast/base/slow trend targets, alpha, both book weights and their difference. Global state contains both NAVs, drawdowns, gross exposures, relative NAV, market-data age, warm-up state, risk scales and episode progress.

The default reward remains interval excess log return. Optional downside and excess-drawdown penalties are explicit. Hybrid and shadow liquidation are not treated as the same terminal event.

## Physical-time configuration

Trend horizons, episode duration and decision cadence can be configured in hours and are resolved through the dataset cadence. PPO discounting can be specified by a real-time half-life:

```text
gamma = exp(log(0.5) * decision_hours / discount_half_life_hours)
```

This prevents a base-timeframe or decision-cadence change from silently changing the economic horizon.

## Walk-forward identity

Stitched OOS output has one of two explicit identities:

- `independent_folds`: fold-local accounts may reset and gaps are recorded;
- `continuous_account`: ranges must be contiguous and opening/closing state digests must form an unbroken chain.

An independent research aggregate can therefore no longer be represented as a continuous live account curve.

## Quality gates

CI runs Ruff, formatting checks, mypy, Import Linter, Vulture advisory reporting, unit and property tests, migration tests and branch coverage. The architecture contract also verifies that legacy execution trees and direct-action mode remain absent.
