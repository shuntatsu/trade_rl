# Action and Environment Comprehensive Hardening Design

## Goal

Turn the maintained residual-RL environment into a causally correct, reproducible, multi-market research environment whose action semantics, reward components, risk constraints, execution state and training identity are explicit and testable.

## Scope

This design implements every actionable finding from the action-space and environment audit while preserving the repository's baseline-anchored safety model. It does not claim profitability or production authorization.

## Architecture

### 1. Market data and calendar contract

`MarketDataset` supports both regular 24/7 bars and irregular session calendars. It stores an explicit calendar kind, derives elapsed wall-clock hours from timestamps, exposes causal lookback and forward-window helpers, and retains an explicit annualization convention. Feature availability remains per feature; feature staleness is represented separately.

### 2. Trend baselines

`TrendStrategy` supports `auto`, `time_series`, `cross_sectional`, `long_only`, `market_neutral`, and `cash_or_trend` modes. `auto` selects time-series behavior for a one-symbol universe and cross-sectional behavior otherwise. This removes the one-symbol zero-baseline failure while preserving the existing multi-asset behavior.

### 3. Residual action schema v2

The environment derives an `ActionSpec` from enabled components. Core controls are independent fast tilt, slow tilt and risk tilt. Alpha scale is present only when alpha is enabled, so disabled alpha cannot create a dead action dimension. Optional factor residual controls are appended when a causal factor basis is configured. Zero action remains exact baseline identity.

Action validation has three modes: clip for training, strict for evaluation and fail-closed for serving. Saturation and action-delta diagnostics are recorded.

### 4. Risk constraints

Pre-trade risk distinguishes hard constraints from soft turnover constraints. Emergency drawdown deleveraging, concentration and gross limits override turnover throttling. Every returned target is validated again and records proposal-to-target distance, constraint reasons and whether turnover was overridden.

### 5. Accounting, margin and economic termination

Book state records cash, signed quantities, collateral use, borrow cost, margin utilization and insolvency status. Market losses, execution costs, margin calls and liquidation are economic terminal states rather than uncaught exceptions. Invalid data remains an exception.

### 6. Execution model

Execution supports per-symbol and per-time fee, spread and participation inputs; minimum notional, lot size, price tick, borrow availability, borrow rate, maker/taker mode, order latency and scheduled funding. Hybrid and shadow books share common random numbers inside one episode but receive a new deterministic episode seed on every reset.

### 7. Episode and initial-state sampling

Episode duration can be sampled from a configured curriculum. Reset modes include cash, baseline, random valid portfolio and stressed drawdown state. Each mode is causal and reproducible. The selected mode, episode seed and duration are returned in reset info.

### 8. Observation schema v3

Per-asset observations include every feature, every feature-availability mask, feature staleness, tradability, trend targets, alpha, actual weights, requested weights, fill ratio, unfilled turnover, participation, execution cost and position metadata. Global observations include cash weight, net and gross exposure, drawdown, margin utilization, previous action and remaining-time fields only when the environment is explicitly finite-horizon.

A fold-fitted `ObservationNormalizer` is content addressed. Statistics may be fitted only on a train range and are frozen for validation, test and serving.

### 9. Reward schema v3

Absolute log-wealth growth is primary. Baseline-relative log growth is secondary. Risk penalties use only incremental drawdown beyond a small dead zone. Baseline underperformance is penalized only when cumulative rolling underperformance exceeds a fixed tolerance, using a progressive hinge. Terminal penalties are continuous in equity and margin deficit rather than fixed jackpots. Every component is returned in `RewardBreakdown`.

### 10. Training and serving identity

Environment identity includes the full action specification, alpha artifact identity, factor basis identity, normalizer digest, calendar contract, reset curriculum, reward configuration and execution constraints. Training validates that configured decision hours match the environment and exposes PPO exploration controls. Serving validates the exact action dimension and strict action mode.

## Error handling

Economic failures return terminal transitions with a structured reason. Contract violations, non-finite market inputs, identity mismatches and serving action violations fail closed with exceptions.

## Verification

Regression tests cover one-symbol trend behavior, episode RNG independence, economic termination, hard-risk precedence, action dimensions, factor composition, reward decomposition, observation shape and masks, normalizer leakage prevention, irregular calendars, reset modes, execution constraints, training cadence checks and serving strictness. Per user instruction, all test commands are run only after all implementation and test code is complete.
