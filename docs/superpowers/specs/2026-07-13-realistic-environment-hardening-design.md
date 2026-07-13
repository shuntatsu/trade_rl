# Realistic Environment Hardening Design

## Status and scope

This design applies to the rebuilt `trade_rl` package on `main`. The removed `mars_lite` tree must not be restored. The objective is to make the maintained residual-policy environment materially closer to a real low-frequency perpetual-futures portfolio while preserving the baseline-anchored research architecture.

The work addresses the previously identified environment defects that remain relevant after the rebuild: irregular time grids, same-close execution, static liquidity-blind costs, weight-only accounting, hidden shadow state, missing pre-trade constraints in training, bar-count time semantics, near-sighted discount configuration, terminal free liquidation, ambiguous stitched OOS identity, missing tradability/availability masks, and misleading trade-count semantics.

## Design principles

1. Market mechanics and accounting are authoritative. Policy code cannot mutate book state directly.
2. One action is decided after a completed bar and first becomes executable at the next bar open.
3. The simulation is self-financing: signed quantities, cash, mark prices, fees, funding, fills, and equity must reconcile.
4. All durations are expressed in real hours at public configuration boundaries and are resolved to exact bar counts from the dataset cadence.
5. Missing data is distinct from a neutral value. Tradability and feature availability are explicit arrays.
6. Hybrid and shadow books use the same execution and risk pipeline. Zero residual action remains exact baseline identity.
7. Independent-fold stitching must never be presented as a continuous live account simulation.
8. Existing `trade_rl` boundaries remain intact: data contracts in `data`, accounting/execution in `simulation`, constraints in `risk`, policy state in `rl`, aggregation in `evaluation`, orchestration in `workflows`.

## Market dataset contract

`MarketDataset` is upgraded to a version-2 in-memory contract with the following mandatory arrays:

- `open`, `high`, `low`, `close`: `(bars, symbols)` positive finite prices.
- `volume`: `(bars, symbols)` non-negative base-asset volume.
- `funding_rate`: `(bars, symbols)` funding applied during each bar.
- `tradable`: `(bars, symbols)` boolean order-entry availability.
- `feature_available`: `(bars, symbols, features)` boolean availability distinct from feature value zero.

Timestamps represent bar close times. They must be strictly increasing and exactly regular. The inferred bar duration must agree with `periods_per_year`; a one-hour 24/7 dataset therefore uses 8,760 periods per year. OHLC invariants are validated (`low <= open/close <= high`).

The dataset exposes `bar_hours` and `bars_for_hours(hours)` so strategy and environment configuration can remain time-based.

## Self-financing account

`BookState` stores:

- signed `quantities` per symbol;
- `cash`;
- current `mark_prices`;
- peak equity and maximum drawdown;
- turnover, cost, funding, rebalance-event and fill counters;
- base-bar return history.

`portfolio_value` and `weights` are derived from quantities, cash and marks. An execution changes quantities and cash at the fill price. Funding changes cash. Mark-to-market changes only marks and derived equity. All mutations validate reconciliation and finite positive equity.

`n_trades` is removed from internal state. Metrics receive `fill_count`; `rebalance_events` is reported separately in environment information. A single fifteen-symbol rebalance is one rebalance event and up to fifteen fills.

## Execution timing and liquidity

At decision time `t`, the policy sees information through close `t`. Existing positions first experience the gap from close `t` to open `t+1`. Orders then execute at open `t+1`. Filled positions experience the intrabar move from open `t+1` to close `t+1`. The same process repeats for each base bar in the decision interval.

Costs are symbol-level:

- taker fee on filled notional;
- half-spread/slippage on filled notional;
- nonlinear impact based on participation rate;
- optional seeded random slippage and tail shocks for training-domain randomization.

Participation is `order_notional / bar_market_notional`, where bar market notional is `open * volume`. A configurable maximum participation rate causes partial fills; unfilled target delta is reported and is not silently assumed filled. A non-tradable symbol cannot receive a new fill. Existing exposure remains marked until trading becomes available.

Execution results report requested turnover, filled turnover, unfilled turnover, cost amount, funding amount, fill count and rebalance count.

## Risk and safety parity

`PreTradeRisk` is mandatory in `ResidualMarketEnv` and is applied independently to hybrid and shadow targets immediately before execution. Defaults are intentionally non-permissive:

- gross exposure <= 1.0;
- absolute symbol weight <= 0.40;
- turnover per decision <= 1.0;
- drawdown deleveraging begins at 10%;
- exposure reaches zero at 20% drawdown.

Targets are additionally masked by next-bar tradability. Both books pass through the same constraint object so zero action remains identity.

Environment observations include the resulting risk scale and both books' portfolio state. Serving remains action-only in the present architecture; release remains blocked until a shared live decision/execution adapter uses the same risk contract.

## Observation schema v2

Per symbol, the policy observes:

- market features;
- feature-availability fraction;
- next-bar tradability flag;
- fast/base/slow trend targets;
- alpha;
- hybrid weight;
- shadow weight;
- hybrid-minus-shadow weight.

Global state includes:

- dataset global features;
- log hybrid equity and log shadow equity;
- hybrid and shadow drawdown;
- log relative NAV;
- hybrid and shadow gross exposure;
- current hybrid and shadow risk scale;
- episode progress.

This removes the hidden-shadow-state defect and makes constraint-induced action changes observable.

## Reward and termination

The base reward remains excess interval log return. A configurable downside penalty and excess-drawdown penalty may be added, both defaulting to zero for backward scientific comparability. Hybrid insolvency receives the terminal negative penalty. Shadow insolvency without hybrid insolvency does not punish the hybrid policy; it produces a capped positive terminal outcome.

Terminal transitions return a valid final observation. Episodic training defaults to explicit close-of-episode liquidation with normal fee, spread and impact costs. Continuous evaluation disables liquidation and carries the closing book into the next deployment segment.

## Time-based strategy and training configuration

`TrendConfig` uses `fast_hours`, `base_hours`, and `slow_hours`, defaulting to 24, 48, and 96 hours. The environment uses `episode_hours` and `decision_hours`. All values must resolve to integral bar counts.

Training can still accept an explicit `gamma`, but the preferred configuration is a discount half-life in hours. The helper

`gamma = exp(log(0.5) * decision_hours / discount_half_life_hours)`

ensures identical real-time discounting across base timeframes and decision intervals. The CLI exposes and validates this conversion.

## Walk-forward identity

`StitchedOOS` receives an explicit mode:

- `independent_folds`: fold-local accounts may reset; gaps are allowed and recorded;
- `continuous_account`: ranges must be contiguous and each fold must declare matching opening and closing state digests.

The default remains independent folds. Metrics and serialized output must carry the mode so an independent research aggregate cannot be described as a continuous live equity curve.

## Testing and migration

Tests are added before implementation and must cover:

1. irregular timestamps rejected;
2. OHLC and availability contracts validated;
3. decision-at-close executes at next open;
4. quantity/cash/equity reconciliation;
5. weight drift without free rebalancing;
6. participation-limited partial fill;
7. non-tradable symbols do not fill;
8. zero action still matches the constrained shadow exactly;
9. shadow state appears in the observation;
10. time-based lookbacks and decisions resolve consistently;
11. half-life discount conversion;
12. end-of-episode liquidation charges costs;
13. hybrid-only and shadow-only termination rewards differ correctly;
14. independent and continuous stitching cannot be confused;
15. fill count and rebalance-event count are distinct.

All existing fixtures migrate to the v2 dataset contract. Ruff, formatting, mypy, import-linter and full branch-coverage tests remain mandatory.
