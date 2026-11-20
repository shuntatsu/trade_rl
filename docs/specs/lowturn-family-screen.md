# Low-turnover strategy family screen (`lowturn_family_screen_v1`)

Status: Active

Status: protocol frozen before any result exists for these families. The screen runs only on known-used Tier D data, so its strongest possible label is `EXPLORATORY_CANDIDATE`. It is not evidence of an edge, and it does not authorize paper trading, live orders, FS1 access, unused-data access or final-data access.

## Why

The Core7 MTF v4 Phase 1 run ([Issue #810 comment 7023026438](https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-7023026438)) traded about 900× turnover. Costs ate about 53% of capital, and every arm lost money at BASE cost. Earlier development work found two weak positives:

- slow time-series momentum: Phase 2 trial T1, 5 perps, 2023–2026, +10.4%;
- volatility targeting: Phase 3 P3-6, post hoc.

The fast Donchian trend-stop study was `NO_WINNER`.

**Hypothesis H-LT:** slow multi-horizon signals rebalanced weekly cut turnover by more than 10×. At that turnover, a diversified trend or momentum family keeps a positive after-cost return and beats passive controls on risk-adjusted terms.

**Counter-hypotheses:**

- The positive trend results are beta to the 2023–2024 bull market. In that case the trend families do not beat `VT_HOLD_PERP` or `EW_HOLD_SPOT` in Sharpe.
- The dev-period pass comes from the 2022 bear market alone. In that case the families fail in the holdout.

**Cheapest falsifier:** every non-control family fails `DEV_PASS`, or no `DEV_PASS` family is `HOLDOUT_CONFIRMED`.

**Prior exposure, disclosed:** none of these families has been run before. They are not data-blind, though:

- The same 11 symbols, the same 2021–2026 window, TSMOM T1 and the Donchian study were all read before this protocol was written.
- The family choice is informed by those results.

This is why the label ceiling is exploratory.

## Data

| Item | Value |
|---|---|
| Dataset | Tier D, `dataset_id 92336a938ac1229316a1e5ecce97f5d12a2f8f4864812b2b415e83389dbbaf8b`, assembled by `tools.core7_mtf_tierd` with the same inputs as the Phase 1 run |
| Legs | 11 pairs × (spot, USD-M perp), alternating |
| Bars | 4h, 2021-10-01T04:00Z..2026-07-31T20:00Z |
| Embargo | No bar at or after 2026-08-01T00:00Z |
| Warm-up | The first 1080 bars (180 days) are never traded |
| DEV segment | First decision at index 1080; stops at the first bar at or after 2024-07-01T00:00Z |
| HOLDOUT segment | From the first bar at or after 2024-07-01T00:00Z to the dataset's last replay index |

- Each segment is a separate replay starting flat with 100,000 USDT.
- HOLDOUT replays run for every family, but only the decision rule below reads them.

## Common execution

| Setting | Value |
|---|---|
| Replay path | `run_shared_portfolio_replay` only |
| Risk | `PreTradeRiskConfig` with the same values as `core7_mtf_shadow_anchor_ppo_v4`: max_gross 2.5, max_abs_weight 0.55, max_turnover 2.5, drawdown_start 0.10, drawdown_stop 0.20 |
| Execution cost | `scenario_cost(...)` from `core7_mtf_study.protocol`, unchanged |
| Scenarios | `BASE`, `HIGH_COST`, `DELAY_1BAR` |
| Terminal settlement | `settle_terminal_position=True` |
| Rebalance | Every 42 bars (weekly), at bars where `(index - segment_start) % 42 == 0` |

- **DELAY_1BAR** lags every signal input by one bar, following the core7 protocol.
- **Between rebalances** every symbol is an exact-quantity HOLD. The exception is when `risk_scale` decreases: the policy then re-targets the last rebalance weights times the new scale, using the same rule as `ShadowAnchorMtfPolicy`.
- **Deadband:** at a rebalance, a symbol whose `|target − current weight|` is below 0.02 is held.
- **Signal timing:** a decision at index `t` may use closes of bars `≤ t` only if that matches the snapshot convention used by `mtf6_signals`. The implementation must follow that convention exactly, and a future-perturbation causality probe must pass for every cell.
- **Spot legs are long-only.** A negative spot target is a bug, not a short.

All lookbacks are in 4h bars:

| Lookback | Bars |
|---|---|
| 7d | 42 |
| 30d | 180 |
| 60d | 360 |
| 90d | 540 |
| 150d | 900 |
| 180d | 1080 |

- Realized vol `vol60_i` is the annualized standard deviation of 4h log returns over 360 bars, using a 2190 bars/year annualization factor, floored at 0.10.

## Families (frozen; no parameter grid)

Controls:
- `CASH`: always flat.
- `EW_HOLD_SPOT`: long every spot leg at 1/11.
- `VT_HOLD_PERP`: long every perp at `w_i = (0.50 / vol60_i) / 11`.

Candidates (perp legs unless stated):
- `TS_TREND_LS`: `s_i = mean(sign(r30), sign(r90), sign(r180))`; `w_i = s_i · (0.50 / vol60_i) / 11`.
- `TS_TREND_LONG_SPOT`: as `TS_TREND_LS` on spot legs, with `s_i` clipped to `[0, 1]`.
- `TS_TREND_BTCETH`: as `TS_TREND_LS` on BTC and ETH perps only, with `/ 2` in place of `/ 11`.
- `XS_MOM`: score is `close[t−42] / close[t−402] − 1`. Long the top 3, short the bottom 3, each at `±(0.50 / vol60_i) / 6`.
- `MA_BTCETH_SPOT`: long the BTC and ETH spot legs at 0.5 each when SMA(180) > SMA(900) for that asset, otherwise 0 for that asset.

**Caps, applied in this order:**
1. Clip each `|w_i|` to 0.40.
2. If gross `Σ|w|` exceeds 2.0, scale every weight down proportionally to 2.0. For `XS_MOM` the gross limit is 1.5.

**Rules common to every family:**
- A symbol whose inputs are unavailable gets weight 0.
- The non-traded leg of each pair has weight 0.

## Decision rule (frozen)

For each non-control family `F`, the DEV stage is read first. `DEV_PASS(F)` holds only if every condition below holds:

1. All three DEV cells are valid: ledger reconciled, causality probe passed, terminal flat, not insolvent.
2. No DEV cell hit the 20% drawdown stop.
3. The DEV BASE annual Sharpe is at least 0.5.
4. The DEV BASE Sharpe is above the DEV BASE Sharpe of both `EW_HOLD_SPOT` and `VT_HOLD_PERP`.
5. The DEV HIGH_COST return and the DEV DELAY_1BAR return are both above 0.

`HOLDOUT_CONFIRMED(F)` applies only to `DEV_PASS` families. It holds only if:

- all three HOLDOUT cells are valid;
- no HOLDOUT cell is stopped;
- the HOLDOUT BASE Sharpe is at least 0.3;
- the HOLDOUT HIGH_COST return is above 0.

**Screen label:**
- `EXPLORATORY_CANDIDATE` if at least one family is both `DEV_PASS` and `HOLDOUT_CONFIRMED`.
- `NO_CANDIDATE` otherwise.
- `INVALID` if any control cell is invalid.

**Reported but not gating:**
- every cell's return, Sharpe, max drawdown, cost, funding and turnover;
- the deflated Sharpe ratio of each DEV BASE candidate Sharpe, with 5 trials;
- per-calendar-year returns.

## Stop rule

The protocol runs once. After any result:

- No family, parameter, lookback, cap, segment boundary, cost or symbol changes on this data.
- On `NO_CANDIDATE` the low-turnover line of this design ends.
- On `EXPLORATORY_CANDIDATE` the only allowed next step is a separately pre-registered forward or paper evaluation of the unchanged candidate. That needs its own owner gate.
- No result is reclassified.
