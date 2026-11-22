# Vol-targeted BTC trend ensemble on a spot cash account (`btc_trend_ensemble_v1`)

Status: Active (not frozen until an independent review approves the exact head). This protocol is frozen before any Bitstamp BTC/USD bar of its BACKCAST window has been read, and before any FORWARD bar exists.

Protocol digest: `e0912608069eaa07115c1cdfb7bf37b51b7f581d03b187b16124b3069755e751`

## Question

Exploratory screens on known-used Binance BTCUSDT spot daily data (2018-11 to 2026-10, issue #810 planner notes) found:
- A long/cash rule, "close above its N-day simple moving average", reaches Sharpe 0.87–1.26 for every N from 30 to 200.
- Macro, stablecoin and multi-coin variants did not improve on it.
- At full exposure its max drawdown is 38–65%, above the owner's 25% budget.

**Hypothesis:** an equal-weight ensemble of six lookbacks, vol-targeted to 20% annualized, keeps a positive and cost-robust return with drawdown inside the budget, on data never used for it.

**Counter-hypothesis:** the known-data result is a 2019–2024 regime effect, and the lookback set and the vol target were fitted to it.

**Cheapest falsifier:** BACKCAST on 2015-01 to 2018-06 Bitstamp BTC/USD daily bars, which have never been read for this rule.

**Prior exposure, disclosed:**
- The six lookbacks, the 20% target, the 0.05 deadband and the 30-day vol window were chosen after seeing known-used 2018-11 to 2026-10 results. On that data: CAGR about 21%, max DD 19–22%, Sharpe about 1.2.
- The planner has public knowledge of BTC history, including the 2017 bubble and the 2018 crash.
- `trend_long_spot_oos_v1` already used 2018-07 onward for a related spot trend family, which is why BACKCAST ends at 2018-06-30.
- That protocol's FORWARD interim reports exposed post-2026-08 spot data, which is why FORWARD here is strictly prospective.

## Rule (frozen)

All quantities are computed at the close of UTC day t from closes up to and including day t.

- **Trend score:** s_t = (1/6) · Σ_{N ∈ {30, 50, 75, 100, 150, 200}} 1[close_t > SMA_N(close)_t], so s_t ∈ [0, 1].
- **Volatility:** σ_t = standard deviation of the last 30 daily log returns × √365. Use population standard deviation (ddof 0).
- **Target exposure:** e_t = s_t · min(1, 0.20 / σ_t), in [0, 1]. It is the fraction of equity in BTC; the rest is cash. There is no leverage and no shorting.
- **Deadband:** the held exposure changes only if |e_t − held| > 0.05, and then it moves to e_t.
- **Execution:**
  - BASE: the exposure decided at the close of day t is held over day t+1, earning close_{t+1}/close_t − 1, with the trade cost charged at the close of t.
  - DELAY: the exposure is first held over day t+2.
- **Cost:** one-way cost × |change in exposure| × equity, charged at each trade.
- **Warm-up:** no position until 200 closes exist.

## Cells (frozen)

| Cell | Cost (one-way) | Execution delay | Gates |
|---|---|---|---|
| `TREND__BASE` | 10 bp | 1 day | yes |
| `TREND__HIGH_COST` | 20 bp | 1 day | yes |
| `TREND__DELAY` | 10 bp | 2 days | yes |
| `HOLD__BASE` | 10 bp at entry only | — | reported |
| `CASH` | — | — | reported (0) |

## Stages

| Stage | Data | Load from (warm-up) | Decisions | Role |
|---|---|---|---|---|
| BACKCAST | Bitstamp BTC/USD daily OHLC, public API `https://www.bitstamp.net/api/v2/ohlc/btcusd/?step=86400` | 2014-06-01 | 2015-01-01 to 2018-06-30 | decisive for the backcast half |
| FORWARD | Binance BTCUSDT spot daily klines, public API `https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d` | 2026-05-01 | 2026-11-23 to 2027-11-22 | decisive, prospective |

- **Daily bars:** a daily bar is the UTC day.
- **Missing days:** a missing day or a non-positive close refuses the run. No fill is applied.
- **Fetched data:** raw responses are cached and hashed in the receipt.
- **FORWARD interims:** FORWARD interim reports, labelled `INTERIM_NOT_A_DECISION`, may run monthly. An interim stops the line (`STOPPED_EARLY`) if `TREND__BASE` max DD is 25% or more.
- **FORWARD final:** the final runs once, at or after 2027-11-24.

## Decision rule (frozen)

For S being BACKCAST or FORWARD, `SUPPORTED(S)` holds iff all of the following hold:

1. All cells are valid.
2. `TREND__BASE` total return is above 0.
3. `TREND__BASE` annual Sharpe is at least 0.5. Sharpe is computed from daily simple returns, population standard deviation, √365, with no risk-free rate.
4. `TREND__HIGH_COST` and `TREND__DELAY` total returns are above 0.
5. The max drawdown of every gating cell is below 25%.

| Label | Condition |
|---|---|
| `TREND_SUPPORTED` | `SUPPORTED(BACKCAST)` and `SUPPORTED(FORWARD)` |
| `TREND_BACKCAST_ONLY` | `SUPPORTED(BACKCAST)`, and FORWARD not yet final |
| `TREND_NOT_SUPPORTED` | otherwise, with all cells valid |
| `INVALID` | any cell invalid |

A cell is invalid if, at any day:
- equity is not positive;
- exposure is outside [0, 1];
- the accounting identity fails by more than 1e-9.

A BACKCAST `TREND_NOT_SUPPORTED` stops this version.

## Authorization

- BACKCAST and FORWARD reads are covered by the owner scope relay [Issue #810 comment 7040247247](https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-7040247247).
- Each stage also needs its own G4 relay on Issue #810 that quotes this protocol id, the digest and the full code commit.
- The harness must fetch both comments twice. Each must be owner-authored and unchanged between the fetches.
- The checkout must be clean.
- Outputs are write-once. A local ledger claim must be created exclusively before any computation.

## Stop rule

- No change to any constant, lookback, target, deadband, cost, window or rule after any stage result.
- No rescue run.
- A refusal for a defect outside the economics needs a result-blind amendment before any rerun.
- Higher exposure is a new protocol.
- No label authorizes orders, paper trading or live trading.
