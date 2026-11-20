# Realistic agy anchor out-of-sample test (`agy_anchor_realistic_v1`)

Status: Active. This protocol was frozen before any trade_rl-engine replay of the ported anchor and before any data in its out-of-sample windows was read for this signal.

## Question

agy's non-RL anchor `ANC_BIDIR_75_25` is a cross-sectional mean-reversion strategy with a carry component over 11 spot/perp pairs. In agy's lab it returned +340% at 6 bp over 2023-01 to 2026-07. The audit in [Issue #810 comment 7034131404](https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-7034131404) found no look-ahead. It also found that the edge is concentrated in 2023–2024, with 2025–2026 at about +5% a year, and that the lab has no participation cap and no financing.

**Hypothesis:** the faithfully ported signal keeps a positive, cost-robust return through the trade_rl engine under realistic constraints, on data never used for it.

**Counter-hypothesis:** the edge was a 2023–2024 regime effect. The 2025–2026 decay will continue out of sample.

**Cheapest falsifier:** the FORWARD stage. Its first 3.7 months of data already exist and have never been read for this signal.

**Prior exposure, disclosed:**
- Tier D 2021–2026 was explored by agy over about 280 arms.
- The audit and the port read known-used pre-2026-08 data, including counterfactual returns of the lab-defect fix: +354% with the fix, +325% as-is, both in agy's lab. The fix was adopted because it is the evident intent of the code (`factor_scores[:, 9, :]`, not `[:, :, 9]`), but its counterfactual was seen first.
- The planner has public knowledge of 2020 market history.

## Signal (frozen)

- **Port:** `trade_rl/strategies/agy_anchor/` at the commit this spec lands in. It is a klines + funding + taker-buy variant, with open interest, Hyperliquid and the 15m jump terms neutral. It uses the fixed per-symbol N4.
- **Parity:** exact parity with the lab is pinned by `tests/fixtures/agy_anchor/` (`fix_*`).
- **Constants:** every lab constant is unchanged.
- **Rebalance stride:** 18 bars (3 days), with the lab's total-turnover deadband. The stride was chosen from agy's known-used diagnostics, where 3-day was the most cost-robust (+246% at 12 bp against +220% daily). This is disclosed as a selection.

## Execution (frozen)

- **Engine:** `run_shared_portfolio_replay` with settlement and ledger capture. Lean-core terminal dust is accepted through the core7 summary with `execution_cost=` passed.
- **Risk:** `PreTradeRiskConfig(max_gross=2.5, max_abs_weight=0.55, max_turnover=2.5, drawdown_start=0.10, drawdown_stop=0.20)`.
- **Costs:** `max_leverage=2.5` and participation cap 0.01. The BASE, HIGH_COST and DELAY_1BAR costs are those of `core7_mtf_study.protocol.scenario_cost`.
- **Cash account:**
  1. After the signal's own caps, the target is scaled proportionally so that spot long gross plus perp gross times the engine's initial-margin rate is at most 1.0.
  2. A cell is `INVALID` (`negative_cash_borrowing`) if, at any interval, cash net of cumulative execution costs falls below −1e-6 × capital. The measure is the one in `trend_long_spot_oos_v1`.
  3. If the engine has no initial-margin rate for perps, the implementation stops and reports instead of inventing one.
- **Arms:** `AGY_ANCHOR` (candidate), `CASH`, and `EW_HOLD_SPOT`. The `EW_HOLD_SPOT` arm is the lowturn control, on a cash account.

## Stages

| Stage | Load start (warm-up) | First decision | Last decision bar | Role |
|---|---|---|---|---|
| DEV | 2021-10-01 | first index with full warm-up | 2026-07-31T20:00Z | known-used diagnostic: port fidelity under realistic constraints; never gates, never tunes |
| BACKCAST_2020 | 2020-01-01 | 2020-05-01T00:00Z | 2020-12-31T20:00Z | unused but short (about 8 months); weak check |
| FORWARD | 2026-02-01 | 2026-08-01T00:00Z | 2027-04-30T20:00Z | decisive |

- **Universe:** the 11 Tier D pairs. A pair is tradable only from when both its spot and its perp markets exist and have the warm-up the signal needs. Before that, it is unavailable.
- **Data:** Binance public Vision archives, checksum-verified, with taker-buy volume. Missing hours are handled as in `trend_long_spot_oos_v1`: daily-archive rebuild, then declared exchange-wide halts from a result-blind coverage report. Any undeclared gap refuses the run.
- **Authorization:** FORWARD and BACKCAST_2020 reads are covered by owner comment 7028416072. Each run needs its own G4 relay quoting this protocol id and digest.

## Decision rule (frozen)

`SUPPORTED(S)`, for S being BACKCAST_2020 or FORWARD, holds iff:

1. All nine cells are valid.
2. No candidate cell has a max drawdown of 0.20 or more.
3. The candidate's BASE total return is above 0.
4. The candidate's BASE annual Sharpe is at least 0.50.
5. The candidate's HIGH_COST and DELAY_1BAR total returns are above 0.

| Label | Condition |
|---|---|
| `ANCHOR_SUPPORTED` | `SUPPORTED(FORWARD)` and `SUPPORTED(BACKCAST_2020)` |
| `ANCHOR_FORWARD_ONLY` | `SUPPORTED(FORWARD)` but not BACKCAST_2020 |
| `ANCHOR_NOT_SUPPORTED` | otherwise, with all cells valid |
| `INVALID` | any cell invalid |

**Timing:**
- BACKCAST_2020 is decided when it runs.
- FORWARD interim reports, `INTERIM_NOT_A_DECISION`, run monthly.
- FORWARD final scoring happens once, at or after 2027-05-02T00:00Z.

**Reported, not gating:**
- per-year returns, turnover, cost, funding and max drawdown;
- the fraction of decisions where the cash-account scale bound;
- the stationary block bootstrap (42 bars, 10,000 resamples, seed 20261121) lower bound of BASE excess over CASH;
- DEV-stage numbers.

## Stop rule

- No change to any signal constant, stride, cap, cost, risk value, universe or window after any stage result, including DEV.
- DEV is reported before BACKCAST_2020. It may only stop the line, if the port proves broken, and it may never retune it.
- No rescue run.
- Higher leverage is a new protocol.
- No label authorizes orders or live trading.
