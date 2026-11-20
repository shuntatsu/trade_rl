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
- **Risk:** `PreTradeRiskConfig(max_gross=2.5, max_abs_weight=0.55, max_turnover=2.5, drawdown_start=0.125, drawdown_stop=0.25)` (Amendment 2; it was 0.10 / 0.20).
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
2. No candidate cell has a max drawdown of 0.25 or more (Amendment 2; it was 0.20).
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

## Amendment 1 (result-blind, before any engine replay or out-of-sample read)

Status: Active. Written before any stage was run and before any number of this protocol was seen. It fixes values the frozen text left to the implementation. It changes no signal constant, stride, cap, cost, risk value, universe or window.

### Initial-margin rate of the cash account

- The engine has no explicit perpetual initial-margin field. `ExecutionCostConfig` carries only `max_leverage`, `maintenance_margin_rate` and `collateral_haircut`.
- The margin model derives its initial margin as `margin_used = gross_notional / max_leverage` (`trade_rl/simulation/execution.py:756`, `_update_margin`). The rate is therefore `1 / max_leverage = 1 / 2.5 = 0.4`, taken from the engine's own formula and not from this protocol.
- The cash-account cap is: spot long gross + 0.4 × perp gross ≤ 1.0. Perp gross is the sum of the absolute perp weights (shorts count).
- The rate is conservative: it is far above the engine's maintenance rate in this protocol (0.05) and the engine does not enforce it as an order constraint.
- The rate is a frozen constant of `protocol.py` and part of the protocol digest. A different rate is a new protocol.

### Warm-up values

- A pair is available to the signal from the first bar at which both its spot and its perp are listed plus 180 bars (the longest per-symbol window in the signal chain, the 30-day volatility and momentum window). A pair listed from the first row of the stage is available from the first row; the stage warm-up covers it.
- The stage warm-up is 540 bars (the longest windowed statistic of the chain). Every first decision must be at or after row 540. DEV's first decision is therefore the row with index 540.
- Before a pair is available its signal inputs are neutral: zero volume, `valid = False`, and a constant price. At activation the pair's prices are divided by one common factor (the spot close at the activation row) so that the sentinel-to-price step is not a return. Both factors use only the activation row and past rows. Pairs listed from the first row are not rescaled.
- The lab's cross-sectional demeans of the hybrid alphas include every symbol column (`all_ones`). A pair that is not yet available is a zero column in those demeans. This is a disclosed approximation of the lab for stages in which pairs list late.

### DEV last row

DEV is bound by the Tier D embargo (2026-08-01). The spec's 2026-07-31T20:00Z is the Tier D last dataset row, so DEV's last dataset row is that close and its last decision is two bars earlier. No hour at or after 2026-08-01 is read for DEV.

### Data

- Spot and perp 1h klines and perp funding are read from the same checksum-verified Binance Vision archives. The taker-buy quote volume of each leg is the sum of the four hourly values of the 4h bar, for hours the leg is tradable, and zero otherwise.
- Funding events are stamped at the close of the 4h bar that holds them. The count-based funding means start cold at the stage's first row (or the perp's first row); the stage warm-up washes this out before the first decision.
- Perp series are strict: no halt calendar and no gap is ever filled. A perp gap refuses the run.
- Declared exchange-wide spot halts are read from a result-blind coverage report as in `trend_long_spot_oos_v1`. The list is empty until such a report is produced and its SHA-256 frozen in `protocol.py`; the digest then changes and the G4 relay quotes the final digest.

### Cash-account validity

The negative-cash check is unchanged (cash net of cumulative execution costs below −1e-6 × capital is INVALID). The engine books perp notional through cash like spot notional, so a book with net exposure above 1.0 can show negative cash although the margin cap holds. That cell is INVALID as written; it is a reported outcome, not a rescue case.

### Sequencing

FORWARD final needs a BACKCAST_2020 `decision.json` of this exact protocol digest, whatever its label. The labels are decided only by the decision rule above.

## Amendment 2 (result-blind, before any engine replay or out-of-sample read)

Status: Active. Made before any result of this protocol existed (no stage has been run and no replay of the ported anchor has produced a number).

The owner raised the research drawdown budget to 25%. Quote, 2026-11-20: 「25％でもいいよ」 (relayed on Issue #810).

- The risk config is `PreTradeRiskConfig(max_gross=2.5, max_abs_weight=0.55, max_turnover=2.5, drawdown_start=0.125, drawdown_stop=0.25)`. The ramp is proportional to the old 0.10 / 0.20 (start at half of the stop).
- Condition 2 of `SUPPORTED(S)` is: no candidate cell has a max drawdown of 0.25 or more.
- Everything else is unchanged: signal, stride, caps, costs, `max_leverage`, participation cap, universe, windows, the cash account of Amendment 1 and the other four conditions.
- `trend_long_spot_oos_v1` and the earlier protocols stay at 0.20: they are already frozen and run, and this amendment changes no value outside this protocol.
- The probe's synthetic decision snapshots come from the shared Core7 helper and carry its 0.10 / 0.20 limits. They only exercise the causality probe; the replay itself runs with the config above.
