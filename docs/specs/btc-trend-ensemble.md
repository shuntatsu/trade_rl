# Vol-targeted BTC trend ensemble on a spot cash account (`btc_trend_ensemble_v1`)

Status: **HOLD / Amendment 3 awaiting independent review (2026-11-23).** This source has not been independently approved for economic execution. The first BACKCAST attempt (digest `e0912608…`) refused before any exposure, return, drawdown or label. **A second BACKCAST under Amendment 1 (digest `7e276c4f…`, code `e0ac8908`) ran once and wrote a result labelled `TREND_BACKCAST_ONLY`** ([Issue #810 comment 7078583793](https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-7078583793)); that result was computed by the defective `_account` that Amendment 2 corrects. It is defective, is not evidence for the rule, is not relabelled and is not re-scored; the label `TREND_BACKCAST_ONLY` stays on record unchanged and plays no part in any decision. BACKCAST is consumed and is refused under every digest. The corrected protocol decides on FORWARD alone, and no FORWARD result exists. The FORWARD decision window starts 2026-12-01 (Amendment 3); the earlier 2026-11-23 start was never run or read.

Protocol digest: `0904174741f5ae7617999a06fa9ac4eb626c4152e2d3377f8054ca0b8dd694f9`

## Question

Exploratory screens on known-used Binance BTCUSDT spot daily data (2018-11 to 2026-10, issue #810 planner notes) found:
- A long/cash rule, "close above its N-day simple moving average", reaches Sharpe 0.87–1.26 for every N from 30 to 200.
- Macro, stablecoin and multi-coin variants did not improve on it.
- At full exposure its max drawdown is 38–65%, above the owner's 25% budget.

**Hypothesis:** an equal-weight ensemble of six lookbacks, vol-targeted to 20% annualized, keeps a positive and cost-robust return with drawdown inside the budget, on data never used for it.

**Counter-hypothesis:** the known-data result is a 2019–2024 regime effect, and the lookback set and the vol target were fitted to it.

**Cheapest originally proposed falsifier:** BACKCAST on 2015-01 to 2018-06 Bitstamp BTC/USD daily bars. That window is now consumed: BACKCAST ran once under Amendment 1 and its (defectively accounted) result is on record. Under the corrected accounting the only falsifier is the prospective FORWARD year (2026-12-01 to 2027-11-30).

**Prior exposure, disclosed:**
- The six lookbacks, the 20% target, the 0.05 deadband and the 30-day vol window were chosen after seeing known-used 2018-11 to 2026-10 results. On that data: CAGR about 21%, max DD 19–22%, Sharpe about 1.2.
- The planner has public knowledge of BTC history, including the 2017 bubble and the 2018 crash.
- `trend_long_spot_oos_v1` already used 2018-07 onward for a related spot trend family, which is why BACKCAST ends at 2018-06-30.
- That protocol's FORWARD interim reports exposed post-2026-08 spot data, which is why FORWARD here is strictly prospective.
- **Defective BACKCAST headline numbers (public prior exposure, labelled defective, not evidence):** the `7e276c4f…` result published in [Issue #810 comment 7078583793](https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-7078583793) showed total return +367.4%, max drawdown 11.8% and Sharpe 2.31 on 2015-01 to 2018-06. They came from the defective `_account` (free daily re-weighting), they decide nothing, and they are disclosed so that no reader treats the later FORWARD verdict as independent of them. [Issue #810 comment 7080608617](https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-7080608617) cited the then-current digest `196d6a4b…`, not the digest of the protocol that produced the result (`7e276c4f…`).

## Rule (frozen)

All quantities are computed at the close of UTC day t from closes up to and including day t.

- **Trend score:** s_t = (1/6) · Σ_{N ∈ {30, 50, 75, 100, 150, 200}} 1[close_t > SMA_N(close)_t], so s_t ∈ [0, 1].
- **Volatility:** σ_t = standard deviation of the last 30 daily log returns × √365. Use population standard deviation (ddof 0).
- **Target exposure:** e_t = s_t · min(1, 0.20 / σ_t), in [0, 1]. It is the fraction of equity in BTC; the rest is cash. There is no leverage and no shorting.
- **Deadband:** the held **target** changes only if |e_t − previous target| > 0.05, then it becomes e_t. A stale unchanged target is **not** permission to rebalance BTC units for free. The actual BTC fraction drifts with prices until the next accepted target change.
- **Execution:**
  - BASE: a changed target decided at close t is executed at close t using post-fee equity; exact BTC units and cash then remain unchanged until another accepted decision, accruing day t+1 price P&L. No daily percentage-weight reset occurs on HOLD days.
  - DELAY: a changed target decided at close t is first executed at close t+1 and held over day t+2, with the same quantity-HOLD and cost rules.
- **Cost:** one-way cost rate c × actual traded BTC notional |new BTC units − old BTC units| × execution close. After-fee equity E1 solves E1 + c × |target × E1 − old BTC notional| = pre-trade equity E0. The target is the fraction of E1. A changed signal causing no actual fill incurs no cost. Turnover is actual fill notional / E0, summed over fills; a held target never triggers a hidden trade.
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
| BACKCAST (**consumed, defective, refused under every digest**) | Bitstamp BTC/USD daily OHLC, public API `https://www.bitstamp.net/api/v2/ohlc/btcusd/?step=86400` | 2014-06-01 | 2015-01-01 to 2018-06-30 | none: historical record only, no part in any decision |
| FORWARD | Binance BTCUSDT spot daily klines, public API `https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d` | 2026-05-01 | 2026-12-01 to 2027-11-30 | decisive, prospective, the only open stage |

- **Daily bars:** a daily bar is the UTC day.
- **Source candle finality (Binance FORWARD only):** every raw 1d Binance kline must pass versioned `STRICT_SOURCE_CLOSE_V2` source-clock admission: native open/close timestamps in the same unit (ms or microseconds), aligned UTC midnight open, and exact native close `(normalized_open_ms + 86400000) * unit - 1`. Missing, shifted or mixed-unit source closes abort the page before daily close admission. **The admission verifies only that the native close field equals the scheduled end of the UTC day; it does not prove the candle is final.** Finality comes from the stage window: a run only loads days up to the last complete UTC day (`now` minus one day for an interim, the fixed last decision day for the final), so an in-progress candle is never a loaded day. Valid microsecond times are normalized to seconds only AFTER preserving microsecond close precision. Raw page bytes stay write-once and hashed even when validation fails. Bitstamp BACKCAST parsing and frozen historical evidence are unchanged.
- **Missing days and direct ledger input:** a missing day or a non-positive close refuses the run. No fill is applied. Both the trend ledger and BTC HOLD reference independently reject all non-positive or non-finite daily closes before account arithmetic; the HOLD reference may not divide by an invalid entry mark.
- **Fetched data:** raw responses are cached and hashed in the receipt.
- **FORWARD interims:** FORWARD interim reports, labelled `INTERIM_NOT_A_DECISION`, may run monthly from 2026-12-02 and before 2027-12-02. An interim refuses before any claim when no evaluated day is complete yet (the last complete UTC day is not after the first decision day 2026-12-01, so the first interim that can run is on 2026-12-03). An interim stops the line (`STOPPED_EARLY`) if `TREND__BASE` max DD is 25% or more; this rule is unchanged.
- **FORWARD final:** the final runs once, at or after 2027-12-02.
- **Freeze:** the protocol is frozen at the reviewed head (the digest above) before the first new FORWARD bar, the 2026-12-01 daily bar. The earlier 2026-11-23 FORWARD start was never run or read.

## Decision rule (frozen)

`SUPPORTED(FORWARD)` holds iff all of the following hold:

1. All cells are valid.
2. `TREND__BASE` total return is above 0.
3. `TREND__BASE` annual Sharpe is at least 0.5. Sharpe is computed from daily simple returns, population standard deviation, √365, with no risk-free rate.
4. `TREND__HIGH_COST` and `TREND__DELAY` total returns are above 0.
5. The max drawdown of every gating cell is below 25%.

The corrected protocol decides on FORWARD alone. The consumed, defective BACKCAST is a disclosed prior, not a condition.

| Label | Condition |
|---|---|
| `TREND_FORWARD_SUPPORTED` | `SUPPORTED(FORWARD)` at the final |
| `TREND_FORWARD_NOT_SUPPORTED` | otherwise, with all cells valid, at the final |
| `INVALID` | any cell invalid |
| `INTERIM_NOT_A_DECISION` / `STOPPED_EARLY` | interims only, as above |

Labels that mention BACKCAST (`TREND_SUPPORTED`, `TREND_BACKCAST_ONLY`, `TREND_NOT_SUPPORTED`) do not apply to this digest. The defective BACKCAST label `TREND_BACKCAST_ONLY` stays on record unchanged and plays no part in any decision.

A cell is invalid if, at any day:
- equity is not positive;
- exposure is outside [0, 1];
- the post-fee cash/filled-unit and mark-to-market accounting identity fails by more than 1e-9;
- the fee rate is nonfinite, negative, or at least 100%.


## Authorization

- FORWARD reads are covered by the owner scope relay [Issue #810 comment 7040247247](https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-7040247247).
- Each FORWARD stage also needs its own G4 relay on Issue #810 that quotes this protocol id, the digest and the full code commit. Earlier relays and reviews do not authorize this digest.
- The harness must fetch both comments twice. Each must be owner-authored and unchanged between the fetches.
- The checkout must be clean.
- A connectivity probe (one BTCUSDT 1d page dated 2026-10-01 to 2026-10-07, before 2026-11-01) and the authorization fetches run **before** the claim. Any failure refuses without claiming, so an outage cannot burn a stage, and no stage-window data is read by the probe. The probe page is known-used pre-FORWARD data covered by the owner scope relay [Issue #810 comment 7040247247](https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-7040247247); it is never scored.
- Outputs are write-once.

## Ledger and single use

- `LEDGER_DIR` is the protocol-specific ledger root. A per-digest folder `<root>/<digest>/` keeps provenance; **single use and the stop state are keyed by protocol id and stage (interims once per UTC month) across every digest.**
- Before any fetch, the harness refuses if the root already holds `<stage>.claim`, or if any digest folder holds that stage's claim or result, or if any digest folder holds a `STOPPED_EARLY` result (the line is stopped for every later interim and the final).
- The claim is created exclusively at the root first (the atomic lock across digests) and then per digest (provenance). The refused claim of digest `e0912608…` stays at the root.
- **Disclosed prior:** FORWARD requires that the consumed defective BACKCAST claim and result exist under `<root>/7e276c4f94647d19b42f761317ca971e86ef0542dc7ca9653e0675765cf1c937/` with label `TREND_BACKCAST_ONLY`. The record is read only to confirm the disclosure exists; its label decides nothing. No same-digest BACKCAST claim or result is required.
- **BACKCAST is refused** before any claim under every digest and every ledger state. There is no path that creates a new BACKCAST.

## Stop rule

- No change to any constant, lookback, target, deadband, cost, window or rule after any stage result.
- No rescue run.
- A refusal for a defect outside the economics needs a result-blind amendment before any rerun.
- **Governance exception (Amendment 3).** The no-change-after-a-stage-result rule has one stated exception: a **software defect in accounting, reproduced on synthetic data**, may be corrected by a reviewed result-blind amendment. The defective result is never relabelled or re-scored, and the amendment may not change any economic constant. Here the only stage still open is FORWARD, which has no data yet, so no result can be rescued. Basis: [Issue #810 comment 7080608617](https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-7080608617) (defect report) and [Issue #810 comment 7078583793](https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-7078583793) (the defective BACKCAST result).
- **Consequential decision-rule change (Amendment 3).** Correcting the accounting voids the only BACKCAST result as evidence, and BACKCAST can never run again. The original label rule `TREND_SUPPORTED = SUPPORTED(BACKCAST) ∧ SUPPORTED(FORWARD)` can then only be satisfied by counting invalid evidence, which is forbidden, or never, which leaves the protocol undecidable. Amendment 3 therefore removes the BACKCAST conjunct, and renames the labels to FORWARD-only names so that no later reader can mistake them for the original two-stage claim. This change is made before any FORWARD bar exists. It does not use the defective BACKCAST result in any way: neither its favourable value nor its label informs the change, and the outcome would be identical had BACKCAST failed. It weakens what a support label can claim (one prospective year instead of a backcast plus a forward year); it does not make support easier to reach on the remaining stage. No economic constant, cell, cost, window length or FORWARD condition changes. This is the only rule change permitted under this exception, and the stage G4 relay must quote and acknowledge it.
- Higher exposure is a new protocol.
- No label authorizes orders, paper trading or live trading.

## Amendment 1 (result-blind, after a refused BACKCAST attempt that produced no result)

Status: Active.

**The refused attempt.**
- The BACKCAST run authorized by [Issue #810 comment 7078446921](https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-7078446921), at code `220bd96b` and digest `e0912608`, refused during data validation with `missing UTC day 2014-06-01 (491 missing)`.
- No exposure, return or label was computed. Its ledger claim and its raw page `bitstamp_000.json` are kept.

**Cause.** Bitstamp's `ohlc` endpoint ignores `start` and returns the last `limit` bars at or before `end`. A probe on known-used 2025 data confirmed this: a 2025-01-01..01-10 request returned 1,000 bars from 2022-04-16 to 2025-01-09.
- The one page returned covered 2015-10-05 to 2018-06-30.
- The adapter treated it as complete because it reached the last day.

**Change.**
- Every page request spans at most `PAGE_LIMIT` days, `end = min(stage end, start + PAGE_LIMIT·86400 − 1)`, and rows outside `[start, end]` are dropped. The paging is therefore correct whether or not the API honours `start`. The same rule applies to Binance.
- The test fake now ignores `start` and returns the last `limit` bars before `end`, as Bitstamp does. The old adapter fails that test; the new one passes.
- **Disclosure:** the refused attempt fetched and parsed BACKCAST-window bars (page `bitstamp_000.json`, 2015-10-05..2018-06-30). The independent reviewer printed two raw closes from it while checking the anchoring. No exposure, return, drawdown or label was computed by anyone.

**Unchanged.** The rule, cells, costs, windows, decision rule and authorization are unchanged.

**Consequence.** This is a new digest. It needs a new independent review and a new G4 relay.
- **Ledger namespace:** the ledger is now keyed by protocol digest, `<ledger>/<digest>/<claim name>.claim`. Each digest gets one claim per claim name: one for BACKCAST, one per month for FORWARD_INTERIM, one for FORWARD_FINAL. A new digest exists only after a reviewed and relayed amendment. The refused claim of digest `e0912608` stays at the ledger root.
- **Cross-digest single use:** BACKCAST refuses before claiming if any digest namespace already holds a `BACKCAST.result.json`. A refusal that wrote no result does not block a reviewed amendment; a result of any label does.

## Amendment 2 (accounting correction, result-blind with respect to FORWARD; a defective BACKCAST result exists)

Status: HOLD pending a new review and G4 (superseded in part by Amendment 3).

**Exposure and result-blind scope.** The first BACKCAST attempt refused during missing-day validation and computed no return, exposure or label, but fetched and disclosed BACKCAST raw closes (2015-10-05 to 2018-06-30). **After Amendment 1, BACKCAST ran once under digest `7e276c4f…` and wrote a `TREND_BACKCAST_ONLY` result (result.json sha256 `e63486ce…`), seen by the planner.** It was computed with the defective `_account` described below, so it is not evidence for the rule; it is neither relabelled nor re-scored, and the cross-digest guard keeps BACKCAST from running again under any digest. Preserve that claim, raw page, historical digest `7e276c4f94647d19b42f761317ca971e86ef0542dc7ca9653e0675765cf1c937` and all earlier review evidence. The window is no longer wholly unseen. This correction uses only synthetic data; no additional market prices, measured performance or stage decision were inspected.

**Reproduced software defect.** The old `_account` did count trades, turnover and fees on target deltas (on the oracle below: one trade, turnover 0.5, fee on the target change). Its defect was that it re-weighted BTC units to the target weight after every daily price change, even when the accepted target did not change, for free: the implied daily trades had no fill and no fee. Synthetic prices `[100, 200, 100]` and targets `[0, 0.5, 0.5]` produced equity 1.125, while an actual one-entry quantity-HOLD cash book returns exactly 1.0. These are synthetic oracle values, not market performance results.

**Correction.** Change accepted target at the preceding close only; otherwise keep BTC units and cash fixed while the actual marked weight drifts. At a real rebalance, solve the post-fee target fraction by charging the frozen one-way fee rate on actual BTC fill notional. Turnover and fill count measure executed notional/units instead of only target-weight deltas. Independent high-precision Decimal cash/quantity/fee tests cover both buys and sells. Values outside the admissible fee-rate domain are invalid.

**Immutable parameters and authorization.** Keep the six lookbacks, volatility window/target, 0.05 decision deadband, 10/20 bp rates, daily timing, universe, study windows, drawdown/Sharpe rules and all previous G4/ledger records. Previously published known-used exploratory scores cannot be adopted as corrected-accounting performance. The protocol digest must change; earlier G4 relays and reviews do not authorize that new digest. Require fresh result-blind external G0–G3/source review and per-stage owner G4 before any new data read. No scientific rescue, FS1/unused/final data permission or live trading is authorized.

## Amendment 3 (result-blind, FORWARD window, prerequisite, labels and single use)

Status: HOLD pending independent review and G4. Review 6020387665 (BLOCKED at head `ff604e26`, digest `9e65e80a…`) is the trigger. No market data was read or computed for this amendment; the tests use synthetic data only.

**Governance basis.** See the stop rule: a reproduced software defect in accounting may be corrected by a reviewed result-blind amendment; the defective BACKCAST result is never relabelled or re-scored; the only open stage, FORWARD, has no data yet. Cited: [Issue #810 comment 7080608617](https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-7080608617) and [comment 7078583793](https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-7078583793).

**B1, FORWARD window.** FORWARD decisions run 2026-12-01 to 2027-11-30, warm-up loaded from 2026-05-01 as before. Monthly interims run from 2026-12-02 (first runnable 2026-12-03, see above) and are `INTERIM_NOT_A_DECISION`; the `STOPPED_EARLY` rule (`TREND__BASE` max DD at least 25%) is unchanged; the final runs once at or after 2027-12-02. The protocol is frozen at the reviewed head before the first new FORWARD bar. The digest had changed five times since the BACKCAST result (`7e276c4f`, `196d6a4b`, `3b80241e`, `236517b1`, `9e65e80a`) while the previous start 2026-11-23 was the review date; that start was never run or read.

**B2, prerequisite and labels.** Removed: the same-digest BACKCAST claim and result requirement (it could never be met, because BACKCAST is refused under every digest). Added: the disclosed-prior requirement and the label set `TREND_FORWARD_SUPPORTED`, `TREND_FORWARD_NOT_SUPPORTED`, `INVALID`. BACKCAST remains refused; no new BACKCAST and no new path to one.

**Single use and stop state.** Keyed by protocol id and stage across all digests, with an exclusive-create root claim, a cross-digest check, authorization and connectivity pre-checks before the claim, and a probe dated before 2026-11-01 (the same pattern as the single-use code in open [PR #1335](https://github.com/shuntatsu/trade_rl/pull/1335), not yet on main).

**M3, L5, L6.** The description of the old `_account` defect is corrected in Amendment 2; source-candle finality is stated as coming from the stage window; the defective BACKCAST headline numbers are disclosed as prior exposure.

**Unchanged.** The rule, cells, costs, deadband, lookbacks, volatility target, decision conditions 1 to 5 and the accounting of Amendment 2. The protocol digest changes; earlier reviews and G4 relays do not authorize it.


## 2026-11-24 software admission amendment (Issue #1347; prospective only)

The single-use stage ledger MUST resolve to one absolute filesystem location before any
claim, owner-relay fetch, exchange connectivity probe, or market request. The legacy
Windows `C:\trade_rl_runs\...` default remains unchanged on Windows, but is NOT
an absolute path on POSIX: fail closed there unless the caller provides a verified
absolute ledger directory. Existing claims and consumed BACKCAST history are preserved;
no cross-machine or shared-filesystem coordination is asserted.

Interim claim months and stage-clock comparisons use the same timezone-aware instant
normalized to UTC. Naive datetimes are refused before side effects. This closes the
local-CWD/UTC identity bypass without changing any strategy, indicator, cost, stage
window or result decision rule. Re-pin the spec/harness digest and repeat independent
software-review and Full CI gates on this prospective source; previous-head receipts
are not reusable. **No economic stage, G4 relay grant, market-data use, or trading is
authorized by this amendment.** Separate per-stage G4 relay binding and relay-comment
ID replay prevention remain OPEN under Issue #1347.
