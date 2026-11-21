# Realistic agy anchor out-of-sample test (`agy_anchor_realistic_v1`)

Status: **Historical v1 / economic execution HOLD (2026-11-21)**. This protocol
was frozen before its original run. The original DEV and BACKCAST_2020 receipts
and source/digest bindings remain immutable. Since the cash-cap and shared
fee-credit/borrowing implementation were subsequently corrected, the new source
must not execute further economic stages under the unchanged
`agy_anchor_realistic_v1` identity. Ordinary `tools.agy_anchor_oos run` and
`coverage` refuse before clock, G4, files or data access. The only callable stage
paths now use explicitly injected synthetic test fixtures. A new real experiment
requires a separately named protocol, genuinely eligible window, result-blind
research review and its own authorization. This HOLD does not retroactively
alter or upgrade the earlier INVALID BACKCAST_2020 outcome.

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
- **Authorization:** BACKCAST_2020 and FORWARD reads are covered by the owner scope relay [Issue #810 comment 7040247247](https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-7040247247), which covers fetching and reading data for any pre-registered out-of-sample or forward protocol (Amendment 5; it was owner comment 7028416072). Every such read binds that comment (owner-authored, unedited between two fetches) and records it in the receipt. DEV reads known-used data and needs no relay of its own. Each run also needs its own G4 relay quoting this protocol id and digest.

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

## Amendment 3 (result-blind, before any engine replay or out-of-sample read)

Status: Active. Reason: the engine's booking of perpetual legs through cash was discovered during implementation, before any result of this protocol existed (no stage has been run and no replay of the ported anchor has produced a number).

- The engine books every leg, perpetuals included, through cash like spot: a fill sets `cash_after = cash - (signed_notional + cost)` over all legs (`trade_rl/simulation/accounting.py:820`) and the book is valued as `cash + sum of signed position values` (`accounting.py:387`). So cash is about `capital * (1 - sum of signed weights)` minus costs. The Amendment 1 cap (a) alone does not keep cash non-negative: a net-long book can sit inside (a) and still borrow.
- The cash-account scale therefore enforces both: (a) spot long gross + 0.4 x perp gross <= 1.0, and (b) the net signed exposure, the sum of all signed weights, <= 0.97 (Amendment 4; it was 1.0), so that cash stays >= 0 before costs. The whole target is scaled proportionally by the tighter of the two factors. A carry-like book (spot long, perp short) has a net near zero and is not affected by (b).
- The negative-cash INVALID check is unchanged (cash net of cumulative execution costs below -1e-6 x capital).
- It is reported, never gating, how often each bound was the binding one (margin, net exposure or both).
- Everything else is unchanged. A proposal that does not trade is never capped. The risk layer can only shrink a book and the anchor's per-name cap (0.55) is applied before this scale, so risk clipping does not undo (b); a residual breach would still be caught by the unchanged check and make the cell INVALID.

## Amendment 4 (result-blind, before any engine replay or out-of-sample read)

Status: Active. Made before any result of this protocol existed.

- The net-exposure bound (b) of Amendment 3 is 0.97, not 1.0. Orders are sized at the decision close and filled at the next bar, so a book at net exactly 1.0 overshoots cash by the price gap plus costs. In a synthetic check at exactly 1.0 the entry fill overshot by about 0.009% of capital, which exceeds the 1e-6 negative-cash tolerance. A 3% buffer covers normal 4h-boundary gaps on a 24/7 market.
- The margin bound (a) stays at 1.0. The negative-cash tolerance (1e-6 x capital, net of cumulative execution costs) is unchanged: any cell that still borrows is INVALID, as the spec intends.
- Everything else is unchanged.

## Amendment 5 (result-blind, before any engine replay or out-of-sample read)

Status: Active. Made before any result of this protocol existed, in response to a report-only pre-review of the harness. It changes no signal constant, stride, cost, risk value, universe or window.

- **Authorization scope.** The data authorization is the owner scope relay [Issue #810 comment 7040247247](https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-7040247247). It is bound for every BACKCAST_2020 and FORWARD read (coverage and run), must be owner-authored and unedited between the first and the last fetch of a run, and is recorded in the coverage report and the data receipt. The dataset identity of those stages carries it.
- **Digest binds the harness.** The protocol digest includes `harness_source_digest`, a SHA-256 over `protocol.py`, `cash.py`, `decision.py`, `data.py`, `runner.py` and `tools/agy_anchor_oos.py`, built like the port digest (name and LF-normalised bytes). It is computed, not pinned, because `protocol.py` is one of the hashed files. `protocol_digest()` also verifies the parity fixture binding. Any edit to the harness is therefore a new digest and needs a new G4 relay.
- **Controls share the cash account.** The controls `CASH` and `EW_HOLD_SPOT` run in the same cash-account wrapper as the anchor (margin bound 1.0 and net-exposure bound 0.97 of Amendments 1, 3 and 4). They are spot-only, so both bounds reduce to their spot gross; the earlier spot-gross-1.0 wrapper would have exposed them to the same entry-gap overshoot Amendment 4 removes from the anchor. Held legs count at their current weight in the net-exposure bound (the effective vector is what is bounded). A leg that is halted or invalid cannot be traded, so a held weight above the bound cannot always be reduced; such a breach is caught by the unchanged negative-cash check.
- **DEV may stop the line.** BACKCAST_2020 is refused unless the DEV result has no invalid cell. An invalid DEV cell means the port or the harness is broken; it may stop the line and never retunes anything.
- **Activation transient (disclosed, not removed).** The anchor's hybrid alphas demean across all eleven symbol columns (`all_ones`, as in the lab). A pair that is not yet available is a zero column in those demeans, and a pair that has just become available enters them with windows that still contain the neutral pre-activation values. Excluding a pair from the demean until its basis window is fully real would need a change inside the port, which is bound by the parity fixture and a pinned digest; it was not done. The 180-bar pair warm-up exceeds the 42-bar basis mean and the 84-bar basis deviation, so those windows are real by the time a pair is available.
- **Cold EMA seed residual (disclosed).** The exponential averages of the chain start cold at zero (late pairs at their activation, early pairs at the first row) and carry the seed forward. With the 126-bar half-life of the carry average, 0.5^(180/126) = 37% of the seed weight remains when a late pair becomes available, and 0.5^(540/126) = 5% at the first decision of an early pair. Shorter averages are flushed. This residual is part of the protocol.
- **FORWARD warm-up is known-used data.** FORWARD loads from 2026-02-01 and decides from 2026-08-01. The rows of 2026-02-01 to 2026-07-31 are the Tier D period, known-used pre-2026-08 data; they warm the signal and are never scored. Only decisions from 2026-08-01 are out of sample.
- **Strides.** The anchor policy of the port keeps the lab default (6, daily); the harness always passes 18 explicitly and a test pins it. The controls keep lowturn's weekly rebalance.
- **Drawdown wording.** Condition 2 reads "max drawdown of 0.25 or more", as amended; any remaining 20% in the code comments refers to the shared core7 helper.

## Amendment 6 (result-blind, before any engine replay or out-of-sample result)

Status: Active. Made from coverage reports only. Coverage builds no signal, no replay and no return, so no result of this protocol exists.

- **BACKCAST_2020 declared halts.** `halts-from-coverage` on the BACKCAST_2020 coverage report gave 8 exchange-wide spot outages (18 hours), and they are frozen in `protocol.DECLARED_HALTS`:
  - 2020-02-09 02–03
  - 2020-02-19 12–17
  - 2020-03-04 10–11
  - 2020-04-25 02–04
  - 2020-06-28 02–05
  - 2020-11-30 06–07
  - 2020-12-21 14–18
  - 2020-12-25 02–03
  
  All times are UTC, with half-open hour ranges.
- **Coverage report.** Coverage ran at code `bed9c8de34342527fec55b460e1244e2b6bbc414` under digest `ce4691446c89d4851434b55bc52bb7e8d9da9b34e4cdfc2e042c7ed6d0b58979`, and the report's SHA-256 `fb8104aea3987be4ab5a4caa975b9f05f6135cb992955aa69073c3f6c29966eb` is frozen as `COVERAGE_EVIDENCE_SHA256`. The report shows no gap that would remain and refuse the run, and no unread chunk.
- **FORWARD.** The FORWARD coverage report (SHA-256 `4752e862ff0e7b40429cd4411133e3ac049d4316a68e70950f94b8959457054c`) found 0 exchange-wide spot gap hours, so nothing is declared for it.
- **Treatment.** The halts are treated exactly like the registered Tier D `SPOT_HALTS`. Any other missing hour still refuses the run.
- **New digest.** Because `protocol.py` is part of the harness digest, this is a new protocol digest and needs a new G4 relay. Nothing else changes.

## Amendment 7 (result-blind, review follow-up before any engine replay or out-of-sample result)

Status: Active. These changes follow review [6001946862](https://github.com/shuntatsu/trade_rl/pull/1304#pullrequestreview-6001946862). They change no signal, cost, risk value, universe, window or decision rule.

- **Disclosure text.** The manifest's cash-account disclosure now states the net signed exposure bound actually enforced, `<= 0.97` (Amendment 4). It previously said 1.0.
- **Coverage re-confirms the data authorization.** `coverage` re-fetches the data-authorization relay after its last read and refuses, writing nothing, if the relay was edited. This is the same check `run` already makes, so Amendment 5's "unedited between the first and the last fetch" now holds for coverage too.
- **`halts-from-coverage` checks provenance.** It refuses a report that was not produced from a clean tree. It reports whether the report's SHA-256 matches the frozen `COVERAGE_EVIDENCE_SHA256`. The frozen BACKCAST_2020 report matches.
- **Line endings.** The branch's files are normalised to LF. The harness digest is built from LF-normalised bytes, so normalising alone does not change it; the edits above do.

## Amendment 8 (made after a refused DEV attempt that produced no result)

Status: Active.

- **The refused attempt.** The DEV run authorized by [Issue #810 comment 7055512104](https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-7055512104), at code `ad696006` under digest `f4369159`, refused before any signal, replay or return was computed. The error was `a 4h bar carries two or more funding events`. It wrote only its data receipt. Its output directory is kept, and the next run writes to a new directory.
- **Cause.** Binance shortened SOLUSDT's funding interval to 2 hours in November 2022, so 37 of its 4h bars hold two events. The other ten DEV perps keep 8-hour funding throughout. The harness derived events from the 4h dataset rows, and that path refuses such a bar by design.
- **Change.** The harness now passes the exact hourly perp funding events to the port, stamped at the close of their hour. The port's documented `funding_events` path counts every event as the lab does from its funding rows.
  - An hour with more than one event refuses the run.
  - The causality probe perturbs the events after its cut as well.
- **What is unchanged.**
  - Where every bar holds at most one event, the count-based funding observations are identical to before, and a test pins this.
  - The engine's funding accounting still sums a bar's events.
  - The count-based means still start cold at the perp's first row.
  - Signal constants, costs, risk, universe, windows and the decision rule are unchanged.
- **Consequence.** This is a new digest, which needs a new review and a new G4 relay.
