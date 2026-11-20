# Out-of-sample test of TS_TREND_LONG_SPOT (`trend_long_spot_oos_v1`)

Status: Active. This protocol was frozen before any data in its windows was fetched or read for this rule.

## Question

`TS_TREND_LONG_SPOT` from `lowturn_family_screen_v1`
([Issue #810 comment 7028195515](https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-7028195515))
met every candidate condition on known-used Tier D data:

| Segment | Return | Sharpe |
|---|---|---|
| DEV | +21.6% | 0.66 |
| HOLDOUT | +22.8% | 0.75 |

The screen label was `INVALID` because two control cells held terminal dust (comment 7028336366). The deflated Sharpe was about 0.51, so the candidate is not distinguishable from selection luck.

**Hypothesis:** the unchanged rule keeps a positive, cost-robust, better-than-passive risk-adjusted return on data that no study in this repository has used. The rule is long spot when the 30/90/180-day trends agree, flat otherwise.

**Counter-hypothesis:** the 2022–2026 result is a selection artefact of one bear-then-bull cycle.

**Cheapest falsifier:** the BACKCAST stage below. It needs no waiting.

## Authorization and data

- **Owner authorization:** [Issue #810 comment 7028416072](https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-7028416072). It covers out-of-sample and forward reads, and fetching public data not held yet. It does not cover orders, API keys or FS1 scoring.
- **Data source:** Binance public spot klines (Vision archives, checksum-verified). Missing hours may only be filled from checksum-verified daily archives, with provenance recorded. There is no forward fill.
- **Data-coverage amendment (BACKCAST, result-blind).**
  - A coverage run (`python -m tools.trend_oos coverage`) read the BACKCAST archives and computed no signal, replay or return; it read prices only to validate rows. It found two data problems, and both were fixed after it and before any run. No economics changed: rule, windows, costs, thresholds and the decision rule are as above.
  - **(A) Unusable monthly archives.**
    - Some monthly archives are unpublished or fail the strict row parse (for example one unaligned row in 2018-02; the 2020-12-21 14:00 outage placeholder in the 2020-12 archives of symbols beyond the six first documented).
    - A month whose monthly archive is unpublished or malformed (never a checksum fault) is rebuilt from checksum-verified daily archives under the same strict parse.
    - Provenance is recorded per day (archive URL and SHA-256).
    - A day that is unpublished or malformed stays missing.
    - A leading month whose monthly archive is unpublished is checked for a daily archive (its last day). If one exists the run refuses, because the listing would otherwise silently move to a later month.
    - The documented placeholder bar is recognised for the further symbols in which the coverage run observed it; the predicate still demands every exact condition.
  - **(B) Exchange-wide outages.**
    - After (A), the ranges of hours that every symbol listed at that time is missing are declared in `trend_oos/protocol.py` as a frozen halt list. The SHA-256 of the coverage report they were read from is frozen with the list, so both enter the protocol digest.
    - The frozen list has 20 ranges totalling 108 hours, from 2018-01-04T04:00Z to 2020-12-25T03:00Z, read from the coverage report with SHA-256 `8680f82140fc6ce0b4c19076acebb4031b956bba52ac3940d9dc911a734c3d49`, produced at code commit `e9dc2eeda508cd7f329897ae08573e29bda2eb62` under protocol digest `535a25162edd843687a845e920d178b0ea7af184eb8fd6968202e5222bc3a1bb` (both recorded in `trend_oos/protocol.py` and in the data receipt); the ranges themselves are in `trend_oos/protocol.py`.
    - A declared halt is treated exactly as the registered Tier D `SPOT_HALTS`: the affected 4h rows are not tradable, carry zero volume and take the preceding published close as their mark, and the signals use them as lowturn does for halt rows. This is the only carried-forward close the protocol permits.
    - A range that is not exchange-wide does not qualify. Any hour missing outside the declared halts refuses the run.
  - **Published zero-volume hours** (for example 2019-06-07 21:00Z in several symbols) are not missing and are not halts. Their bars stay tradable, but they have no participation capacity: with `processing_bar_volume_capacity=False` the capacity of an order is 1% of the volume of the previous completed bar, and a zero volume gives zero capacity, so no fill can follow it. A synthetic test pins this. A bar that sums only some zero-volume hours keeps the capacity of its remaining volume.
- **Legs:** the same 11 pairs, in the same roster order, as the Tier D dataset.
- **Prior exposure:**
  - Every economic study in this repository, including the agy exploration Rounds 1–12, used data from 2021-01-01 onward.
  - The only earlier touch is a 2020-12-21 placeholder-row check, which carries no economic content.
  - The planner has general public knowledge of crypto market history (the 2018 bear market, the 2020 crash). This is disclosed, and it cannot be removed.

| Stage | Load start (warm-up) | First decision | Last decision bar | Settles |
|---|---|---|---|---|
| BACKCAST | 2018-01-01 | 2018-07-01T00:00Z | 2020-12-31T20:00Z | 2021-01-01T00:00Z |
| FORWARD | 2026-02-01 | 2026-08-01T00:00Z | 2027-04-30T20:00Z | 2027-05-01T00:00Z |

- **BACKCAST:**
  - The warm-up covers the 1080-bar (180-day) lookback.
  - A pair whose spot market did not exist yet, or whose inputs are not yet available, gets weight 0. This follows the frozen rule.
- **FORWARD:**
  - The final scoring run happens at or after 2027-05-02T00:00Z.
  - Interim reports (see below) are allowed before that.
  - The window coincides with the FS1 window. FS1's frozen arms and its scoring are untouched.

## Arms (the lowturn v1 policy, unchanged, on a spot cash account)

- **Candidate:** `TS_TREND_LONG_SPOT` on a spot cash account (see the amendment below). It is not "lowturn v1 `TS_TREND_LONG_SPOT`".
- **Controls:** `CASH` and `EW_HOLD_SPOT`.
- **Unchanged from lowturn v1:**
  - lookbacks, vol60, vol target 0.50, the per-11 division, caps (0.40 per name, gross 2.0);
  - weekly rebalance on `(index − first_decision_index) % 42 == 0`;
  - deadband rule 0.02, the pre-image rule and the risk config (0.10→0.20 ramp);
  - BASE, HIGH_COST and DELAY_1BAR costs;
  - `run_shared_portfolio_replay` with settlement and ledger capture.
- **Spot cash account (amendment after review 5988926816, made result-blind before any BACKCAST run).**
  - **Why:** lowturn v1 allowed a spot gross of up to 2.0 (and the replay up to 2.5), while the engine finances negative cash at 0% (it charges borrowing only on shorts and the stage dataset sets no cash or borrow rate). A gross above 1.0 was therefore free leverage in every scenario, which would flatter the BASE, HIGH_COST and DELAY_1BAR results.
  - **What changes:** both stages use a spot cash account with no borrowing. For every arm, after the frozen lowturn caps, if the spot gross target exceeds 1.0 the target is scaled proportionally to gross exactly 1.0. This changes the candidate from "lowturn v1 `TS_TREND_LONG_SPOT`" to "the same rule on a spot cash account". The lowturn v1 result itself (the DEV and HOLDOUT screen) is not relabelled.
  - **How:** a thin wrapper in `trend_oos/cash_policy.py` around the unchanged `LowturnPolicy`; no lowturn code is touched. The cap is applied to the effective target, meaning held legs at their current weight and every other leg times the risk scale, so risk applies the scale once. When the cap binds, every exact-quantity hold is dropped and the whole vector is re-targeted, so the 0.02 deadband cannot let gross creep above 1.0. A proposal that does not trade (a pure hold, or the pre-image of an unchanged risk scale) is never capped, because it cannot change cash.
  - **Invalid on borrowing:** a cell is `INVALID` with reason `negative_cash_borrowing` if, at any ledger interval, its negative cash net of the execution costs paid so far exceeds 1e-6 of the initial capital. The engine debits costs from cash, so a book fully invested at gross 1.0 (such as `EW_HOLD_SPOT`) sits at minus its costs; that overdraft is the cost of the trade and not a loan, and it is not counted.
  - **Reported, not gating:** per cell, the fraction of rebalance decisions where the 1.0 cap bound, the mean and maximum pre-cap gross, the minimum cash and the maximum borrowing.
- **Cell validity:** a cell is valid when its ledger reconciles, its causality probe passes, it is not insolvent, it does not borrow (above), and its terminal book is flat **or** consists only of lean-core terminal dust. Dust is a residual whose terminal close the engine rejected for `below_minimum_notional`. It is valued at mark minus the modelled exit cost, and that cost is deducted from the return. This is the rule fixed after the lowturn diagnosis. It applies to this protocol from the start.
- **Causality probe:** it runs once per (arm, signal lag) and is shared by the cost scenarios of that lag (BASE and HIGH_COST share lag 0, DELAY_1BAR has lag 1), because the probe decisions come from synthetic account paths that do not depend on costs. Each cell's JSON records the probe key and that it is shared.
- **Reported returns:** the per-year returns and the bootstrap use the dust-adjusted return series (the last return carries the terminal dust exit cost), so they agree with the cell's total return. The annual Sharpe is the unchanged core7 summary value, which core7 also computes from the same dust-adjusted series (`core7_mtf_study/runner.py`), as are its max drawdown and stop check.
- **Data that perp-leg code paths need:** if the implementation reuses the 22-leg layout, it marks perp legs unavailable where they do not exist. Neither arm reads them.

## Decision rule (frozen)

Stage S is BACKCAST or FORWARD. `SUPPORTED(S)` holds iff every condition below holds:

1. All nine cells (3 arms × 3 scenarios) are valid.
2. No candidate cell reached max drawdown ≥ 0.20.
3. The candidate's BASE total return is above 0.
4. The candidate's BASE annual Sharpe is at least 0.30.
5. The candidate's HIGH_COST and DELAY_1BAR total returns are both above 0.
6. The candidate's BASE Sharpe is above `EW_HOLD_SPOT`'s BASE Sharpe.

**Labels:**
- `OOS_SUPPORTED`: `SUPPORTED(S)` holds.
- `OOS_NOT_SUPPORTED`: all cells are valid but a condition fails.
- `INVALID`: any cell is invalid.

**Reported but not gating:**
- per-calendar-year returns;
- turnover, cost and max drawdown;
- a one-sided 95% stationary block bootstrap (block 42 bars, 10,000 resamples, seed 20261120) lower bound of the mean BASE excess return over CASH, and of the mean over `EW_HOLD_SPOT`.

## Authorization of each run

- Each stage run needs its own owner G4 comment on Issue #810 that quotes this protocol's id and its digest as whole tokens. The data authorization above (comment 7028416072) predates the digest and quotes neither, so it cannot serve as the G4 comment.
- A FORWARD run also fetches comment 7028416072 as the repository owner and re-fetches it immediately before the run; a changed body or edit time refuses the run, as for the G4 comment.

## Sequencing and stop rule

1. **BACKCAST runs once.**
   - If it is `OOS_NOT_SUPPORTED` or `INVALID`, the `TS_TREND_LONG_SPOT` line ends. No FORWARD scoring is done, and the rule is not changed.
   - If it is `OOS_SUPPORTED`, FORWARD proceeds.
2. **FORWARD interim reports** may be run monthly as `INTERIM_NOT_A_DECISION`. They print the same metrics with no label, and they never stop, extend or alter the protocol.
3. **The FORWARD final scoring** runs once, at or after 2027-05-02T00:00Z.
4. **No parameter, sizing, leverage, symbol, window or cost change in any stage.** Any variant, such as higher leverage, is a new protocol that needs its own windows.
5. **Claim ceiling:**
   - BACKCAST alone: `OOS_SUPPORTED`.
   - BACKCAST and FORWARD both supported: `OOS_AND_FORWARD_SUPPORTED`.
   Neither label authorizes orders or live trading.
