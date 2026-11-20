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
    - The documented placeholder bar is recognised for the further symbols in which the coverage run observed it; the predicate still demands every exact condition.
  - **(B) Exchange-wide outages.**
    - After (A), the ranges of hours that every symbol listed at that time is missing are declared in `trend_oos/protocol.py` as a frozen halt list. The SHA-256 of the coverage report they were read from is frozen with the list, so both enter the protocol digest.
    - The frozen list has 20 ranges totalling 108 hours, from 2018-01-04T04:00Z to 2020-12-25T03:00Z, read from the coverage report with SHA-256 `8680f82140fc6ce0b4c19076acebb4031b956bba52ac3940d9dc911a734c3d49`; the ranges themselves are in `trend_oos/protocol.py`.
    - A declared halt is treated exactly as the registered Tier D `SPOT_HALTS`: the affected 4h rows are not tradable, carry zero volume and take the preceding published close as their mark, and the signals use them as lowturn does for halt rows. This is the only carried-forward close the protocol permits.
    - A range that is not exchange-wide does not qualify. Any hour missing outside the declared halts refuses the run.
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

## Arms (unchanged code from lowturn v1, commit `7fb05514`)

- **Candidate:** `TS_TREND_LONG_SPOT`.
- **Controls:** `CASH` and `EW_HOLD_SPOT`.
- **Unchanged from lowturn v1:**
  - lookbacks, vol60, vol target 0.50, the per-11 division, caps (0.40 per name, gross 2.0);
  - weekly rebalance on `(index − first_decision_index) % 42 == 0`;
  - deadband rule 0.02, the pre-image rule and the risk config (0.10→0.20 ramp);
  - BASE, HIGH_COST and DELAY_1BAR costs;
  - `run_shared_portfolio_replay` with settlement and ledger capture.
- **Cell validity:** a cell is valid when its ledger reconciles, its causality probe passes, it is not insolvent, and its terminal book is flat **or** consists only of lean-core terminal dust. Dust is a residual whose terminal close the engine rejected for `below_minimum_notional`. It is valued at mark minus the modelled exit cost, and that cost is deducted from the return. This is the rule fixed after the lowturn diagnosis. It applies to this protocol from the start.
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
