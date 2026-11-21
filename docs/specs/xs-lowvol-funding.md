# Cross-sectional low-volatility and funding-contrarian perp book (`xs_lowvol_funding_v1`)

Status: Active (not frozen until an independent review approves the exact head). This protocol is frozen before any perp kline or funding data of its DISJOINT or FORWARD universe has been fetched or read. Freezing happens when the protocol digest is pinned and an independent review approves the exact head.

Protocol digest: `8e4ba3d4429730691aca01ca7be21b9b3a1dd681a278889507bf2fa44012453b`

## Question

Diagnostics on known-used Tier D data (the 11 core pairs, 2022-01 to 2026-07; [Issue #810 comment 7046210397](https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-7046210397)) found that the agy anchor's alpha was mostly funding and basis reversion, and that it decayed in 2025–26. Only two cross-sectional effects held in both eras:

- **Low volatility.** Names with lower trailing 30-day volatility had higher relative 3-day returns. The rank IC was about −0.08 before 2025 and −0.13 after.
- **Funding contrarian.** Names with higher recent funding had lower relative returns.

A learning-to-rank diagnostic also attributed most of its test IC to the low-volatility feature.

**Hypothesis.** A weekly perp book that is long low-volatility, low-funding names and short high-volatility, high-funding names earns a positive, cost-robust net return. It should do so on a liquid universe of symbols that no study in this repository has touched.

**Mechanism chain (G1).**

1. *Lottery demand.* Retail and leveraged demand for high-volatility alts overprices them; this is the low-risk / MAX effect. Leverage-constrained participants cannot arbitrage it away.
2. *Crowding.* High funding marks crowded leveraged long positioning, which tends to unwind. The short side of the book also collects that funding.
3. *Structural premium.* Both effects are premia for providing liquidity and leverage to constrained or lottery-seeking flow, not information edges. So they can persist beyond the 11 majors, where retail participation is even higher.
4. *Risk balance.* Inverse-volatility weighting gives each name an equal stand-alone risk. This keeps the book roughly beta-balanced, so it is not simply short the market, and it keeps short notional small on the most explosive names.

**Counter-hypothesis.** The diagnostics searched about 30 features on 11 large caps. The two survivors may be selection luck, or properties of majors only. On a broader alt universe, the short leg of high-volatility alts may be squeezed (alts pump), delistings may hit, and wider spreads may erase the edge.

**Cheapest falsifier.** The DISJOINT stage below. It uses only archived data and needs no waiting.

## Prior exposure (disclosed)

- The two effects were found by known-used diagnostics. Their feature search, about 40 trials counting the anchor-decay, meta-labeling and LTR scripts, is counted as prior trials. It enters the reported deflated Sharpe.
- No perp kline or funding value of any symbol in `trade_rl/evaluation/xs_lowvol_funding/candidates.json` has been read by any study in this repository.
  - The scoping note read only S3 object names and sizes, and cached exchangeInfo metadata (onboard dates, status).
- The calendar window (2021–2026), the crypto-wide beta and the macro regimes are the same ones the prior studies saw. DISJOINT therefore tests generalization to other symbols, not time out-of-sample. Time out-of-sample is the job of FORWARD.
- The planner has general public knowledge of crypto market history (2021 alt season, LUNA, FTX). This is disclosed and cannot be removed.

## Authorization and data

- **Owner authorization.** The owner scope relay, [Issue #810 comment 7040247247](https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-7040247247), covers out-of-sample and forward reads and fetching public data not held yet. It does not cover orders, API keys or FS1 scoring.
- **Data source.**
  - Binance USD-M perpetual 1h klines and monthly funding-rate archives from Vision, checksum-verified. These are the same fetch, parse and daily-fallback paths as `agy_anchor_oos`.
  - A malformed or unpublished monthly kline archive is rebuilt from checksum-verified daily archives, with per-day provenance.
  - There is no forward fill.
- **Clock.** 4h bars aggregated from 1h bars, timestamped at bar close, as in Tier D. Funding events are booked at their funding time.
- **Spot data.** No spot data is read. The book trades perps only.

### Candidate list (frozen)

`trade_rl/evaluation/xs_lowvol_funding/candidates.json`, SHA-256 `fc680208899cd4166c9923f9cf9e8f8ebe829010aae073a935de10d6e6fefdf6`, holds 441 USDT perpetual symbols. It is the scoping note's 450-pair fetch superset with these removed:

- the 16 touched symbols (BTC ETH BNB SOL XRP DOGE ADA AVAX LINK LTC DOT BCH SUI APT ARB OP);
- TRX (a conservative exclusion);
- the six 1000x perps whose spot base differs (a unit mismatch on the spot side, removed by the scoping). 1000CAT, 1000CHEEMS, 1000SATS and 1MBABYDOGE remain: the book is perp-only and returns are scale invariant, so price scaling does not matter;
- every symbol that is not ASCII `[A-Z0-9]+USDT` (one was removed), because the Vision path handling for such symbols is untested.

The list includes 24 pairs that were later delisted. It was built from archive listings, not from the current exchangeInfo, so delisted pairs are present.

### Monthly universe (point-in-time)

On each calendar month start M, a symbol is **eligible** when all of the following hold:

1. its perp has a complete set of 1h bars for the 90 days strictly before M, or the only missing hours lie inside a declared exchange-wide halt;
2. it has at least 9 settled funding events in the 3 days before M.

The **universe** for month M is the 30 eligible symbols with the largest perp quote volume over the 30 days strictly before M. Ties are broken by symbol name. Membership is fixed for the month and uses no information from M or later.

If fewer than 12 symbols are eligible, the month holds no book and the CANDIDATE is flat.

### Missing bars and delisting while held

- A held name is closed when its perp archive ends or a gap starts that no declared halt explains.
  - Its target is forced to 0 from the decision bar **42** 4h bars (7 days) before its last available bar onward. This models Binance's roughly one-week advance delisting notice.
  - The 42-bar lead uses knowledge of the archive end. It models the public delisting announcement and is disclosed as such.
  - Each fill of the forced close is charged the scenario cost plus an extra adverse **5%** of the closed notional. The charge is booked as a cash debit at that fill bar.
- There is no write-off branch. If any position of such a name is still held at its last available bar (no capacity, below minimum notional), the run refuses, naming the symbol.
  - This is a disclosed limitation: the shared replay cannot continue past an engine settlement of a held position, and handling it would need an amendment.
- A name with a gap inside its trailing 90 days is ineligible for that month.
- It is reported, never gating, how many forced closes occurred, their total charge, and the result without them.

### Declared halts

Exchange-wide perp outages are declared result-blind from a coverage report, as in `agy_anchor_oos`. `coverage` reads archives and validates rows. It computes no signal, replay or return. A range qualifies only if every symbol listed at that time is missing it. Any other missing hour of an eligible, held or ranked symbol follows the rules above.

## Stages

| Stage | Universe | Load start (warm-up) | First decision | Last decision bar | Settles |
|---|---|---|---|---|---|
| DEV | the 11 Tier D core pairs (known-used) | 2022-01-01 | first Monday on or after 2022-04-04 | 2026-07-27T00:00Z | 2026-08-01T00:00Z |
| DISJOINT | `candidates.json`, monthly top 30 | 2021-01-01 | 2021-04-05T00:00Z | 2026-07-27T00:00Z | 2026-08-01T00:00Z |
| FORWARD | all USDT perps in the Vision um listing at run time, minus stablecoin and gold bases and non-ASCII symbols, monthly top 30 (touched symbols included) | 2026-04-01 | 2026-08-03T00:00Z | 2027-04-26T00:00Z | 2027-05-01T00:00Z |

- **DEV** checks the harness on known-used data and is not evidence. With 11 names, each leg holds `floor(11/3) = 3` names.
  - DISJOINT is refused unless DEV has no INVALID cell.
  - DEV may stop the line, for a broken port or harness. It never retunes anything.
- **DISJOINT** is the primary falsifier. Its reads stop strictly before 2026-08-01 (the Pre-FS1 embargo).
- **FORWARD**
  - Decisions before 2026-08-01 are warm-up only.
  - Interim reports are allowed monthly. They report the running numbers and are never a decision.
  - The final run happens at or after 2027-05-08T00:00Z, after the 2027-04 funding archive is published.
  - Its reads go through the exact data-authorization relay above.

## Policy (frozen)

Decisions are made weekly, at the 4h bar closing Monday 00:00 UTC. Orders fill at the next bar. Between decisions, quantities are held and weights drift.

1. **Inputs** for each universe member at decision time t:
   - `sigma_30d`: the standard deviation of the last 180 4h log returns.
   - `f_3d`: the mean of the last 9 settled funding rates with funding time ≤ t.
     - This smooths the diagnostic's `f_last` to match the weekly decision. The smoothing is declared, not tuned.
   - A member missing either input is excluded from that decision.
2. **Score.** `score = 0.5 * zr(-sigma_30d) + 0.5 * zr(-f_3d)`. Here `zr` is the cross-sectional rank, mapped to (rank − mean rank) / std of ranks across the members with both inputs.
3. **Legs.**
   - Long the `k = floor(n/3)` highest scores and short the `k` lowest, where n is the number of members with inputs.
   - Within each leg, raw weight `w_i ∝ 1/sigma_i`. Each leg is normalised so that `sum |w_i| * sigma_i = 1`, which gives equal stand-alone risk per leg.
4. **Volatility target.**
   - The book is scaled to an ex-ante annualised volatility of **20%**.
   - The covariance is estimated from the last 360 4h log returns of the held names. It is shrunk 50/50 toward its own diagonal and annualised with 2190 bars per year.
5. **Caps**, applied in this order:
   1. per-name `|w| ≤ 0.25`;
   2. perp gross `≤ 2.0`, scaled proportionally;
   3. the cash account of `agy_anchor_oos`: margin `0.4 × perp gross ≤ 1.0` and net signed exposure `≤ 0.97`, with the whole target scaled by the tighter bound.
6. **Risk.**
   - `PreTradeRiskConfig(max_gross=2.0, max_abs_weight=0.25, max_turnover=2.5, drawdown_start=0.125, drawdown_stop=0.25, emergency_turnover_override=True)`.
   - The risk layer can only shrink the book.
7. **Accounting.**
   - Replay runs only through `run_shared_portfolio_replay`, with `settle_terminal_position=True` and `capture_ledger_evidence=True`.
   - Funding is booked on perp legs.
   - Terminal dust is recognised as in lean-core.
   - Initial capital is 100,000 USDT.

## Costs and scenarios

| Scenario | fee | spread | impact | extra signal lag |
|---|---|---|---|---|
| BASE | 5 bp | 3 bp | 2 bp × sqrt(participation) | 0 |
| HIGH_COST | 5 bp | 8 bp | 6 bp × sqrt(participation) | 0 |
| DELAY | as BASE | as BASE | as BASE | 1 bar |

All scenarios use these settings:

- maximum participation of 1% of the previous completed bar's volume;
- minimum notional of 10 USDT;
- `processing_bar_volume_capacity=False`;
- `maintenance_margin_rate=0.05`.

The fee is Binance's VIP0 perp taker rate. The spread is wider than Core7's 2 bp because the universe is mid-cap alts.

## Arms

| Arm | Role | Book |
|---|---|---|
| `CANDIDATE` | gated | the policy above |
| `LOWVOL_ONLY` | reported | the policy with `score = zr(-sigma_30d)` |
| `FUNDING_ONLY` | reported | the policy with `score = zr(-f_3d)` |
| `EW_LONG` | control, reported | equal-weight long of the month's universe, gross 1.0, rebalanced weekly, same cash account |
| `CASH` | control | flat |

Every arm runs under every scenario. Only `CANDIDATE` is gated. The other arms explain the result and are never promoted from this protocol.

## Decision rule (frozen)

A cell is **INVALID** if any of these hold:

- cash net of cumulative execution costs falls below −1e-6 × capital;
- a non-dust residual remains after terminal settlement;
- the run reads a row outside its stage window or authorization.

**DISJOINT label** for `CANDIDATE`, computed on 4h net returns annualised with `sqrt(2190)`:

| # | Condition |
|---|---|
| C1 | no INVALID cell in any arm or scenario |
| C2 | BASE Sharpe ≥ 0.6 |
| C3 | HIGH_COST Sharpe ≥ 0.3 and DELAY Sharpe ≥ 0.4 |
| C4 | BASE max drawdown < 0.25 (on the dust-adjusted path) |
| C5 | BASE Sharpe > 0 in both halves: 2021-04-05 to 2023-12-31, and 2024-01-01 to 2026-07-31 |
| C6 | the one-sided 90% lower bound of the mean BASE 4h net return is > 0, from a stationary block bootstrap (block 42 bars, 10,000 resamples, seed 20261121) |

- **`XS_SUPPORTED`** if C1–C6 all hold.
- **`INVALID`** if C1 fails.
- **`XS_NOT_SUPPORTED`** otherwise.

These figures are reported and never gate:

- per-year returns;
- turnover and cost drag;
- funding PnL;
- beta to `EW_LONG`, and the correlation with it;
- long-leg and short-leg attribution;
- forced-close counts;
- the deflated Sharpe with 40 declared prior trials.

**FORWARD label** at the final run:

- **`XS_FORWARD_SUPPORTED`** if DISJOINT was `XS_SUPPORTED` and, on FORWARD decisions from 2026-08-03, all of these hold:
  - BASE Sharpe > 0;
  - HIGH_COST net return > 0;
  - max drawdown < 0.25;
  - no INVALID cell.
- **`XS_FORWARD_NOT_SUPPORTED`** otherwise.

## Sequencing and stop rule

1. Freeze this spec and the implementation: pin the protocol digest and obtain an exact-head independent review.
2. Run `coverage` for DISJOINT. It reads archives, validates rows and declares halts result-blind (an amendment and a new digest).
3. Post a G4 relay that quotes the protocol id, the digest and the code commit. Then run DEV.
4. If DEV has no INVALID cell, post a G4 relay and run DISJOINT once.
5. **If the label is `XS_NOT_SUPPORTED`, the line stops.**
   - No constant is retuned on these 441 names.
   - A variant is a new protocol: it counts as further trials, and it must name a fresh holdout.
6. **If the label is `XS_SUPPORTED`,** the CANDIDATE goes to FORWARD as a paper book. A paper-trading shadow of the same policy may be built.
   - This is not authorization for orders, API keys or live trading.

A result is never reclassified after it is seen. An amendment after a result has been seen applies only to future stages, and it is disclosed.

## Oracles (G2)

The implementation must pass synthetic tests that do not call production sizing:

- **Hand-computed scores and legs.** On a 6-name synthetic panel, compute the ranks, z-scores, legs and inverse-volatility weights by hand.
- **Volatility-target scaling.** Compare against an exact two-name covariance.
- **Universe membership.** The monthly top-30 is point-in-time. A name whose volume spikes only inside month M does not enter M.
- **Forced close.** A delisting fixture charges exactly the scenario cost plus 5% on the closed notional and leaves no position.
- **Funding sign.** A short perp leg with positive funding receives funding.
- **Causality.** Appending future rows never changes any earlier decision.
- **Embargo.** DISJOINT refuses any row at or after 2026-08-01, and FORWARD refuses rows after the latest closed bar or outside the authorization.
- **INVALID.** A borrowing cell is INVALID.

## Evidence and claim boundary (G3)

- `XS_SUPPORTED` means the frozen rule worked on other symbols over the same calendar window. It is not proof of a time-stable edge. That claim needs FORWARD.
- The figures above are paper replays under assumed fees, spreads, impact and participation, with close-price marks. They are not executed trades.
- Nothing here authorizes orders, live trading, API keys or FS1 scoring.
