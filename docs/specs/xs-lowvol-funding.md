# Cross-sectional low-volatility and funding-contrarian perp book (`xs_lowvol_funding_v1`)

Status: Active (not frozen until an independent review approves the exact head). This protocol is frozen before any perp kline or funding data of its DISJOINT or FORWARD universe has been fetched or read. Freezing happens when the protocol digest is pinned and an independent review approves the exact head.

Protocol digest: `31b6d474055e868b24ea8cd14aa74349b1e37017b74ea4bc1990bc5511b94739`

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
  - A monthly funding archive may be unpublished only for a month in which the perp has no kline bar at all (before its listing, after its delisting). If the perp has any kline bar in a month whose funding archive is unpublished, the run and `coverage` refuse (`coverage` only records it), naming the symbol and the month. Every unpublished funding month is recorded in the coverage report and the data receipt.
- **Clock.** 4h bars aggregated from 1h bars, timestamped at bar close, as in Tier D. Funding events are booked at their funding time.
- **Spot data.** No spot data is read. The book trades perps only.

### Candidate list (frozen)

`trade_rl/evaluation/xs_lowvol_funding/candidates.json`, SHA-256 `fc680208899cd4166c9923f9cf9e8f8ebe829010aae073a935de10d6e6fefdf6`, holds 441 USDT perpetual symbols. It is derived from the scoping metadata (`symbols.json`: 450 USDT perpetuals with both spot and um monthly 1h archives, 2021-01 to 2026-07, the 16 touched symbols (BTC ETH BNB SOL XRP DOGE ADA AVAX LINK LTC DOT BCH SUI APT ARB OP) already excluded) by removing:

- TRX (a conservative exclusion): 450 − 1 = 449;
- the 7 entries flagged `price_scaled_1000x` (1000BONK, 1000BTTC, 1000FLOKI, 1000LUNC, 1000PEPE, 1000SHIB, 1000XEC), whose spot base differs (a unit mismatch on the spot side): 449 − 7 = 442. 1000CAT, 1000CHEEMS, 1000SATS and 1MBABYDOGE are not flagged and remain: the book is perp-only and returns are scale invariant, so price scaling does not matter;
- every symbol that is not ASCII `[A-Z0-9]+USDT` (one), because the Vision path handling for such symbols is untested: 442 − 1 = 441.

The list includes 24 pairs that were later delisted. It was built from archive listings, not from the current exchangeInfo, so delisted pairs are present.

### Monthly universe (point-in-time)

On each calendar month start M, a symbol is **eligible** when all of the following hold:

1. its perp has a complete set of 1h bars for the 90 days strictly before M, or the only missing hours lie inside a declared exchange-wide halt;
2. it has at least 9 settled funding events in the 3 days before M.

The **universe** for month M is the 30 eligible symbols with the largest perp quote volume over the 30 days strictly before M. Ties are broken by symbol name. Membership is fixed for the month and uses no information from M or later.

If fewer than 12 symbols are eligible, the month holds no book and the CANDIDATE is flat.

### Missing bars and delisting while held

- A held name is closed when its perp archive ends or a gap starts that no declared halt explains.
  - Its target is forced to 0 from the first 4h decision bar whose **open time is at or after** the release of the event's public notice through the event, the close of the leg's last available 4h bar. The notice is the **earliest** recorded notice of that symbol with `release_utc <= event` and `release_utc >= event - 90 days`, so a later postponement notice never delays the start (Amendment 5).
  - An **event** is a zombie-tail effective end, a relisting-gap start, or any other archive end inside the stage. The effective end (Amendment 2) is the close of the leg's last 1h bar with volume. Only the pairing of a notice with the event it announced looks at the event time; each notice is used only from its release. The recorded notices are `trade_rl/evaluation/xs_lowvol_funding/delisting_notices.json` (symbol, release time, article code, title, kind), pinned by `protocol.DELISTING_NOTICES_SHA256` and verified before any read.
  - **Fail closed (Amendment 5):** an event of a leg that was a universe member at or before it, with no causal notice, refuses the run before any replay and names the symbols. A real DEV, DISJOINT or FORWARD run is refused before clock/G4/file/row/output access unless the notice file verifies against its pin. If the notice is released too late for the position to be closed before the event, the rule below applies. This is not an economic rerun, a protocol approval, an unlock of already studied data or a new eligible holdout.
  - Each fill of the forced close is charged the scenario cost plus an extra adverse **5%** of the closed notional. The charge is booked as a cash debit at that fill bar.
- There is no write-off branch. If any position of such a name is still held at its last available bar (no capacity, below minimum notional), the run refuses, naming the symbol.
  - This is a disclosed limitation: the shared replay cannot continue past an engine settlement of a held position, and handling it would need an amendment.
- A name with a gap inside its trailing 90 days is ineligible for that month.
- It is reported, never gating, how many forced closes occurred, their total charge, and the result without them.

### Declared halts

Exchange-wide perp outages are declared result-blind from a coverage report, as in `agy_anchor_oos`. `coverage` reads archives and validates rows. It computes no signal, replay or return. A range qualifies only if every symbol listed at that time is missing it. Any other missing hour of an eligible, held or ranked symbol follows the rules above.

**Amendment 1 (2026-11-21, result-blind).** `coverage` for DISJOINT ran from code `ef11e0b06d108fae34bc6ea0f33f63a9d2a02775` (protocol digest `1fae7b85…`) on a clean tree. It reported 0 exchange-wide perp gap hours and no read error, so `DECLARED_HALTS` stays empty. Ten legs have symbol-specific gaps (AERGO, BNX, CTK, CVC, CVX, ICP, LIT, PUMP, SLP, TLM; 7–122 hours); they follow the eligibility and forced-close rules above. The report's SHA-256 `757cbdd8f4d0a1ce7c783ea91bd3e7a16f719b5e5c3e8d00f1979e6a6b8b7d5c` is pinned as `COVERAGE_EVIDENCE_SHA256`. No signal, replay or return was computed. Amendment 2 supersedes that report as evidence: a new coverage report, produced by the amended tool, will be pinned in a separate result-blind commit before DEV is rerun.

**Amendment 2 (2026-11-21, result-blind).** The single-use DISJOINT attempt at code `21becb5616f036d5f8c10944d7ff8e68550dc4e8`, run under the relay https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-7062196515, refused on `A2ZUSDT` (the monthly funding archive of 2026-04 is unpublished but the perp has kline bars in that month) before any signal, replay or return was computed. It wrote no output directory. No DISJOINT label exists and the single-use run has not been consumed. The cause is a data-format fact found from volumes only: Binance Vision keeps publishing flat 1h kline archives with volume 0 ("zombie bars") after a perp is delisted. The harness changes as follows.

- **Zombie-tail truncation.** The effective listing end of a perp leg is the close of its last 1h bar with volume (base or quote volume above 0). The trailing run of zero-volume bars after it is dropped as not listed before anything else uses the leg: universe eligibility, funding admissibility, gap checks and the forced-close schedule, which now keys on the effective end (a delisted held name is closed before the zombie tail, as the delisting-notice model intends). Only the trailing run is dropped here; interior zero-volume runs follow the relisting-gap rule below. A leg with no bar with volume is never listed. The data receipt and the coverage report record, per leg, `zombie_tail_bars` and `effective_end`. This uses archive-end knowledge exactly like the disclosed forced-close exception and is part of that same exception.
- **Relisting gaps.** An interior run of consecutive 1h bars with base and quote volume both 0 that lasts at least 24 hours (`FUNDING_MAX_GAP_HOURS`) is a delisting followed by a relisting, whether or not Binance funding events fall inside it. Its bars are converted to missing hours before eligibility, the forced-close schedule, funding admissibility and the gap checks, and the funding events strictly inside the run are dropped (counted per leg as `dropped_funding_events_in_gaps`): no position can be held there, because the existing gap rules apply unchanged. An unexplained gap makes the name ineligible through the 90-day complete-history rule, and a held name is force-closed from the gap start on. Left as present flat bars, such a run would let a held name stay unclosable (zero volume, zero participation), jump at the relisting and keep ranking from trailing volume. The conversion uses archive knowledge exactly like the delisting forced close and is part of the same disclosed exception. Shorter zero-volume runs stay as they are. An unpublished funding month with no bar of positive volume left (none remaining, or only zero-volume bars) is not traded and does not refuse; a month with a positive-volume bar keeps the REST fallback and gap check below. The data receipt and the coverage report record `relisting_gap_runs` (`[start, end, hours]`) per leg and the coverage summary counts `relisting_gap_legs`. Result-blind coverage at `90981055` showed the runs that motivate the rule: ICPUSDT from 2022-06-10T10:00Z (2486 h), LITUSDT from 2025-01-31T09:00Z (7815 h) and TLMUSDT from 2022-06-09T09:00Z (7047 h), all with funding events inside, so an earlier funding-event exemption left them unconverted.
- **REST funding fallback.** If a monthly Vision fundingRate archive is unpublished in a month in which the (truncated) leg has at least one bar with volume, the month's events are read from the public Binance endpoint `https://fapi.binance.com/fapi/v1/fundingRate` (no key; only with `--allow-network`), paginated by `startTime` while 1000 events are returned, over the UTC month `[start, next month start)` clipped to the read limit. The response must name the symbol, have strictly increasing times inside the month and finite rates. The raw JSON of each month is cached under `<archive>/rest/fundingRate/<SYMBOL>/<SYMBOL>-fundingRate-<YYYY-MM>.json` and its SHA-256 and event count are recorded in the data receipt (`rest_funding`) and the coverage report, so the run is reproducible offline from the cache (a cache file is used before the network). The events are merged like archive events. Without cache and network, or on an invalid response, the run refuses as before (fail closed). If the REST endpoint returns no event for such a month (a perp that listed at the end of that month), the month is admissible only if the merged funding series has no gap longer than 24 hours around it, measured from the first bar with volume (or the last event before the month) to the next event (or the effective end); otherwise the run refuses.
- **Coverage summary.** `coverage` applies the same truncation and fallback, and its printed summary lists `inadmissible_unpublished_funding_months` (count and `symbol:month`), `rest_funding_months`, `zombie_tail_legs` and `relisting_gap_legs`, so a refusal is visible before a run.
- **Sequence.** DEV is rerun under the new protocol digest, and DISJOINT then runs exactly once under it.

**Amendment 3 (2026-11-22, result-blind software validity).** Before any DISJOINT replay or return was computed, a synthetic review found that the Amendment 2 REST decoder implicitly coerced a malformed floating-point, string or boolean `fundingTime` into an integer, or a boolean/huge integer `fundingRate` into a float. The official USD-M endpoint publishes integer-millisecond `fundingTime` and string `fundingRate`; the updated decoder rejects any wrong field type and refuses malformed network responses and cached months without appending, caching or computing a result. This valid-response-preserving fail-closed change is nevertheless a new **harness source digest** and thus a new **protocol digest**: earlier review/CI/authorization and coverage under the previous digest do not transfer. Freeze, new result-blind coverage and any economic run remain on HOLD until exact-head independent review and fresh authorization; no real candidate data was read by this amendment.

**Amendment 4 (2026-11-22, result-blind coverage evidence).** `coverage` for DISJOINT ran from the clean code `d5e2b6a88dfd6c57e52473d16aa9ba733d2a7d0c` (protocol digest `00530f3a26f042e07e3c795ef8046768f223cf1e812470009b62281ec78c21f6`). It reported 0 exchange-wide perp gap hours, no read error, 0 inadmissible unpublished funding months, 38 REST funding months, 100 zombie-tail legs and 10 relisting-gap legs (AERGO, BNX, CTK, CVC, CVX, ICP, LIT, PUMP, SLP, TLM; 260–21567 hours). `DECLARED_HALTS` stays empty. The report's SHA-256 `64149a6ff2e3770449481e048896d832d6e5eb558c17b950b76de0a06c25d594` replaces the Amendment 1 value as `COVERAGE_EVIDENCE_SHA256`. Only this pin changes, which gives a new protocol digest. No signal, replay or return was computed.

**Amendment 5 (2026-11-22, result-blind, causal delisting-notice evidence).** The earlier forced close used archive-end hindsight, so every real-data DEV, DISJOINT and FORWARD run was refused. This amendment supplies timestamped public notices and makes the forced close causal.

- **Source.** `python -m tools.xs_lowvol_funding notices --allow-network` reads Binance's public CMS API (no key) for the catalogs Delisting (161), Latest Binance News (49), New Cryptocurrency Listing (48) and Maintenance Updates (157) from 2020-10-01: 6,642 articles. It keeps futures delistings, spot delistings whose body carries a Binance Futures close-all-positions or settlement statement, and token swap, migration or rebrand notices. Titles and, when a title does not name the perpetual, the article body give the USDⓈ-M symbols; COIN-M, margin, loan, Alpha and completion notices are skipped. The result is `trade_rl/evaluation/xs_lowvol_funding/delisting_notices.json`: 234 entries (`symbol`, `release_utc`, `article_code`, `title`, `kind`) from 103 articles for 214 symbols, with an empty `overrides` section (an override must carry an article code and its release time and title are read back from the API; none was needed). Its SHA-256 `52b0245a754bf381861d7fe892f34d52c129c4cd911c2113dd098287f89bbcc0` is pinned as `protocol.DELISTING_NOTICES_SHA256`, is part of the protocol payload and is verified before the file is parsed.
- **Rule.** An event is a zombie-tail effective end, a relisting-gap start or any other archive end of a leg that was a universe member at or before it. Its notice is the earliest recorded notice of that symbol with `release_utc <= event` and `release_utc >= event - 90 days` (a later postponement notice does not delay the start; the cases are AKROUSDT 2022-05-27, ANTUSDT 2024-04-01 and OMGUSDT 2025-01-31), where the event time is the close of the leg's last available 4h bar. The target is 0 from the first 4h decision bar whose open is at or after the release through the event; the extra 5% charge is unchanged. The 42-bar archive-end lead is removed. An event with no such notice refuses the run before any replay, naming the symbols. If the notice comes too late to close the position before the event, the existing refusal at the last available bar applies.
- **Coverage of the 109 DISJOINT events** (the 100 zombie-tail effective ends and 10 relisting-gap starts of the DISJOINT coverage report, `COVERAGE_EVIDENCE_SHA256` unchanged; 109 fall in 2021-04-05 to 2026-08-01, the tenth, BTCST on 2021-03-12, precedes the first decision; BNX appears with both an end and a gap): all 109 have a causal notice. 109 matched automatically (42 by a futures-titled delisting notice, 43 by a spot delisting notice whose body carries the Binance Futures settlement statement, 24 by a token swap, migration or rebrand notice with a futures section), 0 by override and 0 unmatched. The lead from the earliest notice ranges from 0.04 to 56 days (median 6.1); the earliest-notice rule changes the start for exactly the three postponement cases AKROUSDT, ANTUSDT and OMGUSDT; one notice was released inside the event's last 4h bar, so a held name there refuses at the last bar rather than closing.
- **Authority.** A real run is refused before clock, G4, file, row and output access unless the notice file matches its pin; otherwise it proceeds only through this path. Notice coverage was checked against the stored coverage report only. No stage was run and no signal, replay or return was computed. This is not a protocol approval, a new holdout or an authorization of any economic run; FORWARD events after the report's window are checked by the same rule at run time.

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
  - The symbol file is `trade_rl/evaluation/xs_lowvol_funding/forward_symbols.json` (684 symbols), SHA-256 `48656a5dfb73491dacd2234490ba7c34c79b5692fcc91812a15095434900ba66`, pinned as `FORWARD_SYMBOLS_SHA256` (part of the protocol digest) before DISJOINT runs. It was generated result-blind from the Vision S3 listing (names only) on 2026-11-21: the union of um monthly and daily kline prefixes, USDT, ASCII `[A-Z0-9]+USDT`, not a stablecoin or gold base; symbols present in the cached USD-M exchangeInfo snapshot with `contractType != PERPETUAL` or `underlyingType != COIN` are excluded (237 TradFi, index and premarket perps; delisted names absent from the snapshot are kept).
  - The tool defaults `--forward-symbols` to this packaged file; an explicit path must hash-match. The file is identical for every interim and the final run and is never edited afterwards. FORWARD coverage and runs refuse when the pin is unset or the file differs.
  - Disclosed: perps listed after 2026-11-21 are not in the FORWARD universe. They could only become eligible after 90 days, that is from about 2027-02.
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
   - Disclosed: in a week where a forced-close name is zeroed after this scaling, the remaining book is not re-targeted, so its ex-ante volatility is below 20%.
   - The covariance is estimated from the last 360 4h log returns of the held names. It is shrunk 50/50 toward its own diagonal and annualised with 2190 bars per year.
5. **Caps**, applied in this order:
   1. per-name `|w| ≤ 0.25`;
   2. perp gross `≤ 2.0`, scaled proportionally;
   3. the cash account of `agy_anchor_oos`: margin `0.4 × perp gross ≤ 1.0` and net signed exposure `≤ 0.97`, with the whole target scaled by the tighter bound.
6. **Risk.**
   - `PreTradeRiskConfig(max_gross=2.0, max_abs_weight=0.25, max_turnover=2.5, drawdown_start=0.125, drawdown_stop=0.25, emergency_turnover_override=True)`.
   - The risk layer can only shrink the book.
7. **Terminal unwind.**
   - At the last weekly decision of every stage window (DEV, DISJOINT, FORWARD final and every FORWARD interim cut: the last Monday decision on or before the stage's last decision bar), every arm's target is 0.
   - The flat target is re-issued on every later bar, so partial fills under the 1% participation cap complete over the remaining days before terminal settlement.
   - Disclosed: the last week or so of every stage holds no book. Terminal settlement is unchanged.
8. **Accounting.**
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

- at any ledger interval, the cash balance after cumulative forced-close debits, less **only still-unpaid execution fees from the current and previously unrecovered intervals**, implies borrowing above 1e-6 × capital. Fees stop being borrowing credit when cash recovers; lifetime historical fees cannot offset a later loan. Cumulative cost boundaries, monotonicity, finiteness and forced-close debit amounts are validated before the borrowed-cash calculation (proposed software-only amendment; the previously published funding/coverage evidence is not reinterpreted);
- a non-dust residual remains after terminal settlement (the terminal unwind above makes this a failure of the engine path, not of a one-bar close; only lean-core terminal dust is excepted);
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
4. If DEV has no INVALID cell, post a G4 relay and run DISJOINT exactly once.
   - DISJOINT runs exactly once for this protocol id on these 441 names. Its label, including `INVALID`, is final. DEV exists to catch harness faults first; a harness fix after any DISJOINT result never reruns DISJOINT on these names (a successor protocol needs a fresh holdout).
   - The tool requires an explicit `--confirm-single-use` for DISJOINT and writes the output directory name and the digest into the result, so a second DISJOINT result is visibly a second run. This is a procedural guard, enforced by the one G4 relay per stage.
5. **If the label is `XS_NOT_SUPPORTED` or `INVALID`, the line stops.**
   - No constant is retuned on these 441 names.
   - A variant is a new protocol: it counts as further trials, and it must name a fresh holdout.
6. **If the label is `XS_SUPPORTED`,** the CANDIDATE goes to FORWARD as a paper book. A paper-trading shadow of the same policy may be built.
   - This is not authorization for orders, API keys or live trading.

A result is never reclassified after it is seen. An amendment after a result has been seen applies only to future stages, and it is disclosed. It never permits rerunning DISJOINT on these names.

## Oracles (G2)

The implementation must pass synthetic tests that do not call production sizing:

- **Hand-computed scores and legs.** On a 6-name synthetic panel, compute the ranks, z-scores, legs and inverse-volatility weights by hand.
- **Volatility-target scaling.** Compare against an exact two-name covariance.
- **Universe membership.** The monthly top-30 is point-in-time. A name whose volume spikes only inside month M does not enter M.
- **Forced close.** A delisting fixture charges exactly the scenario cost plus 5% on the closed notional and leaves no position.
- **Funding sign.** A short perp leg with positive funding receives funding.
- **Causality.** Appending future rows never changes any earlier decision. The one disclosed exception is the forced-close schedule: it pairs each recorded public notice with the archive event it announced (the 90-day pairing window uses the event time), and a notice is used only from its release (Amendment 5). Every other input obeys the append-future-rows invariance.
- **Embargo.** DISJOINT refuses any row at or after 2026-08-01, and FORWARD refuses rows after the latest closed bar or outside the authorization.
- **INVALID.** A borrowing cell is INVALID.

## Evidence and claim boundary (G3)

- `XS_SUPPORTED` means the frozen rule worked on other symbols over the same calendar window. It is not proof of a time-stable edge. That claim needs FORWARD.
- The figures above are paper replays under assumed fees, spreads, impact and participation, with close-price marks. They are not executed trades.
- Nothing here authorizes orders, live trading, API keys or FS1 scoring.
