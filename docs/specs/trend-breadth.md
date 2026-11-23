# Daily trend breadth screen (`trend_breadth_v1`)

Status: Active (not frozen until an independent review approves the exact head). This protocol is fixed before any archive value of the BREADTH stage has been parsed for a trend statistic. No stage has run, no return, exposure, regime or label was computed from the 441-symbol archives to write it (only file listings and the format of a handful of archive files were inspected), and nothing here authorizes an economic run.

Protocol digest: `67e9c2fa75c95efee1a1ebb51f73df1b6a647dfe433929eaa8dd72fc4d3b3117`

## Question

The frozen daily-regime trend rule is the only robust candidate of the research line, but every number behind it comes from 11 large Binance USDT-M perpetuals that the planner has seen. Is the rule's behaviour specific to those 11 majors, or does it carry over to a large cross-section of perpetuals on which it has never been scored?

**Primary question:** over 2021-01-01 to 2026-08-01, does the long/short rule, applied with equal sleeves to every eligible perpetual of the 441-symbol xs DISJOINT candidate list, clear the frozen gates of the decision rule below, including a paired comparison with an equal-weight long control?

**Hypothesis:** the trend mechanism is general across crypto perpetuals. Persistent price regimes show up as the sign of the price against slow averages in small and mid caps as well as in the majors, so the same rule earns a positive net-of-cost, net-of-funding return across the broad cross-section and beats a passive long basket of the same symbols.

**Counter-hypothesis:** the 11-coin result is selection plus the 2023–24 regime. The 11 coins were the survivors a planner looks at; the rule was kept after the planner saw them. Small-cap perpetuals are dominated by listing pumps followed by multi-month decay and by delistings, which favours shorts and punishes long-biased books (a trend-long book buys late and is sold into the decay); a daily rule on such names pays costs on illiquid fills that bind the participation cap. A long/short book may lose, or only an equal-weight short bias may pay.

**Cheapest falsifier:** this stage. It uses archived data that is already on disk, needs no waiting and no new code beyond this harness. `TREND_BREADTH_NOT_SUPPORTED` (any gate failing) falsifies the hypothesis in this universe and period; before it, the synthetic loop oracle, the truncation causality tests and the no-borrowing tests falsify the harness arithmetic.

**Evidence class.** A `TREND_BREADTH_SUPPORTED` label is class 2 evidence: the protocol is pre-registered, but the data are known-used (the symbols and the window were already used by the xs protocol, the rule was chosen after seeing 11-coin results). It is not profitability evidence and does not authorize orders, paper trading, live trading or capital.

## Prior exposure (disclosed)

- **xs DISJOINT on the same 441 symbols.** The xs protocol `xs_lowvol_funding_v1` ran its DISJOINT stage on exactly this list and window (2021-04-05 to 2026-08-01 decisions; Issue #810 comment 7081760356 holds the planner's relay). The planner has seen: the `EW_LONG` control cell of that stage was INVALID (`negative_cash_borrowing`) with a total return of -7.8%, and the xs CANDIDATE, LOWVOL and FUNDING legs' results. These are known facts about a *different* strategy family (monthly top-30 cross-sectional long/short on volatility and funding). No daily-trend statistic, regime, exposure or label of these symbols has ever been computed. The xs control defect is the reason this protocol defines a cash model with a no-borrowing clamp (below); its -7.8% is not adopted as a benchmark and is not evidence for or against the trend rule.
- **The xs single-use and holdout language.** The xs spec states that its DISJOINT stage runs once for its protocol id on these 441 names and that a successor needs a fresh holdout. This is a new protocol id with a different hypothesis and a frozen, never-retuned rule; the 441 names are therefore NOT an untouched holdout for trend. Their use is disclosed here and no claim of independence from xs is made beyond "no trend statistic was computed".
- **The 11-coin known-data numbers (known-data, not evidence).** The research harness smoke test on ALL11 (BTC ETH SOL BNB XRP ADA DOGE LINK DOT AVAX LTC, 2023-01..2026-07, FS1 costs, actual funding, actual units) is a LOCAL, UNREVIEWED exploratory artifact (`C:\trade_rl_runs\research-x\out\smoke_q4_rule.txt` and `.json`, cell `E_FS1cost_actualfunding_units`): the plain long/short rule returned +63.6% with Sharpe 0.55 and max drawdown 40.4%, and its long-only variant +104.3% with Sharpe 0.75 (the harness's capital differs from this protocol's 100,000 USDT). The 2025 figure of -10.2% comes from the same research line's H-EXIT report, which is not in that artifact. The earlier trend-OOS BACKCAST (2018-2020, unused data at that time) is cited from the planner's recollection of the Issue #810 records (about +21%, Sharpe 0.9, tiny exposure, not significant against cash); no #810 comment was located for it. The MTF system screens (q2-q6) on known data found the daily rule beating BTC hold on Sharpe and drawdown. None of these is evidence for this protocol: the universe, sleeve rule, costs and account differ, and the same windows were seen.
- **Rule selection.** The planner chose this exact rule (regime vote, 0.5 volatility target) after seeing 11-coin results. No constant is tuned on the 441 names. The decision constants (Sharpe 0.5, drawdown 40%) were fixed from the 11-coin experience before any 441-name statistic existed.
- The FS1 seal is unaffected: this is a separate protocol on public Binance data and touches no FS1 artifact.

## Data and universe (frozen)

All times are UTC.

- **Universe.** The 441 USDT-M perpetual symbols of `tools/trend_breadth_data/universe_candidates.json`, a byte copy of the xs `candidates.json` (PR #1307 head `9793293462dd141dc489de29071672020460ea5f`, `trade_rl/evaluation/xs_lowvol_funding/candidates.json`), SHA-256 `fc680208899cd4166c9923f9cf9e8f8ebe829010aae073a935de10d6e6fefdf6`, verified before any read. In the xs spec it is derived from a scoping list of 450 USDT perpetuals with spot and um monthly archives in which the 16 touched large caps (BTC ETH BNB SOL XRP DOGE ADA AVAX LINK LTC DOT BCH SUI APT ARB OP) are already excluded, by removing TRX (450 - 1 = 449), seven price-scaled `1000x` symbols (449 - 7 = 442) and one non-ASCII symbol (442 - 1 = 441); it includes 24 later-delisted pairs. The universe is the whole list: no symbol is excluded for liquidity, listing age, price level or any other reason.
- **Window.** Hourly bars opening from 2021-01-01 00:00 up to and excluding 2026-08-01 00:00. The first decision close is 2021-01-01 04:00 and the last mark close 2026-08-01 00:00 (12,227 4h returns after the first close). No bar opening at or after 2026-08-01 is ever read: a file dated at or after it is never opened, and an archive row at or after it refuses the run.
- **Archives.** Local Binance Vision archives under `C:\trade_rl_runs\xs-archives` (read-only): `futures/um/monthly/klines/<SYM>/1h`, `futures/um/daily/klines/<SYM>/1h`, `futures/um/monthly/fundingRate/<SYM>` and `rest/fundingRate/<SYM>`. A month of klines comes from its monthly zip; if that zip is missing or malformed, from the daily zips of that month. A month of funding comes from its Vision zip, else from the cached REST json. There is no forward fill and no network read.
- **Archive integrity.** Before the claim, every zip that may be read must match its `.CHECKSUM` sidecar and hold exactly one member; the sha256 of every readable file is combined into a manifest digest recorded in the result. A mismatch refuses the run without claiming. A zip that is malformed although its checksum matches is tolerated only where its month can be rebuilt (monthly klines from daily kline zips, Vision funding from cached REST json); otherwise the preflight refuses the run without claiming, and a readable zip whose csv cannot be parsed and cannot be rebuilt is a data problem that makes the stage INVALID, never silently missing data.
- **Delisting notices.** `tools/trend_breadth_data/delisting_notices.json`, a byte copy of the xs `delisting_notices.json` (234 entries from 103 public Binance articles; the xs Amendment 5 evidence), SHA-256 `52b0245a754bf381861d7fe892f34d52c129c4cd911c2113dd098287f89bbcc0`, verified before any read. A notice released at or after 2026-08-01 is ignored. The file's `overrides` list is empty and the reader refuses a non-empty one (xs merges overrides; this reader does not).
- **Bars.** 4h and 1d bars are aggregated from 1h bars exactly as the exploratory `feat.agg` does (open of the first hour, max high, min low, close of the last hour, sum of quote volume), labelled by open time. A bar exists only if every hourly bar of it exists; otherwise it is NaN, and so is every value that uses it. At a 4h decision close the latest COMPLETED daily bar is used.
- **Format facts the harness handles.** Kline csv files may carry a header line; Vision funding csv files carry `calc_time, funding_interval_hours, last_funding_rate` and no mark price.

### Listing episodes, delistings and gaps

- A *dead* hour is a missing hour or one without volume (quote and base volume both 0 or non-finite price). A run of at least 24 dead hours ends a *listing episode*; shorter holes stay inside it. Leading and trailing dead hours are not part of an episode. Binance Vision keeps publishing flat zero-volume bars after a delisting (the xs "zombie tail"); they are dead hours and end the episode at the last traded hour.
- Dead hours of a long run are blanked before any bar is aggregated, so no feature spans a delisting or a relisting gap. After a relisting gap the symbol is a NEW episode that needs 200 fresh daily bars. Inside an episode a hole of at most 23 hours makes the affected 4h and 1d bars NaN; a NaN daily bar keeps every window that contains it NaN (the exploratory screens' rule), so a single missing hour blanks the regime for up to 200 days. Marking uses the last close carried over such a hole (no fill is possible there); an episode still listed at the stage end also carries its last close through a trailing hole of less than a day.
- **Eligibility at a decision close.** A symbol is eligible when its regime and 30-day volatility exist (this needs 200 consecutive complete daily bars of its current episode), it has a price and it is not inside a forced-close window. `N_t` is the number of eligible symbols at the close. A symbol enters when it has 200 complete daily bars and leaves at delisting.
- **Delisting (forced close, following the xs Amendments 5 and 6).** An episode followed by a long dead run (interior or trailing) is delisted; its *event* is the close of its last available 4h bar. Its notice is the EARLIEST recorded notice of the symbol with `release <= event` and `release >= event - 90 days`. From the first decision close whose 4h bar opens at or after the release, through the event, the symbol is not eligible and its target is 0 (the position is closed). Each fill of a forced close pays the scenario cost plus an extra adverse 5% of the closed notional. A position still held at the event bar's close (no capacity) is settled at that close at full recovery (the last published price) and pays the same 5% charge (the xs write-off). If the notice is released too late to close before the event, or there is NO qualifying notice, there is no forced-close window and only the write-off applies: the position is held until its last valid bar and settled at that close with the extra 5% charge. This is causal (an instrument is known to have stopped only by observing it stop). The number of events and of symbols that took this path is reported (`notice_none`, `symbols_no_notice`).
- **Deviation from xs, disclosed.** xs refuses a run when an event of a universe member has no causal notice. This protocol has no monthly universe and all 441 names count, so an event with no notice is not refused: it is write-off only (above). The pairing of a notice with its event looks at the event time, which is the disclosed xs exception to the append-future-rows invariance; every other input is causal.

## Rule (frozen; identical to the q4 daily rule)

- **Regime:** `mean over N in {20, 50, 100, 200} of sign(close_1d - SMA_N(close_1d))` on the daily closes.
- **Direction:** +1 if regime >= 0.5, -1 if <= -0.5, else 0 (NaN gives 0).
- **Size:** `min(1, 0.5 / vol_ann)`, `vol_ann` the 30-day std (ddof 1) of daily log returns x sqrt(365). The size is fixed at entry while the direction is unchanged and recomputed on a new entry or a flip. A row whose close or `vol_ann` is not finite makes no decision (the previous position is held), as in the exploratory code.
- **Decisions** are made at 4h closes using the latest completed daily bar.
- **Sleeve.** A new entry (or flip) of a symbol fixes its sleeve `0.9 / N_t` at that close and the target weight of equity is `direction x size x sleeve`. The sleeve is NOT re-sized while the direction is unchanged: a later change of `N_t` (new listings, delistings) never changes the sleeve of an open position. A new entry needs `N_t >= 20`; below that no symbol enters (open positions continue). The budget 0.9 keeps a 10% cash reserve for costs, funding and forced-close charges.
- **Long only (`TREND_LONG`).** The same weights with every short replaced by 0 (the `N_t` of the long/short book is kept, so the sleeves are the same).
- **`EW_LONG` control.** At 00:00 UTC on the first day of every month (and the first time `N_t >= 20`) every eligible symbol is re-targeted to `0.9 / N_t` of equity; a symbol leaving eligibility is closed, nobody is added between rebalances, and the book is flat while `N_t < 20`. Units are re-derived at each rebalance (not only for symbols whose weight changed).

## Accounting (actual units; corrects the xs control defect)

The book holds actual contracts and cash, `equity = cash + sum(units x price)`. It follows `btc_trend_ensemble_v1` Amendment 2 and the #1335 book: a symbol whose target weight is unchanged keeps its units however far prices drift.

- **Capital.** A 100,000 USDT reference account (as in PR #1335). It sizes the participation of every fill, so that illiquid symbols bind the cap.
- **Sizing.** A changed (or re-targeted) weight is converted to units once, at the fill: `units = w x E / price`, `E` the equity marked at the fill prices before the event's fills.
- **Costs.** The repository `ExecutionCostConfig` formula with the FS1 constants: a fill of notional `N` pays `N x multiplier x (0.0005 fee + 0.0002 spread + 0.0001 x sqrt(participation))`, `participation = N / V`, no price slippage; `multiplier` 1 (BASE, DELAY) or 2 (HIGH_COST; the forced-close 5% extra is not doubled).
- **Participation cap.** A fill is capped at 5% of `V`, the quote volume of the last completed 1h bar of the 4h bar. An unfilled residual is carried toward the same target units while the target is unchanged. A bar without volume or without a complete close is not tradable and the target stays pending.
- **Fill order and margin model (the no-borrowing rule).** The account is capped per side, not at gross 1: cash is `equity - net signed notional`, short sales add their proceeds to cash and are not borrowing, and long gross plus short gross can reach about 2 x equity (each side is capped at 1.0 x equity; the book is long/short, not a gross-at-most-1 book). Within a fill event reducing fills go first. A BUY (a long increase or a short cover) is first limited so that `notional x (1 + cost rate + forced extra)` fits in the cash on hand, so no buy can leave cash below zero. An INCREASING fill is then clamped so that, after the WHOLE fill's costs (the reducing part of a flip and the forced extra included), the long notional and the short notional each stay within `1.0 x equity`. A clamped fill is reported (`clamped_fills`). Reducing and closing sells are never blocked; their costs come out of equity and can leave a side marginally above the cap through no increase, which is not an invalidity by itself (it is bounded by the mark check below). Sleeves are `0.9 / N_t` fixed at entry while `N_t` changes (it grows over the window), so the open sleeves of one side can sum to more than 0.9 (and to more than 1.0 in a full book); the clamp then limits later entries, and a long-only book's cash cannot go negative through a fill.
- **Funding.** The ACTUAL funding of every event of every symbol. An event stamped `t` with `c_{j-1} <= t < c_j` is paid in bar `j` on the units held after the fill at `c_{j-1}`: `cash -= units x price x rate` (longs pay a positive rate). `price` is the REST record's mark price when the month comes from REST, else the 1h open of the funding hour (Vision funding files carry no mark; the count of fallbacks is recorded). An event stamped outside every listing episode has no position and is dropped (counted). A data problem makes the stage INVALID when more than 1% of the eligible symbol-bars have no funding event in the preceding 8 hours plus one minute.
- **Timing.** A decision at the close `c_j` fills at that close (BASE, HIGH_COST) or at `c_{j+1}` (DELAY). P&L is marked at every close; the book starts flat with 100,000 USDT. `DELAY` shifts the forced-close fills too.
- **Validity.** A cell is INVALID if at any step: equity is not positive; cash falls below `-2%` of equity at a mark (`negative_cash_borrowing`: funding and cost drift is allowed down to this limit, a fill can never create it); an INCREASING fill leaves its side's notional above 1.0 x equity or leaves cash negative (a defect check of the clamp); the accounting identity `equity = capital + price P&L + funding - costs` breaks by more than 1e-9 relative at a mark or equity after an event differs from equity before it minus its costs by more than 1e-9 relative; a price used by a held symbol is not positive or missing. There is no scaling away of a breach. The stage is INVALID if the data check above fails or a price is not positive.
- **Reporting.** Each cell reports total return, max drawdown, Sharpe, fills, capped and clamped fills, forced fills and charges, write-offs, turnover, costs, funding, price P&L and costs per side (long and short), mean long and short gross, minimum cash fraction and per-year returns; the primary cells also report P&L by listing-age bucket (`<200d`, `200-365d`, `1-2y`, `2-3y`, `3y+` since the start of the current episode, attributing price P&L plus funding minus costs of every symbol-bar to its bucket). The `N_t` summary and the notice-pairing counts (matched, none, late) are reported, together with `blanked_symbol_days`: the symbol-days of listings at least 200 days old that are ineligible only because a missing hour blanked the regime or volatility. This eligibility loss is a known, disclosed conservatism of the frozen rule (open positions of such a symbol are closed), not a defect, and it gates nothing.

## Cells (frozen)

| Cell | Description |
|---|---|
| `TREND_LS__BASE` | long/short rule, BASE costs |
| `TREND_LONG__BASE` | long side only |
| `TREND_LS__HIGH_COST` | costs x2 |
| `TREND_LS__DELAY` | fills one 4h bar later |
| `EW_LONG` | control: equal-weight long of eligible symbols, monthly rebalance |
| `CASH` | control: equity 1 |

Single stage `BREADTH`; it runs once.

## Decision rule (frozen)

`TREND_BREADTH_SUPPORTED` requires ALL of:
1. All cells are valid and the data have no problem.
2. `TREND_LS__BASE` annual Sharpe >= 0.5 (4h simple returns of the equity series, population standard deviation, x sqrt(6 x 365), no risk-free rate).
3. The paired moving-block bootstrap of the 4h return difference `TREND_LS__BASE - EW_LONG` (equity-return series; overlapping blocks of 30 4h bars, 5,000 draws, seed 0) has a lower one-sided 95% bound (the 5% quantile of the bootstrap means) above 0.
4. `TREND_LS__HIGH_COST` and `TREND_LS__DELAY` total returns are above 0.
5. `TREND_LS__BASE` max drawdown (4h close marks) is below 40%.

| Label | Condition |
|---|---|
| `TREND_BREADTH_SUPPORTED` | every cell valid and gates 1-5 pass |
| `TREND_BREADTH_NOT_SUPPORTED` | every cell valid and a gate fails |
| `INVALID` | any cell invalid or a data problem |

Reported but never deciding: per-year results, per-listing-age buckets, the long and short legs, costs, funding, capped and clamped fills, forced closes and write-offs, `TREND_LONG__BASE` and `EW_LONG` statistics.

## Mechanism chain (G1)

1. Daily SMA-vote regimes are persistent in crypto perpetuals, so the sign of price against slow averages predicts the sign of the following weeks' return more often than not; this is the whole entry edge and is claimed for the broad cross-section, not only for majors.
2. Equal sleeves spread the book over many names, so idiosyncratic pumps, squeezes and delistings are diversified; each position is sized by its own volatility.
3. Costs and funding are paid on actual fills and held units; illiquid names bind the participation cap and pay more impact, so a trend that cannot be traded at size earns nothing.
4. Delisted names are closed before the end at a notice-driven time, with an extra 5% charge, and settled at the last price if still held.

Each link can fail: persistence can end or be a feature of 2023-24; listing pumps and decays favour the short leg (or punish longs) and make the long/short result depend on one side; a thin cross-section early (few eligible names) concentrates sleeves; the cap and costs on small caps can consume the edge; the notice-based forced close and the 5% charge can dominate returns near delistings. The counter-hypothesis is that links 1-2 are selection plus regime.

## Independent oracle (G2)

- A pure-Python loop oracle (dicts and floats, no numpy book) recomputes the target weights (episodes, eligibility, `N_t`, sleeves fixed at entry, long-only, `EW_LONG` rebalance) and the equity of every cell from the panel on synthetic data with a listing, a delisting with a notice, a relisting gap, a no-notice (write-off-only) delisting, thin volume that binds the participation cap and funding, and requires agreement to 1e-9.
- Hand-computed cases cover the actual-units rule (price 100 to 200 to 100 returns exactly flat), the repository cost formula (pinned against `ExecutionCostConfig`), the cap and carried residual, the side clamp, funding sign and timing, the one-bar delay, forced-close charges and the write-off settlement, non-positive-equity invalidity, identity-break detection and long/short attribution.
- No-borrowing tests: on synthetic data with a bull market, thin volume and capped fills, a long-only `TREND_LONG` and an `EW_LONG` book never leave cash below zero at a fill and never below `-2%` of equity at a mark, an increasing fill is clamped when the side is full, and a deliberate cash leak is detected as an identity break. The xs `EW_LONG` defect (a fully invested long-only book paying costs) cannot occur because the budget is 0.9 and increasing fills are clamped.
- Causality: the panel truncated at any close equals the full-data panel at that close for every feature and the weights of every cell; a perturbation of every bar after the cut changes nothing before it; an appended delisting changes no earlier decision except through the documented notice pairing, with cuts placed inside a no-notice delisting window (no forced flag, weights and equities unchanged) and inside a notice window (unchanged before the release row).
- The daily-bar construction, regime and `vol_ann` are compared with a verbatim copy of the exploratory pandas code when pandas is installed (the CI core job skips it).

## Evidence and claim boundary (G3)

- `TREND_BREADTH_SUPPORTED` means: on this universe and window the frozen rule passed the gates once. It is class 2 evidence (pre-registered, known-used data; the universe was used by xs, the window is the xs window, the rule was chosen after 11-coin results). It is NOT profitability evidence, not an out-of-sample or forward result, and not authorization of paper trading, live trading or capital. It would justify a further pre-registered, forward protocol only.
- The 441 names and the window are shared with xs, and the regimes of 2021-2026 and the crypto-wide beta are the same ones prior studies saw. The comparison with `EW_LONG` removes the passive long exposure but not the common regime.
- The reference account, the last-completed-hour volume, the Vision funding price proxy and the notice-based forced close are modelling choices; real slippage beyond sqrt impact, partial-fill microstructure, exchange margin, liquidation, borrow and funding-interval changes are not modelled. Drawdown and Sharpe are measured on 4h close marks only.
- `TREND_BREADTH_NOT_SUPPORTED` ends this version; a software or data refusal needs a result-blind amendment before any rerun. No label authorizes orders.

## Authorization

- Reads of the archived data are covered by the owner scope relay [Issue #810 comment 7040247247](https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-7040247247). The same-account review route is [Issue #810 comment 7082120805](https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-7082120805).
- The stage run also needs its own G4 relay on Issue #810 that quotes this protocol id, the digest and the full code commit. The harness fetches both comments twice; each must be owner-authored, on Issue #810 and unchanged between the fetches. The relay is written by the planner and posted from the owner account; the harness verifies identity, issue, quoted tokens and stability, NOT human authorship.
- The checkout must be clean: any modified tracked file or untracked, non-ignored file refuses the run (`git status --porcelain --untracked-files=all`).
- **Gates before the claim:** the output directory must not exist; the universe and notices files must match their pinned sha256; the cross-digest single-use check must pass; the checkout must be clean; the authorization fetches must pass; and the archive preflight (checksums, one member per zip, files present) must pass. Any failure refuses WITHOUT claiming. A failure AFTER the claim (parsing, replay) consumes the stage and needs a result-blind amendment.
- Single use is keyed by protocol id and stage, NOT by digest: before it claims, the run refuses if ANY digest namespace under the protocol id holds a claim or result for `BREADTH`. The atomic lock is an exclusive-create claim at the protocol root (`<ledger>/trend_breadth_v1/BREADTH.claim`); the per-digest subfolder keeps a provenance copy. A harness commit that changes the digest never reopens the consumed stage.
- Outputs are write-once. Run with the locked environment: `uv run --locked python -m tools.trend_breadth run --stage BREADTH --authorization-url <relay> --output-dir <new dir>`. The harness needs only the standard library and numpy.

## Stop rule

- No change to any constant, lookback, sleeve, budget, threshold, cost, window, cell or rule after the stage result. No rescue run.
- A software or data refusal needs a result-blind amendment before any rerun.
- Other rules, other symbols, higher exposure or a fresh holdout are new protocols.
- No label authorizes orders, paper trading or live trading.

## Choices the review should check

- The sleeve is `0.9 / N_t` fixed at entry (the planner suggested `1 / N_t`; the 0.9 budget and the clamp keep long-only cash non-negative). Because `N_t` grows over the window, early sleeves are larger than late ones and the side clamp can block late entries in a full book; `clamped_fills` shows how often.
- `N_MIN = 20` is an added guard (not in the brief): with few eligible names the book would concentrate. It is fixed before any `N_t` was computed from these data.
- Eligibility uses a finite regime and volatility, which means 200 consecutive complete daily bars of the current episode, so an isolated missing hour blanks a symbol for up to 200 days.
- Delistings without a recorded notice are write-off only (no hindsight); the notice pairing uses the event time, the one disclosed exception to append-future-rows invariance.
- The equal-weight control re-targets at month starts and at the first close with `N_t >= 20`.
- Vision funding carries no mark price; the 1h open of the funding hour is the price proxy.
- Listing age is measured from the first complete 4h bar of the current episode.
- `HIGH_COST` doubles the whole per-side rate including impact; the forced-close 5% is unchanged.
- The harness needs 1-2 GB of memory (dense panels of 12,228 closes x 441 symbols).
