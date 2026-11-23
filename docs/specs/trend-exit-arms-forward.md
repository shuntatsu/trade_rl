# Trend short-exit arms, prospective FORWARD test (`trend_exit_arms_v1`)

Status: Active (not frozen until an independent review approves the exact head). This protocol is fixed before the first FORWARD bar (2026-12-01 00:00 UTC) exists. No stage has run. No Binance kline dated 2026-12-01 or later was read to write it, and no economic result was produced for it.

Protocol digest: `d3f2a0b4158ff3b7cb751ff8639587322ed80136ad7828afcb60524ee9a414b4`

## Question

An exploratory research line on known-used Binance USDT-M perpetual data (2023-01 to 2026-07, issue #810 planner notes) found that a daily-regime trend book earned most of its profit from a few long-lived long trades, while its short leg lost money and was squeeze-prone. Three small, frozen ways of treating only the short leg looked better than the plain rule in that history. This protocol runs the plain rule and those three variants prospectively, side by side, on the same capital, the same entries, the same costs and the same funding, and corrects for testing three arms at once.

**Primary question:** does any short-exit arm improve on `A0_RULE` over the 12 months from 2026-12-01?

**Hypothesis:** trend P&L is fat-tailed and side-asymmetric. Long winners are rare and long-lived, so cutting longs early destroys the book; shorts are faster and squeeze-prone, so a de-risking rule on the short leg alone (a trailing stop, half size or a 30-day time limit) loses less to squeezes than it gives up.

**Counter-hypothesis:** selection bias. The arms were chosen after seeing the H-EXIT results on the same 2023–2026H1 data; their gains over the rule were inside bootstrap noise (each 95% bootstrap interval of the 4h mean difference included 0); and 2025 was negative for every arm. A different choice of arm would have looked as good by chance. The apparent short-leg problem may be one bad regime, not a structural asymmetry.

**Cheapest falsifier:** this FORWARD year. It needs no new data source and no code beyond this harness. Nothing else is claimed to test the arms.

**Evidence class of the arms:** class 1: known-used data, chosen post hoc, inside noise. That is the reason only a forward test can decide, and the reason no known-data number below is evidence.

## Prior exposure (disclosed, known-data, not evidence)

- Exploratory numbers from the research registry report `out/h_exit_asym/report.txt` (ALL11, 2023-01..2026-07, the research harness's book with FS1 costs and actual funding; research capital 1.0, so its impact term was near zero). They are known-data and are not evidence for this protocol:

  | Policy | Total return | Sharpe | Max DD | Mean 4h diff vs rule, bp | 95% percentile interval of the diff, bp |
  |---|---|---|---|---|---|
  | rule exit (A0) | +63.6% | 0.55 | 40.4% | 0 | n/a |
  | short trailing stop 3 ATR (A1) | +94.3% | 0.68 | 38.5% | +0.20 | [-0.04, +0.46] |
  | short size halved (A2) | +86.3% | 0.67 | 35.5% | +0.10 | [-0.46, +0.56] |
  | shorts closed after 30 days (A3) | +84.9% | 0.64 | 40.7% | +0.13 | [-0.18, +0.42] |

  Per-year 2025: rule −10.2%, A1 −10.8%, A2 −9.7%, A3 −16.8%. Every interval includes 0.
- The research registry holds 66 exploratory trials for this line (H_EXIT and H_EXIT_ASYM). The arms were chosen post hoc from them.
- The planner sees post-2026-11 prices through other protocols' interim reports. This is why this protocol is strictly prospective: nothing dated 2026-12-01 or later has been read here, and the regime warm-up ends before it.
- The FS1 seal is unaffected: this is a separate protocol on public Binance data and touches no FS1 artifact.
- The learned arms of the exploratory line (an RL exit and a supervised value exit) are NOT included. Their frozen-model provenance cannot be made reproducible in this repository in time; this is a disclosed exclusion, not a finding about them.

## System (frozen)

All times are UTC. A 4h bar is labelled by its open time and a decision happens at its close `c`. Inputs at `c` use bars that closed at or before `c`.

**Universe.** Binance USDT-M perpetuals: BTCUSDT ETHUSDT SOLUSDT BNBUSDT XRPUSDT ADAUSDT DOGEUSDT LINKUSDT DOTUSDT AVAXUSDT LTCUSDT. Each coin has a 1/11 sleeve of equity.

**Data.** 1h klines from the public `https://fapi.binance.com/fapi/v1/klines` endpoint, loaded from 2021-01-01 00:00. 4h and 1d bars are aggregated from 1h bars exactly as the exploratory `feat.agg` does (open of the first hour, max high, min low, close of the last hour, sum of quote volume), labelled by open time. A bar exists only if every one of its hourly bars exists; otherwise it is blank (NaN) and so is every value that uses it. The 1d bar is the UTC day. At a 4h decision close the latest COMPLETED daily bar is used (the causal daily forward-fill alignment).

**Native source-close admission.** Every raw 1h Binance kline (including
history warm-up and the preclaim connectivity probe) uses the versioned strict
BinanceSourceClosePolicy.STRICT_SOURCE_CLOSE_V2 clock contract. Native open_time
and close_time must be integer-form timestamps in the SAME unit (ms or us).
The exact close is (normalized_open_ms + 3600000) * unit - 1. Native us opens
must align to ms before the grid conversion; submillisecond source-close drift
is never rounded away. A premature, late or missing source close, mixed units,
or unaligned opening fails the whole page/stage before any aggregation, volume
sizing, order or decision. This applies equally to interim and final stages;
it is never treated as a tolerated missing-hour gap. Truly absent hourly
rows still use the existing NaN-hole admission rule. Raw HTTP response bytes
are still preserved write-once for hash/readback. Historical frozen Dataset
identities are not reinterpreted.

**Regime and direction.** `regime = mean over N ∈ {20, 50, 100, 200} of sign(close_1d − SMA_N(close_1d))`. Direction is +1 if `regime ≥ 0.5`, −1 if `regime ≤ −0.5`, else 0 (NaN gives 0).

**Size.** `min(1, 0.5 / vol_ann)`, where `vol_ann` is the 30-day std (ddof 1) of daily log returns × √365. The size is fixed at entry while the direction is unchanged and recomputed on a new entry or a flip. A row whose close or `vol_ann` is not finite makes no decision for that coin (the previous position is held), as in the exploratory code. The target weight of a coin is `position × multiplier / 11` of equity.

**Trades.** The rule book starts flat at the first decision, 2026-12-01 00:00. A trade is a maximal run of one position sign; a zero position ends it, and a flip starts a new one. The rule exit is the end of the trade (the regime leaves the direction). The longs of every arm always keep the rule exit.

**Daily ATR.** The causal definition of the exploratory `h_exit_lib`: the 84-bar (14-day) mean of the 6-bar (24h) rolling high-low range built from the 4h bars, `rolling(84).mean()` of `max(high, 6) − min(low, 6)`. It is NaN if any input bar in the windows is missing.

**Arms.** Arms differ only in the short leg. An exit only de-risks: it sets the trade's multiplier to 0 for the rest of the trade, and there is no re-entry until a new trade starts (the rule direction changes, then a new entry occurs), exactly as in the exploratory code.

| Arm | Short leg |
|---|---|
| `A0_RULE` | rule exit |
| `A1_SHORT_TRAIL3` | exit at the first decision row after entry whose close is at least 3 daily ATR (that row's ATR) above the lowest close since entry (entry row included) |
| `A2_SHORT_HALF` | short size halved (multiplier 0.5 for the whole trade) |
| `A3_SHORT_TIME30` | exit at the decision row 180 4h bars after the entry row if the trade is still open |
| `C_HOLD_BTC` | reported only: 1× long BTCUSDT perpetual bought at the first close, actual funding |
| `C_CASH` | reported only: equity 1 |

A short trade whose entry row has no finite positive daily ATR keeps the rule exit under A1 and A3 (the exploratory policy runner skips a trade without entry features). The exit rules are ported from `h_exit_lib.exit_trailing`, `exit_time` and `h_exit_asym.combine`; the tests compare them with a trade-first reformulation, and a local one-off run compared them with the exploratory modules themselves on synthetic data (identical multipliers).

## Accounting (actual units; corrects the screens)

The book holds actual contracts and cash. It follows `btc_trend_ensemble_v1` Amendment 2 and the research harness's `book.py`: a symbol whose target weight is unchanged keeps its units (no free re-weighting), however far prices have drifted.

- **Capital.** 100,000 USDT reference account. The size matters only through the participation of each fill; the research harness used capital 1.0, where the impact term was close to zero.
- **Sizing.** A changed target weight is converted to units once, at the fill: `units = w · E / price`, with `E` the equity marked at the fill prices before that event's fills.
- **Costs.** The repository `ExecutionCostConfig` formula with the FS1 constants: a fill of notional `N` pays `N · multiplier · (0.0005 fee + 0.0002 spread + 0.0001 · sqrt(participation))`, `participation = N / V`, with no price slippage. `multiplier` is 1 (BASE, DELAY) or 2 (HIGH_COST, which doubles the whole per-side rate). The cost is debited from cash.
- **Participation cap.** A fill is capped at 5% of `V`, the quote volume of the 1h bar that closes at the fill close (the last 1h bar closed at or before it). An unfilled residual is carried toward the same target units while the target is unchanged. This differs from the research harness, which used the volume of the 1h bar opening at the fill; the hour that closes at the fill is used here so that nothing after the fill close is read.
- **Funding.** The ACTUAL Binance funding of each symbol from the public `https://fapi.binance.com/fapi/v1/fundingRate` endpoint. An event stamped `t` with `c_{j−1} ≤ t < c_j` is paid in bar `j` on the units held after the fill at `c_{j−1}`: `cash −= units · price · rate` (longs pay a positive rate). `price` is the record's mark price, else the 1h open of the funding hour (the number of fallbacks is recorded). The funding history of every symbol must cover the window without a gap above 8h plus one minute.
- **Timing.** A decision at the close `c_j` fills at that close (BASE, HIGH_COST) or at `c_{j+1}` (DELAY). Decisions are made at the closes from 2026-12-01 00:00 to 2027-11-30 20:00; P&L is marked at every close through 2027-12-01 00:00 (2,190 4h bars). The book starts flat with 100,000 USDT.
- **Validity.** A cell is invalid if at any step: equity is not positive; the gross notional after a fill event in which at least one fill happened exceeds 1.5 times equity (gross is `Σ|units · price| / equity`; nominal gross is at most 1 and exceeds it only through price drift). Gross is NOT an invalid condition at a mark or at a decision event without a fill: drift-only excursions are reported as `max_held_gross` (every mark) and `max_gross_at_fills`, and gate nothing; an accounting identity breaks by more than `1e-9` relative (the cash and mark-to-market reconciliation `equity = capital + price P&L + funding − costs` at every mark, and equity after a fill event equals equity before it minus its costs); a price is not positive. A missing hour from the open of the first decision bar onward, in any symbol, is a data problem that makes the stage INVALID. Missing hours before it only blank the affected bars. There is no gross clamp: a breach is invalid, not scaled.
- **Reporting.** Each cell reports total return, max drawdown, Sharpe, fills, capped fills, turnover, costs, funding, price P&L, funding and costs per side (long and short), max gross at fills and at marks.

## Cells (frozen)

Each of `A0_RULE`, `A1_SHORT_TRAIL3`, `A2_SHORT_HALF`, `A3_SHORT_TIME30` is run in three variants: `BASE`, `HIGH_COST` (costs ×2) and `DELAY` (fill one 4h bar later). That is 12 decision cells (`<arm>__<variant>`), plus the reported-only `C_HOLD_BTC` and `C_CASH`, in BASE.

## Stages

| Stage | Data | Decisions | Role |
|---|---|---|---|
| FORWARD interim | Binance USDT-M 1h klines from 2021-01-01 to the last settled 4h close (the newest 5 minutes are ignored), funding from 2026-12-01 | from 2026-12-01 00:00 | monthly progress, `INTERIM_NOT_A_DECISION` |
| FORWARD final | the same, to 2027-12-01 00:00 | 2026-12-01 00:00 to 2027-11-30 20:00 | decisive, prospective |

- Interims may run from 2026-12-02, once per UTC month. At an interim, all arms stop together (`STOPPED_EARLY`) if the `A0_RULE__BASE` max drawdown is 30% or more. A ledgered `STOPPED_EARLY` blocks every further interim and the final and ends the line with the meaning `NO_ARM_IMPROVES`. An interim also records the decision statistics for information; they decide nothing.
- The 30% stop applies to interims ONLY. The final never stops on it: the arm question is a relative comparison with `A0`, so the final ALWAYS decides it (when all cells are valid), also when `A0_RULE__BASE` crossed 30% after the last interim. Every valid stage result records `a0_base_max_dd` and `a0_stop_crossed` (true when the `A0_RULE__BASE` max drawdown over the whole stage window is 30% or more); in a final it is a reported fact and changes no label.
- The final runs once, at or after 2027-12-02.
- Every run re-downloads the full history (the work directory must be empty). Raw responses are cached write-once and hashed in the receipt.

## Decision rule (frozen)

For each challenger `Ai` (`A1_SHORT_TRAIL3`, `A2_SHORT_HALF`, `A3_SHORT_TIME30`), `IMPROVES(Ai)` requires ALL of:
1. All cells are valid and the data have no problem.
2. The paired moving-block bootstrap of the 4h return difference `Ai − A0` in BASE (the equity-return series of the two cells; block 30 bars, overlapping blocks, 5,000 draws, seed 0, one shared set of block draws for all arms) has a lower one-sided 95% bound (the 5% quantile of the bootstrap means) above 0, AND the arm is rejected by the Holm step-down at α = 0.05 across the three arms, using the one-sided bootstrap p-values `(1 + #{mean ≤ 0}) / (5000 + 1)`: sorted ascending with index `r = 0, 1, 2`, reject while `p_(r) ≤ 0.05 / (3 − r)` (thresholds 0.05/3, 0.05/2, 0.05), stopping at the first failure.
3. The mean 4h difference `Ai − A0` is positive in the HIGH_COST cells and positive in the DELAY cells (each against the same-variant `A0`).
4. `Ai` BASE max drawdown ≤ `A0` BASE max drawdown + 5 percentage points.

| Label | Condition |
|---|---|
| `ARM_IMPROVES:<ids>` | the final with every cell valid, for the comma-separated sorted ids of the arms satisfying 1–4 |
| `NO_ARM_IMPROVES` | the final with every cell valid and no arm satisfying 1–4 |
| `INVALID` | any cell invalid or a data problem |

Secondary, reported but not decisive: whether `A0_RULE__BASE` has a higher Sharpe than `C_HOLD_BTC`. Sharpe is computed from 4h simple returns of the equity series, population standard deviation, × √(6·365), with no risk-free rate. The long and short leg price P&L, costs and funding are reported for every cell.

Interim labels are `INTERIM_NOT_A_DECISION`, `STOPPED_EARLY` (interims only; `A0_RULE__BASE` max drawdown ≥ 30% at that interim) and `INVALID`. The final has no `STOPPED_EARLY` label: its labels are exactly `ARM_IMPROVES:<ids>`, `NO_ARM_IMPROVES` and `INVALID`, whatever `a0_stop_crossed` says. No interim result is a decision, except that `STOPPED_EARLY` is terminal.

## Mechanism chain (G1)

1. Daily SMA-vote regimes are persistent, so the sign of price against slow averages predicts the sign of the next weeks' return more often than not; this is the rule book's whole entry edge.
2. Trend P&L is dominated by few long-lived winners, mostly on the long side; truncating them costs more than it saves (the exploratory long-exit truncation hurt).
3. Short trades in an uptrending or squeeze-prone market are faster and give back gains; a trailing stop, a smaller size or a time limit on the short leg only reduces the giveback without touching the long winners.
4. Costs and funding are paid on the actual trades and positions; a de-risking arm trades more (an exit and no earlier re-entry), so its cost must stay below its saving: HIGH_COST and DELAY test that.

Each link can fail: persistence can end (a mean-reverting year); the short-leg asymmetry can be one regime (a bear year helps shorts and makes any stop a cost); an exit that fires on noise only adds cost. The counter-hypothesis is that link 3 is selection bias.

## Independent oracle (G2)

- The tests recompute the equity of every arm and variant with a separate plain-python loop (dicts and floats, equity as cash plus units times price, a trade-first reformulation of the rule positions and exits) on synthetic data with random regimes, volatility, ATR, volume (so the participation cap binds), funding and prices, and require agreement to 1e-9 for 3 seeds × 4 arms × 3 variants.
- Hand-computed cases cover the actual-units rule (price 100 to 200 to 100 returns exactly flat), the repository cost formula (pinned against `ExecutionCostConfig` and its sqrt-participation impact), the 5% participation cap and carried residual, funding sign and timing, the one-bar delay, gross and non-positive-equity invalidity, identity-break detection and long/short attribution reconciliation.
- The daily-bar construction, regime, `vol_ann` and daily ATR are compared with a verbatim copy of the exploratory pandas code (`feat.agg`, the `q2` regime and the `h_exit_lib` ATR) when pandas is installed, with and without missing hours.
- Causality: the frame truncated at any close equals the full-data frame at that close; the exit multipliers truncated at any row equal the full-data multipliers; the interim replay is an exact prefix of the later replay for every cell.

## Evidence and claim boundary (G3)

- `ARM_IMPROVES:<ids>` means one prospective year, one market regime, 365 days of about 2,190 overlapping 4h observations, passed these gates with the frozen constants and a multiplicity correction over three arms. It is not an edge estimate and not a result about other years, symbols or venues. The decision rule has low power: in the known data the arms' mean 4h differences were 0.1–0.2 bp with intervals that included 0, so `NO_ARM_IMPROVES` is the likely label and is NOT evidence that the arms are worse than the rule.
- A pass does not authorize orders, paper trading, live trading or capital. It would justify a further pre-registered protocol only.
- `NO_ARM_IMPROVES` ends this version. A software or data refusal needs a result-blind amendment before any rerun.
- Drawdown and Sharpe are measured on 4h close marks only; intrabar drawdown, which is larger, is not measured.
- Real slippage beyond the sqrt-impact term, partial-fill microstructure, margin rules, liquidation, borrow and funding-interval changes are not modelled.

## Authorization

- FORWARD reads are covered by the owner scope relay [Issue #810 comment 7040247247](https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-7040247247). The same-account review route is [Issue #810 comment 7082120805](https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-7082120805).
- Each stage run also needs its own G4 relay on Issue #810 that quotes this protocol id, the digest and the full code commit.
- The stage-specific G4 relay additionally requires exactly one standalone line `G4 claim: <claim name>` with the exact UTC `claim_name` (`FORWARD_INTERIM_YYYY-MM` or `FORWARD_FINAL`). An ambiguous/missing/multiple claim line fails closed even when the protocol id, digest and commit match. The relay comment ID is consumed **once across all stages, months and digest namespaces of this protocol**: before any stage-window data access the harness scans existing claims, exclusively creates a `g4-<comment ID>.claim` marker at the protocol ledger root, and refuses reused or malformed historical claims. The stage root claim is already consumed at that point and is never rolled back on a G4 conflict or later failure. These are additional software gates, not a G4 authorization grant. The ledger root must be a durable, shared, absolute filesystem path to enforce cross-run uniqueness.
- The harness must fetch both comments twice. Each must be owner-authored and unchanged between the fetches.
- The checkout must be clean: any modified tracked file or any untracked, non-ignored file refuses the run (`git status --porcelain --untracked-files=all`).
- The G4 relay is written by the planner and posted from the owner's GitHub account. The harness verifies account identity (owner login), the issue, the quoted protocol id, digest and commit, and that the content is unchanged between two fetches. It does NOT verify human authorship; the operator confirms the owner's approval before each stage.
- Before the claim, every run (a) completes the authorization fetches (the G4 relay and the scope comment, each fetched twice) and (b) passes a minimal connectivity probe: one BTCUSDT klines page and one funding page dated 2026-10-01..03, which reads no stage-window data. Any failure refuses the run WITHOUT claiming, so an outage or an authorization problem cannot burn the single-use stage. The operator accepts that a failure AFTER the claim (during the stage-window fetch or the replay) consumes the stage and needs a result-blind amendment.
- Outputs are write-once. A local ledger claim must be created exclusively before any stage-window data is fetched: once per UTC month for interims and once for the final, keyed by protocol id and stage as described below.
- Single use and the stop state are keyed by protocol id and stage (and by UTC month for interims), NOT by digest. Before it claims anything, a stage run refuses if ANY digest namespace under the protocol id already holds a claim or result for the same stage name (the final, or that interim month), or a `STOPPED_EARLY` result. The atomic single-use lock is an exclusive-create claim at the protocol root (`<ledger>/trend_exit_arms_v1/<stage name>.claim`); the per-digest subfolders (`<ledger>/trend_exit_arms_v1/<digest>/`) keep a provenance copy only. A harness commit that changes the digest therefore never reopens a consumed stage or a stopped line.
- Run with the locked environment: `uv run --locked python -m tools.trend_exit_arms_forward run ...`. The receipt records the numpy version. The harness needs only the standard library and numpy.

## Stop rule

- No change to any constant, lookback, multiplier, cost, window, cell or rule after any stage result.
- No rescue run.
- The 30% `A0_RULE__BASE` drawdown stop applies to interims only. The final decides the arm question regardless and records `a0_stop_crossed`; neither the stop nor its crossing is a reason to skip, rerun or reinterpret the final.
- A refusal or invalid result caused by a defect outside the economics needs a result-blind amendment before any rerun.
- Other exits, other arms, the excluded learned exits, other symbols or higher exposure are new protocols.
- No label authorizes orders, paper trading or live trading.

## Choices the review should check

- The arms are ported from the exploratory short-leg experiments (`h_exit_asym`), including the rule that a trade without an entry ATR keeps the rule exit. The exploratory code applies A2 to every short trade including one-row trades; this port does too.
- The book starts flat at the first decision, so a trend that began before 2026-12-01 is entered at that close; A1 trails from that close and A3 counts its 180 bars from it. The exploratory code started flat at its first test bar (2023-01-01), which is the same convention.
- The reference account is 100,000 USDT (not the research harness's 1.0) and the participation volume is the last completed hour (not the hour opening at the fill). Both make the impact term more conservative than the exploratory numbers; the exploratory returns are not adopted.
- There is no gross clamp, unlike `mtf_combo_v1`: the book's nominal gross is at most 1, so a breach of 1.5 at a fill means 50% drift and invalidates the cell instead of being scaled away.
- HIGH_COST doubles the whole per-side rate, including the impact term, as the research harness's `scaled` does.
- `C_HOLD_BTC` and `C_CASH` are reported in BASE only.
- Interim decision statistics are computed on partial data for information; with fewer than a few days they are degenerate.
- Duplicated code: the data fetch, authorization, ledger and clock machinery is copied from `mtf_combo_forward` (PR #1333, not merged) so this PR is self-contained. A shared helper module is a follow-up.
- Dependencies: the tests use pandas only for the optional frame parity oracle (`uv run --with pandas`); the CI core job skips it.
