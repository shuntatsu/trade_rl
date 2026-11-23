# Hierarchical MTF long/short perp system, prospective FORWARD test (`mtf_combo_v1`)

Status: Active (not frozen until an independent review approves the exact head). This protocol is fixed before the first FORWARD bar (2026-12-01 00:00 UTC) exists. No stage has run. No Binance kline dated 2026-12-01 or later was read to write it.

Protocol digest: `f601d84e51cfcf6c31ad674b134cb3cd84ea53e8b0e24f02f285e9d6b62b8fa7`

## Question

An exploratory screen series on known-used Binance USDT-M perpetual data (2021-01 to 2026-07, issue #810 planner notes) found a long/short system whose reported return rested on three layers:
- a daily trend regime (a vote of four SMA lookbacks);
- a gradient-boosting classifier for the 30-bar direction, used as a consistency filter for the regime;
- portfolio volatility targeting.

**Hypothesis:** the system, run as frozen below on a book of actual contracts and cash with actual funding, keeps a positive and cost-robust return with drawdown inside the owner's 25% budget over the 12 months from 2026-12-01.

**Counter-hypothesis:** the known-data edge is selection bias plus beta. About 60 variants (features, filters, barriers, horizons, bands, targets) were tried on 2021–2026H1, and the surviving constants were chosen by looking at that same history. The 2023–24 crypto uptrend with a vol target explains the return, and the ML filter adds nothing outside the window it was tuned on.

**Cheapest falsifier:** this FORWARD year. It needs no new data source and no code beyond this harness. Nothing else is claimed to test the system.

## Prior exposure (disclosed, known-data, not evidence)

- The exploratory screens `q2`–`q6` on 2021-01..2026-07 chose every constant below: the four regime lookbacks and the ±0.5 thresholds, the 30-bar horizon, the 0.4/0.6 bands, the model hyperparameters, `size = min(1, 0.5/vol)`, the 50/50 sleeve mix, the 20% vol target, the 2.0 cap, the 180-bar window and the 20% leverage band.
- The planner's known exploratory results, which are known-data and are not evidence for this protocol:
  - COMBO with a 20% vol target, 2023–2026H1: +133%, Sharpe 1.19, max drawdown 25.5%.
  - With 2022 included: +76% to +119% and drawdown 25–28% across six perturbations.
  - BTC hold over the same span: +36%, drawdown 67%.
- Those figures used the screens' accounting, which re-weighted every book to its nominal fraction every 4h bar without a fee and charged a 1 bp per 8h funding proxy. They are not corrected-accounting performance and are not adopted here.
- The planner has public knowledge of crypto history up to 2026.
- 2026-08..11 perpetual data were partly exposed by other protocols' interim reports and screens. This is why FORWARD here is strictly prospective: nothing dated 2026-12-01 or later has been read, and the training data ends before it.

## System (frozen)

All times are UTC. A 4h bar is labelled by its open time and a decision happens at its close `c`. Features at `c` use bars that closed at or before `c`.

**Universe.** Binance USDT-M perpetuals: BTCUSDT ETHUSDT SOLUSDT BNBUSDT XRPUSDT ADAUSDT DOGEUSDT LINKUSDT DOTUSDT AVAXUSDT LTCUSDT.

**Data.** 1h klines from the public `https://fapi.binance.com/fapi/v1/klines` endpoint, loaded from 2021-01-01 00:00. A 4h or 1d bar exists only if all of its hourly bars exist; otherwise it is blank (NaN) and so are the features that use it. 4h and 1d bars are aggregated from 1h bars (open of the first hour, max high, min low, close of the last hour, sums of quote and taker-buy quote volume). The 1d bar is the UTC day. At a 4h decision close the latest COMPLETED daily bar is used, and the 1h feature of the hour that closes at `c`.

**Native Binance source-close admission.** Every raw 1h candle, including
historical training warm-up and every interim/final stage, passes versioned
BinanceSourceClosePolicy.STRICT_SOURCE_CLOSE_V2. Native open_time and
close_time must be integer-form timestamps in one consistent unit (ms or us),
with exact close (normalized_open_ms + 3600000) * unit - 1. Native us opens
normalize to millisecond UTC grids only when exactly aligned; submillisecond
close errors are never rounded away. Invalid/early/late/missing source close,
mixed units or an unaligned open hard-fails the fetched page BEFORE hourly,
daily or 4h aggregation, feature computation, taker-buy quote volume or
execution sizing. Such rows are not silently treated as completed candles or
tolerated missing gaps. Genuinely absent hours still follow the separate
NaN-hole and invalid-stage policy. Raw downloaded bytes remain write-once for
audit; no frozen Dataset identity is retroactively modified.

**Features.** The 42 columns of the exploratory bank, in this order (`FEATURE_COLUMNS` in the harness):
- for each of 4h, 1h and 1d: RSI(14), MACD histogram / close, PPO(12,26), stochastic(14,3), Ichimoku score (the cloud shifted +26 bars, never −26), bull / bear / net counts of the multi-indicator divergence (9 indicators: RSI, MACD histogram, stochastic, CCI, MFI, OBV, ROC, Williams %R, taker-buy ratio; pivots 5/3, gap 60, life 6), SMA20 and SMA50 log distance, 1- and 6-bar log returns, 30-bar return std;
- the daily `regime`, `dsma200_1d` and `vol_ann` (30-day std × √365).

They are computed in numpy with the same rolling and EWM rules as the exploratory pandas code, including its NaN rules. The tests prove causality by truncation and prove numeric parity against a verbatim pandas copy (G2).

**Regime.** `regime = mean over N ∈ {20, 50, 100, 200} of sign(close_1d − SMA_N(close_1d))`. The rule direction is +1 if `regime ≥ 0.5`, −1 if `regime ≤ −0.5`, else 0 (NaN gives 0).

**ML direction model.**
- `HistGradientBoostingClassifier(max_depth=3, max_iter=300, learning_rate=0.03, l2_regularization=2.0, min_samples_leaf=300, random_state=0)` with the library defaults for everything else (including early stopping `auto`), pooled over the 11 coins on the raw (unaligned) 42 features.
- Label: `close[t+30] / close[t] − 1 > 0` (30 4h bars).
- Training set: every 4h bar from 2021-01-01 with a finite close, a finite regime and a finite label whose label bar `t+30` OPENS before 2026-12-01 00:00. Every hourly bar behind a training row is therefore dated before 2026-12-01.
- The model is trained once at the start of each stage run and never retrained during FORWARD. Because the training set is the same every time, every run of the stage trains the same model. The harness fits twice on the same data and records `deterministic` (exactly equal training probabilities); the receipt carries a sha256 of those probabilities, and a later run whose hash differs from any earlier ledgered run under the protocol id, in any digest namespace, is INVALID (model drift).
- Bands: the 0.4 and 0.6 quantiles of the training probabilities. `mlb = +1` if `p > q0.6`, `−1` if `p < q0.4`, else 0.

**Sleeve A: BTC rule long.** BTCUSDT direction is `max(rule, 0)`. Size `min(1, 0.5/vol_ann)` is fixed at entry while the direction is unchanged (no re-sizing).

**Sleeve B: alt AGREE long/short with BTC filter.** For each of the 11 coins (1/11 slot each):
- enter `rule` only when `rule ≠ 0` and `mlb == rule`;
- hold while `rule == current` and `mlb ≠ −current`; otherwise exit to flat (a flip passes through flat and re-enters on a later bar);
- a non-BTC short is allowed only while the BTC rule direction is −1 (regime ≤ −0.5), checked at every bar, so a short is closed when BTC leaves the down regime;
- size `min(1, 0.5/vol_ann)` at entry, fixed while the direction is unchanged.
A bar whose `vol_ann` is not finite makes no decision for that coin (it holds), as in the screens.

**Portfolio.** COMBO nominal weights are 0.5 × sleeve A and 0.5 × sleeve B (each coin's sleeve-B weight is `size / 11`). Portfolio vol targeting follows the screens' `q5.scale`:
- the input is the trailing 180 4h-bar realized return std (population, × √(6·365)) of the UNSCALED book (a shadow book with leverage 1 and the same fills, fees and funding);
- `L = min(2.0, 0.20 / σ)`, or 2.0 if `σ ≤ 1e-6`; L changes only when `|L_new/L − 1| > 0.20`;
- `L = 1.0` until 180 in-stage bars of the unscaled book exist; the first update is at the decision whose trailing window holds 180 in-stage returns.

## Accounting (actual units; corrects the screens)

The book holds actual contracts and cash. It follows the precedent of `btc_trend_ensemble_v1` Amendment 2 (PR #1331): quantities are held between accepted changes and there is no free re-weighting.

- **Components.** Sleeve A on BTC and each sleeve-B coin are separate components with their own signed units. A component's units are set only when its nominal weight changes (entry, exit or flip) or when the leverage changes, in which case every component's units are scaled by `L_new / L_old`; the only other rescaling is the fill-time cap below.
- **Sizing.** A changed component is set to `units = L · nominal · E / price`, where `E` is the equity marked at the fill price just before the fill (after funding, before that fill's fee). Cash may be negative (margin).
- **Fill-time gross cap.** The cap applies at fills only. If a fill would leave the gross notional above 2 times the post-fee equity, the fill's candidate units are scaled by the largest factor `s ≤ 1` that lands on the cap (the fee depends on `s`). The factor multiplies ALL component units of the candidate book, including components the fill does not change: the cap is a bound on the whole book's gross after the fill, so scaling only the changed components could not always reach it (held units may already sit near or above the cap after a loss). Identity and fees stay exact, because the fee is charged on the actual per-coin trade that results. This binds, for instance, when sleeve A alone runs at leverage 2 with size 1, because the entry fee lowers the equity the fraction was sized on. The number of clamps and the smallest factor are recorded per cell. Between fills the held gross is NOT capped: held units are never force-reduced, so after a loss the marked gross can exceed 2 (a book at gross 2 that loses 10% marks at 2 × 0.9 / 0.8 = 2.25). The largest marked gross is reported as `max_held_gross` and gates nothing.
- **Fees.** `fee · Σ |Δ net units of a coin| · price`, charged at the fill on actual notional per coin (net of the components of that coin), at 6 bp one-way (12 bp in HIGH_COST). Leverage changes pay fees on the notional they trade.
- **Funding.** The ACTUAL Binance funding of each symbol from the public `https://fapi.binance.com/fapi/v1/fundingRate` endpoint. An event stamped `t` with `c_{j−1} ≤ t < c_j` is paid in bar `j` on the units held after the fill at `c_{j−1}`: `cash −= units · price · rate` (longs pay a positive rate). `price` is the record's mark price, else the 1h open of the funding hour (the number of fallbacks is recorded). The funding history of every symbol must cover the window without a gap above 8h plus one minute.
- **Timing.** A decision at the close `c_j` fills at that close (BASE) or at `c_{j+1}` (DELAY). Decisions are made at the closes from 2026-12-01 00:00 to 2027-11-30 20:00 (the last closes before 2027-11-30 23:59); P&L is marked at every close through 2027-12-01 00:00 (2,190 4h bars). The book starts flat at the first decision with equity 1.
- **Validity.** A cell is invalid if at any step: equity is not positive; gross leverage after a fill exceeds `2 + 1e-9` (gross is `Σ|units·price| / equity` over the net per-coin positions; after the fill-time cap above this is an assertion that the clamp worked; the gross at every mark is reported as `max_held_gross` but gates nothing, because held units drift after losses); an accounting identity breaks by more than `1e-9` (the cash and mark-to-market reconciliations of every bar and every fill); a price is not positive. A missing hour from the open of the first decision bar onward, in any symbol, is a data problem that makes the stage INVALID. Missing hours before it only blank the affected bars.

## Cells (frozen)

| Cell | Definition | Gates |
|---|---|---|
| `COMBO__BASE` | COMBO, vol target on, 6 bp per fill, fills at the decision close | yes |
| `COMBO__HIGH_COST` | as BASE, 12 bp per fill | yes |
| `COMBO__DELAY` | as BASE, fills at the next 4h close | yes |
| `BTC_RULE__BASE` | sleeve A alone (weight 1), vol target 20% / cap 2 | reported |
| `ALT_AGREE__BASE` | sleeve B alone (weight 1), unscaled | reported |
| `HOLD_BTC` | 1× long BTCUSDT perp bought at the first close, entry fee, actual funding | reported |
| `CASH` | equity 1 | reported |

## Stages

| Stage | Data | Decisions | Role |
|---|---|---|---|
| FORWARD interim | Binance USDT-M 1h klines from 2021-01-01 to the last settled 4h close (the newest 5 minutes are ignored), funding from 2026-12-01 | from 2026-12-01 00:00 | monthly progress, `INTERIM_NOT_A_DECISION` |
| FORWARD final | the same, to 2027-12-01 00:00 | 2026-12-01 00:00 to 2027-11-30 23:59 | decisive, prospective |

- Interims may run from 2026-12-02, once per UTC month. One stops the line (`STOPPED_EARLY`) if the `COMBO__BASE` max drawdown is 25% or more. A ledgered `STOPPED_EARLY` blocks every further interim and the final: the drawdown is a running maximum, so a final could only fail its drawdown gate, and the line is over.
- The final runs once, at or after 2027-12-02.
- Every run re-downloads the full history (the work directory must be empty). Raw responses are cached write-once and hashed in the receipt.

## Decision rule (frozen)

`MTF_COMBO_SUPPORTED` requires all of the following:
1. All cells are valid and the data have no problem.
2. `COMBO__BASE` total return is above 0.
3. `COMBO__BASE` annual Sharpe is at least 0.5. Sharpe is computed from 4h simple returns of the equity series, population standard deviation, × √(6·365), with no risk-free rate.
4. `COMBO__HIGH_COST` and `COMBO__DELAY` total returns are above 0.
5. The max drawdown of every gating cell is below 25%.

| Label | Condition |
|---|---|
| `MTF_COMBO_SUPPORTED` | the final satisfies 1–5 |
| `MTF_COMBO_NOT_SUPPORTED` | the final, with all cells valid, fails any of 2–5 |
| `INVALID` | any cell invalid, a data problem, a non-deterministic or drifting model |

Interim labels are `INTERIM_NOT_A_DECISION`, `STOPPED_EARLY` and `INVALID`. No interim result is a decision, except that `STOPPED_EARLY` is terminal: it ends the line with the same meaning as `MTF_COMBO_NOT_SUPPORTED` (the final can no longer be run, and its drawdown gate could not pass).

## Mechanism chain (G1)

1. Daily SMA-vote regimes are persistent: the sign of the price against slow averages predicts the sign of the next weeks' return more often than not.
2. A classifier on multi-timeframe oscillators and divergences adds a short-memory direction signal; requiring agreement removes entries against it and exits on a flip.
3. Sizing by `0.5/vol` gives each position a comparable risk.
4. Vol targeting on the book's own trailing vol lowers exposure when the book is turbulent and raises it when quiet.
5. Costs and funding are paid on the actual trades and positions.

Each link can fail: persistence can end (a mean-reverting year); the classifier can be noise; a quiet period can precede a break so that the vol target levers into a drop. The counter-hypothesis is that links 2 and 4 carry no information outside the tuning window.

## Independent oracle (G2)

- The tests recompute the COMBO, BTC-rule and alt-AGREE equity with a separate plain loop (a second implementation of the sleeve state machines, the unscaled-then-scaled account sequence, the leverage schedule and the fills, valuing equity as cash plus units times price) on synthetic data, for BASE, HIGH_COST and DELAY, and require agreement to 1e-9.
- Hand-computed cases cover the actual-units rule (price 100 to 200 to 100 returns exactly flat), fees on actual notional, funding sign and timing, and the one-bar delay.
- The feature bank is compared with a verbatim copy of the exploratory pandas code when pandas is installed, with and without missing hours.
- Causality: features truncated at any close equal the full-data features at that close; the interim replay is an exact prefix of the later replay.

## Evidence and claim boundary (G3)

- `MTF_COMBO_SUPPORTED` means one prospective year, one market regime, 365 days of about 2,190 overlapping 4h observations, passed these gates with the frozen constants. It is not statistical significance, not an edge estimate, not a Sharpe forecast and not a result about other years, symbols or venues.
- A pass does not authorize orders, paper trading, live trading or capital. It would justify a further pre-registered protocol only.
- `MTF_COMBO_NOT_SUPPORTED` ends this version. A software or data refusal needs a result-blind amendment before any rerun.
- Drawdown and Sharpe are measured on 4h close marks only; intrabar drawdown, which is larger, is not measured.
- Funding and fees use Binance public history and a flat 6 bp per fill; real slippage, partial fills, margin rules, liquidation, borrow and funding-interval changes are not modelled.

## Authorization

- FORWARD reads are covered by the owner scope relay [Issue #810 comment 7040247247](https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-7040247247).
- Each stage run also needs its own G4 relay on Issue #810 that quotes this protocol id, the digest and the full code commit.
- The harness must fetch both comments twice. Each must be owner-authored and unchanged between the fetches.
- The checkout must be clean: any modified tracked file or any untracked, non-ignored file refuses the run (`git status --porcelain --untracked-files=all`).
- The G4 relay is written by the planner and posted from the owner's GitHub account. The harness verifies account identity (owner login), the issue, the quoted protocol id, digest and commit, and that the content is unchanged between two fetches. It does NOT verify human authorship; the operator confirms the owner's approval before each stage.
- Outputs are write-once. A local ledger claim must be created exclusively before any data is fetched: once per UTC month for interims and once for the final, keyed by protocol id and stage as described below.
- Single use and the stop state are keyed by protocol id and stage (and by UTC month for interims), NOT by digest. Before it claims anything, a stage run refuses if ANY digest namespace under the protocol id already holds a claim or result for the same stage name (the final, or that interim month), or a `STOPPED_EARLY` result. The atomic single-use lock is an exclusive-create claim at the protocol root (`<ledger>/mtf_combo_v1/<stage name>.claim`); the per-digest subfolders (`<ledger>/mtf_combo_v1/<digest>/`) keep a provenance copy only. A harness commit that changes the digest therefore never reopens a consumed stage or a stopped line.
- Run with the locked environment: `uv run --locked --extra forecast-gbm python -m tools.mtf_combo_forward run ...`. The receipt records the numpy and scikit-learn versions.

## Stop rule

- No change to any constant, lookback, band, hyperparameter, cost, window, cell or rule after any stage result.
- No rescue run.
- A refusal or invalid result caused by a defect outside the economics needs a result-blind amendment before any rerun.
- Higher exposure, other symbols or another model are new protocols.
- No label authorizes orders, paper trading or live trading.

## Choices the review should check

- The screens (`q4`) refit the ML model for each test year on earlier data with per-year bands and a 30-bar embargo. This protocol fits ONE model at the start of each stage run on all rows whose label bar opens before 2026-12-01 and uses its bands for the whole of FORWARD. It is a deliberate simplification (no retraining after the start), not the screens' walk-forward, so screen figures do not describe this model's cadence.
- The vol estimate uses in-stage bars only. Warm-up history is used for features, the regime and model training but NOT for the vol estimate: no shadow book exists before the stage, so `L = 1.0` for the first 180 in-stage bars, as in the screens.
- Component-level actual units with leverage scaling and sizing from pre-fee equity (the fee is a second-order shortfall of the nominal weight).
- Gross leverage is capped and checked at fills only. Held units are not force-reduced when a loss lifts the marked gross above 2 between fills; that is reported as `max_held_gross`, not gated. The clamp is an addition to the screens, which had no cap on a fill.
- The BTC filter treats a non-finite BTC regime as "not down", so a short is blocked; the screens allowed it. This never occurs after the 200-day warm-up.
- A funding event stamped at or just after a decision close is paid by the units held after that close's fill.
- `HOLD_BTC` is a perpetual long with funding, not spot.
- Dependencies: scikit-learn comes from the `forecast-gbm` extra; the harness imports it only when it fits the model. The CI core job installs only the `dev` extra, so the tests that need scikit-learn or pandas skip there and run locally with `uv run --extra forecast-gbm --with pandas`.
