# Trend following with fast stops

Status: Active

Pre-registered development protocol and software for a long/short
Donchian breakout with ATR stops across a fixed set of USD-M perpetuals. The owner's
idea (2026-11-10): trend following loses on many trades, but cutting losses quickly can
still make money overall, a low win rate with positive skew. This rule has not been
run on real data and no price path, return or PnL of it was computed to write this
document. That is narrower than "unexposed": the universe and window were read by
earlier work, listed in section 5a, and the protocol below is frozen in `trade_rl.evaluation.trend_stop_study.protocol` (digest recorded in every
result). The machine-readable G0-G3 author packet is
[`report/trend-stop-research-assurance-20261113-after1216.json`](../../report/trend-stop-research-assurance-20261113-after1216.json);
its review status is `UNREVIEWED` until a review of its exact digest and heads is
posted. This is a source-binding-only reissue after merged #1216 changed a
transitively imported funding reader. The original 2026-11-13 revision-2 author
packet remains unchanged in `report/` (digest `8178da9c...`); both packets are
UNREVIEWED and no G4 authorization or economic run is implied.

Revision 2 (2026-11-13). Review 5870390989 of PR #1201 blocked revision 1 (packet
`report/trend-stop-research-assurance-20261110.json`, digest `894fedb0...`, protocol
digest `f57f2d6d...`, kept as historical evidence): the causality probe ran for one
variant only, operator-supplied roster exclusions were unchecked, and later changes to
imported code were not covered. Revision 2 runs one probe per variant and timeframe,
removes the exclusion path (owner decision, issue #810 comment 6818495567: all eleven
coins, no exclusions, a missing archive makes the run `INVALID`), records the code
digest and refuses a dirty tree, and has a new packet pinned to this revision's code.
Frozen protocol digest of revision 2:
`2e5ceb6bdad76deaab503ebf5be714cff1b7b8611b0c44048bd562897f707fda`. The packet digest
is pinned in `tests/evaluation/test_trend_stop_v2_packet.py` and named in the pull
request. Reviews 5879836463 and 5879837718 of that pull request (the same-account
route) blocked its first head because an incomplete Vision cache halted with a
traceback instead of the `INVALID` stated here (the digests above are those of the
fixed head), and noted that a reversal while a fired stop is pending was unspecified.
Reviews 5880408681 and 5880409010 then blocked the next head because a missing
exchange-information snapshot ended in a traceback and because the outcome depended on
the order of the two timeframes; both are fixed here and the digests above are those of
the fixed head.

Step 1 (this document, software only): does the rule have positive expectancy after
costs, funding and hard risk? Step 2 (design note only, section 9): only if step 1 is
`DEVELOPMENT_PROMISING`, a separate protocol may let PPO learn exposure and stop width
on top of the rule.

## 1. Objective, invariants, failure modes, oracle

- Objective: decide, on known-used exploratory data, whether the rule is worth a
  separate authorization request. Not a winner selection, not unused-data, paper or live
  evidence.
- Invariants: causal bars only (a decision at row `t` uses completed bars `<= t`); one
  shared account through the existing replay (`MarketExecutor + BookState`, Dataset
  funding, `PreTradeRisk`); terminal flat; no parallel backtester; nothing after
  2026-07-31 is ever read; existing strategies, defaults and artifacts are unchanged.
- Failure modes: look-ahead in a channel or ATR; a stop that silently never triggers or
  fills at an impossible price; re-entry churn after a stop; gross cap breach; a
  position without a defined stop; cost or funding omitted; a data file reaching past the
  boundary; running without owner authorization.
- Test oracle: a hand-computed (`Fraction`) replay with a stop hit and a gap through the
  stop; mirror symmetry; future-perturbation probe; gate truth table; refusal tests.

## 2. Frozen protocol

| Item | Value |
|---|---|
| Universe | BTC, ETH, BNB, ADA, LINK, DOGE, DOT, AVAX, LTC, SOL, XRP (USDT-margined perpetuals). Chosen by name and listing date, not from any return of this rule (prior exposure of the universe: section 5a). The roster is exactly these eleven symbols: there is no exclusion path, no minimum roster size and no CLI option for either. A Dataset missing any roster symbol, or a Vision-cache archive with recorded HTTP 404 evidence (below), writes an `INVALID` result with no replay (no rejection of the rule). An archive missing without that evidence (not yet downloaded, a transient fetch failure, an empty file, a failed content check) is an operational halt: the CLI exits non-zero with a message naming what is absent, writes no result and does not consume the run. A symbol outside the roster or a cadence or warm-up violation also halts. |
| Timeframes | 4h and 1d, native bars, one decision per bar |
| Signal | Donchian breakout, long and short. Entry: close beyond the prior N-bar extremum (current bar excluded). Signal exit: close beyond the prior M-bar opposite extremum. Opposite entry signal reverses. |
| Variants (at most 2) | `donchian_20_10` (N=20, M=10) and `donchian_55_20` (N=55, M=20), Turtle conventions in bars; never tuned |
| Initial stop | 2.0 x ATR(14, Wilder) from the observed entry-bar open; ATR frozen at the entry signal bar |
| Trailing stop | 3.0 x the same ATR from the best extreme since entry; the stop level is `max(initial, trailing)` for longs, `min` for shorts, and never loosens |
| Re-entry | After a stop-out the same direction is blocked until the entry signal has been absent for one decision |
| Sizing | Risk 0.5% of equity at the stop distance (`weight = 0.005 / (2 ATR / close)`), per-symbol cap 0.25, portfolio gross cap 0.95 (simultaneous entries are scaled pro rata; no entry if no capacity); positions are then held at exact quantity (no rebalance churn) |
| Hard risk | `PreTradeRisk`: gross 1.0, per-symbol 0.5, no turnover cap, drawdown brake from 20% (scale to zero at 30%) |
| Costs | The repo model: fee 5 bps, spread 2 bps, impact 1 bp, participation cap 5%, next-open execution; funding from the Dataset. Stress `cost_x2` doubles fee, spread and impact |
| Window | 2021-03-01 to 2026-07-31 (last bar closes 2026-08-01T00:00). Known-used and exploratory. Sub-years 2021 (from March) ... 2026 (to July) |
| Account | 10 000 USDT, one shared cash account, terminal flat (engine-settled) |
| Controls | CASH (return 0), BTC buy-and-hold, equal-weight buy-and-hold (95% gross), same engine and costs |

## 3. Stop model (`bar_extreme_touch_next_open_v1`)

The engine has no resting protective-stop order: orders come only from target-weight
reconciliation, and a `STOP_MARKET` entry would be cancelled and replaced on every
re-proposal. Building one would change the native execution path, so the stop is a
decision-layer rule that uses the existing order, fill, cost and accounting path.

1. The stop level uses only completed bars. A bar *touches* the stop when its low (long)
   or high (short) reaches the level. The favourable extreme is applied first (adverse
   ordering), so a bar that makes a new high and then trades through the new trailing
   level stops the position.
2. A touch submits a normal exit order at that decision; it fills at the next bar's open
   under the engine's cost model. A gap through the stop is taken in full; no fill is
   invented at the stop price.
3. **Stop haircut.** Where the exit fill is better than the stop level (the bar wicked
   through and recovered, so a resting stop would have filled at the level), the runner
   removes that price improvement from PnL as a cost on that exit fill. It is applied
   to the PnL stream, so **every gated quantity is computed from the haircut-adjusted
   series**: the equity path (net return, sub-years, drawdown, Calmar, the bootstrap's
   daily returns) and the closed-cycle PnL (profit factor). On the equity path the
   haircut of an interval is a fraction of the engine equity at the start of that
   interval, applied multiplicatively, so it compounds through every later interval (a
   first-order adjustment, not a re-simulation with smaller positions). The engine
   (un-haircut) values stay in each cell as `engine_*` for reference and are never
   gated. Adverse gaps are never credited back. The haircut is computed from the raw
   order events and the policy's stop log, not from the policy's own accounting.
   A fired stop keeps working until the position is flat: if the exit fills only in part
   (participation cap), the remainder is exited at every later decision even when the
   price has recovered above the stop level, and the whole exit is one stop event.
   The stop exit takes precedence over a reversal: on the bar a stop fires, and while
   its exit is pending, no entry is made in either direction (an opposite breakout
   signal included) until the position is flat; from the first flat decision the
   normal re-entry rules apply (the stopped direction stays blocked until its signal
   has been absent, the opposite direction may enter at once).
4. Limits: the entry reference is the observed fill-bar open (slippage is not in the
   stop distance); a stop touched and recovered inside a bar is charged at the level but
   the intrabar path after it is unknown; orders still working on the next bar (partial
   fills) are counted once per position.

## 4. Pass / fail, multiple testing and stop rule

Four configurations (2 variants x 2 timeframes) are tested; Bonferroni adjusts the
bootstrap level to `0.05 / 4`. The bootstrap interval is two-sided with coverage
`1 - 0.05/4` (0.625% in each tail, so the endpoints are the 0.625% and 99.375%
percentiles of the resampled annualised mean daily net return); criterion 7 reads only
its lower endpoint, which makes it a one-sided test of "mean above zero" at the 0.625%
level, stricter than a one-sided test at `0.05/4`. This is the frozen statistic and is
not changed here. A configuration is `DEVELOPMENT_PROMISING` only if every
criterion holds (base = repo costs, `cost_x2` = doubled costs):

1. net return > 0 in base and in `cost_x2` (haircut-adjusted; includes funding);
2. at least 4 of the 6 calendar sub-years positive (base, haircut-adjusted);
3. closed-cycle profit factor >= 1.2 (base; execution-cost and haircut net, funding excluded);
4. maximum drawdown of the adjusted equity <= 20% in base and `cost_x2`;
5. at least 100 closed position cycles (base);
6. Calmar ratio above that of BTC buy-and-hold (base, same window and costs); a missing
   Calmar on either side fails the criterion;
7. lower end of the Bonferroni-adjusted moving-block bootstrap interval of the
   annualised mean daily net return is above 0 (base);
8. validity: ledger reconciliation passed in both scenarios, no termination, terminal
   flat, and the pre-cut future-perturbation probe of **that configuration's own
   variant and timeframe** is byte-identical with positions held (one probe per
   (timeframe, variant) pair; a leaky variant fails only its own criterion 8).

The study label is `DEVELOPMENT_PROMISING` if at least one configuration passes and
`NO_WINNER` otherwise. It never selects a winner among passing configurations. **Stop
rule:** on `NO_WINNER` this line stops: no re-tuning of the channel lengths N and M,
the ATR stop multiples, the stop model, the universe or the timeframes on this window, and step 2 is not started. On
`DEVELOPMENT_PROMISING` the only authorized next action is a separate request (new
protocol, G0-G2 review, owner authorization) for step 2 or for an unused-data protocol.

## 5. Mechanism, counter-hypothesis, falsifiers (G0/G1)

- Mechanism chain: slow diffusion of information and herding in a leveraged market
  produce return autocorrelation over days to weeks; liquidation cascades extend moves;
  a breakout enters after the move starts, and a volatility-scaled stop caps the loss of
  the many false breakouts so that a few large trends can pay for them.
- Counter-hypothesis: crypto trends are already arbitraged at these horizons and whipsaw
  loss plus costs and funding exceed the trend gain; or the result is survivorship (the
  roster is 2026 survivors), a single bull-market regime, a gap/stop artifact, or
  selection by prior exposure: other work in this repository already read this universe
  and part of this window (section 5a), including a related positive time-series
  momentum result, so a pass here is not independent of that exposure.
- Cheapest falsifiers, in order: the synthetic oracle tests (no data); the refusal and
  future-perturbation probes; then the frozen run itself. Pass criteria 1-8 are the
  economic falsifier and are fixed before any run.

## 5a. Prior exposure and attestation

The universe and window are not unexposed. Earlier repository work read them:

- `tests/fixtures/legacy/realdata_20260713_signal_metadata.json`: a legacy GBM
  cross-sectional signal gate over 15 symbols, of which 10 of the 11 here (all but DOT),
  with a 2025-12-06 fit cutoff and a failed information-coefficient gate. It is a signal
  metadata record, not the PnL of a trend rule.
- Phase 2 trial T1 (time-series momentum: a 30d/90d return blend, volatility-scaled, no
  stops, daily decisions) on BTC, ETH, BNB, XRP and ADA over 2023-01-01 to 2026-08-01
  reported a positive return (see current-status.md). It is directly related prior
  evidence: the same rule family on 5 of the 11 symbols and the 2023-2026 part of this
  window, with a different signal and without stops.
- The M34 development Dataset (BTC, ETH, BNB, XRP, ADA, 2021-01 to 2026-08, with 4h and
  1d features) and the Phase 2/3 and directional studies that used it.

Consequences: DOT was never read by those records; the other ten symbols were. All of
2023-2026 is already used. This study cannot be called result-blind with respect to the
universe and window, only with respect to this rule's own outcome.

Attestation about parameters. N (20/55 and 10/20), ATR(14), the 2x and 3x stop
multiples, 0.5% risk per trade, the per-symbol and gross caps, the 20% drawdown level,
the profit-factor threshold of 1.2, the 4-of-6 sub-year rule, the 100-cycle minimum and
the Calmar-versus-BTC comparison are taken from trading convention (Turtle-style
channels, Wilder ATR, round-number thresholds) as given in the planner's brief on
2026-11-10. The implementer attests that it did not read the Phase 2 T1 results, the
legacy fixture or the M34 outcomes when fixing any value, and that no value was tuned on
any data. The implementer cannot attest what the planner or owner had in mind from the
repository's earlier results, so influence of the Phase 2 TSMOM result on the choice of
rule family, 20/55 bars or the thresholds is **not excluded**; it is disclosed rather
than denied. Any pass must be read with that in mind.

## 6. Independent oracles (G2) and evidence validity (G3)

- Stop and accounting: `tests/strategies/test_trend_stop.py` replays a tiny synthetic
  path through the real executor and compares every equity step with `Fraction`
  arithmetic done by hand (entry 103, trailing stop 101, gap exit 94, loss 75). The
  wick-and-recover path is checked against the hand haircut in
  `tests/evaluation/test_trend_stop_study.py`. A stop exit that fills in three parts
  (4, 4 and 1/3 units under a participation cap), after a gap and after a recovery
  above the stop, is checked fill by fill, for a single stop event and for the summed
  haircut. A stop and an opposite breakout on one bar, and an opposite breakout while
  a partly filled stop exit is pending, open nothing until the position is flat.
- Sign: the short side is the exact mirror image of the long side.
- Time: perturbing every bar after a cut leaves earlier decisions byte-identical, in the
  unit test and in the runner's real-data probe, which runs for every variant and
  timeframe (a test shows a variant that peeks 20 bars ahead fails its own probe and
  its own criterion 8 and not the other variant's).
- Risk: simultaneous entries are scaled to the gross cap; positions are held without
  churn; risk per trade equals 0.5% of equity at the stop distance.
- Mutation check performed on the strategy tests: removing the stop trigger, the
  current-bar extreme, the re-entry block, the channel's exclusion of the current bar,
  the gross-cap scaling or the HOLD each fail at least one test.
- G3 claim boundary: the result is development, exploratory evidence on known-used data
  with survivorship in the roster and the prior exposure of section 5a. It cannot
  support an unused-data, paper or live claim, and an engine-PnL positive that fails the
  haircut-adjusted gate is not a pass.

## 7. Software map

| Capability | Location |
|---|---|
| Rule, indicators, stop state machine, policy | `trade_rl/strategies/rules/trend_stop.py` |
| Frozen protocol, costs, hard risk, authorization-URL check | `trade_rl/evaluation/trend_stop_study/protocol.py` |
| Metrics, closed cycles, stop haircut, gate | `.../trend_stop_study/metrics.py` |
| Multi-symbol runner (every cell), per-variant causality probe, code provenance | `.../trend_stop_study/runner.py` |
| Dataset loading / validation / Vision fallback | `.../trend_stop_study/data.py` |
| CLI (write-once outputs) | `python -m trade_rl.evaluation.trend_stop_study` |

The runner refuses, before any replay, a missing or malformed G4 authorization URL
(`https://github.com/shuntatsu/trade_rl/<issues|pull>/<n>#issuecomment-<id>`; only the
shape is checked, the owner lease remains a human gate), a window or Dataset reaching
past 2026-07-31, a symbol outside the roster, a cadence mismatch, a non-empty output
directory and a dirty Git working tree (the run must name the exact code). A Dataset
that lacks any roster symbol, or a Vision cache that lacks an archive with recorded
HTTP 404 evidence, writes an `INVALID` result instead (naming the symbol, URL, status
and time of each such archive). The evidence is written by the download step: when
the venue answers HTTP 404 for a Vision URL, the transport records a
`<entry>.missing.json` sidecar next to the cache entry (schema
`binance_vision_missing_archive_v1`: URL, status 404, observed time, downloader); any
other HTTP failure, a timeout and an offline miss record nothing, a later successful
download removes the record, and a malformed record is never read as a 404. The
record is a file, not an authenticated venue receipt, so it can only turn a run
`INVALID` and can never produce a reduced roster. The CLI decides in one
evidence pass over every archive of both timeframes and funding before it builds any
Dataset, so valid 404 evidence anywhere is `INVALID` whatever the order of the
timeframes or the other gaps. Every other data or metadata failure (absent, empty or
checksum-failing archive, a missing exchange-information snapshot, an unreadable or
corrupt Dataset, any other exception raised while loading data) makes the CLI exit 2
with a message naming the exception type and no result; it never ends in a traceback.
Forward acquisition does not ignore the `.missing.json` sidecar: it reports it as an
unexpected cache member and fails closed. The result records
`code`: the Git HEAD, the tree id of `trade_rl/` at HEAD, the dirty flag (false for any
run that proceeds) and a digest over the raw bytes of every loaded `trade_rl` module,
each of which must lie under the checkout. There is no registry of runs: a repeat into
a new output directory is possible, and the owner's one-run rule is procedural. Every
cell is written as its own JSON file (`cells/<timeframe>__<arm>__<scenario>.json`) with
`equity.npz` and `result.json`.

Data path. Primary: one Market Store export per timeframe, then pass each
`<dir>/dataset` to the CLI:

```bash
python -m trade_rl.integrations.market_store export --market usds_m --interval 4h \
  --bases BTC,ETH,BNB,ADA,LINK,DOGE,DOT,AVAX,LTC,SOL,XRP \
  --start 2021-01-01 --end 2026-08-01 --feature-timeframes 1d --output-dir <dir-4h>
```

The 1d Dataset cannot be produced by this CLI today (it requires a non-empty feature
timeframe and the builder forbids repeating the base interval), so use the offline
Vision fallback for both timeframes or call `export_market_dataset(feature_timeframes=())`
from Python. Fallback: `--vision-cache <dir> --exchange-info-dir <dir>` builds from a
complete local Binance Vision cache of native 4h and 1d klines and funding archives
with the unchanged builder and never uses the network. Check whether another branch
has added a shared multi-symbol 4h/1d bar loader before extending `data.py`.

Expected runtime: about 25 minutes single process (12 000 4h rows x 11 symbols is about
2 minutes per replay cell, the four per-variant probes replay the pre-cut window twice
each, and the 1d cells take seconds). Latency stress is not run:
`order_latency_bars` 0 and 1 are the same next-open fill in this engine, and 2 or more
cancels a still-working entry on every re-proposal (the policy would have to preserve
pending orders as the decision-value policies do).

## 8. Known limitations

Survivorship in the roster; current exchange filters applied to history; the stop
distance excludes entry slippage; bar-resolution stops cannot see intrabar order; funding
is not attributed to individual cycles; a single regime-heavy window with six
sub-years; the gate is a development screen, not a profitability proof; the
authorization URL is a shape check.

## 9. Design note for step 2 (not implemented)

Start condition: step 1 is `DEVELOPMENT_PROMISING` for at least one configuration and a
new protocol has a result-blind G0-G2 review and owner authorization. On `NO_WINNER`
step 2 does not start (reinforcement learning is not a rescue for a failed rule).

**Role of PPO.** A meta-policy on top of the rule, not a free trader. The rule still
proposes entries, exits and stops; PPO only adjusts, per symbol and decision, (a) an
exposure multiplier `m` on the risk budget from a small bounded grid including 0
(skip), and (b) a stop multiplier `k` from a small bounded grid around the rule's 2 ATR.
Actions are residual: the zero action is the rule unchanged. PPO cannot remove a stop,
raise gross above the hard risk, or open a position the rule did not propose. The
existing `PreTradeRisk` and executor remain the final authority.

**Observation (point in time, row `t`).** Per symbol: rule state (side, bars in trade,
unrealised PnL in units of initial risk, distance to the stop in ATR, bars since the last
stop), ATR as a fraction of price, momentum over several horizons scaled by ATR, funding
rate and its sign, volume z-score. Portfolio: gross, net, current drawdown, open
positions, remaining risk budget. Cross-asset (Track D `cross_asset_features_v1`,
point-in-time, admitted only where `available_at <= decision time` with availability
masks): BTC-relative strength and beta, market breadth, cross-sectional momentum rank,
dispersion and correlation regime. Normalisation is fit on the training window only,
following the existing portfolio observation transform. This protocol does not depend on
the Track D package.

**Reward.** The existing gamma-one equity-delta objective after all costs and funding.
No reward term references stop outcomes.

**Overfitting guardrails.**

1. Walk-forward in time with an embargo of at least the longest holding horizon; train,
   validate and test windows are fixed in the protocol; the test pass is one-shot and
   never used for selection; 2026-08 onward stays untouched.
2. One symbol-agnostic policy (no symbol identity input) and a leave-symbols-out
   evaluation: a pass must hold on assets withheld from training for the whole period.
3. Small capacity (a handful of units per layer), at most about 40 features, weight
   decay, five seeds; a pass needs at least four of five seeds beating the rule on the
   paired comparison, with medians over all five.
4. Null and ablation controls: shuffled-observation PPO, PPO without the cross-asset
   block (isolates the value of Track D), a random exposure/stop policy with matched
   turnover, and the best *fixed* `(m, k)` chosen on the training window only. PPO must
   beat the best fixed pair, otherwise learning adds nothing.
5. Multiple-testing accounting over every hyperparameter, seed and variant tried, with
   hyperparameters frozen before any run.
6. Same cost engine with 2x costs and a delay stress (which first requires
   pending-order-preserving proposals); stop and risk invariants covered by tests that
   no action can break them.
7. Immutable evidence: checkpoint digests, one-shot final pass, and the rule alone as
   the paired baseline in every comparison. If the improvement does not survive 2x costs
   and leave-symbols-out, the result is `NO_WINNER` and the RL line stops.

## 10. What this change does not do

No real price, return or PnL of this rule was computed (the prior exposure of the
universe and window by other work is disclosed in section 5a), no order or venue call
exists, FS1, warm-up, unused and final data are untouched, and no existing strategy, default or artifact changed.
Passing CI authorizes nothing: a G4 run needs the owner's authorization and a review
of the packet's exact digest and heads. For this revision the owner decided (issue #810
comment 6818495567) that the same-account route (a fresh Claude Haiku 5.5 agent posting
from the owner's account, model independence only, not principal independence) covers
the packet whose digest this revision's pull request names; the route is
operator-enforced, `tools.research_assurance_github` still rejects same-principal
reviews and the packet stays `UNREVIEWED` for the tooling.
