# Funding carry bot: development contract

Status: Active

The user delegates implementation toward a profitable trading bot. This work
adds an independently identified structural carry experiment, not a retune or
revival of the rejected ridge/PPO studies. No profitability is assumed.

## Frozen first experiment

- Instruments: BTCUSDT and ETHUSDT, each spot long plus equal base quantity
  USD-M perpetual short. No reverse carry, forecasting, training, symbol
  selection, threshold search, or leverage optimization.
- Data: Binance Vision hourly closed bars and actual published funding events,
  from 2022-12-31 through 2025-01-01 exclusive. Evaluate 2023 and 2024 only.
  This is already-used development time, never an unused/final test.
  The original strict-clock build stopped before economic replay: both official
  monthly and daily spot archives lack the 2023-03-24 13:00 UTC open. Preserve
  that failed source attempt. A separately identified halt-aware source revision
  uses Binance's published spot halt [2023-03-24 11:27, 14:00) UTC. Only a missing
  whole hourly bin contained in that declared halt may receive the preceding
  close as a stale valuation mark, zero volume and tradable=false. Every hourly
  bin intersecting the halt is nontradable with zero capacity, including partial
  bins; real published prices remain unchanged. No backfill from reopening,
  removal of elapsed time, or inference of halts from missing data is allowed.
  Unknown/partial-bin gaps and boundary gaps without a preceding mark fail.
  Orders still use information from the preceding decision: the halt mask only
  rejects execution when that processing bar arrives. Previous-bar volume also
  conservatively delays fills for the first reopened hour. Stale spot valuation
  and the coarse halt mask are disclosed limitations of this development study.
  Source: https://www.binance.com/en/blog/from-our-ceo/6789340645608890113
- Capital: 10,000 USDT. At entry and each calendar month, allocate 12.5% of
  current account equity to each spot leg and the same base units to its short
  leg; target gross at decision prices is at most 50%. Use the larger observed
  spot/perpetual price to size each matched pair. Retain quantity between
  rebalances. Round both legs down to the common 0.001 base-unit lot. Matched
  partial fills keep pursuing that fixed target through GTC orders; no daily
  price-driven resizing. Record realized next-open gross separately: a gap can
  exceed the decision budget and is not resized using future prices.
- Costs per fill: spot 10 bp, perpetual 5 bp, plus 5 bp adverse execution per
  leg (the half of a 10 bp quoted full spread); no rebates. The canonical market
  executor charges its whole spread field, so use spread_rate=0.0005 and generic
  fee_rate=0, with venue fees only in maker/taker fields. Minimum notional 10 USDT;
  participation at most 1% of previous completed hourly quote volume. These
  are disclosed research assumptions, not historical account-specific rules.
- Execution: existing MarketExecutor/BookState, next open; no second P&L
  ledger. Spot cannot be shorted. Funding settlement uses perpetual bar-close
  price as a mark proxy; this limitation blocks production eligibility.
- Risk: common pair quantities; irreversible stop and next-open flatten after
  unmatched fills, 10% account drawdown, or inadequate separate futures-wallet
  collateral. Do not count spot mark value as futures collateral. Report an
  intrabar adverse-high margin breach as invalid execution evidence. Preserve
  any residual positions if exit liquidity is unavailable; never invent fills.
- Wallet assumption: all unspent USDT is available to the futures wallet and
  spot sale proceeds transfer without delay/fee. Futures collateral is canonical
  account equity minus current spot market value (book cash plus signed perpetual
  value), never book cash alone. Require collateral at least the perpetual
  notional at observed closes. Maintenance is 50% of perpetual notional; test
  intrabar adverse highs using post-fill cash BEFORE bar-close funding credits.
  Also check previous cash/holdings at the processing open BEFORE exit or
  rebalance fills; flattening cannot erase an existing gap breach.
  An intrabar maintenance breach invalidates evidence, even if the close recovers.
  These conservative research ratios are not historical exchange margin tiers.
- Carry risk stops submit real zero-quantity targets through the existing
  executor. Any canonical economic termination is invalid evidence: retain the
  pre-forced-flatten position evidence and do not accept canonical forced-flat
  output as an executed exit. Existing core termination semantics are unchanged.
- Final bar: schedule closing at the previous decision, execute at the last
  available bar's open, and include exit costs in the return path.
- Report full/year returns, drawdown, funding, trading costs, hedge error,
  collateral checks, termination, per-pair diagnostics, raw equity/quantities,
  config/source/code digests. Existing output is never overwritten.
- Predeclared stress: double fees/spread, one extra initial-entry decision of latency
  (monthly rebalances and exits retain their timing),
  and 10x less participation; run all, without choosing favorable stresses.
- Development qualification requires positive net full and each-year returns
  for the shared account and each pair, drawdown below 10%, all stress full
  returns positive, and no hedge/margin/termination defects. Failure is STOP.
  A pass only permits designing a distinct prospective paper validation.
- Run the shared account and each pair separately for every treatment (12
  replays: three scopes times base/double-cost/delayed-entry/thin-capacity).
  Shared initial capital is 10,000 USDT; each independent pair gets 5,000 USDT
  so its initial leg sizing matches its allocation in the shared account.
- Funding data use published events on maintained bar-close timestamps; no
  forward fill. For the declared BTC/ETH development range, internal gaps over
  nine hourly bins fail closed; this allows an eight-hour event to straddle an
  hourly boundary. First/last source rows are exposure-free boundaries. Missing
  terminal funding cannot qualify a run with an unclosed position.

## Software boundary

`strategies/carry.py` owns deterministic paired sizing/state, with no provider
or evaluation dependency. `integrations/binance/carry.py` owns two-market
source loading and an identity-bound dataset. `evaluation/carry.py` composes
strategy, canonical execution, and evidence. A CLI prepares immutable input
and performs the declared development study. No private exchange API or live
order transmission is included in this research change.

## Verification

Test exact same-price hedge P&L, signed funding, opening/closing cost,
quantity-preserving hold, no-future sizing, monthly rebalancing, asymmetric
fills, unfillable emergency exit, separate collateral, invalid input, source
clock/coverage and evidence overwrite protection. Run full permanent CI,
including distribution closure and clean install, on the final PR head.

Profit on the exposed 2023-2024 data is development evidence only. Prior work
has also used some 2025-2026 data; those periods cannot automatically be
renamed untouched holdout. A prospective paper record is required before
any claim of independently validated operational performance.
