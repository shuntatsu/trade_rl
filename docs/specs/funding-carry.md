# Funding carry bot: development contract

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
- Capital: 10,000 USDT. At entry and each calendar month, allocate 12.5% of
  current account equity to each spot leg and the same base units to its short
  leg; total starting gross exposure is at most 50%. Use the larger observed
  spot/perpetual price to size each matched pair. Retain quantity between
  rebalances. Round both legs down to the common 0.001 base-unit lot.
- Costs per fill: spot 10 bp, perpetual 5 bp, plus 5 bp adverse execution per
  leg through a 10 bp full spread; no rebates. Minimum notional 10 USDT;
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
- Final bar: schedule closing at the previous decision, execute at the last
  available bar's open, and include exit costs in the return path.
- Report full/year returns, drawdown, funding, trading costs, hedge error,
  collateral checks, termination, per-pair diagnostics, raw equity/quantities,
  config/source/code digests. Existing output is never overwritten.
- Predeclared stress: double fees/spread, one extra decision of entry latency,
  and 10x less participation; run all, without choosing favorable stresses.
- Development qualification requires positive net full and each-year returns
  for the shared account and each pair, drawdown below 10%, all stress full
  returns positive, and no hedge/margin/termination defects. Failure is STOP.
  A pass only permits designing a distinct prospective paper validation.

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
