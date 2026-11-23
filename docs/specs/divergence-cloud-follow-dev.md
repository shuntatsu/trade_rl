# Visible-cloud trend continuation and divergence-exit exploratory candidates

Status: Active — software candidate definitions, **not** a verified market winner,
research assurance approval, deployment permission, or part of any existing
frozen study. This design follows exploratory outcomes already inspected on
2022–2024 Binance BTC/ETH futures data. All such comparisons remain
**known-used / post-hoc**. Do not call them prospective or independent.

## G0 question and counterhypotheses

Earlier data suggested that requiring *both* 1h and 4h divergence to enter
misses long rising trends. Does entering on a persistently confirmed daily
visible cloud improve the return/turnover trade-off? Can a strict opposing
4h divergence provide useful early exit information after execution costs?

Counterhypotheses: cloud persistence merely changes exposure timing without
additional information; opposing divergence causes false exits and turnover
that overwhelm its risk benefit; results rely on a few historical regimes.

## Exact opt-in C2 and C3 semantics

- Input: the existing `DivergenceCloudMtfPack`, with prior 1h/4h
  confirmed-pivot detection and point-in-time daily **visible** cloud
  established in `divergence-cloud-mtf-dev.md`.
- C2 (`confirmed_cloud`): only enter after six consecutive 1h decision
  rows with the same positive/negative cloud regime. Target ±0.25 of
  account value, subject to shared hard-risk limits. If a held position
  loses cloud support, request zero immediately. Otherwise explicitly
  hold its exact contract quantity rather than rebalance to a target.
- C3 (`divergence_exit_cooldown`): all C2 behavior, plus close a long
  on *newly confirmed* bearish 4h divergence, or a short on bullish 4h
  divergence; wait 24 subsequent 1h decisions before another entry.
  Divergence does **not** reverse exposure or open trades.
- The original C1 cloud-only and C0 divergence-mandatory variants are
  retained independently. C4 divergence-accelerated entry was also tested
  in the external scratch exploration; its effect was negligible in the
  six tested cohorts, so it is **not** introduced as another production API.
- Configuration is per-policy instance and requires a fresh instance per
  replay; repeated, out-of-order and off-clock decisions are rejected.
- The policy never owns equity, fees or book balances: the existing
  `PortfolioPolicy` / shared native risk-execution-accounting path owns
  every economic result. No other accounting implementation is added.

## Status and research integrity (G1–G3)

Synthetic tests cover six-hour entry confirmation, direction invalidation,
strict 4h opposing exits, 24h cooldown, explicit exact-quantity HOLD,
native replay fills and fees, invalid settings, timestamp mismatch, and
monotonic decisions. No new Dataset artifact, FeatureKind, default feature
roster, frozen protocol, G4 relay, or live/paper order routing is added.

All BTCUSDT and ETHUSDT 2022/2023/2024 observations are **consumed** for
these research ideas; post-hoc double-cost and 1h-latency stress results
must be retained even where they contradict improvements. Exact prior
outcomes and the external scratch scripts have provenance hashes and must
be reported in the PR body / explicit research artifact, not promoted to
unused-data evidence.

Later economic research needs an **independently reviewed, result-blind,
separately authorized** protocol with same-capital, risk, funding, and
execution semantics, explicit long/short attribution, multiple-trial
correction and unknown-future evaluation. Do not unfreeze or rewrite
historical study results for this candidate.
