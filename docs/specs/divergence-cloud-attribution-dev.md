# C2/C3 native-ledger attribution and descriptive exit diagnostics

Status: Active — software-only diagnostic study for already-used historical
data. This work is NOT independent economic evidence, G4/G5 authorization,
or permission to trade live or access sealed/unused data.

## Why this diagnostic exists

The known-used Binance 2022–2024 backcasts showed divergent C2/C3 annual
performance on BTC/ETH. Before adding entry and exit rules, determine how
native realized portfolio equity moved in long/short/flat/reversal intervals.
Separately check the direction of traded-price moves in 24/72/168 hours after
a C3 opposing 4h divergence exit request.

## Existing accounting owns P&L

- The existing run_shared_portfolio_replay function remains the only
  risk, execution, exact book, funding and accounting owner. Enable its
  pre-existing capture_ledger_evidence switch.
- The new trade_rl/evaluation/divergence_cloud_diagnostics.py is read-only:
  no new market engine, shadow book, fill simulation or re-accounting.
- An interval entered/held/exited with a long (short) position, with no
  direct reversal, attributes the *native change in equity* to long (short).
  An interval changing directly from long to short goes to a separate
  reversal bucket. Both endpoints flat go to a flat bucket.
- Refuse bad numeric fields, malformed exact rational quantities, multi-symbol
  ledgers, skipped/duplicate intervals and inconsistent quantity/equity
  chains. Total attribution must conserve the original book equity change.
- Directional amounts are an **accounting partition**, not causally identified
  alpha, separately simulated trade P&L or position-independent performance.

## C3 post-exit event study

- A recorded C3 decision with an open signed position, zero proposal,
  same-side current visible cloud, and newly confirmed opposing strict 4h
  divergence identifies an early-exit **request**, not a hypothetical trade.
- Record whether the native order fully flattened by next hour, whether
  C2 held the same direction, and whether C2 maintained it for the horizon.
- Compare traded close[t] to close[t+24], close[t+72] and close[t+168],
  signed by the prior C3 position, only inside the already-used study range.
- Future horizon prices are used ONLY by offline analysis, never by
  production or decision features. These prices do not calculate missed
  **realized** profit, exclude fees/funding, and overlapping exit events
  must not be regarded as independent statistical trials.

## Software verification / scope

New tests cover native equity conservation, exact native rational quantities,
gapped or discontinuous ledgers, reversal segregation, future-horizon bounds,
and real shared-account replay integration. Existing Dataset, frozen study
identities, facade exports, research authorization and production endpoints
are unchanged.

Only the already-consumed 2022–2024 BTCUSDT/ETHUSDT Binance data may be used
by the external development analysis script. All results are post-hoc,
exploratory, and must be compared against existing saved C2/C3 replay metrics.
