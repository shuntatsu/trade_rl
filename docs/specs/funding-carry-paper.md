# Prospective carry paper validation

Status: Active

The fixed BTC/ETH development carry family passed its twelve preregistered
screens. It is a research candidate, not an operational winner. Preserve that
completed study. This separate prospective path replaces close-price proxies
with data collected when decisions can actually be made.

## First capability: fresh public evidence

Collect only unauthenticated GET market-data endpoints from official Binance
spot and USD-M hosts. No account keys, signed requests, private account calls,
orders, money transfers or retrying HTTP authorization/rate-limit denials.
The maintained bounded transport remains the HTTP owner.

For BTCUSDT and ETHUSDT, preserve raw spot/perpetual depth, actual perpetual
mark/index/next-funding timestamp, settled funding history and both venue clocks.
Record local request-start/receipt UTC times, monotonic duration, source URL,
raw bytes SHA-256 and response bytes. A premium-index lastFundingRate is a quote,
not evidence that a funding payment settled; only published funding history may
later enter the ledger. Preserve exchange metadata separately before any paper
order so current lot, tick, minimum notional and trading status can be enforced.

Use source-time freshness checks for venue clocks, futures depth and mark quotes;
spot REST depth has no exchange timestamp, so disclose receipt-time-only evidence.
Reject crossed/nonpositive/unsorted books, invalid sizes, future/stale source
timestamps, excessive request duration and excessive cross-request span. Never
accept source times older than five seconds or more than one second ahead of
receipt (the latter is an explicit clock-skew tolerance). Recheck all quote ages
at the final response receipt. Enforce total capture span using both wall time
and a capture-wide monotonic anchor; per-request durations alone are insufficient.
Never
turn partial source success into an eligible snapshot. Preserve failure evidence
without pretending the market was flat or tradable. All outputs are write-once.

A read-only probe is connectivity evidence only and is excluded from prospective
strategy evaluation. Freeze strategy, journal, start time, fees, terminal close,
minimum observation duration and decision rule before forward paper positions.

## Paper execution design still to implement

Reuse FundingCarryBot and BookState; do not introduce another financial ledger.
Decisions precede simulated executable quotes. Match each leg against recorded
bid/ask depth after the decision, record partial/asymmetric fills and stop when
the hedge breaks. Do not assume a two-leg atomic exchange fill or fill missing
quotes from a later observation. Funding uses actual published settlement time,
held quantity and the settlement mark; duplicate/revised/late events cannot be
silently counted twice or used as an earlier decision input.

Persist immutable snapshots, intents and ledger checkpoints with a contiguous
chain. Restart resumes exactly once; gaps/stale prices freeze new exposure and
are visible in quality/qualification. Future paper profit and drawdown must be
computed from every accepted observation, with explicit fee/rule/transfer and
mark-to-liquidation limitations. Software CI and a successful source probe are
not forward economic evidence. No production routing is part of this design.
