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

### Durable journal

`evaluation/paper/store.py` owns an append-only SQLite event chain and a separate
immutable protocol manifest. Initialization reserves a new directory before any
publication. Opening requires the caller's expected protocol SHA-256 and never
creates a missing database. Each canonical JSON event binds a contiguous sequence,
kind, unique idempotency key, payload and previous digest; the first parent is
the protocol digest. Rebuild verifies every event and its parent from stored bytes.

An immediate SQLite transaction performs compare-and-append against the expected
tip. Repeating exactly the same key, parent, kind and payload returns the original
event without a new row. Reusing a key with different contents, or appending from
a stale parent, fails. Update/delete are unavailable through the API and blocked
by database triggers. FULL synchronization and rollback journaling preserve the
committed prefix on interruption; no half-published event may become an executed
fill on restart. Opening verifies the protocol before permitting SQLite's hot
rollback-journal recovery; subsequent event inspection uses a read-only connection.
This store is not an alternative financial ledger: paper engine
events will drive the existing strategy and BookState, and only the engine may
assign economic meaning to event kinds.

### Restart inputs and current rules

An offline forward-snapshot reader must revalidate every raw byte hash, response
sidecar, official URL, response roster, timing bound and decoded market field.
It must reproduce the published snapshot from raw responses; the summary alone
is not authority. An optional expected snapshot digest binds a journal parent.
Offline verification does not imply freshness now. Supplying an execution time
additionally enforces every quote's five-second age and rejects future receipts.

Current BTCUSDT/ETHUSDT spot and USD-M exchange information is a separate,
write-once public capture. Require trading status, market-order support, matching
base/quote assets and USDT perpetual collateral; reject duplicate/missing symbols
or filters. Preserve raw rules, source timing and hashes. Derive paper bounds
from the intersection of LOT_SIZE and MARKET_LOT_SIZE, retain every positive
quantity quantum, and distinguish inactive zero market steps from missing rules.
Keep tick/price limits and market-applicable notional flags, including averaging
minutes. Venue-specific averaged notional checks remain explicitly unmodeled by
the depth primitive. Current metadata cannot authorize orders or establish
profitability, and is valid for new paper decisions for at most one hour.

### Recorded-depth execution contract

The provider-independent depth executor belongs in `simulation/depth.py`.
It consumes one signed integer-lot order and a later recorded bid/ask book;
the journal must permit at most one order per instrument per observation.
Quote receipt follows the saved decision, execution follows receipt, receipt
age at execution is at most five seconds, and decision-to-execution delay is
at most ten seconds. These are paper timing rules, not a venue fill guarantee.
Use the correct side, consume only 10% of displayed size, and retain partial
quantities. Capacity sums exact decimal sizes across levels before applying the
order lot quantum; small individual levels must not disappear through per-level
lot rounding. Charge the recorded weighted price plus a fixed adverse 5bp buffer
and the declared taker fee once. Do not separately charge the quoted spread.

Lot/minimum/maximum quantity and minimum/maximum notional are explicit paper
admission rules. A rejected or empty-capacity order leaves the book unchanged.
Published market-notional averaging and account-specific restrictions are not
reproduced by this depth-only primitive and remain production limitations.
Execution updates the canonical BookState using accepted lot evidence. Value
existing positions at the independently supplied marks, never temporarily at the
new fill price: a transient fill-price mark must not manufacture an equity peak.
No invented liquidity or automatic removal of an unfilled residual is allowed.

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
