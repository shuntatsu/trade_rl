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

### Deterministic account composition

The journal engine will replay decision, execution and gap commands against
verified source references beneath its own root. Each committed command contains
its request and complete transition result; restart recomputes the result and
rejects any difference. Compute on an isolated account copy, commit the event,
then adopt it, so a failed append cannot mutate the running account. Repeating
an identical command returns its original event; stale concurrent writers fail.

Decision observations value spot at recorded best-bid/best-ask midpoint and
perpetuals at the published mark. Both are observation marks, not liquidation
proceeds. Retain the existing monthly CarryConfig defaults: gross 0.5, common
lot 0.001 and irreversible 10% drawdown stop. Fees are explicit 10bp spot / 5bp
perpetual paper assumptions. New quotes must be captured after the previous
command and each execution capture must start after its saved decision.
The saved rules must remain fresh at execution. Enforce exact venue lot
compatibility; never round an unmatched residual into a fictitious flat account.

Keep a sparse exact-quantity timeline of actual fills. For each newly published
settlement, use the quantities held strictly before its funding timestamp and
the published settlement mark/rate. Record the amount and first receipt time;
the same symbol/timestamp cannot pay twice. Revisions stop the strategy without
rewriting earlier cash. Funding first received more than 180 seconds after a
post-start settlement, missed expected settlements, or observation gaps over
180 seconds are permanent quality failures. Pre-start history pays zero.
Check observed collateral, insolvency and drawdown before crediting newly known
funding, so a delayed positive payment cannot undo an earlier risk breach.
Apply newly received settlements in chronological timestamp groups, checking
risk after each group. Only simultaneous settlements may net; a later credit
must not hide an earlier debit's drawdown or collateral breach.
Across captures, a new nonzero payment at or before the already processed
settlement-time watermark is an irreversible `funding_out_of_order` failure.
This includes a later fragment of an already processed simultaneous group.
Cash is still settled exactly once at first receipt; historical cash and decisions
are never rewritten. This conservative rule can reject ordinary asynchronous
publication, and that limitation must remain visible in the prospective result.

Retain each attempted leg's partial fill and cost. Assess hedge balance after
all four independent leg attempts; any mismatch permanently stops subsequent
exposure and requests actual liquidation on later quotes. If a new stop occurs
between decision and execution, cancel the old intent and wait for a fresh exit
decision. A gap also cancels pending intent and preserves all existing quantities.
The frozen close time permanently targets zero; final flatness requires actual
accepted exits, including fees and any residual venue-rule rejection. A terminal
time, successful replay or positive interim cash does not imply a passed future
economic gate.

### Public collection supervisor

Seal the complete package-source and runtime provenance before collecting any
study inputs. The supervisor accepts only an expected protocol digest and that
unchanged provenance; verify identity before each network acquisition and each
account command. Use a single advisory process lock to exclude two collectors;
process death releases the lock. Preserve unique immutable capture directories
and re-use rule evidence only while fresh, refreshing before its one-hour limit.
An independent SQLite control record durably begins each cycle before any
request and acknowledges it only after all account commands commit. Reopening
an unfinished cycle permanently halts before network access, including a crash
after a source failure file or after a successful capture but before its command.
Never discard an unconsumed observation by silently trying another one. Existing
source directories without their control database are rejected. This control
record owns operational acknowledgements, never cash, fills or P&L.
An incomplete cycle also blocks another cycle in the same process. Set an
irreversible in-memory halt before attempting failure-file I/O, so even a failed
failure-file write cannot permit additional requests.
An existing pending decision may execute only inside its ten-second lifetime;
otherwise record a permanent gap and cancel it. Clock reversal, source drift,
failed evidence or transport failure writes a durable failure and halts network
activity. A failed collector cannot automatically restart and repeat denied
requests. A report must retain unresolved quantities and last real marks.
The pinned protocol digest is rechecked together with source/runtime identity
before every acquisition. Failure recording uses its own UTC clock so failure
of the injected observation clock cannot suppress durable failure evidence.

Public collection is an operational capability only. Its terminal observation
grace window is 180 seconds after the frozen close time, to capture delayed
settlements and attempt actual exits; it is not permission to extend a failing
study until profit appears. Future minimum duration and economic thresholds
still require a separately frozen prospective protocol before positions start.

### Fixed prospective screen and operator commands

The first economic screen will be ninety consecutive UTC days, with three fixed
thirty-day blocks. Seal at least five minutes before start, use 10,000 virtual
USDT and the existing fixed fees, depth fraction, gross exposure and 10% stop.
The user research ceiling remains 20%; the strategy's stricter stop is retained.
Cash with zero interest is the declared comparison. This is a development paper
screen, not proof of optimal returns or authorization for live deployment.

Require positive net profit after actual simulated exits, positive marked equity
change in every block, and total net profit greater than recorded fees. The last
condition is fee headroom at the realized trajectory, not a recomputed doubled-fee
strategy. Block boundaries use the last causal observation, no more than 180
seconds before the boundary. The last block includes terminal exit costs.
Require observed maximum drawdown below 10%, no quality failures or nonterminal
stop, at least one nonzero funding receipt in each block, no unpaid announced
funding for a previously held position, actual zero quantities and no pending
intent. Minute sampling is not an intraminute drawdown or liquidity guarantee.

Do not qualify before close plus the fixed 180-second grace. Require observations
from within 180 seconds of start through at least close plus 120 seconds, with
no gap over 180 seconds. Audit the externally supplied final event tip, every
raw source consumed by replay, every completed collection cycle, and the exact
source-directory roster. Failed, unfinished or unconsumed captures reject the
screen. A positive short probe or altered duration/settings cannot qualify.

Expose `python -m trade_rl.evaluation.paper.cli` with `seal`, `run`, `status`
and `evaluate`. `run` follows fixed 60-second slots without catch-up quote reuse
and stops at the frozen grace deadline. A permanently failed screen may stop
early once actual exits have made it flat. An error halts; there is no retry loop.
`status` reports journal-chain-checked operational progress and explicitly does
not claim financial replay validation. `evaluate` excludes a running collector,
checks source/runtime, rebuilds the account and applies the frozen screen to the
externally pinned protocol and final tip. A passed paper screen always retains
`production_eligible=false`. Failure evidence and unresolved positions survive.
