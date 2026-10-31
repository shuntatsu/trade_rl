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

The current v2 capture requests 100 depth levels per side. Preserve every
returned level and reject oversized responses. Historical v1 captures remain
20-level evidence and cannot be relabelled as v2. Official endpoint support is
documented in [Spot REST](https://github.com/binance/binance-spot-api-docs/blob/master/rest-api.md#order-book)
and [USD-M market data](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data).
The change follows a rejected software partial-fill probe, before formal future
positions; it leaves participation, execution costs and risk stops fixed.

Fresh snapshot capture limits each of its ten JSON response bodies to 64 KiB;
the two current exchange-info response bodies are limited to 32 MiB each. These
are prospective capture resource budgets, not endpoint size guarantees. The
transport checks the declared length when available and otherwise reads at most
one byte beyond the configured limit in bounded chunks. An oversized response
fails the whole capture without publishing a truncated raw member; completed
response evidence and the failure manifest remain intact. These limits bound
response-body bytes, not peak process memory, disk use or total capture runtime;
conversion to the returned `bytes` value has a finite additional allocation.
The per-call limits do not change historical REST, Vision archive or cache
reads, and they do not establish ninetieth-day storage feasibility.

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
Before beginning any new cycle, compare the collector clock with the last
committed paper command (or the frozen start when none exists). If the elapsed
gap exceeds the configured maximum, append a non-trading gap command before
source acquisition. Preserve exact holdings and the last actual marks at that
gap; do not backfill or infer why the prior process stopped. A fresh current
quote may then be used to attempt actual exits, but the permanent quality failure
keeps the screen ineligible. This also records a stale restart after the last
cycle was fully acknowledged. A later source or transport error still writes a
collection failure and halts acquisition.
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

Each screen protocol binds `carry_paper_screen_v2` and an explicit
`carry_paper_attempt_lineage_v2`. Attempt 1 has no predecessor. Every later
attempt must name exactly the preceding attempt number, protocol SHA-256, final
journal tip, disposition, sorted machine-readable reason codes, and last
observation time. A successor never repairs or replaces a predecessor. The
lineage is part of the sealed research plan and therefore of the protocol
digest. An `observation_gap` disposition is accepted only after the predecessor
has exceeded the fixed 180-second gap limit at the new seal time. `seal` requires
an explicit `--attempt-lineage-json` input; it never guesses that a new root is
the first attempt. Since lineage v3, the v2 label is accepted only for attempt 1
(a tightening of the earlier v2 rule, which accepted any successor with
caller-asserted predecessor values); attempt 2 uses the canonical v3 record below.

Predecessor dispositions are limited to `invalidated` or `incomplete`, and
reason codes are limited to `clock_reversal`, `collector_failure`,
`deadline_missed`, `integrity_failure`, `late_start`, `manual_abort`,
`observation_gap`, `review_blocked` or `source_unavailable`. A lineage cannot
encode screen pass/rejection, returns, drawdown or other economic outcomes, so a
result-blind reviewer does not receive the prior screen decision. The hashes,
attempt number and disposition are still caller assertions: before sealing a
successor, an independent result-blind G0 review must resolve the predecessor's
preserved root with its original reader, verify the protocol and event-chain
identities and elapsed gap, and check for an unreported earlier attempt. The
schema does not authenticate external roots or prevent a caller from claiming a
new attempt 1.

Attempt 1's evidence root (the `funding-carry-forward-20260918` worktree) is lost:
a search of the collection machine found no preserved root, and on 2026-10-30 the
user said in chat that they do not know that data or where its results were saved and
that new data may be collected. Attempt 2 therefore uses lineage
`carry_paper_attempt_lineage_v3`, which adds `predecessor_root.status` with value
`available` or `unavailable` (v2 stays valid and unchanged). `plan` and `seal` require
`--predecessor-root-status`:

* `available` needs `--predecessor-root <PATH>` and is accepted only if the existing
  v1 journal reader verifies the root against the recorded protocol digest, 188
  events, final tip and last observation. The reader opens the database read-only
  (no hot-journal recovery), so a preserved root is not mutated; a root with a hot
  journal fails verification. The sealed protocol embeds that verification and the
  root path;
* `unavailable` accepts no root path. The sealed protocol records that the lineage
  values are taken from the public 2026-09-27/28 result-blind review record
  (`docs/research/current-status.md`, commit
  `b9f1ca65b26dcff7acbb089c7643e95702d474c9`), that they were NOT re-verified
  against the root at seal time, and the user's chat statement (`user_statement`),
  quoted verbatim with its date (2026-10-30). The quote is the two chat lines
  joined by one newline, and the record must equal the canonical record exactly.
  The statement covers only what the user said: attempt 1's data and
  result-storage locations are unknown, and new data may be collected; the record
  says it does not attest the predecessor lineage values or the absence of an
  earlier attempt. A `not_established` list names the predecessor protocol digest,
  final tip, observation gap, and the absence of an earlier unreported attempt.
  The disposition stays `invalidated` with reason `observation_gap`.

Threat model. These gates prevent accidental or non-canonical attempt-2 records
from being sealed or run through the supported entry points (`attempt2`,
`paper.cli`, `seal_paper_study`, `seal_collection`, `PaperEngine`, `PaperCollector`).
They are not authentication against a malicious local operator, who can always edit
code or files; no layer is claimed to be impossible to bypass. All gates use one
shared classifier (`predecessor.lineage_structure_problems`):

* No lineage is allowed. Otherwise the lineage must be an exact `dict` with an exact
  `str` schema. The v2 label requires exactly the v2 key set, so a
  `predecessor_root` (or any other v3-only field) cannot ride under it, and only
  attempt 1 may use it: `attempt_number` >= 2 under the v2 label is refused at every
  gate. This tightens a pre-existing v2 rule (v2 used to accept any successor with
  caller-asserted predecessor values); attempt 2 now has exactly the canonical v3
  record, and a later attempt needs a new, reviewed lineage version. The v3 label
  requires the type-strict canonical record for its status (`available`: for its
  recorded root path). Any other schema, including spelling variants, is refused.
* `seal_collection`, `seal_paper_study` and `paper.cli seal` apply that classifier
  and also refuse a v3 lineage unless given a module-private token that only
  `attempt2` holds (a caller contract, not authentication). `PaperEngine.initialize`
  applies the same rules (it requires a plain `dict` study and research plan, so a
  `dict` subclass cannot hide the plan), and `PaperEngine(root)` refuses to open,
  hence to append to, a root whose lineage the classifier rejects; opening reads no
  predecessor root. `validate_attempt_lineage` (hence `cli evaluate`) uses it for every v3-shaped
  record. `attempt2`'s own sealing preconditions (authorization URL shape, start
  notice, checkout identity, free space) are only enforced by `attempt2 seal`.
* Every `attempt2 launch`, including a supervisor relaunch after the start,
  re-reads an `available` root and refuses if it is lost, unreadable or differs. A
  gap event or heartbeat does not prove an earlier valid predecessor-root read.
* The collector (`PaperCollector`, hence `cli run`, and `cli supervise`) refuses,
  before taking the lock and without writing a failure marker, any lineage the
  classifier rejects, a research plan that is not a plain `dict`, and a
  `protocol.json` that is not canonical (a repeated, extra or reordered key, NaN or
  other non-canonical bytes). An `available` record must also
  re-read its recorded root on **every** collector entry, including restarts after
  the first journal event. A predecessor root that is lost or unreadable (for
  example a corrupt or locked journal) stops the collection and the restart exits
  with the refusal code, rather than being inferred from the attempt's own journal.
  Because that is a permanent refusal, an `available` lineage requires the old root
  to stay intact for the whole screen; the planned attempt 2 uses `unavailable`.

Plainly: attempt 1's evidence root is lost; its lineage rests on a prior recorded
review. The predecessor was not re-verified. For this case only, the runbook's
"fresh review resolves the preserved root" requirement is replaced by a fresh
result-blind review of the lineage documentation (this section and the
current-status record) and the pull request that introduced the rule. That review
cannot confirm the predecessor digests, tip or gap against the root, and it does not
establish that no earlier unreported attempt exists; the sealed record lists these
as `not_established`.

Require positive net profit after actual simulated exits, positive marked equity
change in every block, and total net profit greater than recorded fees. The last
condition is fee headroom at the realized trajectory, not a recomputed doubled-fee
strategy. Block boundaries use the last causal observation, no more than 180
seconds before the boundary. The last block includes terminal exit costs.
Require observed maximum drawdown below 10%, no quality failures or nonterminal
stop, at least one nonzero funding receipt in each block, no unpaid announced
funding for a previously held position, actual zero quantities and no pending
intent. Minute sampling is not an intraminute drawdown or liquidity guarantee.
Require at least one actual fill in each of the four declared instruments:
BTC spot, BTC perpetual, ETH spot and ETH perpetual. Assign each funding payment
to a block by its published settlement timestamp in the half-open screen window
`[start, close)`. Receipt during terminal grace does not move a post-close
settlement into the third block.
Any nonzero settlement at or after `close` rejects the screen as
`post_close_funding`, even when its cash credit is present in terminal account
equity. Preserve the realized ledger and fees; do not let out-of-window funding
qualify the fixed-window result.

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

### Collector liveness and silent-death records

Attempt 1 ended with no failure marker, no shutdown reason and no restart; this
section defines what every end of the collector must now leave behind. It is
operational evidence only: nothing here reads market data, edits the journal or
control database, retries a denied request, or counts toward the economic screen.

`python -m trade_rl.evaluation.paper.cli run` writes an atomic
`collector-heartbeat.json` beside the journal (states `starting`, `cycle_started`,
`cycle_completed`, `sleeping`, `finished`, `rejected`; at most every 15 seconds,
with the last committed command time). The heartbeat file is not part of the
protocol, the source roster or the replay. A heartbeat write error (typically a
Windows sharing violation while a scanner or the check reads the file) is retried
with exponential backoff and then tolerated; it halts the collector, with the real
cause recorded, only when no heartbeat has been written for 120 seconds, which is
before the 150-second stale limit would fire. A supervisor additionally writes a
best-effort `supervisor-heartbeat.json`, whose I/O errors never stop it.

Every non-terminal end must leave the single permanent `collection-failure.json`,
which is created exclusively so the first recorded cause is kept:

* the collector context manager (any exception, `KeyboardInterrupt`, or a
  `SIGTERM`/`SIGBREAK` turned into `CollectorTerminated`), which may additionally
  append the existing non-trading gap command;
* an `atexit` guard (`CollectorExitedUnfinished`) and, on Windows, a console
  control handler for close, logoff and shutdown (`CollectorConsoleClosed`);
  both write only the marker, because the process may be gone within seconds;
* `cli check` (stateless, safe from an independent scheduled task): after the
  frozen start, a heartbeat older than the stale limit (default 150 s, accepted
  range up to the frozen 180 s gap, as for `supervise`), or no valid heartbeat 180
  seconds after start, writes `CollectorHeartbeatStale` or `CollectorNeverStarted`;
  before the start it only reports. While a live supervisor is relaunching a child
  inside its own startup grace (default 300 s, at most 900 s) the stale limit is
  deferred and the state is `relaunch_in_progress`, because a relaunched child
  replays the whole journal before its first heartbeat; the deferral needs a fresh
  supervisor status, a child that has not yet beaten since its launch, and the
  grace not yet spent. `check` exit codes: 0 healthy (including `not_started`,
  `running`, `relaunch_in_progress` and a terminal state); 3 a failure marker exists
  or was just written (`halted`, `failed_stale`, `failed_never_started`); 5 unhealthy
  without a marker (`stale_before_start`, `heartbeat_invalid`), which writes nothing
  and needs a human look;
* `cli supervise`, which writes the marker for a hang, a death inside a cycle, an
  unrecoverable gap, a refused or early-exited child and a restart loop.

Only one collector may own a root. A second `cli run`, a supervisor started
while its predecessor's child is still alive, and `attempt2 launch` against a live
collector must not damage the screen: each detects the held `collector.sqlite`
lock (the supervisor and `launch` probe it before spawning, and a child that loses
the race exits with code 4) and ends with exit code 4, outcome `lock_contention`,
without writing any failure marker, heartbeat or journal event. Exit 2 stays a
real refusal and is recorded as `CollectorRefused`. Lock acquisition retries a busy
result with a short random backoff for 1.5 seconds before reporting contention, so two
processes started at the same instant cannot both lose and leave no owner; a lock
really held by a live collector stays busy through that window. Only a busy/locked
sqlite result counts as contention.

Hard process death (`taskkill /f`, power loss, machine sleep) cannot run any
in-process hook; only `check` or `supervise` can record it, which is why one of
them must always be running. Watchdog-written markers do not append a journal gap
and do not flatten positions: unresolved quantities and the last real marks
survive as evidence and the screen is rejected.

The supervisor relaunches a child only when relaunching cannot lose or invent
evidence: no failure marker, no terminal heartbeat, child exit code neither 0 nor
2, no unacknowledged cycle in the control database, the projected observation gap
(time since the last committed command plus a 60-second launch margin) still
within the frozen 180 seconds, and at most five relaunches. A death inside a
cycle is never resumed, because the control record deliberately halts reopening;
a transient network error is not retried either, because the collector halts on
any source failure. The 90-day screen therefore still ends on the first such
event. Startup replays the entire journal, so relaunch latency grows with the
journal; measure it before relying on the margin.

### Attempt 2 operator runbook

Attempt 2 is prepared offline but is neither sealed nor started. Attempt 1's root
is never resumed, backfilled or edited (and, if it exists, is preserved evidence that
is only ever opened read-only; see the lost-root rule above).
`python -m trade_rl.evaluation.paper.attempt2` has three commands (`plan`, `seal`,
`launch`) and `storage_probe` one more; each of which refuses unless
`--authorization` is an Issue #810 comment URL
(`https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-<id>`). The URL
shape is all that code can check, so a human must post the authorization comment
first; nothing here proves the comment exists or says what it should say.

1. Human prerequisites, before any command:
   * post the Issue #810 authorization comment naming the intended UTC start;
   * if the attempt-1 root is available: obtain a fresh result-blind review that
     resolves it with its v1 reader and confirms protocol, event chain, final tip,
     last observation and elapsed gap, and checks for an unreported earlier attempt;
     the lineage values are assertions until that review exists. If it is lost
     (`--predecessor-root-status unavailable`, the current case): obtain a fresh
     result-blind review of the lineage documentation and the pull request that
     introduced the lost-predecessor rule instead; it does not re-verify the
     predecessor;
   * create a dedicated detached worktree of the merged commit and a private locked
     environment; keep it unmodified until the screen ends (the sealed
     implementation and runtime digests are rechecked before every acquisition);
   * on the collection machine disable sleep and hibernation, postpone automatic
     restarts and updates for 90 days, keep clock synchronisation on (a clock step
     backwards halts collection), and use stable power and network.
2. Storage and runtime check (the screen needs 129,604 sixty-second slots, 90 days
   plus the 180-second grace, without interruption). Measure the bytes per
   observation: from the preserved attempt-1 root, measured without opening it for
   writing (total bytes, including
   rule captures, divided by its 188 observations) if it exists; otherwise with the
   storage probe below. The
   plan multiplies that rate by the slot count and a 1.25 headroom factor and
   compares it with free space on the target volume; `seal` refuses unless the
   result is `ok`. Also reserve enough memory and time for journal replay on a
   restart, which is not measurable offline.
   The storage probe is the only networked step before launch. It runs the real
   collector for a bounded trial (default 20 one-minute slots, about 24 minutes
   including the 60-second lead and the 180-second grace) into a new scratch root
   outside the repository and outside any sealed root, and prints the bytes per
   observation:

   ```
   python -m trade_rl.evaluation.paper.storage_probe --root <NEW_SCRATCH_ROOT> \
     --authorization <COMMENT_URL> --allow-network --output <NEW_REPORT.json>
   ```

   Pass `conservative_bytes_per_observation` from the report as the rate. It is
   `total bytes / observations + mean snapshot bytes + journal bytes / events`: the
   measured rate plus one extra snapshot capture and one extra journal event per
   observation, because a trade cycle takes a second capture (decision, then
   execution) and appends a second event, and a probe that does not trade measures
   neither. The journal term uses the whole database file and an execution event is
   about as large as a decision event (both store the full account status), so it errs
   high, and the fixed files and one rule capture of a short probe are spread over
   few observations. Growth specific to fills beyond that is not measured; the plan's
   own 1.25 headroom is the only further margin. The probe root's research plan is
   `carry_paper_storage_probe_v1` with no attempt lineage: `attempt2 seal` (it only
   writes new roots), `attempt2 launch`, `cli evaluate`, `cli run` and `cli supervise`
   all refuse it, so only the probe's own entry point runs it. It is capacity
   evidence only, consumes no screen data and is never an attempt. A source error in
   the probe halts it (no retry) and yields no rate.
3. Plan (no side effects; `--output` writes a new draft file once):

   ```
   python -m trade_rl.evaluation.paper.attempt2 plan --root <NEW_ROOT> \
     --start-at <FUTURE_UTC_ISO> --authorization <COMMENT_URL> \
     --predecessor-root-status unavailable \
     --bytes-per-observation <MEASURED_RATE> --output <NEW_DRAFT.json>
   ```

   Use `--predecessor-root-status available --predecessor-root <ATTEMPT_1_ROOT>`
   only when the root exists and the v1 reader verifies it.

   The draft binds the exact attempt-1 lineage and its root status, the fixed
   `carry_paper_screen_v2` plan, the checkout commit, and the implementation and
   runtime digests, and lists `seal_blockers`. The start must be at least five
   minutes after the wall clock.
4. Seal (offline) with the same arguments and `seal`. It refuses a missing or
   unknown predecessor-root status, `available` without a verifiable root,
   `unavailable` with a root path, a dirty or
   unidentifiable checkout, an existing root, a missing authorization, a start not
   at least five minutes in the future at seal time, and insufficient storage. It
   prints `protocol_sha256`; record that digest, and post it with the start time.
5. Launch (the only networked step, and only with `--allow-network`):

   ```
   python -m trade_rl.evaluation.paper.attempt2 launch --root <ROOT> \
     --start-at <SAME_START> --authorization <COMMENT_URL> --allow-network \
     --protocol-sha256 <DIGEST> --log-file <PATH_OUTSIDE_ROOT>
   ```

   It re-verifies the root and runs the supervisor in the foreground, so run it
   under Windows Task Scheduler (see below) well before the start; the collector
   waits for the start. A first launch after the start is refused as a late start.
   After the start, `launch` is accepted only to restart a supervisor for a verified
   root whose collector has already written a heartbeat.
6. Windows Task Scheduler recipe (PowerShell; `python.exe` is the locked
   environment's interpreter). Task A runs the launch command at startup with
   restart-on-failure, no execution time limit and no idle stop. Task B runs
   `cli check` every minute under the same account:

   ```
   $a = New-ScheduledTaskAction -Execute <PYTHON> -WorkingDirectory <CHECKOUT> -Argument '-m trade_rl.evaluation.paper.attempt2 launch ...'
   $s = New-ScheduledTaskSettingsSet -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit ([TimeSpan]::Zero) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
   Register-ScheduledTask -TaskName carry-a2-supervisor -Action $a -Settings $s -Trigger (New-ScheduledTaskTrigger -AtStartup) -User <ACCOUNT> -RunLevel Limited
   ```

   Without a supervisor, `start "" /b <PYTHON> -m trade_rl.evaluation.paper.cli run
   --root <ROOT> --protocol-sha256 <DIGEST> > <LOG> 2>&1` still gets the heartbeat
   and failure records, but nothing relaunches it. Exit codes: 3 from `check`,
   `supervise` or `launch` means a failure marker exists or was just written; alert
   on it, do not clear it, and do not start a replacement into the same root. Exit 4
   from `run`, `supervise` or `launch` means another live collector already owns the
   root: nothing was written, the owner is undisturbed, and the right action is to do
   nothing (this is the normal outcome of restarting the Task Scheduler job while the
   previous child survives). Exit 5 from `check` means unhealthy with no marker (a
   stale heartbeat before the start, or an unreadable heartbeat inside the start
   grace); investigate, because the next `check` after the start will write a
   marker. Exit 1 from `run` or `supervise` means the collection ran to a terminal state
   but the screen is rejected or the child finished non-zero (quality failure or
   positions not flat); it is final and needs `evaluate` for the reasons. Exit 2 is a
   refusal or error: `check`, `supervise` and `status` write no marker on it, but `run`
   MAY have written one (any exception inside the collector context, such as a source
   error or provenance change, records `collection-failure.json` before returning 2),
   so always read `check` afterwards. Exit 4 is trusted only when a live owner exists:
   the lock probe counts only a busy/locked database as contention, and any other
   sqlite error is reported as an error (exit 2), not as exit 4. After an exit 4 the
   surviving owner has no supervisor; if it later dies nothing relaunches it, so the
   `check` task must run for the entire screen and is what turns that death into a
   marker.
7. Monitor: `cli check` each minute (exit 0 healthy, 3 marker written or present,
   5 degraded without a marker); `cli status` for
   journal-chain-only progress; daily free-space and log-size checks; compare the
   supervisor log with the heartbeat. Operational status is not economic evidence.
8. Terminal: no earlier than close plus 180 seconds the supervisor reports
   `finished` and the heartbeat is terminal. Run `cli status`, record the final
   `tip` externally, then `cli evaluate --root <ROOT> --protocol-sha256 <DIGEST>
   --expected-tip <TIP> --output <NEW_FILE>`. A pass retains
   `production_eligible=false`.

The screen is invalidated by any `collection-failure.json`; an observation gap over
180 seconds; a first observation more than 180 seconds after start; a last
observation outside close plus 120 to 180 seconds; an unacknowledged cycle or
failed, missing or unconsumed capture; changed source or runtime identity; clock
reversal; machine sleep or a lost network link that ends or stalls a cycle; any
single source error (there is no retry); a late, missing, revised or out-of-order
funding settlement; an unmatched hedge, drawdown at or above 10% or a stop; no
funding receipt in a block; positions not flat at the end; any fill missing from
the four instruments; and nonzero funding settled at or after close. Do not resume
or repair a failed root; a further attempt needs a new lineage, review and seal.
