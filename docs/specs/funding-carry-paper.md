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
Offline verification does not imply freshness now. Two separate bounds apply.
Every quote must be within five seconds old and one second ahead of the capture
completion (`completed_at`), checked at capture and again by the reader; the
capture itself may span up to ten seconds. Supplying a consumption time
(`as_of`, the decision or execution time) additionally requires
`0 <= as_of - completed_at <= 10 s` (`MAX_CONSUME_LATENCY_MS`, equal to the paper
decision-to-execution window) and rejects earlier receipts; quotes are not
re-aged against `as_of`. Consumption beyond the bound fails with
"snapshot consumed too late after capture".

Resulting worst case: a quote may be `MAX_QUOTE_AGE_MS` (5 s) old at capture
completion and consumption may follow by up to `MAX_CONSUME_LATENCY_MS` (10 s),
so a venue-stamped mark can be up to 15 s old at valuation. Fills are tighter:
the depth executor additionally requires the depth receipt to be at most 5 s old
at execution, so futures depth used for a fill is at most 10 s old. These
tolerances are protocol parameters, set from the source constants and venue and
engineering reasoning only (no attempt-2 timing was used). Attempt 3 pre-registers
them in its research-assurance packet (see "Attempt 3 pre-registration" below),
which must pass G0-G2 plus independent review before execution; no execution
under changed tolerances may reuse an earlier seal.

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

Attempt 1's evidence root (the `funding-carry-forward-20260918` worktree) has not
been located: on 2026-10-30 the user said in chat that they do not know that data or
where its results were saved and that new data may be collected, and a documented
read-only implementer search of drive C: (excluding the top-level directories Windows, Program Files, ProgramData, System Volume Information and Recovery and directories named node_modules or site-packages at any depth) on 2026-11-06 found no root
with the attempt-1 protocol digest (scope and limits in the implementer search record
of `docs/architecture/research-assurance.md`). The user's words do not say the root is
lost or unreadable and do not mention G0. Attempt 2 uses lineage
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
  The disposition stays `invalidated` with reason `observation_gap`. The code
  accepts this status on attestation alone and does not check whether the root was
  located or can be read. Using it skips G0 predecessor verification. The owner authorized that
  skip, bounded to funding-carry attempt 2, on Issue #810 on 2026-11-06 (comment
  6680576996, https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-6680576996; the planner's
  record of the owner's corrected chat answer 「この前提で承認する」, quoted verbatim in
  `docs/architecture/research-assurance.md`). It is not a general exception, relaxes no
  criterion, and the four `not_established` items stay `not_established`.

**Search-scope correction (2026-11-06):** The original Issue #810 comment
6673566542 and its question described a search across all of C:, which was
inaccurate. The corrected [owner authorization](https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-6680576996)
records the fresh response 「この前提で承認する」 after the actual exclusions were
disclosed. In a second read-only search, 143,061 directories were visited,
43 paths were unreadable and 50 links/junctions were not followed. The
search covered drive C: excluding the top-level directories Windows,
Program Files, ProgramData, System Volume Information and Recovery and
directories named node_modules or site-packages at any depth, and **included**
Program Files (x86). This corrected record (comment 6680576996), not the
superseded 627-file record, is the search record the sealing conditions rely on. The **616 found** protocol.json files yielded no matching
attempt-1 digest; the prior 627 count was an earlier scoped observation, not
a machine-wide enumeration. Other machines, cloud/backup, Docker volumes,
external media and excluded/unreadable locations were not searched. This
does not establish absence of the root; the four predecessor facts remain
not_established. The question put to the owner listed the exclusions and the
not-searched locations but not the 43 unreadable paths, the 50 unfollowed
junctions or that Program Files (x86) was searched; those details are in the
search record of the same comment. The same owner record makes an
attempt-2-only exception: a review by a model different from every
implementation model (Claude Haiku 5.5), posted from the same GitHub account,
satisfies sealing condition (a). It is operator-enforced (sealing code does not
call the research tool), does not establish different-principal independence and
does not change the existing tools.research_assurance_github same-principal
refusal: the tool does not report PASS for it, the packet is UNREVIEWED and no
PASS is claimed.

Threat model. These gates prevent accidental or non-canonical attempt-2 records
from being sealed or run through the supported entry points (`attempt2`,
`paper.cli`, `seal_paper_study`, `seal_collection`, `PaperEngine`, `PaperCollector`).
They are not authentication against a malicious local operator, who can always edit
code or files; no layer is claimed to be impossible to bypass.

Scope of these checks. In scope are crafted **data** passed through the supported
entry points: JSON files, CLI arguments and plain built-in Python values (`dict`,
`list`, `str`, `int`, `float`, `bool`, `None`) given to `attempt2`, `paper.cli`,
`seal_paper_study`, `seal_collection`, `PaperCollector` and `PaperEngine`
initialization or open. A bypass by such data is a defect. Out of scope are objects
whose classes override dunder methods (`__eq__`, `__hash__`, `__getitem__`, `get`
and so on) to deceive a check, monkeypatching, private internals, and editing code or
sealed roots on disk: these are code-level manipulation by a local operator. The
exact-type checks on containers and mapping keys (an exact `dict`, and `type(key) is
str` at every level, before any lookup) are defence in depth, not a security
boundary.

All gates use one shared classifier (`predecessor.lineage_structure_problems`):

* No lineage is allowed. Otherwise the lineage must be an exact `dict` with an exact
  `str` schema. **Every key** in the in-memory study, research plan and nested
  lineage must be a native `str` (`type(key) is str`) before any `dict.get`
  lookup, including keys inside lists of JSON objects. An equal-looking `str`
  subclass with custom hashing/comparison cannot conceal the v3 token rule.
  The v2 label requires exactly the v2 key set, so a
  `predecessor_root` (or any other v3-only field) cannot ride under it, and only
  attempt 1 may use it: its `attempt_number` must be the exact integer `1` and
  `predecessor` must be null. Float/boolean/subclassed or larger/smaller numbers
  are refused at every gate. This tightens a pre-existing v2 rule (v2 used to accept any successor with
  caller-asserted predecessor values); attempt 2 now has exactly the canonical v3
  record, and a later attempt needs a new, reviewed lineage version. The v3 label
  requires the type-strict canonical record for its status (`available`: for its
  recorded root path). Any other schema, including spelling variants, is refused.
* `seal_collection`, `seal_paper_study` and `paper.cli seal` apply that classifier
  and also refuse a v3 lineage unless given a module-private token that only
  `attempt2` holds (a caller contract, not authentication). `PaperEngine.initialize`
  applies the same rules (it requires a plain `dict` study and research plan
  with native-string keys throughout, so neither a `dict` nor a string-key
  subclass can hide the plan), and `PaperEngine(root)` refuses to open,
  hence to append to, a root whose lineage the classifier rejects; opening reads no
  predecessor root. `validate_attempt_lineage` (hence `cli evaluate`) uses it for every v3-shaped
  record. `attempt2`'s own sealing preconditions (authorization URL shape, start
  notice, checkout identity, free space) are only enforced by `attempt2 seal`.
* Every `attempt2 launch`, including a supervisor relaunch after the start,
  re-reads an `available` root and refuses if it cannot be read or differs. A
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
  to stay intact for the whole screen; attempt 2 would use `unavailable` only if the
  G0 exception is separately authorized by the owner.

Plainly: attempt 1's evidence root has not been located; if it is never found, its
lineage rests on a prior recorded review and the predecessor is not re-verified. Because
the owner authorized the G0 exception for attempt 2 on Issue #810 (2026-11-06, comment
6680576996: the corrected search premise, with the owner's re-approval and the same-account
review route; the earlier comment 6673566542 is superseded for that premise and is cited
only for the start time and the Task Scheduler jobs), the runbook's "fresh review resolves the preserved root" requirement is
replaced by a fresh result-blind review of the lineage documentation (this section and
the current-status record) and the pull request that introduced the rule. That review
cannot confirm the predecessor digests, tip or gap against the root, and it does not
establish that no earlier unreported attempt exists; the sealed record lists these
as `not_established`. If the attempt-1 root is found, the `available` path with the v1
reader must be used instead; for any other family, attempt or predecessor G0
predecessor verification applies unchanged.

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

Attempt 2 was sealed and launched from the frozen checkout described below and ended
INVALID on 2026-11-07 (see "Attempt 3 pre-registration"). This runbook is retained as
the procedure of that frozen attempt and the template for a successor, which needs its
own lineage, packet, review and seal. Attempt 1's root
is never resumed, backfilled or edited (and, if it exists, is preserved evidence that
is only ever opened read-only; see the unavailable-predecessor rule above).
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
     the lineage values are assertions until that review exists. If it cannot be
     found (`--predecessor-root-status unavailable`, the current case): the separate explicit
     owner authorization on Issue #810 that names the G0 predecessor-verification
     exception exists (corrected comment 6680576996, 2026-11-06, https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-6680576996;
     the owner's 2026-10-30 words alone say only that they do not know where the data
     is and that new data may be collected). Then obtain a fresh result-blind review of the lineage
     documentation and the pull request that introduced the unavailable-predecessor rule
     instead; it does not re-verify the predecessor, and the predecessor chain stays
     `not_established` (the predecessor-root-unavailable section in
     `docs/architecture/research-assurance.md` defines the conditions and what is
     recorded; it applies to no other case);
   * obtain an independent `research_assurance_v1` review of the attempt-2 author
     packet (see the packet section below) from a different authenticated GitHub
     principal (for attempt 2 only, the owner-authorized same-account exception of
     Issue #810 comment 6680576996 applies instead: a review by a model different from
     every implementation model posted from the same GitHub account, with a valid
     receipt for the exact digest and heads; see "Review route" below and the
     "Predecessor root unavailable: owner-authorized G0 exception for carry attempt 2"
     section of `docs/architecture/research-assurance.md`); a missing review leaves the
     packet UNREVIEWED and attempt 2 unsealed;
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

The attempt-2 observation and its 90-day `cli status` / `cli evaluate` must run from the
frozen checkout at launch commit `d649fe053f72`, not from
`main`; `main` may evolve after launch, and a change to packet-bound code requires a new
packet and a new seal before any new execution.

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

### Attempt 2 research-assurance author packet

The Research Assurance Gate (Issue #687, `docs/architecture/research-assurance.md`)
requires a versioned author packet before a new economic research protocol. The
attempt-2 packet is
[`report/issue810-carry-attempt2-research-assurance-20261105.json`](../../report/issue810-carry-attempt2-research-assurance-20261105.json). It
describes the existing frozen `carry_paper_screen_v2` plan and the attempt-2
lineage v3 (`predecessor_root` unavailable) exactly as implemented, changes no
criterion, and records G0 (question, counter-hypothesis, cheapest falsifier, stop
rule), the G1 chain from source to decision with all eight boundaries, the G2
independent oracles by test path, the evidence stage and claim boundary, the
attempt-1 `not_established` limits, the threat model and the storage-probe step.

* Canonical digest: `617988282ad6dfbf0ee7188f002676a1675dfac38c9ac969071bee3091b2e770`
  (`python -m tools.agent_repo assurance digest report/issue810-carry-attempt2-research-assurance-20261105.json`).
* Status: **UNREVIEWED**. `assurance check` exits non-zero and reports
  `economic_execution_authorized=false`. No review exists, so this is not a pass.
* The packet binds protocol and implementation head `326188fc0c3cbcc4ff8bf453854e8515ed983abe`, the last commit that
  changes `trade_rl/` in the pull request that adds the packet
  (`git diff 326188fc0c3cbcc4ff8bf453854e8515ed983abe..<PR head> -- trade_rl/` is empty). A packet stored in the tree
  cannot name the commit that contains it, so the packet is committed in a later
  descendant and the PR head differs from the implementation head;
  `tools.research_assurance_github` needs the PR head and the review `commit_id` to equal
  `implementation_head`, so it cannot report PASS for an in-tree packet. The digest is the
  SHA-256 of the packet JSON alone (`assurance_digest`), so a change to `trade_rl/` does
  not by itself alter it: the digest changed in the head re-pin commit `5399cc664`
  because that commit re-pinned the two head fields and rewrote the packet's `stop_rule`,
  `limitations`, one `permitted` claim, `next_authorized_action` and `decision_rule`
  text (review-route and search-scope wording); the code at `326188fc0` was already
  final and `trade_rl/` did not change in it. The historical packet and digest
  remain pinned as published; subsequent changes to `trade_rl/` or a sync merge do
  **not** retroactively alter that packet or invalidate evidence about the code as
  it stood at publication. However, a **new execution using changed dependency
  code** requires a separately frozen packet with a new `implementation_head`,
  digest and review before it can be authorized. Re-pinning the old packet in
  place is forbidden. The dependency scope is `trade_rl/evaluation/paper/`
  plus its transitive `trade_rl` imports (including lazy imports and ancestor
  `__init__.py`), resolved at the pinned commit. The full-history guard
  `test_heads_pin_the_code_as_of_the_frozen_packet_revision` compares that
  dependency closure from the pinned implementation commit through the **last
  commit modifying the frozen packet** (`git log -1 -- <packet path>`), not
  through the current `HEAD`. The public record shows that
  `326188fc0c3cbcc4ff8bf453854e8515ed983abe..5399cc664e561590312c682a67ac277c067c8de3`
  modified only docs, tests and the packet, and no `trade_rl/` path.
  Changes to other code after publication do not invalidate the historical
  receipt, but they also cannot be used to run a new attempt under that receipt.
  The guard runs only with the pinned commits reachable in full history;
  default shallow CI checkouts (`fetch-depth` 1 or 2) skip it, so CI Green alone
  does not establish the historical source-closure check.
  `test_launch_commit_runs_packet_bound_code` additionally asserts, under the same
  full-history condition, that this dependency closure is identical between the
  pinned implementation commit and the frozen launch commit `d649fe053f72`.

Sealing attempt 2 still needs all of the following, and this packet supplies none of
them: (a) an independent `research_assurance_v1` review of the exact digest and
heads (ordinarily verified through `tools.research_assurance_github`; for attempt 2 only,
the owner-excepted review described under "Review route" below); (b) explicit user
authorization on Issue #810 for the seal; and (c) either a verified attempt-1 root
(the `available` path) or a separate explicit owner authorization on Issue #810 that
names the G0 predecessor-verification exception (see
`docs/architecture/research-assurance.md`). Evidence for (b) is the comment
<https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-6659653178>
(2026-11-05, posted from the owner's account and recorded by the planner relaying the
owner; the owner can edit or revoke it; it names no UTC start; code checks only the URL
shape). It is authorization evidence and does not replace (a); it does not satisfy (c),
because the owner's quoted words in it do not mention G0. Evidence for (c)
with `unavailable` is the **corrected** comment
<https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-6680576996>,
which records the owner's fresh 「この前提で承認する」 on the accurately bounded
C: search premise and four `not_established` predecessor facts. The **original**
comment 6673566542 separately records the selected screen start
`2026-11-07T00:00:00Z` and the Windows Task Scheduler supervisor/watchdog
authorization; its description of an all-C: search was inaccurate and is
superseded only for that search premise. Neither comment authorizes FS1 scoring,
unused or final-window reads, live trading, order routing, relaxed criteria,
or use of an attempt-1 economic or status readout. Neither comment is itself the review that (a) requires; the corrected comment only defines which review route satisfies (a) for attempt 2.

Review route. AGENTS.md (#687) ordinarily requires a review from a different
authenticated GitHub principal. The owner's words of 2026-11-05 were
"別アカウントによるレビュー　別になんでもいいよ。agyのレビューでいいよ"; those words alone
did not establish principal independence or authorize a same-principal exception.
The owner **explicitly** made an exception **for attempt 2 only** in corrected
Issue #810 comment 6680576996: a review by a model different from every
implementation model (Claude Haiku 5.5), posted from the shuntatsu account,
satisfies sealing condition (a). It is operator-enforced (sealing code does not
call the research tool) and is not principal independence:
`tools.research_assurance_github` still rejects same-principal reviews and does not
report PASS for that route. The packet stays UNREVIEWED, no PASS is claimed, and an
actual exact-digest/HEAD result-blind review must still exist before any seal. Nothing is sealed, launched or collected under
this packet, and the storage probe (the only networked step before launch)
is not run under it.

### Attempt 3 pre-registration

Attempt 3 is a pre-registration only: nothing is implemented beyond one
behaviour-preserving constant, nothing is sealed, launched, probed or collected, and
the seal, start, authorization comment and launch commit below are
`PENDING_OWNER_AUTHORIZATION`. Its research-assurance
author packet is
[`report/issue810-carry-attempt3-research-assurance-20261108.json`](../../report/issue810-carry-attempt3-research-assurance-20261108.json)
(digest `28d3dc0fead12c6cb0ccb7bcf073191547bf1c9fb88fb8b9665bea4e6fb9a8ca`, status **UNREVIEWED**, `economic_execution_authorized=false`;
`python -m tools.agent_repo assurance digest <packet>`). Its `protocol_head` and
`implementation_head` are both `ec7acfd1f7ca51afa805edf2051e32eff764591b`, the last
commit that changes `trade_rl/` in the pull request that adds it. That head holds no
collection-policy code, so this packet can never support a seal; a later implementation
packet needs a new head, digest and review.

**Why a successor.** Attempt 2 ended INVALID: its collector halted at cycle 61
(`2026-11-07T00:59:07Z`) and wrote `collection-failure.json` because the snapshot reader
re-applied the five-second per-quote age bound at consumption time (root cause on Issue
#810, comment 6711911740; fixed by PR #1164). That is a software defect read from the
failure marker. No attempt-1 or attempt-2 price, equity, profit, drawdown, fill or
funding figure and no attempt-2 timing metadata was read or is used; the attempt-2 root,
its failure marker, frozen checkout and the `carry-a2-*` scheduled tasks stay untouched.
The economic screen is unchanged: same universe (BTCUSDT, ETHUSDT; funding at 00, 08 and
16 UTC), strategy, fees, depth fraction, stop and every economic acceptance criterion.

**Timing tolerances (protocol parameters).** Each value equals a source constant or
literal, and `tests/evaluation/test_paper_attempt3_assurance_packet.py` fails if the
packet, this section and the source diverge. They are justified from venue and
engineering mechanism only and may not change after a seal.

Timing protocol parameters: quote_age_ms=5000; future_skew_ms=1000; request_seconds=5; capture_span_seconds=10; consume_latency_ms=10000; decision_window_seconds=10; depth_receipt_age_seconds=5; worst_case_mark_age_ms=15000.

| Parameter | Source | Bounds |
|---|---|---|
| `quote_age_ms` | `MAX_QUOTE_AGE_MS` | venue-stamped quote age at its receipt and at capture completion |
| `future_skew_ms` | `MAX_QUOTE_FUTURE_SKEW_MS` | quote ahead of local receipt (clock-offset tolerance) |
| `request_seconds` | `MAX_REQUEST_SECONDS` | one request, monotonic; no response slower than the quote-age bound can carry a fresh quote |
| `capture_span_seconds` | `MAX_CAPTURE_SPAN_SECONDS` | first request start to final receipt (cross-request skew) |
| `consume_latency_ms` | `MAX_CONSUME_LATENCY_MS` | capture completion to decision or execution time; not a re-aging of quotes |
| `decision_window_seconds` | `account.py`, `supervisor.py`, `depth.py` literals | decision to execution |
| `depth_receipt_age_seconds` | `depth.py` literal | depth receipt to execution |
| `worst_case_mark_age_ms` | `quote_age_ms + consume_latency_ms` | venue-stamped mark at valuation; futures depth in a fill is at most 10 s |

The flat 5bp adverse buffer is a paper assumption and is not calibrated to these ages.

**Collection-failure, retry and gap policy (adopted under the owner's delegation).**
Attempt 2 halted on a single failure; over 129,604 slots a transient network error is
likely, so that behaviour makes a ninety-day window a bet against transient faults. This
bounded, result-blind policy is derived from engineering and the frozen cadence only.
The owner delegated the four open decisions on 2026-11-08 (「全部お任せで。一番成果が得られるもので」,
relayed by the planner on Issue #810, comment 6715501420; the resolutions are the
planner's, not the owner's drafting) and the policy was adopted as proposed. It is
**not implemented**: the source still halts permanently on the first collection failure
and the assessor still rejects any failed capture. It also relaxes one attempt-2
operational-quality clause (a failed capture rejects the screen) into a registered
budget, pre-registered before any attempt-3 data and relaxing no economic criterion.

Collection policy parameters: retries_per_slot=2; retry_delay_seconds=2; retry_start_window_seconds=30; max_consecutive_skipped_slots=1; max_skipped_slots_per_hour=2; max_skipped_slots_total=129; cadence_seconds=60; frozen_gap_limit_seconds=180.

* Class D (HTTP 401, 403, 418, 429, 451 or any authorization, rate-limit or geo denial)
  and Class I (provenance, protocol digest, clock reversal, journal or lock error,
  malformed or oversized response, crossed or unsorted book, any other exception) are
  never retried and write the permanent failure marker.
* Class T (timeout, connection or DNS error, HTTP 500, 502, 503, 504, and a freshness
  rejection of one capture) may be retried only inside the same slot, no later than
  `retry_start_window_seconds` after the slot start, `retry_delay_seconds` after the
  failed attempt, at most `retries_per_slot` times, only before a capture has been
  consumed, and never after a decision is committed: a failed execution capture lets the
  pending decision expire and skips the slot. Every failed attempt keeps its failure
  evidence and is registered; a consumed capture is never discarded for another.
* A skipped slot appends no financial command, so holdings and marks stay as they were.
  One skipped slot leaves a gap near 120 s; a second consecutive one reaches the frozen
  180-second limit, which is not relaxed. More than `max_skipped_slots_per_hour` in any
  60 consecutive slots marks a persistent fault; `max_skipped_slots_total` is
  floor(129,604 / 1000), 0.1% of the slots. A breach of either, or any Class D or I
  failure, ends the attempt INVALID.
* No settlement-specific tolerance is added: the frozen rule that funding first received
  more than 180 seconds after a settlement is a permanent quality failure still applies
  and a skip cannot extend it. A failed rules refresh does not skip a slot while the held
  rules are at most one hour old. A death inside a cycle is still never resumed.
* Still to build and review before any seal: the collector and attempt registry, a
  `carry_paper_screen_v3` plan whose economic fields equal `carry_paper_screen_v2`, an
  evaluator audit that every failed capture is registered and within budget, an
  attempt-lineage version naming attempt 2 (its root is available, so ordinary G0
  predecessor verification applies; the predecessor-root-unavailable G0 exception of Issue
  #810 comment 6680576996 does not extend to attempt 3), and fake-transport tests for each class and
  budget.
* Accepted side effects and G3 claim boundary: a failed rules refresh proceeds on held
  rules up to one hour old, and skipped slots may coincide with fast markets, so the
  retained observations can lean toward calm periods; the budgets bound this bias but do
  not remove it. No claim may extend beyond the retained slots, and every result must
  report the skip count with its funding-proximity distribution.

**Run identity and seal procedure (nothing computed).** Run root
`C:\trade_rl_runs\attempt3\root-<UTC seal date YYYYMMDD>` (new, outside every repository
and not beneath the attempt-2 root); storage-probe scratch root
`C:\trade_rl_runs\attempt3\storage-probe-<UTC date YYYYMMDD>`; frozen checkout
`C:\trade_rl_frozen\attempt3-<first 8 hex of the launch commit>`; scheduled tasks
`carry-a3-supervisor` and `carry-a3-check`. The attempt-2 seal and protocol digest are
never reused and the attempt-3 protocol digest must differ from every earlier digest.
Order: review and merge this pre-registration; then, in a separate pull request,
implementation; implementation packet re-pinned to that head with a second review; independent
review of that exact digest and heads (the attempt-2 same-account receipt route of Issue
#810 comment 6680576996, extended to attempt 3 by comment 6715501420: a fresh Claude Haiku
5.5 agent, a different model from the Opus planner and the Sonnet implementer, posts a
pull request review from the shuntatsu account; operator-enforced, not different-principal
independence, and `tools.research_assurance_github` still rejects it, so the packet stays
UNREVIEWED with no PASS claimed; the predecessor-root-unavailable G0 exception is not
extended); fresh storage probe with its own
authorization (not an attempt-2 byte rate); owner authorization comment on Issue #810
naming the start; `plan`, then offline `seal` at least five minutes before the start;
`launch` from the frozen checkout.

| Field | Value |
|---|---|
| Collection policy decision | Issue #810 comment 6715501420 (owner delegation, adopted) |
| Review route | Issue #810 comment 6715501420 (extends the receipt route of comment 6680576996) |
| Authorization comment | `PENDING_OWNER_AUTHORIZATION` |
| Screen start (UTC) | `PENDING_OWNER_AUTHORIZATION` |
| Protocol digest (seal) | `PENDING_OWNER_AUTHORIZATION` |
| Launch commit | `PENDING_OWNER_AUTHORIZATION` |
