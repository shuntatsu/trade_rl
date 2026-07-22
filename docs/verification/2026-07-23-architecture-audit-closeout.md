# Architecture Audit Closeout — 2026-07-23

## 1. Purpose and source boundary

This document closes or reclassifies the findings recorded in
`docs/verification/2026-07-22-post-merge-architecture-audit.md` after the
independent remediation pull requests were integrated.

The current source baseline is `main` merge commit
`a661b8e222a1fa7bb3dd69ae2263195a9786755c`, which merged PR #107 after the
sequence-projection guard, telemetry identity, stream-generation, process-lock,
Live Training, environment-runtime, and architecture-boundary remediations.

The 2026-07-22 report remains a historical record of what was observed at its
audited commit. It must not be read as the current defect list. Current status is
established by this closeout and the finding-specific verification documents.

Production remains `NO-GO`. Closing a software-architecture finding does not
establish profitability, exchange-equivalent fills, operational authorization, or
direct exchange capability.

## 2. Closeout summary

| Finding | Original priority | Current status | Remediation |
| --- | --- | --- | --- |
| `AUD-DOC-001` | P3 | RESOLVED | PR #82 and executable documentation contracts |
| `AUD-STUDIO-001` | P1 | RESOLVED | PR #85 environment/current-episode isolation; PR #103 producer episode identity |
| `AUD-ARCH-001` | P2 | RESOLVED | PR #79 explicit maintained imports and removal of initializer mutation |
| `AUD-SIM-001` | P2 | RESOLVED | PR #107 stateful execution phase decomposition |
| `AUD-CI-002` | P2 | RESOLVED | PR #88 stable numerical contract; PR #106 permanent cross-platform guard |
| `AUD-RL-001` | P2 risk | OPEN RISK, REDUCED | PR #92 extracted environment runtime/step services; facade-size risk remains |

No confirmed P0 or P1 product defect remains from the 2026-07-22 finding set.
The only open item is a maintainability risk without a reproduced behavioral
failure.

## 3. Finding-by-finding disposition

### AUD-DOC-001 — RESOLVED

PR #82 synchronized the maintained English and Japanese entry points, START guide,
architecture, research status, Binance, GPU-operation, and Studio documentation
with the post-stateful implementation. It also added executable documentation
contracts derived from source constants and Import Linter boundaries.

The remediation prevents the specific observation-v3, bundle-v4, stale-layer, and
closed-finding drift reproduced by the audit. This closeout additionally updates
the architecture description for Live Training episode isolation and the
stateful-execution phase services.

Documentation can drift again as code evolves, so the executable contract remains
the control. This is not retained as an open finding.

### AUD-STUDIO-001 — RESOLVED

PR #85 changed the maintained Live Training path so replay state is derived from
one selected vector environment and only that environment's current episode. The
same selected records drive the chart, cursor, price, PnL, baseline, drawdown,
events, playback, and jump controls.

PR #103 added nullable producer-issued `episode_id` values while preserving
historical `training_telemetry_v1` JSONL compatibility. The producer owns one
active identity per environment, rotates after terminal or truncated records, and
continues above existing stream identities after sampler restart. The browser
prefers explicit identity and retains terminal/counter-rollback segmentation only
for historical null-ID records.

Supporting boundaries are also current:

- PR #95 serializes cooperating process append/read/index transactions and fails
  closed on incomplete tails or replaced writer inodes;
- PR #99 binds cursors to an opaque stream generation and discards mixed-generation
  Status/Events responses before publishing records.

The confirmed false continuous price/equity/PnL path is no longer present in the
maintained UI. Telemetry remains exploratory and is still excluded from fitting,
selection, sealed evaluation, promotion, release, Serving activation, and order
routing.

### AUD-ARCH-001 — RESOLVED

PR #79 removed import-time `setattr` replacement from simulation, telemetry,
Studio, and catalog package initializers. Maintained consumers now import explicit
facades/readers, and public compatibility aliases are ordinary declarations rather
than runtime mutation.

For example, `trade_rl.simulation.__init__` now explicitly binds
`MarketExecutor = StatefulCompatibilityMarketExecutor`; no package-import side
effect rewrites a class defined in another public module. Architecture contracts
forbid restoration of the prior mutation pattern.

The public compatibility surface remains intentional, but its ownership is now
locally discoverable and statically testable.

### AUD-SIM-001 — RESOLVED

PR #107 replaced the 614-line `execute_stateful_orders()` function with a 111-line
orchestration protocol and four invocation-local responsibilities:

- `StatefulExecutionRuntime` owns cloned books, immutable order-book replacement,
  ordered events/capacity evidence, accumulators, cancellation, and result payload;
- `StatefulBarLifecycle` owns splits, inactive assets, open revaluation, carry,
  mark-to-market, margin, and insolvency phases;
- `StatefulOrderTransitionProcessor` owns expiry, latency, admission, eligibility,
  projected reservation, and attempted-remainder expiry;
- `StatefulSymbolFillProcessor` owns conservative paths, triggers, shared symbol
  capacity, rounding, cost, fills, and fill evidence.

A pre-refactor mixed-order run fixed the complete final books, 13 ordered events,
three capacity records, interval metrics, counters, and per-symbol arrays behind
canonical digest
`3856e696c998e727c78690222d418e070c71eeb56f7f747f0932a17eb8ff2cc2`.
The decomposed path matches that payload and retains focused tests for no capacity,
partial-fill carry disabled, fail-closed starting equity, and runtime evidence
ownership.

The exact implementation head passed 1,223 tests with 83.56% total coverage and a
`67 / 72 = 93.06%` branch result for the four services. The configured group
threshold is 93.0%.

The execution coordinator is no longer the confirmed monolithic hotspot. Further
exchange realism remains outside the repository's OHLCV evidence boundary, but
that is a capability limitation rather than this architecture finding.

### AUD-CI-002 — RESOLVED

PR #88 replaced the backend-sensitive single float32 equality assertion with two
explicit contracts:

- strict float64 historical output, input-gradient, and parameter-gradient
  equivalence;
- bounded float32 output plus cosine-similarity and relative-L2 semantic
  invariants, including exact zero output/input gradient for a fully unavailable
  row.

PR #106 made those contracts permanent through an AST guard and a path-filtered,
read-only, exact-head workflow which repeats the focused tests ten times on both
Ubuntu and Windows. Its final targeted run passed 10/10 repetitions on both
platforms, and normal CI also passed.

The original numerical flake is therefore closed without weakening the semantic
policy-parity boundary.

### AUD-RL-001 — OPEN RISK, REDUCED

No behavior defect was reproduced in the original audit, so this remains a risk
rather than a confirmed bug.

PR #92 reduced the concentration by extracting typed decision planning, risk
projection, reward coordination, and information construction. Earlier runtime
work also separated episode sampling, stateful target execution, observation
assembly, and termination coordination. `step()` is now an orchestration facade
which delegates those responsibilities while retaining mutable Gymnasium state.

The remaining risk is construction and wiring density. `trade_rl/rl/environment.py`
is currently 1,503 source lines, and `ResidualMarketEnv.__init__` still owns a
large amount of provider validation, identity binding, builder construction, and
runtime wiring. A mechanical split is not justified without a behavior-preserving
seam.

Control:

- retain the existing environment runtime and step-service architecture tests;
- do not add new action, risk, reward, execution, or observation policy directly
  to the facade when a typed service owns that concern;
- extract constructor wiring only alongside a concrete feature or configuration
  change with characterization tests.

This risk does not block research use, but it remains the next architecture
maintenance watchpoint.

## 4. Current architecture judgment

The remediated core now has explicit maintained imports, bounded stateful execution
phases, typed environment step services, generation-bound and process-safe
telemetry, producer-issued environment episode identity, and a permanent
cross-platform numerical stability guard.

The original audit's P0 non-findings remain supported by the maintained regression
suite: no future-data observation path, accounting conservation break, sealed-test
feedback path, runtime private-key path, fail-open promotion path, or
pull-request-controlled privileged GPU execution was introduced by the
remediations.

The codebase is suitable for causal research workflows and eligible attested paper
Serving when its independent gates pass. It is not an exchange emulator. Queue
priority, hidden liquidity, auctions, adverse selection, event-level intrabar
ordering, and L2 depth are not proven by OHLCV simulation.

## 5. Remaining capability and operational gates

The following are not architecture-audit defects and remain explicitly open:

- authenticated exchange or broker account access;
- direct order submit, cancel, replace, and reconciliation;
- venue kill switches and operational alerting;
- production secret distribution;
- paper/live reconciliation evidence;
- required out-of-sample duration and statistical uplift;
- signed fresh confirmation and conservative complete execution evidence;
- demonstrated profitability.

Production status remains `NO-GO` until the maintained research, evidence,
operational, and authorization gates all pass independently.
