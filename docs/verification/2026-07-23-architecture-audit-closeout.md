# Architecture Audit Closeout — 2026-07-23

## 1. Purpose and source boundary

This document closes or reclassifies the findings recorded in
`docs/verification/2026-07-22-post-merge-architecture-audit.md` after the
independent remediation pull requests were integrated.

The current integrated source baseline is `main` merge commit
`9d7790f8d570d94220b40c903a32e43ae5fad5a7`, which includes the canonical
observation, provider, runtime-service, portfolio-risk, policy/schedule,
initial-state, reward/execution-resource, and corrected verification remediations
through PR #152. PR #165 is the documentation-contract closeout for the original
deferred `AUD-RL-001` risk; it does not change Python production behavior.

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
| `AUD-ARCH-001` | P2 | RESOLVED | PR #79 removed initializer mutation; PR #111 made public modules canonical maintained owners |
| `AUD-SIM-001` | P2 | RESOLVED | PR #107 stateful execution phase decomposition |
| `AUD-CI-002` | P2 | RESOLVED | PR #88 stable numerical contract; PR #106 permanent cross-platform guard |
| `AUD-RL-001` | P2 risk | RESOLVED | PR #92 typed step services; PR #114 observation contracts; PR #120 provider contracts; PR #122 runtime-service wiring; PR #125 portfolio-risk contracts; PR #126 policy/schedule contracts; PR #140 initial-state factory; PR #152 reward/execution resources; PR #165 executable resolution contract |

No confirmed P0 or P1 product defect remains from the 2026-07-22 finding set.
All recorded architecture findings are now resolved. Capability, operational,
authorization, evidence-duration, and profitability gates remain independent and do
not become resolved by this architecture closeout.

## 3. Finding-by-finding disposition

### AUD-DOC-001 — RESOLVED

PR #82 synchronized the maintained English and Japanese entry points, START guide,
architecture, research status, Binance, GPU-operation, and Studio documentation
with the post-stateful implementation. It also added executable documentation
contracts derived from source constants and Import Linter boundaries.

The remediation prevents the specific observation-v3, bundle-v4, stale-layer, and
closed-finding drift reproduced by the audit. This closeout additionally updates
the architecture description for Live Training episode isolation, the
stateful-execution phase services, and canonical public-contract ownership.

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
Studio, and catalog package initializers. Maintained consumers moved to explicit
facades/readers, and architecture contracts prohibited restoration of runtime
class mutation.

PR #111 completed the ownership boundary rather than stopping at explicit aliases:

- `trade_rl.simulation.execution.MarketExecutor` now directly owns the maintained
  stateful compatibility chain, target identity, persistent order-book state,
  reset behavior, stateful target execution, and `ExecutionResult` projection;
- `trade_rl.telemetry.training` now directly owns strict record parsing,
  process-safe indexed writing, indexed reads, status, and stream-generation
  contracts, while standard-library-only `_indexed_storage.py` owns private sparse
  index and OS-lock mechanics;
- `trade_rl.studio.telemetry.StudioTelemetryReader` now directly owns duplicate
  seed-stream and candidate-symlink rejection.

The former compatibility modules are behavior-free identity aliases:

- `StatefulCompatibilityMarketExecutor is MarketExecutor`;
- `StrictTrainingTelemetryRecord is TrainingTelemetryRecord`;
- `IndexedTrainingTelemetryWriter is TrainingTelemetryWriter`;
- `StrictStudioTelemetryReader is StudioTelemetryReader`.

Package imports and direct-module imports therefore resolve to the same canonical
objects and canonical `__module__` identities. Static review of the declared public
module can no longer see a superseded implementation while package import selects
a different one.

The implementation exact head passed `1,236` tests with `83.78%` total coverage
and `70.74%` total branch coverage. The moved indexed-storage implementation
retained the existing `69.0%` ratchet and measured `75 / 104 = 72.12%` branch
coverage. Ubuntu, Windows-native telemetry locking, training-image, Import Linter,
CLI, and PostgreSQL verification all passed.

The public compatibility surface remains intentional, but behavior ownership is
now local, direct, identity-tested, and statically discoverable.

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

### AUD-RL-001 — RESOLVED

The original audit classified this item as a P2 maintainability risk, not a
reproduced behavior defect. It explicitly prohibited a mechanical split and deferred
remediation until concrete action, risk, or reward work supplied testable seams. That
deferred condition is now satisfied.

PR #92 extracted typed decision planning, risk projection, reward coordination, and
information construction. Episode sampling, stateful target execution, observation
assembly, and termination coordination are also dedicated services. `step()` remains
an orchestration facade: it calls the typed decision, risk, execution, termination,
reward, observation, and information owners, then applies their returned mutable state
through the stable Gymnasium API.

PR #114 extracted deterministic observation construction, schemas, digests,
normalizer identity, sequence windows, policy planes, and Gymnasium spaces. PR #120
extracted provider reconciliation, alpha and factor identities, static factor-basis
validation, and provider-derived causal minimum indices. PR #122 extracted the typed
eight-service construction graph.

PR #125 extracted portfolio-risk model/provider selection, SHA-256 identity and
minimum-index validation. PR #126 extracted configuration, emergency-monitor,
action-layout, reward-config, and episode/decision schedule policy. PR #140 extracted
fresh invocation-local initial state. PR #152 extracted reward-tracker, full reward
pre-roll, independent hybrid/shadow executor, maintained executor-alias, and reward
cache construction.

`reset()` retains mutable Gymnasium state, episode-state validation, stable reset
semantics, and application of fresh books, orders, arrays, diagnostics, and reward
history. This is the protected invariant from the original audit, not unremediated
construction debt. No further mechanical split is justified: future extraction now
requires a concrete behavior-preserving seam and characterization evidence.

The audited constructor spanned 479 lines. The maintained constructor now has a
150-line architecture limit and contains typed contract construction, assignment, and
orchestration rather than inline action, risk, reward, execution, observation, or
information policy.

Architecture tests enforce:

- one typed owner for each extracted constructor and runtime responsibility;
- delegation of decision, risk, execution, termination, reward, observation, and info
  from the stable facade;
- absence of extracted policy and direct construction from the constructor;
- preservation of construction order, identities, public behavior, and the 150-line
  architecture limit;
- continued local ownership of mutable reset/step state application.

The provider, runtime-service, portfolio-risk, policy/schedule, initial-state, and
reward/execution-resource modules all have permanent 100.0% critical coverage
ratchets. PR #152 exact-head verification passed 1,343 tests with 84.25% total
coverage and 71.24% total branch coverage, plus Ruff, formatting, Mypy, Import
Linter, Serving smoke, CLI, Ubuntu, Windows, complete training-image/non-root, and
PostgreSQL Catalog checks.

The executable documentation contract in
`tests/test_current_documentation_contract.py` prevents this resolved finding from
being presented as open again and requires the protected reset/step boundary,
architecture controls, and production `NO-GO` statement to remain explicit.

`AUD-RL-001` is therefore resolved. Future facade regression is governed by maintained
architecture tests and coverage ratchets rather than retained as an open finding.

## 4. Current architecture judgment

The remediated core now has direct canonical public-contract ownership, bounded
stateful execution phases, typed environment step services, typed static
observation, provider, portfolio-risk input, policy/schedule, reward/execution
resources, runtime-service wiring, and invocation-local initial-state boundaries,
generation-bound and process-safe telemetry, producer-issued environment episode
identity, and a permanent cross-platform numerical stability guard.

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
