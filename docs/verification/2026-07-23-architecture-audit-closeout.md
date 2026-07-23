# Architecture Audit Closeout — 2026-07-23

## 1. Purpose and source boundary

This document closes or reclassifies the findings recorded in
`docs/verification/2026-07-22-post-merge-architecture-audit.md` after the
independent remediation pull requests were integrated.

The current integrated source baseline is `main` merge commit
`7d5be020940ab7349d53a419c370a720448006bf`, which merged PR #125 after the
stateful-execution decomposition, sequence-projection guard, telemetry identity,
stream-generation, process-lock, Live Training, environment-runtime, canonical
public-contract, observation-contract, provider-contract, runtime-service wiring,
and portfolio-risk input remediations. The latest verified `AUD-RL-001`
remediation candidate is PR #126 exact head
`08d8bdf6b39f00adaaca6d3f65e3183404083447`.

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
| `AUD-RL-001` | P2 risk | OPEN RISK, FURTHER REDUCED | PR #92 extracted runtime/step services; PR #114 observation contracts; PR #120 provider contracts; PR #122 typed runtime-service wiring; PR #125 portfolio-risk input contracts; PR #126 policy/schedule contracts |

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

### AUD-RL-001 — OPEN RISK, FURTHER REDUCED

No behavior defect was reproduced in the original audit, so this remains a risk
rather than a confirmed bug.

PR #92 reduced the concentration by extracting typed decision planning, risk
projection, reward coordination, and information construction. Earlier runtime
work also separated episode sampling, stateful target execution, observation
assembly, and termination coordination. `step()` is now an orchestration facade
which delegates those responsibilities while retaining mutable Gymnasium state.

PR #114 extracted the deterministic observation seam. The typed
`EnvironmentObservationContractBuilder` owns builder/layout construction, flat and
structured schemas and digests, normalizer identity validation, sequence windows
and policy plane, Gymnasium spaces, and the sequence-derived minimum index.
Architecture tests prohibit those low-level operations from returning to
`ResidualMarketEnv.__init__()`.

PR #120 extracted the provider seam. The typed
`EnvironmentProviderContractBuilder` owns trend and `MarketInputResolver`
reconciliation, causal alpha compatibility wrapping, alpha and factor artifact
identities, static factor-basis validation and copying, factor-count inference, and
provider-derived minimum indices. The legacy provider protocols are imported by the
environment facade from the maintained owner rather than redefined locally.

PR #122 extracted the runtime-service wiring seam. The frozen
`EnvironmentRuntimeServices` contract and `EnvironmentRuntimeServicesBuilder` own
the existing eight-service construction graph and preserve the original episode,
execution, observation, decision, risk, reward, information, and termination order.
The builder receives validated collaborators but no mutable books, order books,
indices, pending targets, episode seeds, diagnostics, or reset state.

PR #125 extracted the portfolio-risk input seam. The typed
`EnvironmentPortfolioRiskContractBuilder` owns default model construction,
advanced-input provider selection, provider SHA-256 identity validation, provider
minimum-index validation, and aggregation with the existing causal minimum. Digest
validation remains ordered before minimum-index access, and supplied model/provider
identities are preserved.

PR #126 extracts the deterministic policy and schedule seam. The frozen
`EnvironmentPolicyScheduleContract` and
`EnvironmentPolicyScheduleContractBuilder` own config resolution, emergency-monitor
construction, leverage and random-gross validation, action-spec resolution and
validation, action names, episode/decision bars, reward-config resolution, resolved
decision hours, and episode-hour-choice validation. Existing identities, exception
text, and validation order are preserved. Reward-tracker and executor construction
remain outside this boundary.

The provider extraction reduced the constructor from 321 to 262 source lines. The
runtime-services extraction further reduced it to 232 lines. The portfolio-risk
contract reduced it to 218 lines. The policy/schedule contract reduces it to 186
lines and enforces a 190-line architecture limit. The facade retains existing
attributes while direct construction and validation policy are prohibited from
returning inline.

The provider contract measured 109 / 109 statements and 44 / 44 branches covered.
The runtime-service module measured 61 / 61 statements with no executable branch
points. The portfolio-risk contract measured 29 / 29 statements and 6 / 6 branches.
The policy/schedule contract measured 53 / 53 statements and 16 / 16 branches. All
four have permanent 100.0% critical coverage ratchets. PR #126 exact-head
verification passed 1,315 tests, 84.18% total coverage, 71.28% total branch
coverage, Ubuntu, Windows, training-image, Import Linter, CLI, and PostgreSQL
Catalog checks.

The remaining construction density is reward-tracker and reward-preroll
construction, hybrid and shadow market-executor construction, observation/runtime
contract assignment, and mutable Gymnasium-state initialization. A mechanical split
is still not justified without another concrete behavior-preserving seam and
characterization evidence.

Control:

- retain the environment runtime, step-service, observation-contract,
  provider-contract, runtime-service, portfolio-risk-contract, and
  policy/schedule-contract architecture tests;
- retain the 100.0% critical coverage ratchets for the extracted contract and
  wiring modules;
- do not add new action, risk, reward, execution, observation, provider,
  portfolio-risk input, policy/schedule, or service-wiring policy directly to the
  facade when a typed owner exists;
- extract further constructor responsibilities only with characterization tests
  that preserve validation order, identities, and public behavior.

This risk does not block research use, but it remains the next architecture
maintenance watchpoint.

## 4. Current architecture judgment

The remediated core now has direct canonical public-contract ownership, bounded
stateful execution phases, typed environment step services, typed static
observation, provider, portfolio-risk input, policy/schedule, and runtime-service
wiring boundaries, generation-bound and process-safe telemetry, producer-issued
environment episode identity, and a permanent cross-platform numerical stability
guard.

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
