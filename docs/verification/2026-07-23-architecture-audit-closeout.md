# Architecture Audit Closeout — 2026-07-23

## 1. Purpose and source boundary

This document closes the findings recorded in
`docs/verification/2026-07-22-post-merge-architecture-audit.md` after their
independent remediations were integrated and verified.

The current integrated source baseline is `main` merge commit
`9d7790f8d570d94220b40c903a32e43ae5fad5a7`, which includes the canonical
remediations through PR #152. PR #165 is the executable documentation-contract
closeout for the original deferred `AUD-RL-001` risk; it changes no Python
production behavior or public API.

The 2026-07-22 audit remains a historical record of what was observed at its
audited commit. It must not be read as the current defect list. Current status is
established by this closeout and the finding-specific verification documents.

Production remains `NO-GO`. Closing software-architecture findings does not
establish profitability, exchange-equivalent fills, operational authorization,
paper/live evidence, or direct exchange capability.

## 2. Closeout summary

| Finding | Original priority | Current status | Remediation |
| --- | --- | --- | --- |
| `AUD-DOC-001` | P3 | RESOLVED | PR #82 and executable documentation contracts |
| `AUD-STUDIO-001` | P1 | RESOLVED | PR #85 environment/current-episode isolation; PR #103 producer episode identity |
| `AUD-ARCH-001` | P2 | RESOLVED | PR #79 removed initializer mutation; PR #111 made public modules canonical maintained owners |
| `AUD-SIM-001` | P2 | RESOLVED | PR #107 stateful execution phase decomposition |
| `AUD-CI-002` | P2 | RESOLVED | PR #88 stable numerical contract; PR #106 permanent cross-platform guard |
| `AUD-RL-001` | P2 risk | RESOLVED | PR #92 runtime/step services; PR #114 observation contracts; PR #120 provider contracts; PR #122 runtime-service wiring; PR #125 portfolio-risk inputs; PR #126 policy/schedule contracts; PR #140 initial-state factory; PR #152 reward/execution resources; PR #165 executable resolution contract |

No confirmed P0 or P1 product defect remains from the 2026-07-22 finding set.
No architecture-audit finding remains open. Capability, evidence, operational,
authorization, and profitability gates remain separate from this closeout.

## 3. Finding-by-finding disposition

### AUD-DOC-001 — RESOLVED

PR #82 synchronized the maintained English and Japanese entry points, START guide,
architecture, research status, Binance, GPU-operation, and Studio documentation.
It added executable documentation contracts derived from source constants and
Import Linter boundaries. Those controls prevent the reproduced schema,
Serving-bundle, stale-layer, and closed-finding drift from silently returning.

### AUD-STUDIO-001 — RESOLVED

PR #85 made one selected vector environment and only its current episode drive the
Live Training chart, cursor, price, PnL, baseline, drawdown, events, playback, and
jump controls. PR #103 added producer-issued nullable `episode_id` values while
preserving historical telemetry compatibility.

PR #95 serializes cooperating process append/read/index transactions and fails
closed on incomplete tails or replaced writer inodes. PR #99 binds cursors to an
opaque stream generation and rejects mixed-generation responses. Telemetry remains
exploratory and is excluded from fitting, selection, sealed evaluation, promotion,
release, Serving activation, and order routing.

### AUD-ARCH-001 — RESOLVED

PR #79 removed import-time class mutation from package initializers. PR #111 made
`MarketExecutor`, `trade_rl.telemetry.training`, and `StudioTelemetryReader` the
direct maintained behavior owners; former compatibility modules are identity
aliases rather than alternate implementations. Architecture contracts prohibit
restoration of runtime mutation or split behavior ownership.

### AUD-SIM-001 — RESOLVED

PR #107 replaced the 614-line stateful execution hotspot with a bounded
orchestration protocol and independently owned runtime, bar-lifecycle,
order-transition, and symbol-fill phases. A fixed pre-refactor characterization
digest preserves final books, ordered events, capacity records, metrics, counters,
and per-symbol arrays.

Exchange realism beyond the OHLCV evidence boundary remains a capability
limitation, not this architecture finding.

### AUD-CI-002 — RESOLVED

PR #88 replaced backend-sensitive float32 equality with strict float64 historical
equivalence and bounded float32 semantic invariants for outputs and gradients. PR
#106 made those contracts permanent through an AST guard and repeated exact-head
Ubuntu and Windows checks.

### AUD-RL-001 — RESOLVED

The original audit recorded a P2 maintainability risk rather than a reproduced
behavior defect. It observed a 1,620-line `environment.py`, a 479-line
`ResidualMarketEnv.__init__()`, and a 223-line `step()`. It protected the stable
Gymnasium API and mutable episode-state ownership, and deferred remediation until
concrete feature seams could be characterized instead of splitting mechanically.

The deferred condition is now satisfied. PR #92 extracted typed decision planning,
risk projection, reward coordination, information assembly, episode sampling,
execution coordination, observation assembly, and termination. `step()` remains an orchestration facade which applies those service results without re-owning their
policy.

PR #114 extracted observation layout, schemas, digests, normalizer identity,
sequence windows, policy planes, spaces, and the observation-derived causal
minimum. PR #120 extracted provider reconciliation, causal compatibility, artifact
identities, factor validation, and provider-derived minima. PR #122 extracted the
eight-service runtime construction graph.

PR #125 extracted portfolio-risk model/provider selection, digest validation, and
causal-minimum aggregation. PR #126 extracted configuration, emergency-monitor,
action-spec, schedule, reward-config, decision-hours, and episode-choice
validation. PR #140 extracted invocation-local initial books, indices, episode
defaults, arrays, order books, execution state, diagnostics, and reset state. PR
#152 extracted reward-tracker construction, optional complete reward pre-roll,
independent hybrid and shadow executors, the compatibility alias, and the
reward-history cache.

`reset()` retains mutable Gymnasium state and installs fresh invocation-local values
under the maintained attributes. That is an intentional framework-facing facade
responsibility, not unowned construction policy.

The constructor fell from the audited 479 lines to a maintained 150-line architecture limit. The remaining constructor statements are validated typed
contract assignment and orchestration. Architecture tests prohibit extracted
action, risk, reward, execution, observation, provider, portfolio-risk,
policy/schedule, service-wiring, initial-state, and resource responsibilities from
returning inline.

The extracted contract and wiring modules retain permanent 100.0% critical coverage ratchets. PR #152 final exact-head verification passed `1343` tests with
`84.25%` total coverage and `71.24%` total branch coverage. CI run `30016488539`
and PostgreSQL Catalog run `30016490046` passed Studio, workflow security, Ruff,
formatting, Mypy, Import Linter, dead-code reporting, complete pytest and coverage,
critical coverage, CLI, Ubuntu, Windows, the complete training image and non-root
probe, Compose validation, readiness, migrations, PostgreSQL unit/integration
tests, and cleanup.

The original deferred risk is therefore closed. `step()` remains an orchestration facade, `reset()` retains mutable Gymnasium state, and no further mechanical split
is required without a newly reproduced responsibility or behavioral defect.
Production remains `NO-GO`.

## 4. Current architecture judgment

The remediated core now has canonical public-contract ownership, bounded stateful
execution phases, typed environment runtime and step services, typed observation,
provider, portfolio-risk, policy/schedule, reward/execution, runtime-service, and
initial-state boundaries, generation-bound process-safe telemetry, producer-issued
environment episode identity, and a permanent cross-platform numerical stability
guard.

The original audit's P0 non-findings remain supported by maintained regression
contracts: no future-data observation path, accounting-conservation break,
sealed-test feedback path, runtime private-key path, fail-open promotion path, or
pull-request-controlled privileged GPU execution was introduced by the
remediations.

The codebase is suitable for causal research workflows and eligible for attested
paper Serving when its independent gates pass. It is not an exchange emulator.
Queue priority, hidden liquidity, auctions, adverse selection, event-level intrabar
ordering, and L2 depth are not proven by OHLCV simulation.

## 5. Remaining capability and operational gates

The following are not architecture-audit defects and remain explicitly open:

- authenticated exchange or broker account access;
- direct order submit, cancel, replace, and reconciliation;
- venue kill switches and operational alerting;
- production secret distribution;
- maintained paper/live reconciliation evidence;
- required out-of-sample duration and statistical uplift;
- signed fresh confirmation and conservative complete execution evidence;
- demonstrated profitability.

Production status remains `NO-GO` until the maintained research, evidence,
operational, and authorization gates all pass independently.
