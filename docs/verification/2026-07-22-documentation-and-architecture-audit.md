# Documentation and Architecture Audit — 2026-07-22

## Scope

This audit was performed after reconciling the maintained explanatory Markdown with repository `main` at:

```text
audited main head: 5c029f0b57c0a628990f223cf8265d86fdc602f3
latest product change: feat: add conservative stateful order simulation (#75)
```

The inspection covered:

- `README.md`, `README.ja.md`, `START.md`, `studio/README.md`;
- `docs/ARCHITECTURE.md`, `docs/RESEARCH_STATUS.md`;
- `.importlinter`, packaging, optional dependency boundaries, and architecture tests;
- PostgreSQL catalog and persistent sealed-test ledger;
- training telemetry writer, reader, Studio discovery, and API boundary;
- target reconciliation, persistent orders, OHLC path selection, liquidity allocation, stateful execution, and promotion evidence;
- reward pre-roll, normal environment transitions, terminal liquidation, Training–Serving parity, and serving bundle v5.

The audit is source-based. The prior exact-head product verification for PR #75 is preserved in `docs/verification/2026-07-21-conservative-order-simulator.md`; this documentation branch requires its own CI result before merge.

## Documentation corrections completed

The maintained documentation now:

- uses Serving bundle **v5** instead of stale v4 references;
- installs `train-sb3` in the Quickstart before invoking actual PPO training;
- documents the optional PostgreSQL catalog without implying that models or arrays are stored as mutable database BLOBs;
- distinguishes exploratory telemetry, deterministic checkpoint evidence, sealed evaluation, paper serving, and direct exchange execution;
- documents persistent market/limit/stop orders, current-bar capacity, partial-fill carry, OHLC path assumptions, and execution-promotion evidence;
- makes the documented Import Linter order match `.importlinter` exactly;
- records that `trade_rl.telemetry` is currently outside the enforced layer stack;
- preserves `NO-GO` and makes no profitability or live-routing claim.

## Architecture strengths

### A1 — Enforced responsibility layers

`.importlinter` defines an explicit top-down dependency order from CLI and Studio through workflows, integrations, serving, learning, RL, risk, simulation, strategies, data, catalog, evaluation, release, artifacts, and domain. Additional forbidden contracts keep domain standard-library only, prevent serving from importing training/workflows, keep learning and core training framework-independent, isolate offline signers, and keep catalog contracts independent of psycopg, NumPy, model frameworks, and upper application layers.

Assessment: **strong and testable**.

### A2 — Immutable artifact and release identities

Dataset and run artifacts use deterministic serialization, file closure, hashes, staging, and atomic publication. Serving bundle v5 separates candidate identity from detached approval, and runtime paths accept public verification material rather than offline signing secrets.

Assessment: **strong trust boundary**.

### A3 — Persistent sealed-test access

The PostgreSQL-backed ledger reserves a unique experiment-plan, dataset, and fold tuple atomically across processes. Duplicate access fails rather than silently generating a second outer-test result.

Assessment: **strong P0 research boundary**.

### A4 — Training–Serving parity

The environment exports the actual current observation state, including persistent-order coordinates, and serving rebuilds and verifies the same identity-bound contract. The existing P0 verification uses non-zero environment trajectories rather than synthetic zero arrays.

Assessment: **strong anti-drift boundary**.

### A5 — Stateful execution decomposition

Order domain types, target reconciliation, admission, bar-path interpretation, liquidity allocation, accounting, promotion evidence, and deterministic replay are separated into focused modules. `BookState` remains the accounting authority, avoiding a second independent cash/PnL implementation inside the order simulator.

Assessment: **strong direction with one important incomplete migration**.

## Prioritized findings

## P1 — Reward pre-roll still uses the compatibility execution engine

**Evidence**

- Normal `ResidualMarketEnv.step()` calls `_execute_stateful_target()`, reconciles target exposure against active residual orders, and carries `OrderBookState` between decisions.
- `_baseline_reward_history()` constructs a separate `MarketExecutor` and calls `execute_interval()` directly.
- `execute_interval()` still contains its own target-to-quantity, latency, previous-bar-capacity, touch, fill, and carry loop. It is not a thin adapter over `execute_orders()`.
- The approved stateful-execution design says the compatibility API should convert targets to intents and execute through the new engine.

**Impact**

Reward history at reset can follow different economics from the episode that consumes it. The divergence is material when latency is non-zero, partial fills persist across decisions, order type is limit/stop, cancel-and-replace matters, or processing-bar capacity differs from the compatibility path's preceding-bar capacity. Because the baseline-underperformance hinge is initialized from this history, the mismatch can change reward state and therefore the MDP.

**Required remediation**

1. Make `execute_interval()` a real adapter over target reconciliation plus stateful execution, or remove maintained callers.
2. Migrate `_baseline_reward_history()` to the same persistent-order path as normal baseline transitions.
3. Add a parity test comparing pre-roll and ordinary baseline execution under non-zero latency, constrained capacity, and partial-fill carry.
4. Bind the resulting compatibility/migration mode into evidence until the old implementation is removed.

## P1 — `trade_rl.telemetry` is outside the enforced dependency layer stack

**Evidence**

`trade_rl.telemetry` is a first-party runtime package used by training integrations and Studio, but it is absent from `.importlinter`'s layer list and has no separate forbidden contract.

**Impact**

The package is currently standard-library only, but that property is not enforced. A future import from workflows, integrations, Studio, model frameworks, or serving could create an unnoticed cycle or invert a trust boundary while Import Linter still passes.

**Required remediation**

Choose and enforce one of these designs:

- place framework-independent telemetry contracts below integrations and Studio in the layer stack, preferably near artifacts/domain-facing diagnostic contracts; or
- split `telemetry/contracts.py` from adapter/writer code and give each part an explicit forbidden-import contract.

Add an architecture test that proves the chosen placement remains enforced.

## P2 — Live telemetry polling performs whole-file rescans

**Evidence**

`read_training_telemetry()` opens the JSONL file and iterates from the beginning even when `after_sequence` is large. `training_telemetry_status()` also scans every line. Studio calls these functions repeatedly for status and event polling.

**Impact**

For long multi-seed runs, cumulative polling cost grows with total file size. Repeated polling can approach quadratic total bytes read over the lifetime of a run, increase API latency, and compete with training I/O. The browser's 2,048-record cap does not limit backend scan cost.

**Required remediation**

- maintain a sidecar index with sequence-to-byte offsets, or persist byte cursors in the job state;
- use a bounded tail scan for status;
- return file generation/inode/size with the cursor so truncation or replacement fails closed;
- add a large synthetic telemetry benchmark and assert that incremental reads scale with appended bytes.

## P2 — Telemetry booleans are not parsed fail closed

**Evidence**

`TrainingTelemetryRecord.from_json_dict()` uses `bool(raw.get(...))` for `emergency_deleverage`, `terminated`, and `truncated`.

**Impact**

Strings and integers are silently coerced. For example, the JSON string `"false"` becomes `True`. This contradicts the otherwise strict typed artifact boundary and can display incorrect risk or terminal state.

**Required remediation**

Add a strict `_required_bool()` parser and reject missing or non-boolean values. Add malformed-record tests for strings, integers, null, and omitted fields.

## P2 — Duplicate seed telemetry streams are resolved by discovery order

**Evidence**

Studio recursively discovers `training-telemetry.jsonl` files and stores them with `streams.setdefault(seed, resolved)`. A second path for the same seed is ignored.

**Impact**

If staging, published, failed, or nested directories contain ambiguous copies, the UI may show an arbitrary stream while appearing authoritative. This is especially risky around interrupted publication or manual artifact copying.

**Required remediation**

- reject multiple distinct files for one seed as `artifact_invalid` unless an explicit namespace precedence and identical content digest are verified;
- bind stream discovery to the job's declared current run namespace/state rather than recursive first-match behavior;
- add tests for duplicate staging/runs/failed copies and symlink variants.

## P2 — `ResidualMarketEnv` remains a large orchestration hotspot

**Evidence**

The environment owns provider resolution, action migration, sequence setup, episode sampling, initial books, reward pre-roll, stateful reconciliation, risk, execution, liquidation, observation snapshots, diagnostics, and reward transitions in one large module. Configuration and economic transition helpers have been extracted, but the main class remains highly coupled.

**Impact**

Changes to execution, reward, serving parity, and episode reset can interact through shared mutable state. The P1 pre-roll divergence is an example of a cross-cutting concern that is hard to see inside the facade.

**Required remediation**

Extract focused collaborators with explicit state contracts:

- episode initialization and causal pre-roll;
- portfolio/order transition engine;
- terminal liquidation coordinator;
- observation snapshot/export adapter.

Keep Gymnasium adaptation in the facade and test collaborators independently.

## P3 — Canonical JSON logic is duplicated

**Evidence**

`trade_rl.artifacts.codec` and `trade_rl.catalog.contracts` each define canonical JSON conversion/encoding and JSON value types.

**Impact**

The implementations currently agree for ordinary catalog payloads, but they support different object types and container freezing behavior. Future changes can make cache-key digests diverge from artifact digests or create two definitions of "canonical".

**Required remediation**

Move the smallest framework-independent canonical JSON primitive to one lower-level module and make both artifact and catalog code depend on it. Add cross-module digest vectors for nested mappings, tuples/lists, Unicode, floats, and invalid non-finite values.

## P3 — PostgreSQL adapter combines generic catalog and evaluation-specific ledger operations

**Evidence**

`PostgresArtifactCatalog` implements generic artifact registration and directly accepts `SealedTestAccessRecord` from the evaluation package.

**Impact**

This is allowed by the current layer order, but it couples a reusable catalog adapter to one evaluation workflow and makes migration/testing responsibilities broader than the `ArtifactCatalog` protocol suggests.

**Required remediation**

Prefer a shared PostgreSQL connection/migration service with separate `PostgresArtifactCatalog` and `PostgresSealedTestReservationStore` adapters. Keep the generic protocol unaware of evaluation records.

## Documentation drift prevention

The documentation branch adds tests that should fail when:

- maintained documents regress to Serving bundle v4;
- the Quickstart invokes training without installing the `train-sb3` extra;
- the architecture document omits a currently enforced Import Linter layer.

These tests do not replace code review. They cover high-value claims that previously drifted.

## Recommended remediation order

1. **Unify reward pre-roll and `execute_interval` with stateful execution.** This is the only finding that can change training reward state and evaluation economics.
2. **Put telemetry inside an enforced architecture boundary.** Do this before more telemetry dependencies are added.
3. **Replace full-file telemetry polling and reject ambiguous streams.** This protects long-running 12M-step use cases.
4. **Make telemetry parsing strict.** Small change, high confidence.
5. **Decompose the environment around episode/pre-roll and transition collaborators.** Perform after semantic parity tests exist.
6. **Unify canonical JSON and split PostgreSQL adapter roles.** Maintenance hardening after the behavioral P1/P2 items.

## Verification interpretation

A successful CI result for this branch will establish that the documentation and contract tests integrate with the repository at the branch head. It will not close the findings above unless code changes explicitly remediate them. It also will not establish profitable trading or authorize direct exchange execution.
