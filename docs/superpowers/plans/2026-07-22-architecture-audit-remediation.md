# Architecture Audit Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the confirmed execution, telemetry, dependency-boundary, environment-responsibility, canonical-JSON, and PostgreSQL-adapter findings recorded by the 2026-07-22 architecture audit without changing public training, Studio, or Serving contracts unnecessarily.

**Architecture:** Preserve existing import paths and public classes, but move new behavior into focused collaborators. Package initializers install explicit compatibility adapters only after the original modules are loaded, allowing direct submodule imports to receive the maintained implementation while avoiding large unsafe rewrites. All behavior changes are introduced test-first and remain fail closed.

**Tech Stack:** Python 3.12, NumPy, Gymnasium, Pydantic, pytest, Import Linter, PostgreSQL/psycopg, GitHub Actions.

## Global Constraints

- Production status remains `NO-GO`.
- No direct exchange routing or profitability claim is introduced.
- Existing `trade_rl.simulation.execution`, `trade_rl.telemetry.training`, and `trade_rl.studio.telemetry` import paths remain valid.
- Stateful execution remains deterministic and uses processing-bar capacity.
- Telemetry remains diagnostic-only and must not influence training, selection, serving approval, or execution.
- New lower-level contracts must remain standard-library only.

---

### Task 1: Unify compatibility and reward pre-roll execution

**Files:**
- Create: `trade_rl/simulation/target_execution.py`
- Create: `trade_rl/simulation/execution_adapter.py`
- Modify: `trade_rl/simulation/__init__.py`
- Test: `tests/simulation/test_stateful_execution_adapter.py`
- Test: `tests/rl/test_reward_preroll_stateful_parity.py`

**Interfaces:**
- Produces: `execute_target_statefully(executor, book, order_book, target, *, start_index, bars, target_identity, time_in_force=TimeInForce.GTC) -> StatefulExecutionResult`.
- Produces: maintained `MarketExecutor.execute_interval()` implemented over target reconciliation plus `execute_orders()`.
- Maintains compatibility order state only when the caller chains the exact returned `BookState` into the same executor; unrelated books reset the compatibility chain.

- [ ] Add failing tests proving `execute_interval()` uses processing-bar volume, produces the same result as `execute_orders()`, and carries partial residual orders across chained calls.
- [ ] Add a failing reward-pre-roll parity test with non-zero latency, constrained capacity, and partial-fill carry.
- [ ] Implement the shared target-execution collaborator.
- [ ] Implement the `MarketExecutor` compatibility subclass and install it through `trade_rl.simulation.__init__`.
- [ ] Run focused simulation and reward tests, then the full suite.

### Task 2: Enforce telemetry dependency placement

**Files:**
- Modify: `.importlinter`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `tests/test_architecture_contract.py`

**Interfaces:**
- Places `trade_rl.telemetry` below `trade_rl.artifacts` and above `trade_rl.domain`.
- Forbids telemetry from importing NumPy, Gymnasium, model frameworks, database adapters, or upper application layers.

- [ ] Add a failing architecture test requiring telemetry in the documented and enforced layer order.
- [ ] Add the layer and forbidden-import contract.
- [ ] Update the architecture document and verify Import Linter.

### Task 3: Make telemetry parsing and polling fail closed and incremental

**Files:**
- Create: `trade_rl/telemetry/indexed_training.py`
- Modify: `trade_rl/telemetry/__init__.py`
- Test: `tests/telemetry/test_training.py`

**Interfaces:**
- Produces strict `TrainingTelemetryRecord.from_json_dict()` boolean parsing.
- Produces a sparse sidecar index bound to device, inode, and indexed byte size.
- Preserves `read_training_telemetry(path, after_sequence, limit)` and `training_telemetry_status(path)` signatures.
- Rebuilds the index on replacement/truncation and scans only appended bytes when identity remains valid.

- [ ] Add failing tests for string, integer, null, and omitted boolean fields.
- [ ] Add a failing large-stream test proving a second poll starts near the appended byte range rather than byte zero.
- [ ] Add failing replacement/truncation tests.
- [ ] Implement strict record parsing, sparse checkpoints, atomic sidecar publication, incremental refresh, and seek-based reads.
- [ ] Run telemetry and integration tests.

### Task 4: Reject ambiguous Studio telemetry streams

**Files:**
- Create: `trade_rl/studio/strict_telemetry.py`
- Modify: `trade_rl/studio/__init__.py`
- Test: `tests/studio/test_telemetry_api.py`

**Interfaces:**
- Produces a `StudioTelemetryReader` subclass that rejects multiple distinct files for the same seed.
- Preserves existing response models and API schemas.

- [ ] Add failing tests for duplicate `.staging`/`runs`/`failed` streams and symlink variants.
- [ ] Implement deterministic discovery and `ArtifactInvalid` on ambiguity.
- [ ] Verify Studio API and frontend tests.

### Task 5: Unify canonical JSON primitives

**Files:**
- Create: `trade_rl/domain/canonical_json.py`
- Modify: `trade_rl/artifacts/codec.py`
- Modify: `trade_rl/catalog/__init__.py`
- Test: `tests/artifacts/test_canonical_json_shared.py`

**Interfaces:**
- Produces one standard-library-only `to_json_value()` and `canonical_json_bytes()` implementation.
- Artifact encoding and catalog cache-key digests use the same byte representation.

- [ ] Add failing cross-module vectors for nested mappings, tuple/list equivalence, Unicode, floats, and non-finite rejection.
- [ ] Implement the domain primitive and re-export it from artifacts.
- [ ] Install the shared encoder into catalog contracts and verify unchanged cache identities.

### Task 6: Separate PostgreSQL sealed-test reservation role

**Files:**
- Create: `trade_rl/catalog/postgres_sealed_test.py`
- Modify: `trade_rl/catalog/sealed_test.py`
- Test: `tests/catalog/test_postgres_integration.py`

**Interfaces:**
- Produces `PostgresSealedTestReservationStore` with `reserve_sealed_test_access(record) -> None`.
- Keeps `PostgresArtifactCatalog.reserve_sealed_test_access()` as a temporary compatibility delegate until workflow callers migrate.

- [ ] Add a failing integration test using the dedicated store across two independent instances.
- [ ] Implement the dedicated adapter over the shared database connection contract.
- [ ] Mark the generic catalog method as compatibility-only in documentation.

### Task 7: Verification and audit closure

**Files:**
- Modify: `docs/verification/2026-07-22-documentation-and-architecture-audit.md`
- Modify: `docs/RESEARCH_STATUS.md`

- [ ] Record exact remediated behavior and any deliberately retained compatibility path.
- [ ] Run Ruff, format, Mypy, Import Linter, full pytest/coverage, critical branch ratchets, Studio checks, Ubuntu/Windows compatibility, PostgreSQL integration, and training-image probe.
- [ ] Open a stacked Draft PR against `agent/docs-architecture-audit-20260722` and preserve `NO-GO`.