# Architecture Follow-up Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove runtime monkey patching, unify stateful target execution, fail closed on unavailable regime data, remove duplicated catalog implementations, and strengthen PostgreSQL exact-head evidence.

**Architecture:** Public package facades re-export maintained implementations without mutating modules. Environment target execution delegates to the shared reconciliation helper. Catalog compatibility is preserved through explicit methods instead of import-time replacement, and CI records the exact tested head.

**Tech Stack:** Python 3.12, NumPy, Gymnasium, pytest, Import Linter, GitHub Actions, PostgreSQL/psycopg.

## Global Constraints

- Preserve public user-facing APIs where practical.
- Do not change reward, action, observation, or execution evidence schemas.
- Do not introduce direct exchange routing.
- Production remains `NO-GO`.
- Follow RED-GREEN-REFACTOR for each behavioral change.
- Final verification must run full CI and PostgreSQL integration on the exact final head.

---

### Task 1: Add failing architecture contracts

**Files:**
- Create: `tests/architecture/test_architecture_followup.py`

**Interfaces:**
- Consumes: package initializer source, `EpisodeContractSampler`, PostgreSQL workflow text.
- Produces: regression contracts for explicit facades, shared execution delegation, fail-closed sampling, catalog single ownership, and CI exact-head configuration.

- [ ] **Step 1: Write tests that assert package initializers do not contain runtime `setattr` replacement.**
- [ ] **Step 2: Assert `EnvironmentExecutionCoordinator.execute_target` calls `execute_target_statefully`.**
- [ ] **Step 3: Build a minimal dataset whose selected global feature is unavailable for every candidate and assert regime/stress sampling raises `ValueError`.**
- [ ] **Step 4: Assert catalog canonical JSON is imported from `trade_rl.domain.canonical_json`, sealed-test SQL exists only in `postgres_sealed_test.py`, and the PostgreSQL workflow includes main push plus exact-head checkout.**
- [ ] **Step 5: Run `pytest -q tests/architecture/test_architecture_followup.py` and record the intended failures.**
- [ ] **Step 6: Commit the RED tests.**

### Task 2: Remove package-initializer runtime replacement

**Files:**
- Modify: `trade_rl/simulation/__init__.py`
- Modify: `trade_rl/telemetry/__init__.py`
- Modify: `trade_rl/studio/__init__.py`
- Modify: `trade_rl/catalog/__init__.py`
- Modify direct consumers that require maintained facades.

**Interfaces:**
- Consumes: `StatefulCompatibilityMarketExecutor`, indexed telemetry implementations, strict Studio reader, dedicated catalog adapter.
- Produces: explicit imports with no runtime mutation.

- [ ] **Step 1: Replace `setattr` calls with ordinary aliases and exports.**
- [ ] **Step 2: Update maintained consumers to import from the explicit package facade or strict implementation module.**
- [ ] **Step 3: Run focused simulation, telemetry, Studio, and catalog tests.**
- [ ] **Step 4: Commit the explicit facade migration.**

### Task 3: Unify environment and compatibility target execution

**Files:**
- Modify: `trade_rl/rl/environment_execution.py`
- Modify: `trade_rl/simulation/target_execution.py` only if an explicit parameter is required.
- Test: `tests/architecture/test_architecture_followup.py`
- Test: `tests/simulation/test_stateful_execution_adapter.py`

**Interfaces:**
- Consumes: `execute_target_statefully(executor, book, order_book, target, *, start_index, bars, target_identity, time_in_force=..., expiry_index=None)`.
- Produces: `EnvironmentExecutionCoordinator.execute_target(...) -> StatefulExecutionResult` through the same helper.

- [ ] **Step 1: Delegate coordinator execution to `execute_target_statefully`.**
- [ ] **Step 2: Remove duplicated `reconcile_target` and `execute_orders` logic from the coordinator.**
- [ ] **Step 3: Run focused environment and simulation parity tests.**
- [ ] **Step 4: Commit the shared execution path.**

### Task 4: Fail closed on unavailable regime data

**Files:**
- Modify: `trade_rl/rl/environment_episode.py`
- Test: `tests/architecture/test_architecture_followup.py`

**Interfaces:**
- Consumes: `global_feature_available` and configured episode sampling mode.
- Produces: explanatory `ValueError` when no candidate has an available regime feature.

- [ ] **Step 1: Remove the fallback from an empty available candidate set to all valid starts.**
- [ ] **Step 2: Raise `ValueError("episode sampling feature is unavailable for every valid start")`.**
- [ ] **Step 3: Run episode sampling tests and confirm uniform sampling is unchanged.**
- [ ] **Step 4: Commit the fail-closed sampling behavior.**

### Task 5: Remove catalog duplication

**Files:**
- Modify: `trade_rl/catalog/contracts.py`
- Modify: `trade_rl/catalog/postgres.py`
- Modify: `trade_rl/catalog/__init__.py`
- Keep: `trade_rl/catalog/postgres_sealed_test.py`

**Interfaces:**
- Consumes: `canonical_json_bytes` from `trade_rl.domain.canonical_json`; `PostgresSealedTestReservationStore`.
- Produces: one canonical JSON implementation and one sealed-test SQL owner while retaining `PostgresArtifactCatalog.reserve_sealed_test_access(record)`.

- [ ] **Step 1: Import canonical JSON from the domain module and remove the local encoder.**
- [ ] **Step 2: Replace the SQL body in `PostgresArtifactCatalog.reserve_sealed_test_access` with an explicit delegate to `PostgresSealedTestReservationStore`.**
- [ ] **Step 3: Remove catalog initializer mutation.**
- [ ] **Step 4: Run catalog unit and PostgreSQL integration tests.**
- [ ] **Step 5: Commit catalog boundary cleanup.**

### Task 6: Strengthen PostgreSQL workflow evidence

**Files:**
- Modify: `.github/workflows/postgres-catalog.yml`
- Test: `tests/architecture/test_architecture_followup.py`

**Interfaces:**
- Produces: PR and main-push execution with exact-head checkout and complete relevant path filters.

- [ ] **Step 1: Add `push: branches: [main]`.**
- [ ] **Step 2: Add evaluation sealed-test and workflow paths to the PR path filter.**
- [ ] **Step 3: Set checkout `ref` to `${{ github.event.pull_request.head.sha || github.sha }}` with credentials disabled.**
- [ ] **Step 4: Run workflow-security and workflow contract tests.**
- [ ] **Step 5: Commit CI evidence hardening.**

### Task 7: Ratchet coverage and complete verification

**Files:**
- Modify: `pyproject.toml`
- Modify: PR documentation and verification evidence.

**Interfaces:**
- Produces: critical coverage group for environment runtime services and final exact-head evidence.

- [ ] **Step 1: Add the environment runtime service files to a critical coverage group with a non-regressing threshold supported by current tests.**
- [ ] **Step 2: Run Ruff, format, Mypy, Import Linter, dead-code report, focused tests, full pytest with branch coverage, critical coverage, Studio tests/build/layout, compatibility, and training-image probe.**
- [ ] **Step 3: Run PostgreSQL Catalog on the exact final head.**
- [ ] **Step 4: Record run IDs, counts, coverage, and artifact digests in `docs/verification/2026-07-22-architecture-followup-remediation.md`.**
- [ ] **Step 5: Remove any temporary patch workflow or script.**
- [ ] **Step 6: Open or update a draft PR and leave it unmerged for review.