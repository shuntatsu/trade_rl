# Stage A PostgreSQL Sealed-Test Ledger Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Authorize the complete Stage A sealed-test triplet × fold closure exactly once and persist it atomically in PostgreSQL, including plan, evaluation-dataset manifest, evaluation identity, selected candidate, selected policy, dataset, fold, and exact test-range identities.

**Architecture:** Introduce a framework-independent Stage A authorization batch contract in `trade_rl.evaluation`. The batch deterministically owns sorted cell authorizations and their existing generic `SealedTestAccessRecord` values. The default in-memory ledger and the PostgreSQL ledger implement the same batch protocol. PostgreSQL migration `0003` creates one batch table and one child-cell table; one transaction inserts the batch, reserves every generic plan/dataset/fold key, inserts every Stage A cell, reads the rows back, and commits only after exact equality checks. The orchestrator authorizes one batch before any test evaluation and binds the returned batch digest into every Stage A access record and published artifact.

**Tech Stack:** Python 3.12, dataclasses, canonical SHA-256 digests, PostgreSQL/psycopg, pytest, Ruff, Mypy, Import Linter.

## Global Constraints

- Validation must never touch the sealed-test ledger.
- Recompute and verify validation selection before constructing the authorization batch.
- One authorization call must contain every declared test triplet × fold cell.
- Sort cells by `(triplet_id, fold_index)` before hashing or persistence.
- Reject duplicate, missing, undeclared, or substituted cells before opening test data.
- Use one database transaction for batch row, generic reservations, Stage A cell rows, and read-back verification.
- Any conflict or mismatch must roll back the complete batch; partial opening is forbidden.
- Preserve the existing generic plan/dataset/fold reservation table to prevent cross-path reopening.
- The primary one-shot key is the immutable Stage A plan digest. Manifest and selection identities are persisted and verified, not used to permit a second opening.
- Persist no mutable status flag that could reopen or overwrite an authorization.
- Store deterministic digests; timestamps remain audit metadata and are excluded from identities.
- Keep catalog independent of workflows. Shared batch contracts live under `trade_rl.evaluation`.
- Update maintained documentation under `docs/operations` only.

---

## File Structure

- Create `trade_rl/evaluation/stage_a_sealed_test.py`: cell/batch contracts, builder, protocol, and in-memory batch ledger.
- Create `trade_rl/catalog/sql/0003_stage_a_sealed_test_ledger.sql`: immutable batch and cell tables.
- Create `trade_rl/catalog/postgres_stage_a_sealed_test.py`: atomic PostgreSQL batch ledger.
- Modify `trade_rl/workflows/stage_a_zero_shot_runner.py`: authorize one complete batch before test evaluation.
- Modify `trade_rl/workflows/stage_a_zero_shot_runner_contracts.py`: bind authorization batch digest into access records and sealed-test runs.
- Modify `trade_rl/workflows/stage_a_zero_shot_artifacts.py`: publish batch digest and bump access-record package schema.
- Modify `trade_rl/catalog/__init__.py`: export the PostgreSQL Stage A ledger.
- Add `tests/evaluation/test_stage_a_sealed_test.py`.
- Add `tests/catalog/test_postgres_stage_a_sealed_test.py`.
- Modify `tests/catalog/test_postgres_integration.py`.
- Modify `tests/workflows/test_stage_a_zero_shot_runner.py` and artifact tests.

### Task 1: Stage A Authorization Batch Contract

- [ ] Add RED tests for deterministic cell ordering, deterministic batch digest, duplicate-cell rejection, generic-record closure, and one-shot in-memory authorization.
- [ ] Verify tests fail because `trade_rl.evaluation.stage_a_sealed_test` does not exist.
- [ ] Implement `StageASealedTestCellAuthorization`, `StageASealedTestAuthorizationBatch`, `StageASealedTestLedgerProtocol`, `StageASealedTestLedger`, and `build_stage_a_sealed_test_authorization_batch`.
- [ ] Require non-empty cells, one selected policy digest, exact generic-record identities, and unique `(triplet_id, fold_index)` keys.
- [ ] Run focused evaluation tests to GREEN.

### Task 2: PostgreSQL Schema and Atomic Ledger

- [ ] Add migration `0003_stage_a_sealed_test_ledger.sql` with immutable batch and child-cell tables.
- [ ] Add RED PostgreSQL tests for first authorization, cross-instance duplicate rejection, exact row persistence, and full rollback when any generic reservation already exists.
- [ ] Implement `PostgresStageASealedTestLedger` using a single connection transaction.
- [ ] Insert the batch with `ON CONFLICT DO NOTHING RETURNING batch_digest`; failure means the plan was already opened.
- [ ] Insert every existing generic access record into `catalog_sealed_test_access` in the same transaction.
- [ ] Insert Stage A cells and read batch/cells back in deterministic order.
- [ ] Compare every stored identity and digest before commit.
- [ ] Run the real PostgreSQL matrix to GREEN.

### Task 3: Orchestrator Batch Integration

- [ ] Replace per-cell generic authorization in `StageAZeroShotEvaluationOrchestrator` with one complete Stage A batch authorization.
- [ ] Keep default behavior through the new in-memory batch ledger.
- [ ] Add tests proving exactly one authorization occurs before any test evaluator call.
- [ ] Add tests proving authorization failure performs zero sealed-test evaluations.
- [ ] Add tests proving the batch contains the complete manifest-derived triplet × fold closure.
- [ ] Keep selected-only policy evaluation and shared baseline behavior unchanged.

### Task 4: Evidence and Artifact Binding

- [ ] Bump `StageASealedTestAccessRecord` schema and add `authorization_batch_digest`.
- [ ] Require every access record in one `StageASealedTestRun` to share the same batch digest.
- [ ] Include the batch digest in access-record digest payloads and the sealed-test package.
- [ ] Bump the access-record package schema and update artifact tests.
- [ ] Reject legacy access-record payloads without a batch identity.

### Task 5: Verification and Merge

- [ ] Run focused evaluation, catalog, workflow, artifact, migration, Ruff, Format, Mypy, and Import Linter checks.
- [ ] Run exact-head normal CI and PostgreSQL Catalog workflows.
- [ ] Require full pytest, branch coverage, Studio, Windows/Ubuntu compatibility, training image, CLI smoke, and PostgreSQL integration success.
- [ ] Record exact head and totals in the PR.
- [ ] Merge only after no unresolved review thread remains.

## Self-Review

- Atomicity: one transaction covers both Stage A-specific and existing generic reservations.
- Concurrency: the immutable batch insert serializes competing openings by plan digest.
- Recovery: any exception rolls back every row, leaving the test unopened.
- Identity closure: batch and cell digests bind all manifest-derived cells and the selected policy.
- Architecture: catalog imports evaluation contracts, never workflows.
- Compatibility: generic non-Stage-A ledgers remain unchanged; Stage A moves to a dedicated batch protocol.
