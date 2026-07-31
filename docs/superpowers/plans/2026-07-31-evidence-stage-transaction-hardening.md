# Execution Evidence and Stage Transaction Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make execution promotion depend on a strict, replay-bound artifact and make symbol-triplet stage advancement crash-consistent through one atomic generation pointer.

**Architecture:** The execution side reconstructs maintained domain objects from canonical bytes, binds all traces and evaluation identities into `StatefulReplayEvidence`, and derives promotion evidence only from that verified root. The stage side stores immutable completion/cursor generations and commits one content-addressed pointer, with legacy two-file state accepted only through validated migration.

**Tech Stack:** Python 3.12, dataclasses, canonical JSON, SHA-256 content identities, POSIX/Windows file locks, filesystem fsync/atomic replace, pytest, Ruff, MyPy, GitHub Actions.

## Global Constraints

- Do not weaken existing fail-closed schema field closure.
- Do not accept caller-claimed completeness, trace counts, or trace digests.
- Do not deserialize unsafe model artifacts from unverified public paths.
- Preserve Windows and POSIX compatibility.
- Do not add a PostgreSQL dependency to filesystem-only stage workflows.
- Every new artifact write must be exclusive, fsynced, and content-addressed where applicable.
- Final verification must run on one unchanged head commit.

---

### Task 1: Strict order-event reconstruction

**Files:**
- Modify: `trade_rl/simulation/orders.py`
- Modify: `trade_rl/simulation/execution_replay.py`
- Test: `tests/simulation/test_execution_replay_strict_contract.py`

**Interfaces:**
- Produces: `OrderEvent.from_mapping(value: Mapping[str, object]) -> OrderEvent`
- Produces: `validate_order_event_stream(events: Sequence[OrderEvent]) -> tuple[OrderEvent, ...]`
- Consumes: existing `OrderStatus`, `OrderEvent`, and canonical payload identities.

- [ ] **Step 1: Write failing tests** for minimal dictionaries, missing fields, sequence gaps, duplicate sequences, invalid status transitions, inconsistent fill arithmetic, and mismatched replacement identities.
- [ ] **Step 2: Run** `pytest tests/simulation/test_execution_replay_strict_contract.py -q` and confirm the loose dictionary artifact currently passes or lacks the required API.
- [ ] **Step 3: Implement** exact field closure and typed reconstruction in `OrderEvent.from_mapping`; reject booleans as integers, non-finite numeric fields, invalid enum values, and noncanonical tuple/list shapes.
- [ ] **Step 4: Implement** stream validation: sequences equal `range(len(events))`, events for each order form a legal maintained state path, cumulative fills never exceed requested quantity, remaining quantity matches requested minus cumulative fill, and terminal events cannot be followed by later events.
- [ ] **Step 5: Run** the focused tests plus `pytest tests/simulation/test_stateful_execution_characterization.py tests/e2e/test_stateful_order_replay.py -q`.
- [ ] **Step 6: Commit** with `fix: validate canonical order event streams`.

### Task 2: Replay-bound execution artifact and exclusive publication

**Files:**
- Modify: `trade_rl/simulation/execution_replay.py`
- Modify: `trade_rl/simulation/execution_promotion.py`
- Modify: `trade_rl/artifacts/verified_file.py`
- Test: `tests/evaluation/test_execution_promotion_replay_binding.py`
- Test: `tests/evaluation/test_execution_promotion_audit_hardening.py`

**Interfaces:**
- Produces: `ExecutionReplayIdentity(candidate_config_digest, evaluation_run_digest, fold, seed)`
- Produces: `ExecutionEventArtifact` schema `execution_order_event_artifact_v2`
- Produces: `write_execution_event_artifact_content_addressed(root: Path, artifact: ExecutionEventArtifact) -> Path`
- Produces: `load_execution_event_artifact_bytes(raw: bytes) -> ExecutionEventArtifact`
- Produces: `ExecutionEvidence` schema `execution_promotion_evidence_v4`

- [ ] **Step 1: Write failing tests** for candidate/run/fold/seed substitution, action/equity/observation trace substitution, incomplete terminal state, duplicate publication, partial publication, symlink/FIFO inputs, and event count disagreement.
- [ ] **Step 2: Run** the focused tests and confirm the v3 artifact does not bind those identities.
- [ ] **Step 3: Extend** the artifact with exact replay identity and embedded `StatefulReplayEvidence`; recompute action, order-event, equity, and observation digests from supplied canonical traces and reject caller-provided mismatches.
- [ ] **Step 4: Cross-validate** terminal book fill count against fill events, terminal order IDs/statuses against the final stream state, and replay step counts against actions/equity/observations.
- [ ] **Step 5: Replace** the write path with basename `<sha256>.execution-replay.json`, `open("xb")`, file fsync, parent-directory fsync, and idempotent acceptance only when existing bytes have the exact same digest.
- [ ] **Step 6: Derive** promotion evidence exclusively from the loaded replay artifact; remove public parameters that claim `order_event_count` or `complete_order_evidence` for promotable evidence.
- [ ] **Step 7: Run** all execution promotion, replay, and serving-package tests.
- [ ] **Step 8: Commit** with `fix: bind promotion evidence to deterministic replay`.

### Task 3: Maintained evaluation production path

**Files:**
- Create: `trade_rl/workflows/execution_promotion_artifacts.py`
- Create: `trade_rl/workflows/stage_a_execution_observation.py`
- Modify: `trade_rl/workflows/training_run.py`
- Modify: `trade_rl/cli/extended.py`
- Test: `tests/workflows/test_execution_promotion_artifact_workflow.py`
- Test: `tests/evaluation/test_stage_a_zero_shot_contracts.py`
- Test: `tests/e2e/test_research_to_serving_v2.py`

**Interfaces:**
- Produces: `ExecutionPromotionArtifacts(event_artifact_path, event_artifact_digest, evidence_path, evidence_digest)`
- Produces: `write_execution_promotion_artifacts(...) -> ExecutionPromotionArtifacts`
- Produces: `StageAEvaluationObservation.create_from_execution_artifacts(...)`

- [ ] **Step 1: Write failing tests** that build one completed stateful evaluation result and require the workflow to emit both replay artifact and evidence without accepting a caller-authored JSON file.
- [ ] **Step 2: Implement** one workflow function that accepts the actual actions, observation digests, equity curve, order events, terminal book/order book, candidate config digest, evaluation run digest, fold, seed, dataset, and execution cost; it builds, writes, reloads, and revalidates both artifacts before returning identities.
- [ ] **Step 3: Add** a Stage A observation constructor that accepts the verified artifact result and checks candidate configuration, checkpoint seed, fold, dataset, execution identity, and plan identities before recording growth values.
- [ ] **Step 4: Change** selected-final training to consume a replay artifact root/digest pair and derive the evidence path from the artifact workflow; retain the old two-path CLI only as an explicitly rejected legacy combination for selected-final runs.
- [ ] **Step 5: Run** Stage A contract/gate, training-run, CLI, and research-to-serving tests.
- [ ] **Step 6: Commit** with `feat: emit promotion evidence from evaluation results`.

### Task 4: Crash-consistent stage generation store

**Files:**
- Create: `trade_rl/workflows/symbol_triplet_stage_state.py`
- Modify: `trade_rl/workflows/symbol_triplet_stage_orchestrator.py`
- Modify: `trade_rl/workflows/symbol_triplet_stage_training.py`
- Modify: `trade_rl/workflows/binance_symbol_triplet_stage_command.py`
- Test: `tests/workflows/test_symbol_triplet_stage_state.py`
- Test: `tests/workflows/test_symbol_triplet_stage_orchestrator.py`

**Interfaces:**
- Produces: `SymbolTripletStageStateStore`
- Produces: `load_current(plan) -> tuple[SymbolTripletStageCompletion | None, SymbolTripletTrainingCursor]`
- Produces: `commit(expected_cursor_digest, completion, cursor) -> StageStatePointer`
- Produces: `migrate_legacy(cursor_path, completion_path, plan) -> StageStatePointer`

- [ ] **Step 1: Write failing tests** for interruption after completion write, interruption after cursor write but before pointer replacement, post-pointer directory-fsync failure, stale writer CAS, orphan generation handling, and legacy mismatch.
- [ ] **Step 2: Implement** immutable generation directories under `generations/<content-digest>/`, containing canonical completion and cursor files whose cross-reference is validated after reload.
- [ ] **Step 3: Implement** a canonical `current.json` pointer with expected previous pointer/cursor digest CAS under the existing cross-platform lock.
- [ ] **Step 4: Ensure** pre-pointer failures leave only unreachable generations; post-replace durability uncertainty does not delete the committed generation or completion.
- [ ] **Step 5: Implement** one-time legacy migration that loads and cross-validates the old cursor/completion pair before publishing a generation pointer.
- [ ] **Step 6: Route** the maintained stage command/training/orchestrator through the state store and stop publishing separate maintained cursor/completion writes.
- [ ] **Step 7: Run** all symbol-triplet and Stage A scheduling tests on Linux-compatible paths; rely on CI for the Windows lock implementation.
- [ ] **Step 8: Commit** with `fix: publish triplet stage state by generation`.

### Task 5: Verified structured manifest bytes

**Files:**
- Modify: `trade_rl/rl/structured_export.py`
- Modify: `trade_rl/serving/structured_policy.py`
- Modify: `trade_rl/artifacts/verified_file.py`
- Test: `tests/rl/test_structured_export.py`
- Test: `tests/rl/test_structured_manifest_trust_boundary.py`

**Interfaces:**
- Produces: `read_verified_bytes(path: Path, expected_digest: str, expected_size_bytes: int, field: str) -> bytes`
- Produces: `load_structured_export_manifest_bytes(raw: bytes) -> StructuredExportManifest`

- [ ] **Step 1: Write a failing race test** that swaps the manifest after digest verification but before the old path parser reopens it.
- [ ] **Step 2: Implement** a byte-based canonical parser and make the path loader open once through the regular-file boundary.
- [ ] **Step 3: Make** the serving loader compare bundle digest/size and parse the exact verified bytes; do not reopen the public manifest path.
- [ ] **Step 4: Replace** the structured export fixed temporary filename with the shared unique atomic writer.
- [ ] **Step 5: Run** structured export, serving loader, and recovery smoke tests.
- [ ] **Step 6: Commit** with `fix: parse verified structured manifest bytes`.

### Task 6: Full verification and review

**Files:**
- Modify: `docs/operations/audit-hardening-verification.md`
- Modify: this plan checklist as tasks complete.

**Interfaces:**
- Consumes all preceding task interfaces.
- Produces a draft PR with exact-head evidence.

- [ ] **Step 1: Run** focused tests for every task and `python -m compileall trade_rl tests`.
- [ ] **Step 2: Run** Ruff check, Ruff format check, MyPy, Import Linter, and dead-code report.
- [ ] **Step 3: Run** the complete pytest suite with branch coverage and critical-branch ratchet.
- [ ] **Step 4: Run** CLI smoke, serving/recovery smoke, Ubuntu/Windows compatibility, Training image, and PostgreSQL Catalog workflows through GitHub Actions.
- [ ] **Step 5: Request** an independent code review against base `6b2abf993abab157f4e8aabf9c173e5c039d7880` and resolve every Critical/Important finding.
- [ ] **Step 6: Update** the PR body with the unchanged final head SHA, exact test counts, coverage, and all workflow conclusions.
- [ ] **Step 7: Keep** the PR draft until every required workflow passes on the same head; do not merge without an explicit user request.
