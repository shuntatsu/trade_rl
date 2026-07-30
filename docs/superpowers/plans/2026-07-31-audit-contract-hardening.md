# Audit Contract Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make execution promotion, staged transfer, artifact publication, and model deserialization cryptographically and concurrently fail closed.

**Architecture:** Reapply the non-conflicting audit fixes from PR #301 to current `main`, then introduce explicit content-chain anchors and shared verified-file primitives. Stage state uses a cursor-anchored completion digest plus an exclusive compare-and-swap commit. Unsafe deserializers consume private verified copies rather than reopening validated paths.

**Tech Stack:** Python 3.12, dataclasses, pathlib, SHA-256 canonical artifacts, Stable-Baselines3, PyTorch TorchScript, pytest, GitHub Actions, PostgreSQL integration tests.

## Global Constraints

- Preserve the current layered import architecture and Import Linter contracts.
- Keep selected-final workflows fail closed; do not silently migrate legacy evidence or cursors.
- Do not add a new runtime dependency.
- Continue supporting Windows, Ubuntu, non-root containers, and optional PostgreSQL tests.
- Write regression tests before each behavioral change.
- Use unique exclusive temporary files and reject symlink/non-regular deserialization inputs.

---

### Task 1: Reapply current audit fixes

**Files:**
- Modify: `trade_rl/simulation/execution.py`
- Modify: `trade_rl/simulation/execution_promotion.py`
- Modify: `trade_rl/workflows/selection_authorization.py`
- Modify: `trade_rl/workflows/training_run.py`
- Modify: `trade_rl/artifacts/store.py`
- Modify: `trade_rl/rl/checkpointing.py`
- Modify: `trade_rl/rl/replay.py`
- Modify: `trade_rl/integrations/sb3_checkpoint_assembly.py`
- Modify: `trade_rl/integrations/sb3_training.py`
- Test: the corresponding PR #301 regression files

**Interfaces:**
- Produces `execution_policy_v2` identity.
- Produces verified checkpoint and replay-buffer private-copy helpers.
- Rejects multi-asset isolated margin.

- [x] Copy the final PR #301 file versions onto the current-main branch.
- [x] Verify that the three commits after PR #301's base do not modify the copied paths.
- [x] Preserve current-main additions unchanged.
- [x] Commit as `Reapply audit execution and artifact hardening`.

### Task 2: Bind real order-event evidence

**Files:**
- Modify: `trade_rl/simulation/execution_promotion.py`
- Modify: `trade_rl/simulation/execution_replay.py`
- Modify: `trade_rl/workflows/selection_authorization.py`
- Modify: `trade_rl/workflows/training_run.py`
- Test: `tests/evaluation/test_execution_promotion_audit_hardening.py`
- Test: `tests/workflows/test_selection_authorization.py`
- Test: `tests/e2e/test_research_to_serving_v2.py`

**Interfaces:**
- Produces `ExecutionEvidence` schema `execution_promotion_evidence_v3`.
- Adds `validate_execution_event_artifact(evidence, event_path) -> None`.
- Adds `execution_evidence_digest` to `SelectionProposal`.

- [ ] Add failing tests that forge `order_event_count`, substitute the event artifact, alter terminal book digests, and pass a signed proposal with a different evidence digest.
- [ ] Confirm the tests fail on v2 behavior.
- [ ] Add event artifact digest, size, schema, terminal book digest, and terminal order-book digest fields.
- [ ] Validate a regular non-symlink event artifact and recompute count and terminal digests.
- [ ] Bind the v3 evidence digest into the selection proposal and selected-final execution path.
- [ ] Run focused execution-promotion, selection, and selected-final E2E tests.
- [ ] Commit as `Bind order-event artifacts to promotion evidence`.

### Task 3: Make stage advancement cursor-anchored and compare-and-swap

**Files:**
- Modify: `trade_rl/workflows/symbol_triplet_training_cursor.py`
- Modify: `trade_rl/workflows/symbol_triplet_stage_orchestrator.py`
- Modify: `trade_rl/workflows/symbol_triplet_stage_training.py`
- Modify: `trade_rl/workflows/binance_symbol_triplet_stage_runner.py`
- Modify: `trade_rl/workflows/binance_symbol_triplet_stage_command.py`
- Test: `tests/workflows/test_symbol_triplet_training_cursor.py`
- Test: `tests/workflows/test_symbol_triplet_stage_orchestrator.py`
- Test: `tests/workflows/test_symbol_triplet_stage_training.py`
- Test: `tests/workflows/test_binance_symbol_triplet_stage_runner.py`
- Test: `tests/workflows/test_binance_symbol_triplet_stage_command.py`

**Interfaces:**
- Produces cursor schema `symbol_triplet_training_cursor_v2` with `last_completion_digest`.
- Changes `advance_symbol_triplet_training_cursor(..., completion_digest: str)`.
- Adds an exclusive cursor lock and persisted-digest compare-and-swap inside `commit_symbol_triplet_stage_completion`.

- [ ] Add failing tests for a substituted previous completion and two commits using the same stale cursor.
- [ ] Confirm both tests fail on the current behavior.
- [ ] Add the completion digest anchor to cursor serialization, validation, and advancement.
- [ ] Require the prior completion digest to equal the cursor anchor before transfer.
- [ ] Acquire an exclusive lock, reload the persisted cursor under the lock, and compare its digest before publishing.
- [ ] Exclusively publish completion and atomically replace cursor; retain rollback on pre-cursor failure.
- [ ] Update runner and command callers.
- [ ] Run all symbol-triplet workflow tests.
- [ ] Commit as `Anchor and serialize symbol-triplet stage commits`.

### Task 4: Close TorchScript and bundle TOCTOU windows

**Files:**
- Create: `trade_rl/artifacts/verified_file.py`
- Modify: `trade_rl/rl/checkpointing.py`
- Modify: `trade_rl/rl/replay.py`
- Modify: `trade_rl/serving/structured_policy.py`
- Modify: `trade_rl/serving/policy_loader.py`
- Test: `tests/rl/test_checkpoint_trust_boundary.py`
- Test: `tests/serving/test_structured_policy_trust_boundary.py`
- Test: `tests/serving/test_policy_loader.py`

**Interfaces:**
- Produces `open_regular_binary(path, field)` and `verified_private_copy(path, expected_digest, field, filename)` context managers.
- All unsafe deserializers load only the yielded private path.

- [ ] Add failing tests that replace a source file after verification but before deserialization and tests for symlink/non-regular inputs.
- [ ] Confirm the tests fail with path reopening.
- [ ] Extract regular-file and verified-copy logic into the architecture-legal shared artifact module.
- [ ] Refactor checkpoint and replay loading to use the shared helper.
- [ ] Refactor single and ensemble TorchScript loading to deserialize private verified copies.
- [ ] Run checkpoint, replay, structured-serving, and research-to-serving tests.
- [ ] Commit as `Deserialize only verified private artifact copies`.

### Task 5: Make pointer publication state-aware

**Files:**
- Create: `trade_rl/artifacts/atomic_pointer.py`
- Modify: `trade_rl/artifacts/store.py`
- Modify: `trade_rl/serving/registry.py`
- Modify: `trade_rl/workflows/full_research_state.py`
- Modify: `trade_rl/serving/bundle.py`
- Modify: `trade_rl/serving/policy_loader.py`
- Test: `tests/artifacts/test_store_audit_hardening.py`
- Test: `tests/serving/test_registry.py`
- Test: `tests/workflows/test_full_research_state.py`

**Interfaces:**
- Produces `atomic_replace_bytes(path, payload) -> AtomicReplaceResult`.
- Raises `AtomicReplaceDurabilityError` only after replacement has occurred.
- Callers roll back content only when replacement did not occur.

- [ ] Add failing tests for failure before replacement and directory-fsync failure after replacement.
- [ ] Confirm the old rollback produces a dangling pointer after post-replace failure.
- [ ] Add a state-aware atomic replacement primitive using unique exclusive temporary files.
- [ ] Roll back published runs only on definite pre-replace failure.
- [ ] Keep installed content in place on post-replace durability uncertainty and raise an explicit error.
- [ ] Replace fixed `.tmp` writers in the touched publication paths.
- [ ] Run artifact, serving registry, bundle, and workflow state tests.
- [ ] Commit as `Make artifact pointer publication state aware`.

### Task 6: Validate lazy exports and complete verification

**Files:**
- Modify: `tests/architecture/test_import_references.py`
- Create: `tests/architecture/test_public_lazy_exports.py`
- Update: PR description and audit documentation

**Interfaces:**
- Verifies every name in lazy `__all__` maps to an importable module attribute.

- [ ] Add a test importing every public lazy export.
- [ ] Run Ruff and formatting checks.
- [ ] Run Mypy and Import Linter.
- [ ] Run the vulture dead-code report.
- [ ] Run full pytest with branch coverage and critical-coverage validation.
- [ ] Confirm Ubuntu, Windows, container, and PostgreSQL workflow results.
- [ ] Update the pull-request body with exact evidence and remaining compatibility breaks.
