# Causal Alpha V3 Architecture Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development for every behavioral change and superpowers:verification-before-completion before completion claims.

**Goal:** Close the architectural gaps found in the V3 research runner so persisted evidence is self-contained and reloadable, resume is bound to production execution semantics, V3 admission is fail-closed on net economics and hard-risk evidence, and downstream consumers cannot accidentally train on the admission holdout.

**Architecture:** Preserve the research-only V3 lane and canonical U6/V2 behavior. Add durable leaf artifacts and loaders, introduce one execution-identity contract that binds the maintained runtime plus replay semantics, make admission V3-specific rather than weakening/changing the maintained V2 admission contract, cluster signal uncertainty by chronological episode, enforce single-writer output-root ownership, and reduce orchestration coupling after correctness contracts are closed.

**Tech Stack:** Python 3.12, dataclasses, NumPy, existing content-digest/canonical JSON/atomic-write primitives, pytest, Ruff, Mypy, import-linter, GitHub Actions.

## Global Constraints

- Canonical U6 remains unchanged: `target_weight` + `causal_alpha_ridge`, pure net-log-growth reward, one-decision signal delay, hard `max_position_to_market_notional=0.02`.
- V3 remains `research_only=true` and `promotion_eligible=false`.
- No validation/test symbol or admission holdout may influence fit, signal gating, candidate freeze, or economic selection.
- Existing V2 admission and V2 package consumer behavior must not change.
- No DAgger, BC, critic warm start, PPO, Lagrangian, or discounted PPO execution in this change.
- Tests may not be weakened, skipped, or rewritten to match an incorrect implementation.

---

### Task 1: Persist and reload complete V3 evidence graph

**Files:**
- Modify: `trade_rl/workflows/universal_causal_alpha_v3_signal.py`
- Modify: `trade_rl/workflows/universal_causal_alpha_v3_contracts.py`
- Modify: `trade_rl/workflows/universal_causal_alpha_v3_store.py`
- Modify: `trade_rl/workflows/universal_causal_alpha_v3_runner.py`
- Create/modify tests under `tests/workflows/`

**Interfaces:**
- Produce strict `from_payload()` loaders for V3 signal metrics/evidence and package manifest.
- Persist each signal scope under `signal/records/<fit_digest>/<symbol>/<episode>.json`.
- Persist each selected-symbol teacher batch as a durable artifact containing contracts, target arrays, dataset/teacher/sampling identity and digest.
- `teacher/package.json` must bind durable batch artifact digests, sample digests, partition digests, selection/admission identity, and be reconstructable in a fresh process.

- [ ] Add failing tests proving signal leaf metrics are absent today and package reload cannot reconstruct batches from disk.
- [ ] Run targeted tests and verify RED for the missing persistence/loader behavior.
- [ ] Add strict serializers/loaders and durable batch artifacts without changing V3 numerical outputs.
- [ ] Re-run targeted tests and verify GREEN.

### Task 2: Close resume provenance over execution semantics

**Files:**
- Modify: `trade_rl/workflows/universal_causal_alpha_v3_contracts.py`
- Modify: `trade_rl/workflows/universal_causal_alpha_v3_runner.py`
- Modify: tests under `tests/workflows/`

**Interfaces:**
- Add `CausalAlphaV3ExecutionIdentity` / equivalent digest to the run manifest.
- Bind `training_contract_digest`, `instrument_context_schema_digest`, action/environment/risk/execution/replay evaluator source identities and runtime settings used by the replay.

- [ ] Add failing tests that mutate runtime/execution identity while reusing the same output root and expect rejection.
- [ ] Verify RED.
- [ ] Implement the identity closure and manifest validation.
- [ ] Verify GREEN.

### Task 3: Strengthen V3 admission without changing V2

**Files:**
- Modify: `trade_rl/workflows/universal_causal_alpha_v3_contracts.py`
- Modify: `trade_rl/workflows/universal_causal_alpha_v3_runner.py`
- Modify: tests under `tests/workflows/`

**Interfaces:**
- V3 admission records include hard-risk violation and execution rejection reasons/counts.
- V3 aggregate admission requires non-negative aggregate gross and net return, no hard-risk violations, no unexplained execution rejections, and no majority-negative gross symbols.
- Maintained `evaluate_causal_alpha_teacher_admission()` remains unchanged for V2/U6.

- [ ] Add failing tests for gross-positive/net-negative holdout, hard-risk violation, and unexplained rejection.
- [ ] Verify RED.
- [ ] Add V3-specific admission evidence/gate.
- [ ] Verify GREEN and V2 regression tests.

### Task 4: Verify replay initial-state identity

**Files:**
- Modify: `trade_rl/workflows/universal_causal_alpha_v3_runner.py` or a focused replay helper module.
- Modify: tests under `tests/workflows/`.

**Interfaces:**
- Before executing a contract, resolve the concrete environment initial weights for the contract mode/start and require exact numerical agreement with `contract.initial_weights` within the existing deterministic tolerance.

- [ ] Add a failing drift test.
- [ ] Verify RED.
- [ ] Implement the fail-closed initial-state check.
- [ ] Verify GREEN.

### Task 5: Replace flat signal bootstrap with chronological episode clusters

**Files:**
- Modify: `trade_rl/workflows/universal_causal_alpha_v3_signal.py`
- Modify: config/docs only if schema semantics require clarification.
- Modify: tests under `tests/workflows/`.

**Interfaces:**
- Aggregate same-episode/time scopes across symbols into one cluster statistic before bootstrap.
- Preserve chronological ordering by episode contract start/index, not symbol-major flattening.
- Bootstrap block size applies to chronological episode clusters.

- [ ] Add failing tests where duplicating correlated symbols cannot artificially tighten the CI/pass the gate.
- [ ] Verify RED.
- [ ] Implement cluster aggregation and deterministic bootstrap.
- [ ] Verify GREEN.

### Task 6: Enforce single-writer resume semantics

**Files:**
- Modify: `trade_rl/workflows/universal_causal_alpha_v3_store.py`
- Modify: `trade_rl/workflows/universal_causal_alpha_v3_runner.py`
- Modify: tests under `tests/workflows/`.

**Interfaces:**
- Acquire an output-root run lock before any stage side effect.
- A second live writer fails closed before replay/admission evaluation.
- Stale ownership is not silently stolen; operator must remove/recover explicitly unless a proven process-safe lease mechanism exists.

- [ ] Add failing two-writer test.
- [ ] Verify RED.
- [ ] Implement exclusive run lock with cleanup in `finally`.
- [ ] Verify GREEN.

### Task 7: Strict schemas and immutable in-memory contracts

**Files:**
- Modify: `trade_rl/workflows/universal_causal_alpha_v3_contracts.py`
- Modify: `trade_rl/workflows/universal_causal_alpha_v3_runner.py`
- Modify: `trade_rl/workflows/universal_causal_alpha_v3_signal.py`
- Modify: tests under `tests/workflows/`.

**Interfaces:**
- Every persisted loader checks exact schema version and required/unknown fields.
- Frozen contracts expose immutable mappings/tuples rather than mutable `dict` instances after digest construction.

- [ ] Add failing schema-tamper and post-construction mapping-mutation tests.
- [ ] Verify RED.
- [ ] Implement strict parsing and immutable mappings.
- [ ] Verify GREEN.

### Task 8: Split orchestration after contracts are closed

**Files:**
- Create focused stage modules as needed under `trade_rl/workflows/`.
- Reduce `universal_causal_alpha_v3_runner.py` to preparation + stage sequencing.
- Preserve public runner/CLI signatures.

**Interfaces:**
- Stage outputs are explicit immutable types: prepared -> signal accepted -> frozen -> selected -> admitted/package.
- No stage can be called with a logically earlier artifact shape.

- [ ] Add/retain contract tests before moving behavior.
- [ ] Refactor only after all prior behavioral tests are GREEN.
- [ ] Run targeted and full regression suites.

## Quality Gate

- All above regression tests pass on the exact final HEAD.
- Ruff and format checks pass.
- Mypy passes without ignores added to silence the new code.
- Import architecture keeps all maintained contracts.
- Full pytest passes; no V3 architecture test is skipped.
- Overall and critical coverage gates pass.
- Ubuntu and Windows compatibility pass.
- Training image/runtime probe and PostgreSQL Catalog pass.
- Final diff is reviewed against this plan and the V3 design spec.
- Independent/falsification review explicitly attempts runtime-drift resume, artifact tampering, duplicate writer, holdout leakage, and net-negative/hard-risk admission bypass.

## What this verification will and will not prove

Passing this plan proves the V3 software workflow is self-contained, auditable, reloadable, and fail-closed under the enumerated software failure modes. It does not prove positive alpha, positive net returns on real data, teacher admission success, RL uplift, profitability, or Production GO.
