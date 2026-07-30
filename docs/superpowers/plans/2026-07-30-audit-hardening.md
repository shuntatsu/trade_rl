# Post-audit Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make execution promotion, margin configuration, artifact publication, and model-loading boundaries fail closed against the audited defects.

**Architecture:** Preserve existing module boundaries. Add validation and identity fields at the existing contracts, reject unsupported semantics rather than emulating them, and isolate mutable filesystem inputs before deserialization. Every task begins with a regression test and ends with targeted verification.

**Tech Stack:** Python 3.12, dataclasses, NumPy, pytest, Stable-Baselines3, canonical JSON and SHA-256 artifact contracts.

## Global Constraints

- Keep selected-final training fail closed.
- Do not silently accept legacy execution-promotion schema versions.
- Do not add a partial isolated-margin implementation.
- Do not weaken existing checkpoint architecture, dataset, or policy identity checks.
- Do not introduce new runtime dependencies.

---

### Task 1: Execution configuration boundary

**Files:**
- Modify: `tests/simulation/test_critical_branch_coverage.py`
- Modify: `trade_rl/simulation/execution.py`

**Interfaces:**
- Produces: `ExecutionCostConfig.execution_economics_digest: str`
- Preserves: `ExecutionCostConfig.execution_policy_digest: str`

- [ ] Add failing tests proving isolated margin is rejected, non-zero tail probability rejects multipliers below one, and economic parameter changes alter the economics digest without altering the mechanics digest.
- [ ] Run the targeted simulation tests and confirm the new tests fail for the audited behavior.
- [ ] Add the minimal validation and canonical economics payload/digest.
- [ ] Run the targeted simulation tests and confirm they pass.

### Task 2: Execution promotion evidence v2

**Files:**
- Modify: `tests/evaluation/test_execution_promotion.py`
- Modify: `trade_rl/simulation/execution_promotion.py`
- Modify callers and fixtures found by repository search for `ExecutionEvidence(` and `execution_evidence_from_cost(`.

**Interfaces:**
- `ExecutionEvidence.execution_economics_digest: str`
- `execution_evidence_from_cost(..., order_event_count: int, complete_order_evidence: bool, ...) -> ExecutionEvidence`

- [ ] Add failing tests proving zero order events cannot promote and mismatched economics identity cannot promote.
- [ ] Run the targeted promotion tests and confirm failures are caused by the missing validation and field.
- [ ] Advance the schema to `execution_promotion_evidence_v2`, bind the economics digest, require positive event count at promotion, and update canonical serialization.
- [ ] Update all maintained fixtures and selected-final validation callers.
- [ ] Run targeted promotion, workflow, serving, and e2e tests.

### Task 3: Artifact publication concurrency

**Files:**
- Modify: artifact-store implementation located by `class ArtifactStore`.
- Modify: corresponding artifact-store tests.

**Interfaces:**
- Preserves public `ArtifactStore` methods.
- Adds internal unique temporary path and store-scoped exclusive lock helpers.

- [ ] Add failing tests for unique temporary files, same-second run-id uniqueness, and concurrent latest-pointer writes.
- [ ] Run the targeted tests and confirm failure against the fixed `.tmp` and second-resolution implementation.
- [ ] Use exclusive, process-unique temporary files and microsecond-plus-random run identifiers; serialize store mutation with an exclusive lock.
- [ ] Run targeted artifact tests.

### Task 4: Trusted checkpoint and replay-buffer loading

**Files:**
- Modify checkpoint manifest/load modules located by `load_checkpoint_manifest` and `load_replay_buffer`.
- Modify corresponding learning/checkpoint tests.

**Interfaces:**
- Adds internal regular-file/no-symlink validation.
- Adds private verified-copy preparation for deserialization.

- [ ] Add failing tests for symlinked manifest/policy/replay files and for source replacement after verification.
- [ ] Run targeted tests and confirm audited behavior is exposed.
- [ ] Validate path containment and regular files, copy verified bytes to a private temporary directory, and deserialize only from that copy.
- [ ] Run targeted checkpoint, resume, transfer, and replay-buffer tests.

### Task 5: Dead and misleading API cleanup

**Files:**
- Modify: `trade_rl/simulation/execution.py`
- Modify: `tests/simulation/test_critical_branch_coverage.py`

**Interfaces:**
- Remove private `_capacity_notional` and obsolete `rate_per_turnover` if repository search confirms no production consumers.

- [ ] Confirm definitions have no production consumers.
- [ ] Remove tests that directly preserve dead private helpers and replace them with public `MarketDataset.market_notional` coverage where needed.
- [ ] Run Vulture and targeted tests.

### Task 6: Full verification and publication

- [ ] Run targeted tests for every changed module.
- [ ] Run full pytest.
- [ ] Run Ruff, Mypy, Import Linter, Vulture, serving smoke, and security-oriented workflow checks available in the repository.
- [ ] Compare branch to main and inspect every changed file for scope drift.
- [ ] Open a draft pull request with root causes, compatibility impact, and exact verification evidence.