# Environment Facade Audit Resolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development and superpowers:verification-before-completion task by task.

**Goal:** Reclassify `AUD-RL-001` as resolved only after an executable documentation contract proves that the maintained facade boundary and production status are described correctly.

**Architecture:** Keep Python production code unchanged. Add one RED documentation contract, then update the closeout and a finding-specific verification document to describe the typed construction/runtime boundaries, protected mutable-state facade, permanent architecture controls, and exact-head evidence.

**Tech Stack:** Python 3.12, pytest, Markdown, GitHub Actions, PostgreSQL.

## Global Constraints

- Do not mechanically split `reset()` or `step()`.
- Do not move mutable Gymnasium state ownership out of `ResidualMarketEnv`.
- Do not change Python production behavior or public APIs.
- Preserve production `NO-GO`.
- Require exact-head CI and PostgreSQL success before merge.

---

### Task 1: Add the RED resolution contract

**Files:**
- Modify: `tests/test_current_documentation_contract.py`

- [ ] Add `test_environment_facade_audit_is_resolved_with_protected_state_boundary()`.
- [ ] Require the closeout summary row to use `RESOLVED`.
- [ ] Require a `### AUD-RL-001 — RESOLVED` section.
- [ ] Require the section to mention the 150-line constructor limit, typed owners, `step()` orchestration, `reset()` mutable-state ownership, architecture tests, permanent coverage ratchets, and no further mechanical split.
- [ ] Require production `NO-GO` to remain documented.
- [ ] Run the focused test and confirm it fails because the closeout still says `OPEN RISK, FURTHER REDUCED`.

### Task 2: Record the resolution

**Files:**
- Modify: `docs/verification/2026-07-23-architecture-audit-closeout.md`
- Create: `docs/verification/2026-07-23-environment-facade-audit-resolution.md`

- [ ] Update the integrated main baseline to PR #152 merge commit.
- [ ] Change the summary row and section heading to `RESOLVED`.
- [ ] Explain why the original deferred-remediation condition is now satisfied.
- [ ] Preserve `reset()` mutable-state ownership and `step()` orchestration as intentional facade responsibilities.
- [ ] Record constructor reduction from 479 audited lines to 150 maintained lines.
- [ ] Record typed service and contract owners, architecture prohibitions, permanent ratchets, and exact-head verification.
- [ ] Remove statements that call `AUD-RL-001` the remaining open maintenance watchpoint.
- [ ] Keep production `NO-GO` and capability limitations unchanged.

### Task 3: Verify and integrate

**Files:**
- Modify: PR body only after evidence exists.

- [ ] Run focused documentation and architecture tests.
- [ ] Run full CI, Ubuntu, Windows, training image/non-root probe, CLI, and PostgreSQL Catalog at the exact final head.
- [ ] Confirm the diff contains only intended docs/tests and no temporary workflow material.
- [ ] Confirm no unresolved review threads.
- [ ] Sync current main if it advances and rerun exact-head verification.
- [ ] Mark ready and squash merge with the expected head SHA.
