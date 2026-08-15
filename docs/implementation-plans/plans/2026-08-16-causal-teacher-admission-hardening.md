# Causal Teacher Admission Hardening Implementation Plan

**Goal:** Harden the maintained pre-BC teacher admission gate, reuse it as the V3 base economic gate, invalidate stale admission evidence, and remove admission-related dead compatibility code.

**Design:** `docs/implementation-plans/specs/2026-08-16-causal-teacher-admission-hardening-design.md`

## Quality contract

### Objective

Prevent economically broken teachers from reaching BC while preserving the deliberate distinction between lightweight teacher admission and statistically stricter post-BC gates.

### Acceptance criteria

- Common admission rejects aggregate after-cost net return below zero.
- Common admission rejects any symbol holdout below `-0.05` net return.
- Common admission rejects zero total trades.
- Existing aggregate-gross and majority-negative-gross checks remain.
- V3 uses the common gate for base economics and keeps hard-risk/unexplained-execution checks.
- Old common admission schema is rejected by current Universal pretraining evidence validation.
- Existing durable V3 per-symbol V2 admission records remain reusable without replay.
- Obsolete V3 v1 admission record/store code with no maintained runtime call path is removed.
- No signal/selection/reward/risk/execution/BC/PPO numerical semantics change.

### Invariants

- One untouched holdout per train symbol; no new bootstrap or holdout sampling.
- Holdout evidence never participates in model/candidate/threshold selection.
- V3 admission remains fail-closed for hard risk and unexplained execution rejection.
- Immutable artifact identity remains deterministic.

### Failure modes

- Cost turns gross-positive teacher net-negative.
- Profitable peers mask one catastrophic symbol.
- Teacher produces no meaningful trades.
- Old evidence is reused under a new gate contract.
- Refactor accidentally drops V3-specific risk/rejection checks.
- Cleanup removes a still-reachable public/runtime dependency.

### Test oracle

Observe `passed`, rejection reasons, aggregate/worst-symbol/trade-count evidence, schema version/digest, no replay of persisted V3 admission records, and unchanged V3-specific rejection counts.

### Required test layers

Unit, integration, workflow regression, static analysis/type/import checks, dead-code analysis, full pytest/coverage, build/compatibility CI.

### Quality gate

No completion claim until exact final HEAD has required GitHub Actions/required checks completed successfully or any unavailable checks are explicitly reported as unverified.

---

## Task 1: Red tests for the stronger common gate

**Files:**
- Modify `tests/learning/test_causal_alpha_teacher_admission.py`
- Modify `tests/integrations/test_universal_causal_teacher_admission_reuse.py`

1. Add regressions for aggregate-net-negative, per-symbol catastrophic net loss, zero-trade, exact `-0.05` boundary, and evidence schema v2.
2. Add a stale-v1 evidence rejection test at the Universal pretraining bundle boundary.
3. Push tests without production changes and confirm the intended failures.

## Task 2: Implement the common gate and artifact revision

**Files:**
- Modify `trade_rl/learning/causal_alpha_teacher.py`
- Modify `trade_rl/integrations/universal_pretraining.py`

1. Add the new checks and observed evidence needed for auditability.
2. Bump common admission evidence schema to v2.
3. Require v2 at the pretraining evidence boundary.
4. Run targeted checks and confirm green.

## Task 3: Connect V3 to the common gate

**Files:**
- Modify `trade_rl/workflows/universal_causal_alpha_v3_admission.py`
- Modify V3 tests under `tests/workflows/`.

1. Use `CausalAlphaV3AdmissionRecordV2.to_holdout_metric()` to feed the common evaluator.
2. Derive V3 base rejection reasons/aggregate evidence from the common result.
3. Keep only V3-specific hard-risk and unexplained-execution checks locally.
4. Revise V3 aggregate evidence schema so stale aggregate evidence cannot be confused with the stronger contract.
5. Verify persisted per-symbol V2 admission records still avoid replay.

## Task 4: Remove dead admission compatibility code

**Candidate files:**
- `trade_rl/workflows/universal_causal_alpha_v3_contracts.py`
- `trade_rl/workflows/universal_causal_alpha_v3_store.py`
- related tests/exports if applicable.

1. Trace all maintained runtime imports/calls after Task 3.
2. Remove legacy V3 v1 admission record/store APIs only if no maintained path requires them.
3. Remove redundant helper/property surfaces made unnecessary by the new connection only when production/tests no longer rely on them as a meaningful contract.
4. Do not remove unrelated historical documentation or unrelated compatibility contracts.

## Task 5: Falsification and whole-branch verification

1. Re-read the final diff against the quality contract.
2. Search for bypasses: stale schema acceptance, alternate admission evaluator, duplicate gross/net logic, record replay on resume, or dead v1 APIs.
3. Run/inspect targeted tests, full test suite, Ruff/format, Mypy, import architecture, vulture/dead-code report, package/build/compatibility checks.
4. Inspect exact-head CI and required checks.
5. Update PR description to reflect the expanded actual scope, verification, remaining risks, and exact HEAD.
