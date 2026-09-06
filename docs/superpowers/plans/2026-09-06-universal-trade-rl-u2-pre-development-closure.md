# Universal Trade RL U2 Pre-Development Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a metadata-only U2 pre-development contract and Development lock that freeze research role cardinality, common evaluation RNG, exact Selection formulas, segmented panel bootstrap semantics, non-resume semantics, training exposure evidence, and exact checkpoint/runtime identity before Development numeric access.

**Architecture:** Add one focused workflow module that owns only pre-Development research closure and lock identity. Do not modify U0/U1 economics or generic universe semantics. Existing deterministic replay remains the economic execution authority; the new contract derives a scope-common evaluation seed from immutable U2/scope identity and future Selection/Development orchestration must consume this contract before numeric evaluation.

**Tech Stack:** Python 3.12, dataclasses, existing `content_digest`, U0/U1/U2 artifact contracts, pytest, Ruff, MyPy, GitHub Actions.

**Spec:** `docs/implementation-plans/specs/2026-09-06-universal-trade-rl-u2-pre-development-closure-amendment.md`

## Global Constraints

- Production remains `NO-GO`.
- Admission remains `SEALED`.
- Development numeric evaluation remains unopened while implementing/verifying this plan.
- No U0/U1 reward/action/risk/execution/accounting change.
- No PPO recipe, architecture, seed set, final timestep, time-partition, or economic-threshold change.
- U2 role minimums are Train `>=9`, Development `>=3`, Admission `>=3`.
- Development evaluation RNG is independent of candidate training seed and is scope-common.
- Exact mid-episode PPO resume is unsupported; Selection-authoritative interrupted members restart from timestep zero.

---

### Task 1: RED contract tests

**Files:**
- Create: `tests/workflows/test_universal_trade_rl_u2_predevelopment.py`
- No production files in this task.

**Interfaces:**
- Expects future module `trade_rl.workflows.universal_trade_rl_u2_predevelopment`.
- Expects `build_universal_trade_rl_u2_predevelopment_contract`, `universal_trade_rl_u2_evaluation_seed`, `build_universal_trade_rl_u2_development_lock`.

- [ ] **Step 1: Write failing role-cardinality tests**

Create a synthetic frozen U0 manifest helper and assert the future pre-development builder rejects each of:

```python
(train_count, development_count, admission_count) in (
    (8, 3, 3),
    (9, 2, 3),
    (9, 3, 2),
)
```

before any numeric-loader surface exists.

- [ ] **Step 2: Write failing common-RNG tests**

Assert:

```python
seed_a = universal_trade_rl_u2_evaluation_seed(
    u2_contract_digest="a" * 64,
    scope_digest="b" * 64,
)
seed_b = universal_trade_rl_u2_evaluation_seed(
    u2_contract_digest="a" * 64,
    scope_digest="b" * 64,
)
assert seed_a == seed_b
assert seed_a in (0, 1, 2)
```

and the API has no training-seed/checkpoint input surface.

- [ ] **Step 3: Write failing exact metric/bootstrap contract tests**

Assert the pre-development artifact carries canonical payloads for:

```text
positive scope strict > 0
CVaR K=max(1, ceil(.10*N)) on leaf net log growth
turnover/day = turnover_total / (decision_count*.25/24)
p95 method=linear
meaningful execution = executed_change_count>0 OR turnover_total>1e-6
retention = balanced net log growth / balanced gross log growth
symbol reduction = equal-weight mean
seed reduction = median
D1/D2 bootstrap segments cannot cross
resamples=2000, seed=0, confidence=.95
```

- [ ] **Step 4: Write failing resume/training-exposure tests**

Assert exact mid-episode resume is false, timestep-zero restart is true, and the required exposure fields are exact/canonical.

- [ ] **Step 5: Write failing Development-lock tests**

Assert lock construction requires exactly seeds `(0,1,2)`, canonical symbol/dataset mappings, all SHA-256 identities, and zero Development/Admission numeric-open counts. Reordered/duplicate/missing seeds and non-zero counts fail closed.

- [ ] **Step 6: Verify RED on exact commit**

Run via the `U2 Contracts` GitHub Actions workflow and require failure because `trade_rl.workflows.universal_trade_rl_u2_predevelopment` does not exist. A syntax/import failure unrelated to the missing contract is not acceptable RED evidence.

---

### Task 2: GREEN pre-development contract

**Files:**
- Create: `trade_rl/workflows/universal_trade_rl_u2_predevelopment.py`
- Test: `tests/workflows/test_universal_trade_rl_u2_predevelopment.py`

**Interfaces:**
- Consumes `UniversalTradeRLUniverseManifest`, `UniversalTradeRLU2Contract`, `content_digest`, `require_sha256`.
- Produces:
  - `UniversalTradeRLU2PreDevelopmentContract`
  - `UniversalTradeRLU2DevelopmentLock`
  - `build_universal_trade_rl_u2_predevelopment_contract(...)`
  - `universal_trade_rl_u2_evaluation_seed(...) -> int`
  - `build_universal_trade_rl_u2_development_lock(...)`

- [ ] **Step 1: Implement canonical role counts and machine payloads**

Use immutable constants for role minimums and exact metric/bootstrap/resume/exposure payloads. All nested payloads are content-addressed by the pre-development artifact digest; no numeric market arrays or loaders are accepted.

- [ ] **Step 2: Implement scope-common evaluation seed**

Use:

```python
material = {
    "schema_version": "universal_trade_rl_u2_evaluation_crn_v1",
    "u2_contract_digest": u2_contract_digest,
    "scope_digest": scope_digest,
    "allowed_evaluation_seeds": (0, 1, 2),
}
index = int(content_digest(material)[:8], 16) % 3
return (0, 1, 2)[index]
```

Do not accept training seed or checkpoint identity.

- [ ] **Step 3: Implement strict canonical codecs**

Both artifacts use frozen dataclasses, exact schema versions, exact-key `from_payload`, SHA validation, deterministic tuple ordering, and self-recomputed artifact digests.

- [ ] **Step 4: Implement Development lock**

Require exact checkpoint mapping:

```python
((0, seed0_digest), (1, seed1_digest), (2, seed2_digest))
```

and sorted unique evaluation dataset mappings. Require `development_numeric_open_count == 0` and `admission_numeric_open_count == 0`.

- [ ] **Step 5: Run targeted GREEN**

`uv run pytest tests/workflows/test_universal_trade_rl_u2_predevelopment.py -q`

Expected: all pass.

- [ ] **Step 6: Static verification**

Run Ruff, Ruff format check, and MyPy on the new production/test files.

---

### Task 3: Bind the new contract into U2 focused CI and documentation authority

**Files:**
- Modify: `.github/workflows/u2-contracts.yml`
- Modify: `docs/implementation-plans/plans/2026-09-03-universal-trade-rl-u2-base-ppo-selection.md`
- Modify or annotate: `docs/implementation-plans/specs/2026-09-05-universal-trade-rl-u2-development-replay-seed-amendment.md`

**Interfaces:**
- Consumes the Task 2 tests/artifacts.
- Produces exact-head focused CI evidence and removes the obsolete `evaluation_seed = candidate training seed` rule from normative authority.

- [ ] **Step 1: Add the new test to focused U2 CI**

Add `tests/workflows/test_universal_trade_rl_u2_predevelopment.py` to the exact U2 contract test list. Do not weaken existing test coverage.

- [ ] **Step 2: Supersede the old replay-seed amendment**

Add a top-level notice that the 2026-09-06 pre-development closure supersedes only the old seed-coupling rule; all same-scope/RNG-isolation requirements remain.

- [ ] **Step 3: Update the implementation plan gate order**

Normatively require:

```text
real U0/U1 freeze
→ pre-development contract freeze
→ real seed 0/1/2 training
→ exact-final checkpoint + exposure evidence closure
→ Development lock publication
→ Development numeric open
→ immutable Selection
```

and mark exact mid-episode resume as unsupported/restart-from-zero.

---

### Task 4: Verification and falsification review

**Files:** No production changes unless a defect is found. Implementation-time falsification did find and close three additional defects before Development numeric opening: authoritative-lock enforcement at the numeric session boundary, same-path gross/net replay evidence, and removal of the unlocked Development builder from the supported public API.

- [ ] **Step 1: Targeted tests**

Run the new pre-development tests plus existing U2 contract/evaluation/replay tests.

- [ ] **Step 2: Static/type/architecture**

Run the same Ruff/format/MyPy/import-architecture checks required by the U2 Contracts workflow.

- [ ] **Step 3: Full repository gate**

Require exact-head `U2 Contracts`, `CI`, `PostgreSQL Catalog`, and `Nautilus Capability` success. A success from an older SHA is not evidence for the final HEAD.

- [ ] **Step 4: Falsification review**

Reconstruct from the amendment rather than implementation assumptions and explicitly try to prove:

- a small real role set can pass;
- changing training seed changes evaluation RNG for the same scope;
- baseline and candidate can use different scope RNG;
- changing metric/quantile/bootstrap semantics leaves artifact identity unchanged;
- a block can cross D1/D2;
- exact-resume can be claimed;
- missing/extra/reordered checkpoints can lock Development;
- Development/Admission numeric opens can occur before lock;
- an unlocked numeric Development builder remains in the supported public API;
- gross economics can be generated from a different replay than net economics;
- an existing U0/U1 economic contract changed.

Fix any reproducible issue, then rerun targeted and exact-head gates.

- [ ] **Step 5: Final evidence**

Record final HEAD, diff, workflow runs, unresolved limitations, and the explicit guarantee boundary: this closes the pre-Development contract surface but does not prove profitability, Admission success, hardware-cross-version bitwise training reproducibility, or Production readiness.