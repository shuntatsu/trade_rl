# Paper Reconciliation Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a typed, content-addressed paper-trading reconciliation artifact and require verified reconciliation evidence before selected-final Serving packaging.

**Architecture:** Keep normalized external paper logs outside the repository and accept one sealed summary artifact. The evaluation layer owns artifact validation and release-tolerance policy; the Serving package layer only binds the verified artifact to confirmation and training identities and copies it into immutable bundle closure.

**Tech Stack:** Python 3.12, dataclasses, canonical content hashing, pytest, Ruff, MyPy, GitHub Actions.

## Global Constraints

- Production status remains `NO-GO`.
- Direct exchange routing remains `NOT_IMPLEMENTED`.
- Do not add broker credentials, private signing keys, or network order operations.
- Reconciliation `passed` must be derived from retained observations, never trusted as an arbitrary caller assertion.
- Release tolerances for position-notional, cash, and equity relative differences must each be no greater than `1e-6`.
- Missing, malformed, failed, identity-mismatched, or over-tolerant evidence must fail closed.

---

### Task 1: Add the reconciliation artifact contract

**Files:**
- Create: `trade_rl/evaluation/paper_reconciliation.py`
- Create: `tests/evaluation/test_paper_reconciliation.py`
- Modify: `trade_rl/evaluation/__init__.py`

**Interfaces:**
- Produces: `PaperReconciliationEvidence.create(...)`, `PaperReconciliationEvidence.require_promotable()`, `load_paper_reconciliation_evidence(path)`, and `write_paper_reconciliation_evidence(path, evidence)`.
- Consumes: `trade_rl.artifacts.codec.canonical_json_bytes`, `trade_rl.artifacts.hashing.content_digest`, and domain digest/date validators.

- [ ] **Step 1: Write failing construction and round-trip tests**

Create tests that construct exact-count, zero-difference evidence, assert `passed is True`, write/load it, and assert equality.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
uv run pytest tests/evaluation/test_paper_reconciliation.py -q
```

Expected: collection fails because `trade_rl.evaluation.paper_reconciliation` does not exist.

- [ ] **Step 3: Implement the minimal immutable artifact**

Implement strict field validation, derived conditions, digest recomputation, immutable write behavior, and strict load-field closure.

- [ ] **Step 4: Add tamper and promotion-policy tests**

Cover unknown fills, mismatched counts, changed observations with stale digest, tolerance caps above `1e-6`, malformed JSON, and overwrite refusal.

- [ ] **Step 5: Run focused tests and verify GREEN**

```bash
uv run pytest tests/evaluation/test_paper_reconciliation.py -q
uv run ruff check trade_rl/evaluation/paper_reconciliation.py tests/evaluation/test_paper_reconciliation.py
uv run ruff format --check trade_rl/evaluation/paper_reconciliation.py tests/evaluation/test_paper_reconciliation.py
uv run mypy trade_rl/evaluation/paper_reconciliation.py
```

Expected: all commands pass.

### Task 2: Require reconciliation at Serving packaging

**Files:**
- Modify: `trade_rl/serving/package.py`
- Modify: `tests/serving/test_package.py`

**Interfaces:**
- Consumes: `load_paper_reconciliation_evidence()` and `PaperReconciliationEvidence.require_promotable()`.
- Produces: `package_selected_training_run(..., paper_reconciliation_path: Path | None = None)` with sibling-file fallback to `paper-reconciliation.json`.

- [ ] **Step 1: Write failing package tests**

Update the valid fixture to create reconciliation evidence and bind its digest into confirmation. Add cases for missing evidence, confirmation-digest mismatch, identity mismatch, and a correctly digested but failed report.

- [ ] **Step 2: Run focused tests and verify RED**

```bash
uv run pytest tests/serving/test_package.py -q
```

Expected: new tests fail because packaging does not load or validate the reconciliation artifact.

- [ ] **Step 3: Implement fail-closed packaging**

Resolve the explicit path or sibling default, load and require promotable evidence, compare all confirmation/training identities and interval boundaries, copy the verified artifact into the staged bundle, and include it in artifact closure.

- [ ] **Step 4: Run focused tests and verify GREEN**

```bash
uv run pytest tests/serving/test_package.py tests/evaluation/test_paper_reconciliation.py -q
```

Expected: all focused tests pass.

### Task 3: Document and verify the independent PR

**Files:**
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/RESEARCH_STATUS.md`
- Verify all changed Python and test files.

**Interfaces:**
- Produces: truthful current-state documentation stating that typed paper reconciliation is required by Serving packaging while production remains `NO-GO` pending real evidence.

- [ ] **Step 1: Update current documentation**

Describe the schema, derived pass conditions, release tolerance cap, identity binding, and bundle file closure. Do not claim that real reconciliation evidence exists.

- [ ] **Step 2: Run complete verification**

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run lint-imports
uv run pytest --cov=trade_rl --cov-branch
npm test --prefix studio -- --run
npm run typecheck --prefix studio
npm run build --prefix studio
npm run check:layout --prefix studio
```

Expected: all commands pass at the exact PR head.

- [ ] **Step 3: Inspect and publish**

```bash
git diff main...HEAD --check
git diff --stat main...HEAD
```

Open one Draft PR. Record the RED and GREEN head SHAs and exact-head workflow runs. Mark ready and squash-merge only after all required checks succeed.
