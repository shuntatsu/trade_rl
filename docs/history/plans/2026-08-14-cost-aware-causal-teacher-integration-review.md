# Cost-aware Causal Teacher Integration Review Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Integrate the remaining `codex/universal-real-data-training` work into current `main`, review the resulting tree, repair any correctness or maintainability defects, and verify the exact final head before merging to `main`.

**Architecture:** Use an integration branch based on current `main` so the reviewed tree includes the already-merged Universal documentation and the newer cost-aware causal-teacher implementation. Preserve the existing target-weight action contract, pure net-log-growth reward, hard risk rules, train-only causality, and fail-closed teacher admission. Review the actual merged tree rather than only the divergent source branch.

**Tech Stack:** Python 3.12, NumPy, Stable-Baselines3 integration, GitHub Actions, Ruff, Mypy, Import Linter, pytest/coverage.

## Global Constraints

- Do not change the scalar reward from pure net-log growth.
- Keep `max_position_to_market_notional=0.02` as a hard downstream risk rule.
- No validation/test-symbol leakage into fitting, selection, BC, or teacher admission.
- Preserve `target_weight` as the policy action semantic and the one-decision execution delay.
- Do not weaken lower-tail, turnover, risk, rejection, BC, or teacher-admission gates to obtain a pass.
- Any liquidity estimate used before execution must be causal and derived only from information available at decision time.
- Do not merge a failing or unreviewed integration head to `main`.

---

### Task 1: Integrate the divergent source branch

**Files:** all files changed by `codex/universal-real-data-training` relative to `main`.

**Interfaces:**
- Consumes: `main@8fe891d7a86fe42a76094dd562b8a144fbd7d8d3` and source branch tip.
- Produces: one integration tree containing both histories without dropping current-main documentation.

- [ ] Merge `codex/universal-real-data-training` into `integration/cost-aware-causal-teacher-review`.
- [ ] Confirm the source branch is no longer ahead of the integration branch.
- [ ] Open an integration-to-main Draft PR so CI evaluates the merged tree.

### Task 2: Review causal liquidity and cost-aware target construction

**Files:**
- `trade_rl/learning/causal_alpha_teacher.py`
- `trade_rl/learning/causal_alpha_diagnostics.py`
- `trade_rl/workflows/universal_causal_alpha_costs.py`
- `trade_rl/workflows/universal_causal_alpha_fitting.py`
- `trade_rl/workflows/universal_causal_alpha_selection.py`
- `trade_rl/workflows/universal_causal_alpha_teacher.py`
- `trade_rl/workflows/universal_causal_alpha_contracts.py`

**Interfaces:**
- Consumes: causal alpha predictions, past-only market/equity context, hard risk configuration.
- Produces: target paths whose pre-execution capacity/cost/no-trade decisions are causal and whose evidence remains reproducible.

- [ ] Verify capacity estimation never reads future execution-bar volume or future equity.
- [ ] Verify predicted/executable capacity is applied before cost hurdle and no-trade decisions while hard execution risk remains authoritative.
- [ ] Verify initial baseline weights, minimum executable targets, zero-fill/no-fill classifications, and liquidity projections cannot produce hidden repeated rebalancing.
- [ ] Verify candidate ranking/rejection remains fail-closed and lower-tail evidence cannot improve by adding later episodes after a recorded minimum.
- [ ] Add a failing regression test first for every defect found, then implement the smallest fix.

### Task 3: Review evidence durability, identity, and monitoring

**Files:**
- `trade_rl/learning/evaluation.py`
- `trade_rl/learning/rollout_evaluation.py`
- `trade_rl/learning/episode_oracle_bc.py`
- `trade_rl/operations/universal_training_monitor.py`
- related tests and research report/spec files.

**Interfaces:**
- Consumes: replay/selection/economic results.
- Produces: resumable, immutable, auditable evidence that does not change training semantics.

- [ ] Verify checkpoint/progress rows are sufficient for safe resume and never mix source-revision/controller semantics.
- [ ] Verify explained no-fills, hard-risk failures, execution rejections, liquidity caps, cost suppression, and zero-trade cases are classified consistently.
- [ ] Verify monitor aggregation is bounded and diagnostics cannot be mistaken for promotion evidence.
- [ ] Add regression tests before any correction.

### Task 4: Exact-head verification and merge

**Files:** no new product behavior unless Task 2/3 found defects.

**Interfaces:**
- Consumes: reviewed integration head.
- Produces: verified `main` merge and cleaned branch state.

- [ ] Run exact-head CI: Ruff, format, Mypy, Import Linter, dead-code, full pytest with branch coverage, critical coverage, package identity, Ubuntu/Windows compatibility, training image, and PostgreSQL catalog where triggered.
- [ ] Self-review the complete `main...integration` diff for architecture, naming, duplication, boundary conditions, leakage, numerical behavior, and temporary assets.
- [ ] Fix any review finding with TDD and rerun the affected checks plus full CI.
- [ ] Merge to `main` only after the exact head is green.
- [ ] Confirm `main` contains the source/integration branch tips, then remove fully contained branches and re-list branches.
