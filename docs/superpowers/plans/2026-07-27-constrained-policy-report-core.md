# Constrained Policy Report Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use test-driven development and verification-before-completion.

**Goal:** Add a pure typed report core that summarizes constrained-policy fold evidence and computes fail-closed eligibility without touching active workflow, C3, or sequence-policy lanes.

**Architecture:** Immutable input observations are normalized into fold and scenario summaries. Required-scenario eligibility checks member identity, canonical cost closure, model diagnostics, budgets, and completed-episode support. Ordinary PPO preserves explicit absence of constraint evidence.

**Tech Stack:** Python 3.12, dataclasses, existing artifact hashing and Lagrangian statistics contracts, pytest, Ruff, Mypy.

## Constraints

- Modify no existing production file.
- Do not import workflow, CLI, Serving, release, or SB3 integration modules.
- Preserve canonical seven-cost ordering.
- Do not concatenate fold return series.
- Do not produce partial averages for incomplete optional diagnostics.
- Lower-bound multiplier occupancy is not upper-cap saturation.
- Production remains `NO-GO`.

### Task 1: RED typed report contracts

**Files:**
- Create `tests/evaluation/test_constrained_policy_report.py`

- [ ] Define constrained evidence fixtures covering two folds, two required scenarios, two seeds, and one deployable ensemble.
- [ ] Assert deterministic aggregate mean, worst-seed, worst-fold, support, and bound occupancy semantics.
- [ ] Assert budget, support, missing-scenario, and member-identity failures.
- [ ] Assert ordinary PPO uses `constraints=None` and no synthetic penalty diagnostics.
- [ ] Assert lower-bound occupancy alone does not cause rejection.
- [ ] Assert non-finite input fails during construction.
- [ ] Assert report digest is invariant to fold input order.
- [ ] Run focused tests and confirm collection failure because the production module does not exist.

### Task 2: GREEN pure implementation

**Files:**
- Create `trade_rl/evaluation/constrained_policy_report.py`

- [ ] Implement immutable cost, policy observation, and fold evidence inputs.
- [ ] Implement fold/scenario cost summaries and aggregate scenario summaries.
- [ ] Implement stable eligibility reasons for required scenarios.
- [ ] Implement complete-only optional diagnostic aggregation.
- [ ] Implement deterministic payloads and report digest.
- [ ] Run focused Pytest, Ruff, format, and Mypy.

### Task 3: Repository verification

- [ ] Confirm final diff contains only two documents, one module, and one test file.
- [ ] Confirm zero changed-file overlap with active PR #225 and PR #227.
- [ ] Run exact-head CI including full Pytest, critical coverage, Ubuntu, Windows, and Training image.
- [ ] Run PostgreSQL workflow if triggered.
- [ ] Record exact test and coverage evidence in the PR.
- [ ] Squash merge only after all exact-head checks pass.

The follow-up workflow-integration PR will be planned from the merged core and will re-check active lane file ownership before modifying any existing workflow file.
