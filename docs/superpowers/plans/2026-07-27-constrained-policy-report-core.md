# Constrained Policy Report Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use test-driven development and verification-before-completion.

**Goal:** Add a pure typed report core that summarizes constrained-policy fold evidence and computes fail-closed eligibility without touching active workflow, C3, or sequence-policy lanes.

**Architecture:** Immutable input observations are normalized into fold and scenario summaries. Required-scenario eligibility checks member identity, canonical cost closure, model diagnostics, budgets, and completed-episode support. Ordinary PPO preserves explicit absence of constraint evidence.

**Tech Stack:** Python 3.12, dataclasses, existing artifact hashing and framework-independent domain constraint contracts, pytest, Ruff, Mypy.

**Current status:** The RED collection failure was confirmed on both compatibility runners. The pure implementation and repository-formatted tests are committed. Ruff, format, Mypy, Ubuntu, Windows, and the Training image passed before Import architecture exposed the forbidden `evaluation -> rl` dependency. Canonical constraint names, aggregation semantics, and units are now being moved to `trade_rl.domain.constraint_contracts`; exact-head full verification remains pending after that boundary migration.

## Constraints

- Existing production files may change only to re-export the domain-owned constraint identity.
- Do not import workflow, CLI, Serving, release, or SB3 integration modules.
- Preserve canonical seven-cost ordering and existing RL public API object identity.
- Do not concatenate fold return series.
- Do not produce partial averages for incomplete optional diagnostics.
- Lower-bound multiplier occupancy is not upper-cap saturation.
- Production remains `NO-GO`.

### Task 1: RED typed report contracts

**Files:**
- Create `tests/evaluation/test_constrained_policy_report.py`

- [x] Define constrained evidence fixtures covering two folds, two required scenarios, two seeds, and one deployable ensemble.
- [x] Assert deterministic aggregate mean, worst-seed, worst-fold, support, and bound occupancy semantics.
- [x] Assert budget, support, missing-scenario, and member-identity failures.
- [x] Assert ordinary PPO uses `constraints=None` and no synthetic penalty diagnostics.
- [x] Assert lower-bound occupancy alone does not cause rejection.
- [x] Assert non-finite input fails during construction.
- [x] Assert report digest is invariant to fold input order.
- [x] Run focused tests and confirm collection failure because the production module does not exist.

### Task 2: GREEN pure implementation

**Files:**
- Create `trade_rl/evaluation/constrained_policy_report.py`
- Create `trade_rl/domain/constraint_contracts.py`
- Update `trade_rl/rl/environment_constraints.py`
- Update `trade_rl/rl/lagrangian_statistics.py`

- [x] Implement immutable cost, policy observation, and fold evidence inputs.
- [x] Implement fold/scenario cost summaries and aggregate scenario summaries.
- [x] Implement stable eligibility reasons for required scenarios.
- [x] Implement complete-only optional diagnostic aggregation.
- [x] Implement deterministic payloads and report digest.
- [x] Add domain-owned canonical constraint metadata and identity tests.
- [ ] Switch the environment, Lagrangian statistics, and report imports to the domain contract.
- [ ] Run focused Pytest, Ruff, format, Mypy, and Import architecture.

### Task 3: Repository verification

- [ ] Confirm final diff contains only the approved documents, domain contract, report core, two existing-file re-exports, and focused tests.
- [ ] Confirm zero changed-file overlap with active PR #227.
- [ ] Run exact-head CI including full Pytest, critical coverage, Ubuntu, Windows, and Training image.
- [ ] Run PostgreSQL workflow if triggered.
- [ ] Record exact test and coverage evidence in the PR.
- [ ] Squash merge only after all exact-head checks pass.

The follow-up workflow-integration PR will be planned from the merged core and will re-check active lane file ownership before modifying any existing workflow file.
