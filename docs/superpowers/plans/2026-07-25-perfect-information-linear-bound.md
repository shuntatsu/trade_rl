# Perfect-Information Linear Bound Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fail-closed, research-only linear-programming benchmark that certifies an optimistic perfect-information log-growth bound without changing the maintained PPO training path.

**Architecture:** A public evaluation module validates the problem and independently replays the solution. A private sparse-LP module performs a primary economic optimization and a secondary lexicographic turnover minimization using optional SciPy HiGHS.

**Tech Stack:** Python 3.12, NumPy 1.26, optional SciPy HiGHS, Pytest, pytest-cov, Ruff, MyPy.

## Global Constraints

- Do not modify `examples/binance-multitimeframe/walk-forward-full.json` or the current residual PPO candidate.
- Do not expose future-informed actions to BC or Serving.
- Every production behavior is introduced by a failing test first.
- All solver and numeric failures are fail-closed.
- Arrays returned from public results are read-only.
- The exact replay log return must never exceed the primary linearized bound outside tolerance.

---

### Task 1: Define validated benchmark contracts

**Files:**
- Create: `trade_rl/evaluation/perfect_information_bound.py`
- Test: `tests/evaluation/test_perfect_information_bound_contracts.py`

- [x] Write failing module and configuration tests.
- [x] Verify the tests fail for missing contracts.
- [x] Implement immutable normalized configuration and result contracts.
- [x] Add canonical configuration, problem, and result identities.
- [x] Verify contract and digest tests pass.

### Task 2: Build the primary and lexicographic HiGHS programs

**Files:**
- Create: `trade_rl/evaluation/_perfect_information_lp.py`
- Modify: `pyproject.toml`
- Test: `tests/evaluation/test_perfect_information_bound_solver.py`

- [x] Add failing monotonic, flat-market, exposure, switching-cost, and infeasibility tests.
- [x] Implement lazy SciPy loading and sparse LP construction.
- [x] Implement the primary economic optimum.
- [x] Implement the secondary minimum-turnover solve constrained to the primary tolerance.
- [x] Add optional dependency `oracle = ["scipy>=1.14,<1.18"]`.
- [x] Verify focused solver tests pass.

### Task 3: Add independent replay and fail-closed evidence checks

**Files:**
- Modify: `trade_rl/evaluation/perfect_information_bound.py`
- Test: `tests/evaluation/test_perfect_information_bound_fail_closed.py`

- [x] Add failing tests for non-positive wealth, LP/replay disagreement, primary-objective drift, invalid bound ordering, constraint violations, missing SciPy, and malformed solver vectors.
- [x] Implement independent turnover, costs, simple-return, log-return, and constraint reconstruction.
- [x] Reject every inconsistent solver or replay state.
- [x] Verify fail-closed tests pass.

### Task 4: Add mathematical verification

**Files:**
- Test: `tests/evaluation/test_perfect_information_bound_solver.py`

- [x] Add a tiny brute-force grid comparison.
- [x] Add 50 deterministic randomized small problems.
- [x] Verify all exposure constraints, objective tolerance, and upper-bound inequalities.
- [x] Achieve 100% statement and branch coverage for both new production modules.

### Task 5: Publish API and run gates

**Files:**
- Modify: `trade_rl/evaluation/__init__.py`
- Modify: `pyproject.toml`
- Create: `docs/superpowers/specs/2026-07-25-perfect-information-linear-bound-design.md`
- Create: `docs/superpowers/plans/2026-07-25-perfect-information-linear-bound.md`

- [x] Add and verify the public import test.
- [x] Verify focused Pytest: 52 passed.
- [x] Verify focused branch coverage: 100% for both new modules.
- [x] Verify Python compilation succeeds.
- [x] Run `ruff check trade_rl/evaluation/perfect_information_bound.py trade_rl/evaluation/_perfect_information_lp.py tests/evaluation/test_perfect_information_bound_*.py` in a complete development environment.
- [x] Run `ruff format --check trade_rl/evaluation/perfect_information_bound.py trade_rl/evaluation/_perfect_information_lp.py tests/evaluation/test_perfect_information_bound_*.py`.
- [x] Run repository-wide Mypy, including both new modules.
- [x] Run `pytest tests/evaluation -q` in the complete repository checkout.
- [x] Run `pytest -q` in the complete repository checkout: 1464 passed, 2 skipped.
- [x] Pass critical branch-coverage ratchets, CLI smoke, Windows/Ubuntu compatibility, training-image build, packaged runtime probe, import architecture, dead-code checks, and PostgreSQL catalog gates on the exact PR head.
