# Environment Branch Coverage Ratchet Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Raise and permanently protect branch coverage for the mutable `ResidualMarketEnv` facade without changing runtime behavior.

**Architecture:** Keep mutable Gymnasium state in `ResidualMarketEnv`; add deterministic contract tests around provider, reset, pre-roll, and terminal branches; add a per-file critical branch ratchet.

**Tech Stack:** Python 3.12, NumPy, Gymnasium, Pytest, pytest-cov.

### Task 1: Add RED architecture contract

- Create `tests/rl/test_environment_branch_contract.py`.
- Assert `pyproject.toml` contains an environment branch target of at least 75.0%.
- Add focused branch tests using deterministic synthetic data.
- Run focused tests and confirm RED is caused by the missing ratchet.

### Task 2: Measure and ratchet

- Run the complete suite with branch coverage.
- Confirm the focused tests raise `environment.py` from the prior 56.25% to at least 75.0%.
- Add the measured safe threshold to `pyproject.toml` without lowering any existing target.

### Task 3: Full verification

- Run Ruff, format, Mypy, Import Linter, dead-code report, Serving smoke, all Pytest, critical coverage, CLI smoke, Studio, Ubuntu, Windows, training image/non-root probe, and PostgreSQL.
- Record exact-head evidence in the PR.
- Squash merge only after all gates pass.