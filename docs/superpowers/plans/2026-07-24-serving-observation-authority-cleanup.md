# Serving Observation Authority Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the unused Serving observation pipeline and lock the actual Bundle/Runtime observation authority in an architecture test.

**Architecture:** `load_serving_bundle()` remains the normalizer artifact verifier; `ServingRuntime._predict_action()` remains the flat/structured observation gate. No replacement abstraction is introduced.

**Tech Stack:** Python 3.12, Pytest, existing Serving bundle/runtime contracts.

### Task 1: Add RED architecture contract

- Create `tests/architecture/test_serving_observation_authority.py`.
- Assert the obsolete module does not exist and no production source imports or names it.
- Assert bundle and runtime source retain their maintained responsibilities.
- Run the focused test and confirm RED because `trade_rl/serving/observations.py` still exists.

### Task 2: Remove duplicate authority

- Delete `trade_rl/serving/observations.py`.
- Run focused Serving and architecture tests.
- Confirm no public import, bundle, runtime, or observation-parity regression.

### Task 3: Full verification

- Run Ruff, format, Mypy, Import Linter, dead-code report, Serving smoke, all Pytest, critical coverage, CLI smoke, Studio, Ubuntu, Windows, and training-image/non-root probe.
- Record exact-head evidence in the PR.
- Squash merge only after all applicable gates pass.