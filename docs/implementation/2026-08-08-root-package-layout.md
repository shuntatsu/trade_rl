# Root Python Package Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the redundant `src/` directory while preserving the installed/import namespace `trade_rl.*` and all runtime, artifact, checkpoint, financial, and CI contracts.

**Architecture:** The repository root directly owns the Python package at `trade_rl/`. The browser UI remains `frontend/`, repository helper commands remain `scripts/`, and tests remain `tests/`. This slice changes physical source location only; it does not rename internal Python packages or alter behavior.

**Tech Stack:** Python 3.12, uv, pytest, Ruff, Mypy, import-linter, Docker, GitHub Actions, Vite/React/TypeScript.

## Global Constraints

- Preserve all `trade_rl.*` import paths.
- Do not change trading, accounting, reward, policy, funding, execution, or time semantics.
- Do not reinterpret persisted artifacts, checkpoints, schemas, manifests, or wire formats.
- Do not add compatibility shim directories for `src/`.
- Do not reduce lint, type, architecture, coverage, compatibility, or CI thresholds.
- Keep `frontend/`, `trade_rl/`, `tests/`, `scripts/`, `docs/`, and `examples/` as explicit repository roots.
- Production authority remains legacy-authoritative / NO-GO.
- PR #366 may target `main` temporarily only to trigger the existing `pull_request: main` CI; restore its stacked base `refactor/repository-layout-v1` before merge-readiness review.

---

### Task 1: Make repository layout contract require root `trade_rl/`

**Files:**
- Modify: `tests/architecture/test_repository_layout.py`

**Interfaces:**
- Consumes: repository root filesystem layout.
- Produces: a contract requiring `trade_rl/` and forbidding `src/`.

- [ ] **Step 1: Write the failing contract**

Change the required roots to include `ROOT / "trade_rl"` and the forbidden roots to include `ROOT / "src"`.

- [ ] **Step 2: Run the narrow compatibility/architecture path and verify RED**

Expected failure: root `trade_rl/` does not exist and/or `src/` still exists.

- [ ] **Step 3: Commit the RED contract separately**

Commit message: `test: require root Python package layout`.

### Task 2: Move the Python package and update package discovery

**Files:**
- Move: `src/trade_rl/**` -> `trade_rl/**`
- Modify: `pyproject.toml`
- Modify: Dockerfiles and build contexts that refer to `src/trade_rl`
- Modify: repository scripts/configuration that inspect physical source paths.

**Interfaces:**
- Consumes: unchanged `trade_rl.*` Python namespace.
- Produces: importable/installable `trade_rl` package from repository root.

- [ ] **Step 1: Move the Git tree without changing Python file contents**

Perform a tree-level move so source bytes remain unchanged except for files that explicitly encode repository paths.

- [ ] **Step 2: Update package discovery and physical-path configuration**

Remove src-layout package discovery and point physical-path checks to `trade_rl/`.

- [ ] **Step 3: Run the narrow layout/import/package checks until GREEN**

Verify repository layout, package import, and package identity on the changed head.

- [ ] **Step 4: Commit the move separately**

Commit message: `refactor: move Python package to repository root`.

### Task 3: Remove stale `src/trade_rl` knowledge from tests, CI, and docs

**Files:**
- Modify: `.github/workflows/*.yml` where physical paths are referenced.
- Modify: `tests/architecture/**` and other tests that open source files by physical path.
- Modify: `tests/examples/**` where source files are inspected.
- Modify: `README.md`, `ARCHITECTURE.md`, and active docs that describe repository layout.

**Interfaces:**
- Consumes: `trade_rl/` as the physical Python source root.
- Produces: one canonical physical path with no stale `src/trade_rl` assumptions.

- [ ] **Step 1: Search for physical-path references**

Classify each `src/trade_rl` occurrence as physical path, documentation, generated/history text, or intentionally external text. Do not replace import namespace strings.

- [ ] **Step 2: Update only physical-path knowledge**

Use shared path helpers in architecture tests where available; do not duplicate repository-root calculation.

- [ ] **Step 3: Run related tests, Ruff, format, Mypy, and import-linter**

All checks must pass without exclusions or threshold reductions.

### Task 4: Final same-head verification and reviewer pass

**Files:**
- Review all changed files.

**Interfaces:**
- Produces: merge-ready evidence for this layout slice only.

- [ ] **Step 1: Inspect the full diff**

Check for unintended source-content changes, generated files, debug output, stale compatibility shims, and stale `src/` references.

- [ ] **Step 2: Run full required checks on the same final head**

Required: Ruff, Ruff format, Mypy, import-linter, dead-code, serving smoke, frontend tests/typecheck/build/layout, Ubuntu/Windows compatibility, training image, full pytest/branch coverage, critical coverage, package identity, Nautilus Capability, PostgreSQL Catalog, and Sequence Projection Stability.

- [ ] **Step 3: Report only observed results**

Do not claim completion if any final-head check is pending, skipped unexpectedly, or failing.
