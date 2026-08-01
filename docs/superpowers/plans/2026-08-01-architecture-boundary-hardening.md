# Architecture Boundary Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove Studio-to-workflow and Serving-to-export-implementation coupling without changing runtime behavior or implementing Stage B.

**Architecture:** Move reusable configuration parsing into the framework-independent RL contract layer, move structured export manifest validation into the artifact layer, and retain compatibility re-exports at existing public import paths. Enforce the new boundaries with Import Linter and source-level architecture tests.

**Tech Stack:** Python 3.12, dataclasses, Import Linter, pytest, Ruff, MyPy, GitHub Actions.

## Global Constraints

- Stage B implementation is out of scope.
- Future Stage B market roles are Spot long-only and USDⓈ-M futures short-only.
- No reward, action, execution, training algorithm, evaluation, or serving behavior changes.
- Existing maintained public import paths remain compatible.
- All new dependency rules fail closed in CI.

---

### Task 1: Add failing architecture boundaries

**Files:**
- Create: `tests/architecture/test_config_and_export_boundaries.py`
- Modify: `.importlinter`

**Interfaces:**
- Consumes: current source tree and Import Linter layer names.
- Produces: regression tests forbidding Studio-to-workflow and Serving-to-RL-export coupling.

- [x] **Step 1: Write tests asserting the intended module ownership**

The tests assert that `trade_rl/rl/training_run_config.py` and `trade_rl/artifacts/structured_policy_contract.py` exist, Studio config discovery imports the former, Serving does not import `trade_rl.rl.structured_export`, and compatibility imports remain in place.

- [x] **Step 2: Run the focused tests and confirm RED**

Run: `uv run pytest tests/architecture/test_config_and_export_boundaries.py -q`

Observed before implementation: four expected failures for the missing lower-level modules and future market-role documentation.

- [x] **Step 3: Commit the RED tests**

### Task 2: Move generic field validation below workflows

**Files:**
- Create: `trade_rl/domain/config_fields.py`
- Modify: `trade_rl/workflows/config_fields.py`
- Test: `tests/architecture/test_config_and_export_boundaries.py`

**Interfaces:**
- Produces: `require_exact_fields(...)` and `require_dataclass_fields(...)` in `trade_rl.domain.config_fields`.
- Compatibility: `trade_rl.workflows.config_fields` re-exports both names.

- [x] **Step 1: Implement the domain helper module by relocating the existing standard-library implementation**
- [x] **Step 2: Replace the workflow helper with a compatibility re-export**
- [x] **Step 3: Run focused helper and architecture tests**
- [x] **Step 4: Commit**

### Task 3: Separate TrainingRunConfig from orchestration

**Files:**
- Create: `trade_rl/rl/training_run_config.py`
- Modify: `trade_rl/workflows/training_run.py`
- Modify: `trade_rl/studio/config_catalog.py`
- Test: existing training-config, Studio catalog, walk-forward, and end-to-end tests.

**Interfaces:**
- Produces: `TRAINING_RUN_CONFIG_SCHEMA` and `TrainingRunConfig` from `trade_rl.rl.training_run_config`.
- Compatibility: `trade_rl.workflows.training_run` continues to expose `TrainingRunConfig` while orchestration imports the lower contract.

- [x] **Step 1: Move only parsing, validation, path resolution, and digest identity logic into the new module**
- [x] **Step 2: Import and re-export the maintained config class from the workflow module**
- [x] **Step 3: Change Studio config discovery to import the lower contract module**
- [x] **Step 4: Run focused tests**
- [x] **Step 5: Commit**

### Task 4: Separate structured export contract from Torch implementation

**Files:**
- Create: `trade_rl/artifacts/structured_policy_contract.py`
- Modify: `trade_rl/rl/structured_export.py`
- Modify: `trade_rl/serving/structured_policy.py`
- Modify: `trade_rl/serving/policy_loader.py`
- Test: structured export and serving tests.

**Interfaces:**
- Produces neutral schema constants, `StructuredInputSpec`, `StructuredExportManifest`, and manifest loaders.
- The RL module continues to export those names for compatibility while retaining Torch export functions.

- [x] **Step 1: Move contract-only definitions and decoding into the artifact module**
- [x] **Step 2: Import the contract into the RL exporter**
- [x] **Step 3: Point Serving loaders directly at the artifact contract**
- [x] **Step 4: Run focused tests**
- [x] **Step 5: Commit**

### Task 5: Document future market-role boundary

**Files:**
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/RESEARCH_STATUS.md`
- Test: documentation and architecture contract tests.

**Interfaces:**
- Produces an explicit non-implemented Stage B boundary: Spot long-side book and USDⓈ-M futures short-side book.

- [x] **Step 1: Add the future market-role statement without claiming implementation**
- [x] **Step 2: Run focused documentation and architecture tests**
- [x] **Step 3: Commit**

### Task 6: Full verification and PR

**Files:**
- Review all changed files.

**Interfaces:**
- Produces a draft PR with exact-head evidence.

- [x] **Step 1: Run focused Ruff and format checks**
- [x] **Step 2: Run focused MyPy and Import Linter**
- [ ] **Step 3: Run full pytest with branch coverage on the final PR head**
- [ ] **Step 4: Run Studio typecheck and build through maintained CI on the final PR head**
- [x] **Step 5: Open a draft PR and record RED/focused-GREEN evidence**

Focused verification before the final connector-authored head:

- focused pytest: 229 passed
- Import Linter: 12 contracts kept, 0 broken
- MyPy: no issues in 330 source files
- Ruff and format: passed

The PR must not be merged automatically. Full exact-head CI remains the final gate.
