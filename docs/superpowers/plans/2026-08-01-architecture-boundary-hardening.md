# Architecture Boundary Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove Studio-to-workflow and Serving-to-export-implementation coupling without changing runtime behavior or implementing Stage B.

**Architecture:** Move reusable configuration parsing into the framework-independent RL contract layer, move structured export manifest validation into the artifact layer, and retain compatibility re-exports at existing public import paths. Enforce the new boundaries with Import Linter and source-level architecture tests.

**Tech Stack:** Python 3.12, dataclasses, Import Linter, pytest, Ruff, MyPy, GitHub Actions.

## Global Constraints

- Stage B implementation is out of scope.
- Future Stage B market roles are Spot long-only and USDⓈ-M futures short-only.
- No reward, action, execution, training algorithm, evaluation, or serving behavior changes.
- Existing public import paths remain compatible.
- All new dependency rules fail closed in CI.

---

### Task 1: Add failing architecture boundaries

**Files:**
- Create: `tests/architecture/test_config_and_export_boundaries.py`
- Modify: `.importlinter`

**Interfaces:**
- Consumes: current source tree and Import Linter layer names.
- Produces: regression tests forbidding Studio-to-workflow and Serving-to-RL-export coupling.

- [ ] **Step 1: Write tests asserting the intended module ownership**

The tests must assert that `trade_rl/rl/training_run_config.py` and `trade_rl/artifacts/structured_policy_contract.py` exist, Studio config discovery imports the former, Serving does not import `trade_rl.rl.structured_export`, and compatibility imports remain in place.

- [ ] **Step 2: Run the focused tests and confirm RED**

Run: `uv run pytest tests/architecture/test_config_and_export_boundaries.py -q`

Expected: failure because the new modules and dependency boundaries do not yet exist.

- [ ] **Step 3: Commit the RED tests**

```bash
git add tests/architecture/test_config_and_export_boundaries.py .importlinter
git commit -m "test: require neutral config and export boundaries"
```

### Task 2: Move generic field validation below workflows

**Files:**
- Create: `trade_rl/domain/config_fields.py`
- Modify: `trade_rl/workflows/config_fields.py`
- Test: `tests/architecture/test_config_and_export_boundaries.py`

**Interfaces:**
- Produces: `require_exact_fields(...)` and `require_dataclass_fields(...)` in `trade_rl.domain.config_fields`.
- Compatibility: `trade_rl.workflows.config_fields` re-exports both names.

- [ ] **Step 1: Implement the domain helper module by relocating the existing standard-library implementation**
- [ ] **Step 2: Replace the workflow helper with a compatibility re-export**
- [ ] **Step 3: Run focused helper and architecture tests**

Run: `uv run pytest tests/architecture/test_config_and_export_boundaries.py tests/workflows/test_training_run_config.py -q`

- [ ] **Step 4: Commit**

```bash
git add trade_rl/domain/config_fields.py trade_rl/workflows/config_fields.py tests/architecture/test_config_and_export_boundaries.py
git commit -m "refactor: lower configuration field validation"
```

### Task 3: Separate TrainingRunConfig from orchestration

**Files:**
- Create: `trade_rl/rl/training_run_config.py`
- Modify: `trade_rl/workflows/training_run.py`
- Modify: `trade_rl/studio/config_catalog.py`
- Test: existing training-config, Studio catalog, walk-forward, and end-to-end tests.

**Interfaces:**
- Produces: `TRAINING_RUN_CONFIG_SCHEMA` and `TrainingRunConfig` from `trade_rl.rl.training_run_config`.
- Compatibility: `trade_rl.workflows.training_run` imports and exposes both names.

- [ ] **Step 1: Move only parsing, validation, path resolution, and digest identity logic into the new module**
- [ ] **Step 2: Import and re-export the contract from the workflow module**
- [ ] **Step 3: Change Studio config discovery to import the lower contract module**
- [ ] **Step 4: Run focused tests**

Run: `uv run pytest tests/workflows/test_training_run_config.py tests/workflows/test_training_run_transfer_config.py tests/studio -q`

- [ ] **Step 5: Commit**

```bash
git add trade_rl/rl/training_run_config.py trade_rl/workflows/training_run.py trade_rl/studio/config_catalog.py
git commit -m "refactor: separate training config from workflow orchestration"
```

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

- [ ] **Step 1: Move contract-only definitions and decoding into the artifact module**
- [ ] **Step 2: Import the contract into the RL exporter**
- [ ] **Step 3: Point Serving loaders directly at the artifact contract**
- [ ] **Step 4: Run focused tests**

Run: `uv run pytest tests/rl/test_structured_export.py tests/serving/test_structured_policy.py tests/serving/test_structured_ensemble_loader.py -q`

- [ ] **Step 5: Commit**

```bash
git add trade_rl/artifacts/structured_policy_contract.py trade_rl/rl/structured_export.py trade_rl/serving/structured_policy.py trade_rl/serving/policy_loader.py
git commit -m "refactor: neutralize structured export contracts"
```

### Task 5: Document future market-role boundary

**Files:**
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/RESEARCH_STATUS.md`
- Test: documentation contract tests.

**Interfaces:**
- Produces an explicit non-implemented Stage B boundary: Spot long-side book and USDⓈ-M futures short-side book.

- [ ] **Step 1: Add the future market-role statement without claiming implementation**
- [ ] **Step 2: Run documentation tests**

Run: `uv run pytest tests/test_current_documentation_contract.py -q`

- [ ] **Step 3: Commit**

```bash
git add docs/ARCHITECTURE.md docs/RESEARCH_STATUS.md
git commit -m "docs: define future asymmetric market roles"
```

### Task 6: Full verification and PR

**Files:**
- Review all changed files.

**Interfaces:**
- Produces a draft PR with exact-head evidence.

- [ ] **Step 1: Run Ruff and format checks**
- [ ] **Step 2: Run MyPy and Import Linter**
- [ ] **Step 3: Run full pytest with branch coverage**
- [ ] **Step 4: Run Studio typecheck and build through maintained CI**
- [ ] **Step 5: Open a draft PR and record RED/GREEN evidence**

The PR must not be merged automatically.
