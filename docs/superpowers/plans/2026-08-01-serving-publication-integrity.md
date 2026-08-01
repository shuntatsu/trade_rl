# Serving Publication Integrity Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make serving publication reject changed training artifacts and incomplete structured-policy bundles before atomic publication.

**Architecture:** Keep publication orchestration in `trade_rl.workflows.release_packaging`. Add a private verified-copy primitive that validates each `RunFile` identity before and after copying, and strengthen the existing sequence-policy branch so the structured loader is mandatory and identity-compatible.

**Tech Stack:** Python 3.12, pathlib, hashlib, os/fsync, pytest, GitHub Actions.

## Global Constraints

- Do not alter policy outputs, training algorithms, evaluation metrics, or serving bundle schema v6.
- Preserve atomic staging cleanup and immutable output-root behavior.
- Reject symlinks, root escapes, size drift, digest drift, and missing structured loaders fail-closed.
- Write failing regression tests before production changes.

---

### Task 1: Add release-publication integrity regressions

**Files:**
- Create: `tests/workflows/test_release_packaging_integrity.py`
- Read: `tests/serving/test_package.py`
- Read: `trade_rl/workflows/release_packaging.py`

**Interfaces:**
- Consumes: `package_selected_training_run(...) -> ServingBundleManifest`
- Produces: Regression expectations for missing structured loader and source mutation.

- [ ] **Step 1: Write a failing missing-loader test**

Build a selected-final sequence training fixture using the established package-test helpers, omit `structured-policy-loader.json`, and assert `package_selected_training_run` raises `ValueError` matching `structured policy loader`.

- [ ] **Step 2: Write a failing source-mutation test**

Patch the publication copy seam so `policy.zip` is modified after initial run validation but before the file is accepted. Assert publication raises `ValueError` matching `source artifact identity changed` and no output bundle exists.

- [ ] **Step 3: Verify RED in GitHub Actions**

Commit only the tests, open the pull request, and confirm the focused tests fail for the expected missing behavior rather than fixture errors.

### Task 2: Implement verified artifact copying

**Files:**
- Modify: `trade_rl/workflows/release_packaging.py`
- Test: `tests/workflows/test_release_packaging_integrity.py`

**Interfaces:**
- Produces: `_copy_verified_run_file(*, training_root: Path, stage: Path, item: RunFile) -> str`

- [ ] **Step 1: Add identity helpers**

Add SHA-256 streaming, safe-path resolution, and source identity verification using the exact `RunFile.digest` and `RunFile.size_bytes` values.

- [ ] **Step 2: Copy through a temporary destination**

Open the verified source, stream bytes to `.<name>.tmp`, flush and fsync, verify destination size and digest, then `os.replace` the temporary file into place. Remove the temporary file on failure.

- [ ] **Step 3: Replace `shutil.copy2` for manifest files**

Use `_copy_verified_run_file` for every `manifest.files` entry. Keep confirmation, reconciliation, and renamed run-manifest copies under their existing external-evidence validation contracts.

- [ ] **Step 4: Verify GREEN**

Run focused tests and existing package tests. Expected: all pass.

### Task 3: Require complete structured-policy closure

**Files:**
- Modify: `trade_rl/workflows/release_packaging.py`
- Test: `tests/workflows/test_release_packaging_integrity.py`

**Interfaces:**
- Consumes: `load_structured_policy_loader_manifest(path)` payload with `architecture_digest` and `action_size`.

- [ ] **Step 1: Fail when loader is absent or undeclared**

For `SEQUENCE_OBSERVATION_SCHEMA`, require `STRUCTURED_POLICY_LOADER_NAME` in the training manifest file set and require the staged path to be a regular file.

- [ ] **Step 2: Validate loader identity**

Require loader `architecture_digest` to equal the ensemble value and loader `action_size` to equal `action_spec.size`.

- [ ] **Step 3: Verify focused and complete CI**

Run Ruff, Ruff format, MyPy, Import Linter, focused package tests, critical branch coverage, and the complete Python suite.

### Task 4: Final review and publication

**Files:**
- Modify: PR description only if implementation details differ from this plan.

- [ ] **Step 1: Review the diff against the design**

Confirm no unrelated training, evaluation, Studio, Stage A, or runtime behavior changed.

- [ ] **Step 2: Verify exact-head CI**

Confirm CI and PostgreSQL Catalog complete successfully on the unchanged final head.

- [ ] **Step 3: Merge intentionally**

Merge only after unresolved review threads are zero and all required checks pass.