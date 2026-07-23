# Serving Package Canonical Loader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Serving's hand-maintained execution-cost reconstruction with one strict canonical loader and enforce critical branch coverage on the packaging boundary.

**Architecture:** `trade_rl.serving.training_environment` parses and validates `training_environment_v2`; `trade_rl.serving.package` delegates to it. Existing bundle, confirmation, reconciliation, and execution-promotion types remain unchanged.

**Tech Stack:** Python 3.12, dataclasses, JSON, Pytest, pytest-cov, existing critical-coverage tooling.

## Global Constraints

- Preserve `serving_bundle_v5` and every existing bundle identity field.
- Preserve current execution and release semantics.
- Reject missing or unknown `ExecutionCostConfig` fields instead of applying defaults.
- Keep production status `NO-GO`.
- Use test-first RED/GREEN evidence.

---

### Task 1: Specify the fail-closed artifact contract

**Files:**
- Create: `tests/serving/test_training_environment_contract.py`

**Interfaces:**
- Consumes: existing private `trade_rl.serving.package._execution_cost(Path)` during RED.
- Produces: behavioral expectations for strict schema and field validation.

- [ ] Write tests for a complete round trip, unsupported schema, missing field, unknown field, malformed root/environment/execution mappings, delegation source ownership, and critical-coverage configuration.
- [ ] Run `pytest -q tests/serving/test_training_environment_contract.py` and record the expected failures caused by permissive defaults and missing coverage declarations.
- [ ] Commit the RED tests.

### Task 2: Add the canonical loader

**Files:**
- Create: `trade_rl/serving/training_environment.py`
- Modify: `trade_rl/serving/package.py`

**Interfaces:**
- Produces: `load_training_execution_cost(path: Path) -> ExecutionCostConfig`.
- `package._execution_cost(training_root)` delegates to `load_training_execution_cost(training_root / "environment.json")`.

- [ ] Parse JSON and require a mapping root.
- [ ] Require `schema_version == "training_environment_v2"`.
- [ ] Require mapping values at `environment` and `environment.execution_cost`.
- [ ] Compare execution keys with `dataclasses.fields(ExecutionCostConfig)` and report missing/unknown names.
- [ ] Convert `trigger_volume_fractions` from JSON list to tuple.
- [ ] Construct `ExecutionCostConfig(**payload)` so semantic range validation remains canonical.
- [ ] Remove field-by-field execution parsing and its local primitive conversion helpers from `package.py` where no longer used.
- [ ] Run the focused tests and commit GREEN.

### Task 3: Ratchet the full packaging boundary

**Files:**
- Modify: `pyproject.toml`
- Modify or create focused tests under `tests/serving/`.

**Interfaces:**
- Produces per-file critical branch targets of 90.0% for `package.py` and 100.0% for `training_environment.py`.

- [ ] Add focused rejection tests for uncovered package identity, confirmation, reconciliation, output, and staging-cleanup branches.
- [ ] Add the two per-file critical coverage entries.
- [ ] Run `pytest -q tests/serving/test_package.py tests/serving/test_training_environment_contract.py --cov=trade_rl --cov-branch --cov-report=json`.
- [ ] Run the repository critical-coverage command used by CI and confirm both targets pass.
- [ ] Commit the coverage ratchet.

### Task 4: Full verification

**Files:**
- Modify verification documentation only if final evidence values are available.

- [ ] Run Ruff and format checks.
- [ ] Run Mypy and Import Linter.
- [ ] Run the complete Pytest suite and coverage.
- [ ] Run Studio tests, typecheck, production build, fixed-viewport validation, Ubuntu and Windows compatibility, training-image/non-root probe, and PostgreSQL catalog workflow.
- [ ] Record exact-head workflow runs and artifact digests in the PR.
- [ ] Squash merge only after all gates pass.