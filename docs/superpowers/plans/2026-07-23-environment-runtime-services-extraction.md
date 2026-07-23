# Environment Runtime Services Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the existing eight runtime-service constructor calls from `ResidualMarketEnv.__init__()` into one typed, behavior-preserving bundle.

**Architecture:** Add a frozen `EnvironmentRuntimeServices` contract and a deterministic builder which receives already-validated construction dependencies, preserves current service creation order, and never receives mutable Gymnasium state. The environment facade invokes the builder once and assigns the same existing private attributes.

**Tech Stack:** Python 3.12, dataclasses, Gymnasium, NumPy, pytest, Ruff, Mypy, Import Linter, pytest-cov.

## Global Constraints

- Preserve the public `ResidualMarketEnv` constructor signature.
- Preserve all eight service classes, environment attribute names, collaborator identities, construction order, validation behavior, and error messages.
- Do not create providers, configuration, action specs, reward trackers, market executors, observation contracts, or mutable environment state in the new builder.
- Do not pass books, order books, current indices, pending targets, episode seeds, diagnostics, or reset state to the builder.
- Keep production status `NO-GO` and `AUD-RL-001` as `OPEN RISK, FURTHER REDUCED`.

---

### Task 1: Add RED architecture and wiring characterization

**Files:**
- Create: `tests/architecture/test_environment_runtime_services_decomposition.py`
- Create: `tests/rl/test_environment_runtime_services.py`

**Interfaces:**
- Consumes: the current `ResidualMarketEnv` constructor and the design contract.
- Produces: failing tests that require `EnvironmentRuntimeServices` and `EnvironmentRuntimeServicesBuilder`.

- [ ] **Step 1: Write the architecture test**

Require local ownership in `trade_rl.rl.environment_runtime_services`, one builder invocation in the facade constructor, absence of direct eight-service construction calls, and a constructor source span no greater than 240 lines.

- [ ] **Step 2: Write the wiring characterization**

Create a deterministic two-symbol dataset and environment. Assert the exact service types, supplied collaborator identities, shared execution coordinator identity, shared reward tracker identity, executor identities, observation-contract objects, config values, and minimum index.

- [ ] **Step 3: Verify RED**

Run:

```bash
uv run ruff check tests/architecture/test_environment_runtime_services_decomposition.py tests/rl/test_environment_runtime_services.py
uv run ruff format --check tests/architecture/test_environment_runtime_services_decomposition.py tests/rl/test_environment_runtime_services.py
uv run pytest -q tests/architecture/test_environment_runtime_services_decomposition.py tests/rl/test_environment_runtime_services.py
```

Expected: Ruff and formatting pass; pytest collection fails only because `trade_rl.rl.environment_runtime_services` does not exist.

- [ ] **Step 4: Commit RED tests**

```bash
git add tests/architecture/test_environment_runtime_services_decomposition.py tests/rl/test_environment_runtime_services.py
git commit -m "test: define environment runtime services boundary"
```

### Task 2: Implement the typed runtime-service bundle

**Files:**
- Create: `trade_rl/rl/environment_runtime_services.py`
- Modify: `trade_rl/rl/environment.py`

**Interfaces:**
- Consumes: `EnvironmentObservationContract`, validated config/action/risk/reward/executor collaborators, and the RED tests.
- Produces: `EnvironmentRuntimeServices` and `EnvironmentRuntimeServicesBuilder.build() -> EnvironmentRuntimeServices`.

- [ ] **Step 1: Add the frozen contract**

Define eight typed fields in the exact existing order.

- [ ] **Step 2: Add the builder**

Store explicit constructor inputs and instantiate the services in this order: episode, execution, observation, decision, risk, reward, info, termination.

- [ ] **Step 3: Delegate from the facade**

Import only `EnvironmentRuntimeServicesBuilder`, replace the eight inline constructors with one builder call, and assign the returned fields to the existing private attributes.

- [ ] **Step 4: Verify GREEN for focused tests**

Run:

```bash
uv run ruff check trade_rl/rl/environment.py trade_rl/rl/environment_runtime_services.py tests/architecture/test_environment_runtime_services_decomposition.py tests/rl/test_environment_runtime_services.py
uv run ruff format --check trade_rl/rl/environment.py trade_rl/rl/environment_runtime_services.py tests/architecture/test_environment_runtime_services_decomposition.py tests/rl/test_environment_runtime_services.py
uv run mypy trade_rl/rl/environment.py trade_rl/rl/environment_runtime_services.py
uv run pytest -q tests/architecture/test_environment_runtime_services_decomposition.py tests/rl
```

Expected: all commands pass.

- [ ] **Step 5: Commit implementation**

```bash
git add trade_rl/rl/environment.py trade_rl/rl/environment_runtime_services.py
git commit -m "refactor: delegate environment runtime service wiring"
```

### Task 3: Complete branch characterization and permanent controls

**Files:**
- Modify: `tests/rl/test_environment_runtime_services.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: coverage JSON from the complete suite.
- Produces: 100% branch coverage for the new module and a permanent critical-coverage ratchet.

- [ ] **Step 1: Run the complete suite with branch coverage**

```bash
uv run pytest -q --cov=trade_rl --cov-branch --cov-report=json:coverage.json
```

- [ ] **Step 2: Inspect the new module's missing branches**

Read `coverage.json` and add only behavior-relevant characterization tests for uncovered constructor-validation branches.

- [ ] **Step 3: Verify 100% branch coverage**

Expected: all statements and branches in `trade_rl/rl/environment_runtime_services.py` are covered.

- [ ] **Step 4: Add the ratchet**

Add:

```toml
"trade_rl/rl/environment_runtime_services.py" = 100.0
```

under `[tool.trade_rl.critical_coverage.files]`.

- [ ] **Step 5: Commit coverage controls**

```bash
git add tests/rl/test_environment_runtime_services.py pyproject.toml
git commit -m "test: guard environment runtime service wiring"
```

### Task 4: Full exact-head verification and audit documentation

**Files:**
- Create: `docs/verification/2026-07-23-environment-runtime-services-extraction.md`
- Modify: `docs/verification/2026-07-23-architecture-audit-closeout.md`

**Interfaces:**
- Consumes: exact commit SHA, CI run IDs, PostgreSQL run ID, test counts, and coverage totals.
- Produces: reviewable evidence and updated `AUD-RL-001` disposition.

- [ ] **Step 1: Run all maintained checks**

Require success for Ruff, format, Mypy, Import Linter, dead-code reporting, serving smoke, complete pytest/coverage, critical coverage, CLI, Ubuntu, Windows, training image/non-root probe, and PostgreSQL Catalog.

- [ ] **Step 2: Record exact evidence**

Document RED commit/run, implementation head, test counts, total statement/branch coverage, new-module statement/branch coverage, constructor reduction, changed-file scope, and non-goals.

- [ ] **Step 3: Update the closeout**

State that typed runtime-service wiring is extracted, but config/action/reward/executor construction and mutable Gymnasium initialization remain. Keep `OPEN RISK, FURTHER REDUCED` and production `NO-GO`.

- [ ] **Step 4: Review the final diff**

Confirm there are no temporary workflows, triggers, generated coverage files, unrelated source changes, or public constructor changes.

- [ ] **Step 5: Ready and squash merge**

Use the exact verified PR head SHA as the merge guard.