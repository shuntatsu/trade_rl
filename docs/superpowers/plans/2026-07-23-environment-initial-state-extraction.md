# Environment Initial State Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the final invocation-local mutable-state construction from `ResidualMarketEnv.__init__()` into a typed, independently tested factory without changing behavior.

**Architecture:** Add one request dataclass, one returned assembly dataclass, and one static factory in `trade_rl.rl.environment_initial_state`. The environment facade invokes the factory once after digest construction and copies values into the existing attributes. Reward history, reward/executor construction, observation contracts, runtime services, and reset logic remain unchanged.

**Tech Stack:** Python 3.12, dataclasses, NumPy, Gymnasium, pytest, Ruff, Mypy, Import Linter, coverage.py, GitHub Actions.

## Global Constraints

- Preserve the public `ResidualMarketEnv` constructor signature.
- Preserve all existing attribute names, values, dtypes, and object-independence relationships.
- Do not move `_reward_history_cache` in this PR.
- Do not change reset, step, reward, risk, execution, observation, or digest semantics.
- Keep `ResidualMarketEnv.__init__()` at or below 170 source lines.
- Maintain 100.0% branch coverage for `trade_rl/rl/environment_initial_state.py`.
- Production remains `NO-GO`.

---

### Task 1: Commit RED characterization and architecture contracts

**Files:**
- Create: `tests/rl/test_environment_initial_state.py`
- Create: `tests/architecture/test_environment_initial_state_decomposition.py`

**Interfaces:**
- Consumes: current `MarketDataset`, `ResidualMarketEnvConfig`, `ActionSpec`, and `ResidualMarketEnv` APIs.
- Produces: the required names `EnvironmentInitialStateRequest`, `EnvironmentInitialState`, and `EnvironmentInitialStateFactory.create()`.

- [ ] **Step 1: Write the failing direct factory tests**

Create a two-symbol hourly dataset and assert that two factory calls return equivalent values but independent books, arrays, order books, execution states, and diagnostics. Assert exact indices, cash, marks, multipliers, dtypes, shapes, episode defaults, pending targets, and reset flag.

- [ ] **Step 2: Write the failing facade architecture tests**

Require local ownership of the three new types, exactly one factory invocation in `ResidualMarketEnv.__init__()`, absence of direct initial-state constructors from the facade source, and a constructor span no greater than 170 lines.

- [ ] **Step 3: Open a draft PR and verify RED**

Run the normal complete CI. Expected failure: pytest collection errors because `trade_rl.rl.environment_initial_state` does not exist. Ruff, formatting, Mypy, Import Linter, Studio, compatibility, and training-image checks should remain green.

### Task 2: Implement the minimal state factory and facade delegation

**Files:**
- Create: `trade_rl/rl/environment_initial_state.py`
- Modify: `trade_rl/rl/environment.py`
- Test: `tests/rl/test_environment_initial_state.py`
- Test: `tests/architecture/test_environment_initial_state_decomposition.py`

**Interfaces:**
- Consumes: `EnvironmentInitialStateRequest(dataset, config, action_spec, minimum_start_index)`.
- Produces: `EnvironmentInitialStateFactory.create(request) -> EnvironmentInitialState`.

- [ ] **Step 1: Implement the request and state dataclasses**

Use `@dataclass(frozen=True, slots=True)`. Include only the 17 values assigned by the existing constructor tail; exclude `_reward_history_cache`.

- [ ] **Step 2: Implement `EnvironmentInitialStateFactory.create()`**

Construct the hybrid book with `BookState.zero`, clone shadow, and create fresh arrays, order books, execution state, and diagnostics with the exact current dtypes and defaults.

- [ ] **Step 3: Delegate once from the environment facade**

Import only `EnvironmentInitialStateFactory` and `EnvironmentInitialStateRequest`, call the factory after `_environment_digest` construction, and assign every returned field to the existing attribute name.

- [ ] **Step 4: Run focused GREEN verification**

Run:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run lint-imports
uv run pytest -q \
  tests/rl/test_environment_initial_state.py \
  tests/architecture/test_environment_initial_state_decomposition.py \
  tests/rl/test_environment.py \
  tests/rl/test_environment_identity.py \
  tests/rl/test_environment_timing.py
```

Expected: all commands pass with no new warnings.

### Task 3: Ratchet coverage and record evidence

**Files:**
- Modify: `pyproject.toml`
- Create: `docs/verification/2026-07-23-environment-initial-state-extraction.md`
- Modify: `docs/verification/2026-07-23-architecture-audit-closeout.md`

**Interfaces:**
- Consumes: final coverage JSON and exact-head CI evidence.
- Produces: permanent branch-coverage enforcement and current architecture disposition.

- [ ] **Step 1: Run complete tests with branch coverage**

Run the maintained complete CI command and inspect `coverage.json`. Confirm all statements and branches in the new module are covered.

- [ ] **Step 2: Add the exact coverage ratchet**

Add:

```toml
"trade_rl/rl/environment_initial_state.py" = 100.0
```

under `[tool.trade_rl.critical_coverage.files]`.

- [ ] **Step 3: Run full exact-head verification**

Require Core quality, full pytest and coverage, critical coverage, CLI, Ubuntu/Windows compatibility, complete training image and non-root probe, plus PostgreSQL Compose, migration, and integration tests.

- [ ] **Step 4: Write verification and closeout documentation**

Record RED head/run, GREEN head/run, exact test totals, total and module coverage, constructor source span, artifact identities, review status, and production `NO-GO`.

### Task 4: Review and integrate

**Files:**
- Review all PR changes.

- [ ] **Step 1: Confirm final diff scope**

The final diff must contain only the new module, two tests, design/plan/verification docs, `environment.py`, `pyproject.toml`, and architecture closeout update. No temporary workflows, triggers, artifacts, or generated coverage files may remain.

- [ ] **Step 2: Check reviews and threads**

Confirm no unresolved review thread or requested change remains.

- [ ] **Step 3: Squash merge with expected head SHA**

Merge only after main has not advanced incompatibly and all maintained verification is successful.

- [ ] **Step 4: Reclassify `AUD-RL-001`**

Keep it `OPEN RISK, FURTHER REDUCED` unless the remaining reward/executor resource construction is separately characterized and extracted. Production remains `NO-GO`.
