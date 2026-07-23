# Environment Policy and Schedule Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move deterministic environment config, action-layout, and episode/decision schedule construction out of `ResidualMarketEnv.__init__()` into one typed contract without behavior changes.

**Architecture:** Add a frozen `EnvironmentPolicyScheduleContract` and a builder that owns the current validation order. The environment facade invokes the builder once, assigns the same attributes, and keeps reward-tracker/executor construction and mutable state local.

**Tech Stack:** Python 3.12, dataclasses, Gymnasium, NumPy, pytest, pytest-cov, Ruff, Mypy, Import Linter, GitHub Actions, PostgreSQL Compose.

## Global Constraints

- Preserve the public `ResidualMarketEnv` constructor signature.
- Preserve supplied config and action-spec object identities.
- Preserve exact exception text and validation order.
- Do not change action, reward, risk, execution, observation, or reset semantics.
- Keep reward-tracker, executor, and mutable Gymnasium-state construction outside the new boundary.
- Enforce a 190-line maximum for `ResidualMarketEnv.__init__()`.
- Require 100.0% statement and branch coverage for the new module.
- Production remains `NO-GO`.

---

### Task 1: Characterize the policy and schedule contract

**Files:**
- Create: `tests/rl/test_environment_policy_schedule_contract.py`
- Create: `tests/architecture/test_environment_policy_schedule_contract_decomposition.py`

**Interfaces:**
- Consumes: existing `MarketDataset`, `ResidualMarketEnvConfig`, `ActionSpec`, `PreTradeRisk`.
- Produces: failing expectations for `EnvironmentPolicyScheduleContract` and `EnvironmentPolicyScheduleContractBuilder`.

- [ ] **Step 1: Write direct behavior tests**

Cover supplied identity preservation, default action-spec resolution, monitor config identity, action names, nominal bars, resolved hours, and exact validation errors/order.

- [ ] **Step 2: Write architecture tests**

Require local module ownership, one facade builder call, absence of inline extracted policy, ordered builder validation markers, and the 190-line constructor limit.

- [ ] **Step 3: Run clean RED verification**

Run:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run pytest -q
```

Expected: static checks pass and pytest collection fails only because `trade_rl.rl.environment_policy_schedule_contract` does not exist.

- [ ] **Step 4: Commit the RED tests**

```bash
git add tests/rl/test_environment_policy_schedule_contract.py \
  tests/architecture/test_environment_policy_schedule_contract_decomposition.py
git commit -m "test: define environment policy schedule contract"
```

### Task 2: Implement the typed contract

**Files:**
- Create: `trade_rl/rl/environment_policy_schedule_contract.py`
- Modify: `trade_rl/rl/environment.py`
- Test: `tests/rl/test_environment_policy_schedule_contract.py`
- Test: `tests/architecture/test_environment_policy_schedule_contract_decomposition.py`

**Interfaces:**
- Produces: `EnvironmentPolicyScheduleContract` with config, emergency monitor, action spec/names, nominal bars, reward config, and resolved decision hours.
- Produces: `EnvironmentPolicyScheduleContractBuilder(dataset, pre_trade_risk, alpha_enabled, factor_count, action_spec, config).build()`.

- [ ] **Step 1: Add the frozen contract dataclass**

Use exact fields from the design spec and export both public names through `__all__`.

- [ ] **Step 2: Implement the builder in the preserved order**

Resolve config, construct monitor, perform leverage/random-gross validations, resolve/validate action spec, derive names and bars, resolve reward config/hours, validate episode choices, and return the contract.

- [ ] **Step 3: Delegate the environment facade**

Replace the inline block from config resolution through resolved decision hours with one builder invocation and assignments to the existing attributes/local `reward_config`.

- [ ] **Step 4: Run focused GREEN verification**

```bash
uv run ruff format trade_rl/rl/environment.py \
  trade_rl/rl/environment_policy_schedule_contract.py \
  tests/rl/test_environment_policy_schedule_contract.py \
  tests/architecture/test_environment_policy_schedule_contract_decomposition.py
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run pytest -q tests/rl tests/architecture/test_environment_policy_schedule_contract_decomposition.py
```

Expected: all commands succeed.

- [ ] **Step 5: Commit the verified implementation**

```bash
git add trade_rl/rl/environment.py \
  trade_rl/rl/environment_policy_schedule_contract.py \
  tests/rl/test_environment_policy_schedule_contract.py \
  tests/architecture/test_environment_policy_schedule_contract_decomposition.py
git commit -m "refactor: extract environment policy schedule contract"
```

### Task 3: Add the coverage ratchet and full verification

**Files:**
- Modify: `pyproject.toml`
- Modify: `tests/rl/test_environment_policy_schedule_contract.py` when uncovered branches require characterization.

**Interfaces:**
- Produces: permanent 100.0% critical branch threshold for `trade_rl/rl/environment_policy_schedule_contract.py`.

- [ ] **Step 1: Run complete coverage**

```bash
uv run pytest -q --cov=trade_rl --cov-branch --cov-report=json:coverage.json
```

Expected: complete suite succeeds and the new module has no uncovered statements or branches.

- [ ] **Step 2: Add missing characterization only when coverage proves a gap**

Add the smallest behavior test for each uncovered branch and rerun complete coverage.

- [ ] **Step 3: Add the permanent ratchet**

Add:

```toml
"trade_rl/rl/environment_policy_schedule_contract.py" = 100.0
```

to `[tool.trade_rl.critical_coverage.files]`.

- [ ] **Step 4: Run maintained CI-equivalent checks**

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run lint-imports
uv run pytest -q --cov=trade_rl --cov-branch --cov-report=json:coverage.json
uv run python scripts/check_critical_coverage.py coverage.json
uv run trade-rl --help
```

Expected: all commands succeed.

- [ ] **Step 5: Commit the ratchet**

```bash
git add pyproject.toml tests/rl/test_environment_policy_schedule_contract.py
git commit -m "test: ratchet policy schedule contract coverage"
```

### Task 4: Record exact-head evidence and integrate

**Files:**
- Create: `docs/verification/2026-07-23-environment-policy-schedule-contract-extraction.md`
- Modify: `docs/verification/2026-07-23-architecture-audit-closeout.md`

**Interfaces:**
- Produces: exact RED/GREEN/final SHA and workflow evidence, constructor reduction, coverage numbers, and updated `AUD-RL-001` disposition.

- [ ] **Step 1: Verify GitHub Actions on the exact implementation head**

Require successful CI, Ubuntu/Windows, training image/non-root probe, and PostgreSQL Catalog on the same SHA.

- [ ] **Step 2: Record exact evidence**

Document test counts, total coverage, total branch coverage, module statement/branch coverage, constructor line count, and unchanged scope.

- [ ] **Step 3: Audit the final diff**

Confirm only intended source, tests, config, plan/spec, and verification documents remain; no temporary workflows, triggers, diagnostics, or generated coverage files remain.

- [ ] **Step 4: Rerun final documentation-head CI**

Require CI and PostgreSQL Catalog success on the documentation-inclusive final SHA.

- [ ] **Step 5: Ready and squash-merge the PR**

Require no unresolved review threads and merge with the exact expected head SHA.
