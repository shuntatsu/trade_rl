# Environment Reward and Execution Resources Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development and superpowers:verification-before-completion task by task.

**Goal:** Extract reward-tracker, optional reward pre-roll, dual-executor, compatibility-alias, and reward-cache construction from `ResidualMarketEnv.__init__()` without changing behavior.

**Architecture:** A focused `EnvironmentRewardExecutionResourcesBuilder` consumes already-resolved policy and schedule values and returns a frozen typed resource container. The Gymnasium facade installs the existing six attributes and continues passing them into observation and runtime-service construction.

**Tech Stack:** Python 3.12, dataclasses, NumPy, Gymnasium, pytest, Ruff, MyPy, Import Linter, GitHub Actions, PostgreSQL.

## Global Constraints

- Production remains `NO-GO`.
- Preserve the public `ResidualMarketEnv` constructor signature.
- Preserve exact reward, pre-roll, execution, observation, reset, and step semantics.
- Preserve construction and validation order.
- Keep observation contracts, runtime-service wiring, and mutable initial state outside the new module.
- Require 100.0% critical coverage for the new module.
- Require a constructor source span no greater than 150 lines.

---

### Task 1: Commit the RED behavioral and architecture contracts

**Files:**
- Create: `tests/rl/test_environment_reward_execution_resources.py`
- Create: `tests/architecture/test_environment_reward_execution_resources_decomposition.py`

**Interfaces:**
- Consumes: current inline `RewardTracker`, `minimum_reward_start_index`, and `MarketExecutor` construction.
- Produces: executable requirements for `EnvironmentRewardExecutionResources` and `EnvironmentRewardExecutionResourcesBuilder`.

- [ ] **Step 1: Write direct failing characterization tests**

Create a regular hourly two-symbol `MarketDataset` and assert that the missing builder API will:

```python
resources = EnvironmentRewardExecutionResourcesBuilder(
    dataset,
    config=config,
    reward_config=reward_config,
    resolved_decision_hours=2.0,
    minimum_start_index=8,
    execution_rule_stress=stress,
).build()
```

Cover no-pre-roll minimum preservation, full pre-roll derivation through `minimum_reward_start_index`, reward window sizing, distinct executor instances, equal execution-policy digests, `executor is hybrid_executor`, stress identity, and fresh cache/tracker objects across two builds.

- [ ] **Step 2: Write the failing facade integration test**

Construct `ResidualMarketEnv` and assert the existing attributes remain present and equivalent:

```python
assert env.executor is env.hybrid_executor
assert env.hybrid_executor is not env.shadow_executor
assert env.hybrid_executor.execution_policy_digest == env.shadow_executor.execution_policy_digest
assert env._reward_history_cache == {}
```

- [ ] **Step 3: Write architecture tests**

Require local type ownership, exactly one builder invocation, no direct resource construction in `ResidualMarketEnv.__init__()`, preserved builder source order, and `len(inspect.getsource(ResidualMarketEnv.__init__).splitlines()) <= 150`.

- [ ] **Step 4: Run RED verification**

Run through the draft PR CI:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run lint-imports
uv run pytest -q
```

Expected: static checks pass; complete pytest collection fails only because `trade_rl.rl.environment_reward_execution_resources` does not exist.

- [ ] **Step 5: Commit RED tests**

```bash
git add tests/rl/test_environment_reward_execution_resources.py \
  tests/architecture/test_environment_reward_execution_resources_decomposition.py
git commit -m "test: define reward execution resource boundary"
```

### Task 2: Implement the minimal typed resource builder

**Files:**
- Create: `trade_rl/rl/environment_reward_execution_resources.py`
- Modify: `trade_rl/rl/environment.py`

**Interfaces:**
- Consumes: `MarketDataset`, `ResidualMarketEnvConfig`, `RewardConfig`, resolved decision hours, incoming minimum start index, and optional `ExecutionRuleStress`.
- Produces: `EnvironmentRewardExecutionResourcesBuilder.build() -> EnvironmentRewardExecutionResources`.

- [ ] **Step 1: Add the frozen result contract**

Implement fields for reward tracker, resolved minimum, hybrid executor, shadow executor, executor alias, and reward-history cache.

- [ ] **Step 2: Implement the builder in maintained order**

Use exactly:

```python
reward_tracker = RewardTracker(
    self.reward_config,
    decision_hours=self.resolved_decision_hours,
)
minimum_start_index = self.minimum_start_index
if (
    self.config.require_full_reward_preroll
    and self.reward_config.baseline_underperformance_weight > 0.0
):
    minimum_start_index = minimum_reward_start_index(
        self.dataset,
        signal_minimum=minimum_start_index,
        window_hours=self.reward_config.baseline_window_hours,
    )
hybrid_executor = MarketExecutor(
    self.dataset,
    self.config.execution_cost,
    rule_stress=self.execution_rule_stress,
)
shadow_executor = MarketExecutor(
    self.dataset,
    self.config.execution_cost,
    rule_stress=self.execution_rule_stress,
)
```

Return `executor=hybrid_executor` and `reward_history_cache={}`.

- [ ] **Step 3: Delegate from the facade once**

Replace only the current reward/executor/cache block. Assign the six returned values to the same attributes. Do not modify observation, runtime-service, initial-state, reset, or step code.

- [ ] **Step 4: Run focused GREEN verification**

```bash
uv run ruff check trade_rl/rl/environment.py \
  trade_rl/rl/environment_reward_execution_resources.py \
  tests/rl/test_environment_reward_execution_resources.py \
  tests/architecture/test_environment_reward_execution_resources_decomposition.py
uv run ruff format --check trade_rl/rl/environment.py \
  trade_rl/rl/environment_reward_execution_resources.py \
  tests/rl/test_environment_reward_execution_resources.py \
  tests/architecture/test_environment_reward_execution_resources_decomposition.py
uv run mypy trade_rl/rl/environment.py \
  trade_rl/rl/environment_reward_execution_resources.py
uv run pytest -q tests/rl/test_environment_reward_execution_resources.py \
  tests/architecture/test_environment_reward_execution_resources_decomposition.py \
  tests/rl/test_environment_identity.py \
  tests/rl/test_environment_timing.py
```

Expected: all focused checks pass.

- [ ] **Step 5: Commit implementation**

```bash
git add trade_rl/rl/environment.py \
  trade_rl/rl/environment_reward_execution_resources.py
git commit -m "refactor: extract environment reward execution resources"
```

### Task 3: Ratchet coverage, document evidence, and integrate

**Files:**
- Modify: `pyproject.toml`
- Create: `docs/verification/2026-07-23-environment-reward-execution-resources-extraction.md`
- Modify: `docs/verification/2026-07-23-architecture-audit-closeout.md`
- Modify: PR body for the canonical branch.

**Interfaces:**
- Consumes: exact final implementation SHA and CI artifacts.
- Produces: permanent coverage and architecture controls plus auditable verification evidence.

- [ ] **Step 1: Run complete verification at the implementation head**

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run lint-imports
uv run pytest -q --cov=trade_rl --cov-branch --cov-report=json --cov-report=term-missing
uv run python tools/check_critical_coverage.py coverage.json
```

Also require Studio tests/typecheck/build/fixed-layout, Ubuntu and Windows compatibility, complete training image/non-root probe, CLI smoke, and PostgreSQL Catalog.

- [ ] **Step 2: Add the permanent 100.0% ratchet**

Add:

```toml
"trade_rl/rl/environment_reward_execution_resources.py" = 100.0
```

under `[tool.trade_rl.critical_coverage.files]` only after the exact coverage artifact proves the threshold.

- [ ] **Step 3: Record verification evidence**

Document RED SHA/run/artifact, GREEN SHA/run, final synchronized SHA, test count, warnings, total and branch coverage, new-module coverage, constructor line count, artifact IDs/digests, and production `NO-GO`.

- [ ] **Step 4: Update architecture closeout**

Record the new typed boundary, constructor reduction, permanent ratchet, remaining constructor responsibilities, and `AUD-RL-001: OPEN RISK, FURTHER REDUCED`.

- [ ] **Step 5: Verify final diff and reviews**

Require only intended source, tests, docs, and `pyproject.toml`; no temporary workflows, triggers, diagnostics, or generated coverage files. Require no unresolved review threads.

- [ ] **Step 6: Synchronize current main and repeat full verification**

If main advanced, merge current main into the branch and rerun CI and PostgreSQL at the new exact head.

- [ ] **Step 7: Mark ready and squash merge with expected head SHA**

```bash
git merge --squash <verified-head>
```

Use GitHub's squash merge only when the expected head SHA, behind count zero, and all exact-head checks are confirmed.
