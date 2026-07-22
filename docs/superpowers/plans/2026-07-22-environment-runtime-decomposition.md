# Environment Runtime Decomposition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract action planning, risk projection, reward transition, and information construction from `ResidualMarketEnv.step()` into independently tested services without changing runtime behavior or public contracts.

**Architecture:** Four stateless or narrowly stateful services consume immutable request dataclasses and return immutable results. `ResidualMarketEnv` remains the sole owner of Gymnasium state and applies service results in the existing order.

**Tech Stack:** Python 3.12, NumPy, Gymnasium, dataclasses, Pytest, Ruff, Mypy, coverage.py, GitHub Actions.

## Global Constraints

- Preserve the `ResidualMarketEnv` constructor, public properties, Gymnasium tuple contract, schema versions, and environment digest payload.
- Preserve target identity, signal-delay, risk reason ordering, reward ordering, terminal accounting, and every existing `info` key and optional-key condition.
- Keep books, order books, pending targets, indices, action history, position ages, and diagnostics mutable only in `ResidualMarketEnv`.
- Do not add direct exchange routing or change production `NO-GO` status.
- Use TDD: architecture and service tests must fail before production modules are added.
- Do not lower the existing `environment_runtime` critical branch-coverage threshold of `64.0`.

---

## File map

- Create `trade_rl/rl/environment_decision.py`: action parsing, composition, signal delay, and decision-bar planning.
- Create `trade_rl/rl/environment_risk.py`: emergency, pre-trade, and portfolio risk projection.
- Create `trade_rl/rl/environment_reward.py`: reward transition input mapping.
- Create `trade_rl/rl/environment_info.py`: stable step and terminal information dictionaries.
- Modify `trade_rl/rl/environment.py`: instantiate and orchestrate the four services; retain thin private delegates where compatibility requires them.
- Create `tests/architecture/test_environment_step_decomposition.py`: source and ownership contracts.
- Create `tests/rl/test_environment_decision_service.py`: action migration, delay, and bar-count tests.
- Create `tests/rl/test_environment_risk_service.py`: shape, emergency, and advanced-risk tests.
- Create `tests/rl/test_environment_reward_service.py`: reward input-mapping tests.
- Create `tests/rl/test_environment_info_service.py`: stable optional and terminal key tests.
- Modify `pyproject.toml`: add a separate measured branch-coverage group for the new step services.
- Create `docs/verification/2026-07-22-environment-runtime-decomposition.md`: RED/GREEN and exact-head evidence.

---

### Task 1: Add RED architecture contracts

**Files:**
- Create: `tests/architecture/test_environment_step_decomposition.py`

**Interfaces:**
- Consumes: current `trade_rl/rl/environment.py` source.
- Produces: failing contracts requiring `EnvironmentDecisionPlanner`, `EnvironmentRiskProjector`, `EnvironmentRewardCoordinator`, and `EnvironmentInfoBuilder`.

- [ ] **Step 1: Write failing source contracts**

Require the four module paths, assert `ResidualMarketEnv` imports and constructs all four services, and assert the facade no longer directly calls `composer.compose`, `emergency_risk_monitor.assess`, `reward_tracker.step`, or constructs the large step `info` literal.

- [ ] **Step 2: Run RED test**

Run:

```bash
uv run pytest -q tests/architecture/test_environment_step_decomposition.py
```

Expected: FAIL because the four service modules and delegates do not yet exist.

- [ ] **Step 3: Commit RED evidence**

```bash
git add tests/architecture/test_environment_step_decomposition.py
git commit -m "test: require environment step services"
```

### Task 2: Implement decision planning service

**Files:**
- Create: `trade_rl/rl/environment_decision.py`
- Create: `tests/rl/test_environment_decision_service.py`

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True, slots=True)
class EnvironmentDecisionRequest:
    action: np.ndarray
    trends: TrendTargets
    alpha: np.ndarray
    factor_basis: np.ndarray
    hybrid_weights: np.ndarray
    shadow_weights: np.ndarray
    pending_hybrid_target: np.ndarray | None
    pending_shadow_target: np.ndarray | None
    current_index: int
    end_index: int

@dataclass(frozen=True, slots=True)
class EnvironmentDecisionPlan:
    parsed_action: ResidualAction | ResidualActionV2 | TargetWeightAction
    maintained_action: np.ndarray
    saturated_count: int
    raw_max_abs: float
    submitted_hybrid_target: np.ndarray
    submitted_shadow_target: np.ndarray
    executed_hybrid_target: np.ndarray
    executed_shadow_target: np.ndarray
    next_pending_hybrid_target: np.ndarray | None
    next_pending_shadow_target: np.ndarray | None
    execution_delay_warmup: bool
    bars: int

class EnvironmentDecisionPlanner:
    def plan(self, request: EnvironmentDecisionRequest) -> EnvironmentDecisionPlan: ...
```

- [ ] **Step 1: Write failing unit tests**

Cover maintained actions, legacy two-value migration, zero-delay targets, one-decision delay warm-up, pending-target execution, regular cadence, irregular cadence, invalid ended intervals, and returned-array copy isolation.

- [ ] **Step 2: Run tests and confirm RED**

```bash
uv run pytest -q tests/rl/test_environment_decision_service.py
```

Expected: import failure for `trade_rl.rl.environment_decision`.

- [ ] **Step 3: Implement minimal planner**

Constructor dependencies are `dataset`, `action_spec`, `composer`, `pre_trade_max_gross`, `alpha_enabled`, `accept_legacy_actions`, `signal_delay_decisions`, `decision_every`, and `decision_hours`. Reuse existing action classes and `BaselineResidualComposer` without changing their behavior.

- [ ] **Step 4: Run focused tests**

```bash
uv run pytest -q tests/rl/test_environment_decision_service.py tests/rl/test_target_weight_action.py tests/rl/test_environment_time_config.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add trade_rl/rl/environment_decision.py tests/rl/test_environment_decision_service.py
git commit -m "refactor: extract environment decision planner"
```

### Task 3: Implement risk projection service

**Files:**
- Create: `trade_rl/rl/environment_risk.py`
- Create: `tests/rl/test_environment_risk_service.py`

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True, slots=True)
class EnvironmentRiskRequest:
    proposal: np.ndarray
    book: BookState
    current_index: int

class EnvironmentRiskProjector:
    def project(self, request: EnvironmentRiskRequest) -> RiskConstrainedTarget: ...
```

- [ ] **Step 1: Write failing unit tests**

Cover symbol-shape rejection, non-finite rejection, quote-notional and base-volume conversion, emergency flattening, pre-trade reason ordering, advanced covariance/beta/stress input forwarding, missing-provider failure, and projection-distance calculation.

- [ ] **Step 2: Run tests and confirm RED**

```bash
uv run pytest -q tests/rl/test_environment_risk_service.py
```

Expected: import failure for `trade_rl.rl.environment_risk`.

- [ ] **Step 3: Implement minimal projector**

Constructor dependencies are `dataset`, `emergency_risk_monitor`, `pre_trade_risk`, `portfolio_risk`, and `portfolio_risk_inputs_provider`. Move `_market_notional` and `_constrain_target` behavior without changing reason order.

- [ ] **Step 4: Run focused tests**

```bash
uv run pytest -q tests/rl/test_environment_risk_service.py tests/risk tests/rl/test_emergency_drawdown.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add trade_rl/rl/environment_risk.py tests/rl/test_environment_risk_service.py
git commit -m "refactor: extract environment risk projector"
```

### Task 4: Implement reward coordinator

**Files:**
- Create: `trade_rl/rl/environment_reward.py`
- Create: `tests/rl/test_environment_reward_service.py`

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True, slots=True)
class EnvironmentRewardRequest:
    hybrid_log_return: float
    shadow_log_return: float
    hybrid: BookState
    shadow: BookState
    projection_distance: float
    hybrid_terminated: bool
    shadow_terminated: bool

class EnvironmentRewardCoordinator:
    def step(self, request: EnvironmentRewardRequest) -> RewardBreakdown: ...
```

- [ ] **Step 1: Write failing mapping tests**

Use a recording reward tracker and assert exact drawdown, margin-deficit fraction, equity fractions, projection distance, returns, and termination flags.

- [ ] **Step 2: Run tests and confirm RED**

```bash
uv run pytest -q tests/rl/test_environment_reward_service.py
```

Expected: import failure for `trade_rl.rl.environment_reward`.

- [ ] **Step 3: Implement minimal coordinator**

Constructor dependencies are `RewardTracker` and `initial_capital`. Keep drawdown computation numerically identical to the facade implementation.

- [ ] **Step 4: Run focused tests**

```bash
uv run pytest -q tests/rl/test_environment_reward_service.py tests/rl/test_reward_design.py tests/rl/test_reward_time_scaling.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add trade_rl/rl/environment_reward.py tests/rl/test_environment_reward_service.py
git commit -m "refactor: extract environment reward coordinator"
```

### Task 5: Implement information builder

**Files:**
- Create: `trade_rl/rl/environment_info.py`
- Create: `tests/rl/test_environment_info_service.py`

**Interfaces:**
- Produces immutable request dataclasses for step and terminal data plus:

```python
class EnvironmentInfoBuilder:
    def step_info(self, request: EnvironmentStepInfoRequest) -> dict[str, object]: ...
    def terminal_info(self, request: EnvironmentTerminalInfoRequest) -> dict[str, object]: ...
```

- [ ] **Step 1: Write failing contract tests**

Assert the complete existing key set, reward-context fields, optional discarded target, optional hybrid/shadow liquidation, terminal metric fields, fresh dictionary creation, and copied NumPy targets.

- [ ] **Step 2: Run tests and confirm RED**

```bash
uv run pytest -q tests/rl/test_environment_info_service.py
```

Expected: import failure for `trade_rl.rl.environment_info`.

- [ ] **Step 3: Implement minimal builder**

Move `_book_metrics`, `_terminal_info`, and the step `info` literal. Use `evaluate_performance` and `ReturnSeries` exactly as before.

- [ ] **Step 4: Run focused tests**

```bash
uv run pytest -q tests/rl/test_environment_info_service.py tests/rl/test_environment_timing.py tests/rl/test_pending_order_environment.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add trade_rl/rl/environment_info.py tests/rl/test_environment_info_service.py
git commit -m "refactor: extract environment info builder"
```

### Task 6: Integrate services into `ResidualMarketEnv`

**Files:**
- Modify: `trade_rl/rl/environment.py`
- Modify: `tests/architecture/test_environment_step_decomposition.py`

**Interfaces:**
- Consumes all service interfaces from Tasks 2-5.
- Preserves existing private compatibility methods as thin delegates where tests call them.

- [ ] **Step 1: Add service construction in `__init__`**

Instantiate the planner, projector, reward coordinator, and info builder after their dependencies are validated.

- [ ] **Step 2: Replace direct step logic with orchestration**

Call services in the existing order, apply mutable state only in the facade, and preserve pending-target discard and terminal-liquidation ordering.

- [ ] **Step 3: Run architecture and environment regression tests**

```bash
uv run pytest -q \
  tests/architecture/test_environment_step_decomposition.py \
  tests/rl \
  tests/serving/test_observation_parity.py \
  tests/serving/test_observation_snapshot_fail_closed.py \
  tests/e2e/test_stateful_order_replay.py
```

Expected: PASS.

- [ ] **Step 4: Run static checks**

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run lint-imports
```

Expected: all commands exit `0`.

- [ ] **Step 5: Commit**

```bash
git add trade_rl/rl/environment.py tests/architecture/test_environment_step_decomposition.py
git commit -m "refactor: delegate environment step responsibilities"
```

### Task 7: Add coverage ratchet and exact-head verification

**Files:**
- Modify: `pyproject.toml`
- Create: `docs/verification/2026-07-22-environment-runtime-decomposition.md`

- [ ] **Step 1: Run full coverage before setting threshold**

```bash
uv run pytest -q --cov=trade_rl --cov-branch --cov-report=json:coverage.json
```

Expected: full suite passes and writes `coverage.json`.

- [ ] **Step 2: Calculate aggregate branch percentage**

Aggregate covered and total branches for:

```text
trade_rl/rl/environment_decision.py
trade_rl/rl/environment_risk.py
trade_rl/rl/environment_reward.py
trade_rl/rl/environment_info.py
```

Set `[tool.trade_rl.critical_coverage.groups.environment_step_services]` to the measured percentage rounded down to a safe tenth. Do not change the existing `environment_runtime` group.

- [ ] **Step 3: Run complete local-equivalent checks**

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run lint-imports
uv run pytest -q --cov=trade_rl --cov-branch --cov-report=json:coverage.json
uv run python .github/check_critical_coverage.py coverage.json pyproject.toml
uv run trade-rl --help
```

Expected: every command exits `0`.

- [ ] **Step 4: Record RED and GREEN evidence**

Document commit SHAs, workflow run IDs, test count, coverage, new ratchet, artifact IDs/digests, compatibility jobs, training image, and PostgreSQL run.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml docs/verification/2026-07-22-environment-runtime-decomposition.md
git commit -m "test: ratchet environment step service coverage"
```

- [ ] **Step 6: Create stacked Draft PR**

Base: `agent/fix-architecture-followup-20260722`

Head: `agent/decompose-environment-runtime-20260722`

The PR body must state that it is stacked on PR #79, remains `NO-GO`, and will be retargeted to `main` after PR #79 merges.