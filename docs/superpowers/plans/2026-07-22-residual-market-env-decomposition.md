# ResidualMarketEnv Decomposition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce `ResidualMarketEnv` to a Gymnasium orchestration facade while preserving its public API, economic behavior, deterministic replay, observation parity, reward semantics, and info contract.

**Architecture:** Extract four focused low-level services inside `trade_rl.rl`: episode contract sampling, observation assembly, target execution/accounting, and terminal transition resolution. The environment owns mutable episode state and applies immutable service results; services do not import `ResidualMarketEnv` and therefore cannot reach through the facade implicitly.

**Tech Stack:** Python 3.12, Gymnasium, NumPy, pytest, Mypy, Import Linter, GitHub Actions.

## Global Constraints

- Production remains `NO-GO`; no direct exchange routing is added.
- `ResidualMarketEnv.reset()`, `step()`, `observation_snapshot()`, action/observation spaces, info keys, environment digest payload, and deterministic seed behavior remain compatible.
- Existing stateful `OrderBookState` reconciliation and `BookState` accounting remain the only maintained economic execution path.
- No coverage threshold may be reduced.
- New services remain in the existing `trade_rl.rl` Import Linter layer.

---

### Task 1: Architecture contract

**Files:**
- Create: `tests/architecture/test_environment_decomposition.py`

**Interfaces:**
- Produces: required service class names and delegation markers used by later tasks.

- [ ] Add a failing contract test importing `EpisodeContractSampler`, `EnvironmentObservationAssembler`, `EnvironmentExecutionCoordinator`, and `EnvironmentTerminationCoordinator`.
- [ ] Assert `ResidualMarketEnv` delegates episode sampling, observation construction, stateful target execution, and terminal resolution to those services.
- [ ] Run the architecture test and confirm it fails because the services do not exist.

### Task 2: Episode contract sampler

**Files:**
- Create: `trade_rl/rl/environment_episode.py`
- Modify: `trade_rl/rl/environment.py`
- Test: `tests/rl/test_environment_episode_sampler.py`

**Interfaces:**
- Produces: `EpisodeContract(start_index: int, end_index: int, hours: float)` and `EpisodeContractSampler.sample(options, rng)`.

- [ ] Add deterministic tests for explicit starts, duration choices, regime-balanced sampling, stress-tail sampling, and invalid options.
- [ ] Move episode-end calculation, valid-start caching, and episode contract sampling into `EpisodeContractSampler`.
- [ ] Delegate the compatibility private methods from `ResidualMarketEnv` to the sampler.

### Task 3: Stateful execution coordinator

**Files:**
- Create: `trade_rl/rl/environment_execution.py`
- Modify: `trade_rl/rl/environment.py`
- Test: `tests/rl/test_environment_execution_coordinator.py`

**Interfaces:**
- Produces: `TargetExecutionRequest`, `EnvironmentExecutionCoordinator.execute_target()`, `merge_liquidation_return()`, `liquidation_complete()`, and `execution_observation_state()`.

- [ ] Add tests proving target identity stability, order-book carry, liquidation merging, completion detection, and position-age updates.
- [ ] Move target reconciliation and execution-observation bookkeeping out of the environment.
- [ ] Keep `MarketExecutor.execute_orders()` and `BookState` as the economic authorities.

### Task 4: Observation assembler

**Files:**
- Create: `trade_rl/rl/environment_observation.py`
- Modify: `trade_rl/rl/environment.py`
- Test: `tests/rl/test_environment_observation_assembler.py`

**Interfaces:**
- Produces: `EnvironmentObservationRuntime` and `EnvironmentObservationAssembler` methods `pending_order_state()`, `flat_pair()`, `snapshot()`, and `observation()`.

- [ ] Add parity tests for flat observations, normalized observations, structured sequence observations, pending-order state, and serving snapshots.
- [ ] Move observation construction and snapshot export into the assembler.
- [ ] Keep dataset/action/normalizer identities and dtypes unchanged.

### Task 5: Terminal transition coordinator

**Files:**
- Create: `trade_rl/rl/environment_transition.py`
- Modify: `trade_rl/rl/environment.py`
- Test: `tests/rl/test_environment_transition_coordinator.py`

**Interfaces:**
- Produces: `EnvironmentTransitionOutcome` and `EnvironmentTerminationCoordinator.resolve()`.

- [ ] Add tests for drawdown emergency liquidation, end-of-episode liquidation, incomplete liquidation failure, minimum-equity termination, and ordinary time-limit truncation.
- [ ] Move liquidation and economic-transition classification out of `step()`.
- [ ] Return immutable outcomes and apply them in the environment facade.

### Task 6: Exact-head verification

**Files:**
- Create: `docs/verification/2026-07-22-residual-market-env-decomposition.md`
- Modify: PR #78 body.

- [ ] Run Ruff, format, Mypy, Import Linter, dead-code checks, serving smoke, full pytest with branch coverage, critical coverage ratchet, CLI smoke, Ubuntu/Windows compatibility, Studio verification, training-image probe, and PostgreSQL tests.
- [ ] Record the exact head SHA, workflow run IDs, test counts, coverage, and artifact digests.
- [ ] Keep PR #78 draft until review or explicit merge instruction.
