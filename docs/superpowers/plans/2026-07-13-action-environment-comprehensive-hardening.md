# Action and Environment Comprehensive Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Per user instruction, implementation and test authoring precede all test execution.

**Goal:** Implement a causally correct action/environment v3 with explicit market calendar, action, risk, execution, observation, reward, training and serving contracts.

**Architecture:** Extend the existing baseline-anchored design through focused immutable configuration objects and structured result types. Keep zero-action baseline identity and paired hybrid/shadow evaluation while making dynamic action dimensions, economic terminal states and fold-fitted normalization explicit.

**Tech Stack:** Python 3.12, NumPy, Gymnasium 0.29.1, Stable-Baselines3 2.3.2, PyTorch 2.3.1, pytest, Hypothesis, Ruff, mypy.

## Global Constraints

- Do not execute tests until all implementation and test files are complete.
- Preserve exact zero-action baseline identity.
- Preserve causal next-open execution and shared hybrid/shadow random numbers.
- Fail closed on serving identity or action violations.
- Keep production status NO-GO.

---

### Task 1: Market calendar and feature metadata

**Files:**
- Modify: `trade_rl/data/market.py`
- Modify: `trade_rl/data/__init__.py`

- [ ] Add `MarketCalendarKind` and explicit regular/session validation.
- [ ] Add causal elapsed-hours, lookback-index and forward-window helpers.
- [ ] Add optional feature staleness and execution metadata arrays with immutable validation.

### Task 2: Trend baseline modes

**Files:**
- Modify: `trade_rl/strategies/trend.py`
- Modify: `trade_rl/strategies/__init__.py`

- [ ] Add trend mode configuration and one-symbol-safe time-series targets.
- [ ] Preserve existing cross-sectional behavior for multi-symbol `auto` mode.

### Task 3: Residual action schema v2

**Files:**
- Modify: `trade_rl/rl/actions.py`
- Modify: `trade_rl/rl/__init__.py`

- [ ] Add `ActionSpec`, dynamic dimensions and strict/clip validation modes.
- [ ] Add independent fast, slow, risk, optional alpha and optional factor controls.
- [ ] Preserve exact zero-action baseline identity and emit composition diagnostics.

### Task 4: Hard and soft pre-trade risk

**Files:**
- Modify: `trade_rl/risk/pretrade.py`
- Modify: `trade_rl/risk/__init__.py`

- [ ] Apply hard concentration, gross and emergency drawdown limits after soft turnover throttling.
- [ ] Add final invariant validation and projection-distance diagnostics.

### Task 5: Economic book and execution constraints

**Files:**
- Modify: `trade_rl/simulation/accounting.py`
- Modify: `trade_rl/simulation/execution.py`
- Modify: `trade_rl/simulation/__init__.py`

- [ ] Represent insolvency, margin calls and liquidation as economic outcomes.
- [ ] Add minimum notional, lot size, tick, borrow, margin, latency and per-bar execution inputs.
- [ ] Preserve paired deterministic randomness while allowing episode-to-episode variation.

### Task 6: Reward schema v3

**Files:**
- Modify: `trade_rl/rl/rewards.py`

- [ ] Add absolute log growth primary component and excess growth secondary component.
- [ ] Add incremental drawdown dead-zone penalty.
- [ ] Add rolling cumulative baseline-underperformance progressive hinge.
- [ ] Add continuous terminal equity/margin penalty and structured breakdown.

### Task 7: Observation schema v3 and normalizer

**Files:**
- Modify: `trade_rl/rl/observations.py`
- Create: `trade_rl/rl/normalization.py`

- [ ] Add per-feature masks, staleness, execution state, requested targets, previous action, cash/net/gross and margin fields.
- [ ] Add train-range-only fitted immutable normalizer with content digest.

### Task 8: Environment v3

**Files:**
- Modify: `trade_rl/rl/environment.py`

- [ ] Integrate dynamic action spec and alpha/factor identity.
- [ ] Add episode curriculum and causal initial-state samplers.
- [ ] Generate deterministic unique episode seeds and paired executor RNG streams.
- [ ] Convert economic failures to structured terminal transitions.
- [ ] Emit reward, action, projection, execution and termination diagnostics.

### Task 9: Training and serving contracts

**Files:**
- Modify: `trade_rl/rl/training.py`
- Modify: `trade_rl/cli/app.py`
- Modify: `trade_rl/domain/policies.py`
- Modify: `trade_rl/serving/bundle.py`
- Modify: `trade_rl/serving/runtime.py`

- [ ] Validate decision-hour/gamma identity.
- [ ] Add PPO log-std, target-KL, SDE and network controls.
- [ ] Bind action dimension, alpha/factor/normalizer identities to training and serving artifacts.
- [ ] Enforce strict serving actions.

### Task 10: Documentation and exports

**Files:**
- Modify: `README.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify relevant package `__init__.py` files.

- [ ] Document v3 contracts, migration behavior and NO-GO limitations.

### Task 11: Regression and contract tests

**Files:**
- Modify existing tests under `tests/data`, `tests/strategies`, `tests/rl`, `tests/risk`, `tests/simulation`, `tests/serving`, and `tests/cli`.
- Create focused new test files where isolation improves clarity.

- [ ] Add all regression tests without executing them.

### Task 12: Final verification

- [ ] Run `ruff check .`.
- [ ] Run `ruff format --check .`.
- [ ] Run `mypy trade_rl`.
- [ ] Run `lint-imports`.
- [ ] Run `pytest --cov=trade_rl --cov-branch`.
- [ ] Fix failures and rerun the complete verification sequence.
- [ ] Review the final diff against every design requirement.
