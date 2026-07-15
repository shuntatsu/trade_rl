# Complete 96-Feature Multi-Timeframe Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the exact 24-feature-per-clock causal contract and make the maintained Binance example run 96-feature, one-hour, three-seed, two-fold research.

**Architecture:** A shared causal feature engine calculates both base and auxiliary native-clock events. The existing multi-timeframe alignment layer carries only available events onto the one-hour clock. The example validates the exact contract before training and uses risk-tilt-free relative allocation actions.

**Tech Stack:** Python 3.12, NumPy, Gymnasium, Stable-Baselines3 PPO, pytest, Ruff, Mypy.

## Global Constraints

- Four native clocks: `15m`, `1h`, `4h`, `1d`.
- Exactly 24 features per clock and 96 dataset features total.
- No future or backward-shifted Ichimoku input.
- One-hour decision interval.
- `risk_tilt_enabled=false` and three net-zero factors.
- Three seeds and two sealed outer folds.
- Production status remains `NO-GO`.

---

### Task 1: Lock the indicator contract

**Files:** `tests/data/test_indicator_features.py`, `tests/integrations/test_binance_multitimeframe.py`

- [x] Add failing tests for indicator numerical validity and prefix causality.
- [x] Add an exact ordered 96-feature preset assertion.
- [x] Verify RED against the four-kind legacy feature implementation.

### Task 2: Implement shared causal indicators

**Files:** `trade_rl/data/features.py`, `trade_rl/data/contracts.py`, `trade_rl/data/builder.py`, `trade_rl/data/multitimeframe.py`

- [x] Add maintained indicator feature kinds.
- [x] Implement causal return, volatility, volume, funding, momentum, trend, range, volume-flow, and Ichimoku events.
- [x] Return source-window lineage and use it for native availability.
- [x] Run focused data and multi-timeframe tests.

### Task 3: Publish the 96-feature Binance preset

**Files:** `trade_rl/integrations/binance.py`, `trade_rl/data/config.py`

- [x] Generate the same ordered 24 features for each requested clock.
- [x] Align funding causally on every native clock through the shared event cache.
- [x] Parse feature timeframe from JSON configurations.
- [x] Reject duplicate/base timeframe requests and retain default lightweight compatibility.

### Task 4: Upgrade the complete example

**Files:** `examples/binance-multitimeframe/run_full_research.py`, `training-full.json`, `walk-forward-full.json`, `tests/examples/test_binance_multitimeframe_full_assets.py`

- [x] Require exactly 96 features and reject legacy datasets.
- [x] Set one-hour training and environment decisions.
- [x] Disable risk tilt and attach three relative factors.
- [x] Set 262,144 full timesteps and 65,536 walk-forward timesteps.
- [x] Preserve three seeds, two folds, repeated dataset identity, and artifact verification.

### Task 5: Verify and publish

- [ ] Run focused and full pytest suites.
- [ ] Run Ruff, format check, Mypy, import architecture, and dead-code checks.
- [ ] Build a deterministic synthetic 96-feature dataset and validate finite observations.
- [ ] Publish the change on an isolated GitHub branch and open a draft PR.
- [ ] Inspect GitHub Actions and report any remaining environment-only limitations.
