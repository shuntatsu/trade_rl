# Absolute Growth Reward v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a cost-adjusted absolute log-growth reward with rolling baseline non-inferiority shaping, staged drawdown-increase shaping, reward-state observations, and terminal emergency liquidation.

**Architecture:** Keep reward mathematics in `trade_rl/rl/rewards.py` as pure typed functions. The environment resolves time windows, computes before/after reward contexts from base-bar book histories, exposes the context in observations, and records a complete reward breakdown. Hard drawdown stopping remains an environment transition rather than a large synthetic reward.

**Tech Stack:** Python 3.12, NumPy, Gymnasium 0.29.1, Stable-Baselines3 2.3.2, pytest, mypy, Ruff.

## Global Constraints

- Zero action must preserve exact hybrid/shadow book identity.
- Reward primary objective is net absolute interval log growth.
- Baseline shaping uses a 720-hour rolling window, 168-hour minimum history, and 0.015 full-window tolerance.
- Drawdown shaping is free through 5%, then piecewise linear with slopes 1, 3, and 8 through the 20% hard stop.
- Only increases in baseline hinge and drawdown severity are penalized.
- No fixed terminal reward or penalty is introduced.
- Reward-relevant rolling state must be observable.
- Environment identity must change when any reward parameter changes.

---

### Task 1: Pure reward model

**Files:**
- Modify: `trade_rl/rl/rewards.py`
- Create: `tests/rl/test_absolute_growth_rewards.py`

**Interfaces:**
- Produces `AbsoluteGrowthRewardConfig`, `RewardContext`, `RewardBreakdown`, `drawdown_severity`, `build_reward_context`, and `absolute_growth_reward`.

- [ ] Write failing tests for validation, rolling tolerance warm-up, hinge increase-only behavior, drawdown continuity/slopes, and reward breakdown arithmetic.
- [ ] Run `uv run pytest tests/rl/test_absolute_growth_rewards.py -v` and confirm the new imports or assertions fail.
- [ ] Implement the smallest pure dataclasses and functions that satisfy the tests.
- [ ] Re-run the targeted tests and confirm they pass.
- [ ] Commit with `feat: add absolute growth reward model`.

### Task 2: Reward state in observations

**Files:**
- Modify: `trade_rl/rl/observations.py`
- Modify: `tests/rl/test_observation_v2.py`

**Interfaces:**
- Consumes `RewardContext`.
- Produces observation schema `baseline_residual_observation_v3` with five additional global fields.

- [ ] Add failing tests that assert the new schema, shape, reward-state values, and emergency flag.
- [ ] Run `uv run pytest tests/rl/test_observation_v2.py -v` and confirm failure.
- [ ] Extend `build_observation` and `observation_layout` without changing per-symbol ordering.
- [ ] Re-run the observation tests and confirm they pass.
- [ ] Commit with `feat: expose reward state to policy`.

### Task 3: Environment reward integration

**Files:**
- Modify: `trade_rl/rl/environment.py`
- Modify: `tests/rl/test_environment_timing.py`
- Modify: `tests/rl/test_environment_identity.py`

**Interfaces:**
- Resolves reward windows to base bars.
- Calls `build_reward_context` before and after execution.
- Calls `absolute_growth_reward` and places all breakdown fields in `info`.

- [ ] Add failing tests for zero-action absolute reward, no repeated hinge/drawdown level penalty, full reward diagnostics, and environment digest changes.
- [ ] Run the targeted environment tests and confirm failure.
- [ ] Replace `relative_interval_reward` integration with the new reward model and configuration.
- [ ] Preserve time-limit truncation and explicit evaluation liquidation semantics.
- [ ] Re-run the targeted tests and confirm they pass.
- [ ] Commit with `feat: integrate hierarchical growth reward`.

### Task 4: Hard drawdown stop

**Files:**
- Modify: `trade_rl/rl/environment.py`
- Modify: `tests/rl/test_environment_timing.py`

**Interfaces:**
- Produces `termination_reason="drawdown_stop"` after complete emergency liquidation.

- [ ] Add failing tests that create a drawdown-stop transition and assert true termination, no truncation, flat hybrid and shadow books, liquidation costs in reward, and no fixed terminal jackpot.
- [ ] Run the focused test and confirm failure.
- [ ] Implement current-close zero-target emergency liquidation for both paired books, require completeness, merge liquidation returns, and then compute reward.
- [ ] Re-run the focused and full environment timing tests.
- [ ] Commit with `feat: terminate through emergency deleveraging`.

### Task 5: Documentation and full verification

**Files:**
- Modify: `README.md`
- Modify: `docs/RESEARCH_STATUS.md`
- Modify: `trade_rl/rl/__init__.py` if public exports are maintained there.

- [ ] Document absolute-growth reward semantics, zero-action reward behavior, rolling baseline shaping, drawdown shaping, and hard-stop semantics.
- [ ] Run `uv run ruff check .`.
- [ ] Run `uv run ruff format --check .`.
- [ ] Run `uv run mypy trade_rl`.
- [ ] Run `uv run lint-imports`.
- [ ] Run `uv run pytest --cov=trade_rl --cov-branch`.
- [ ] Inspect the final diff for unrelated changes, placeholders, stale schema names, and missing digest fields.
- [ ] Commit with `docs: describe absolute growth reward v2`.