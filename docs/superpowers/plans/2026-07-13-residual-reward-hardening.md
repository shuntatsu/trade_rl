# Residual Reward Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the rebuilt residual RL core optimize, observe, terminate, and statistically evaluate the same paired excess-log-growth objective.

**Architecture:** Add a strict long-horizon gamma contract at the typed training boundary, enrich only the residual observation with paired shadow-relative state, separate hybrid and shadow insolvency in the environment and reward function, and centralize paired per-period log excess in evaluation. The removed direct-action and DSR paths remain absent.

**Tech Stack:** Python 3.12, dataclasses, NumPy, Gymnasium, Stable-Baselines3, pytest, Hypothesis, Ruff, mypy, Import Linter.

## Global Constraints

- Production remains **NO-GO**.
- Preserve exact zero-action baseline identity.
- Do not reintroduce `mars_lite`, direct-action PPO, turnover reward penalties, DSR, or compatibility adapters.
- One action continues to control one complete decision interval.
- Reward scaling remains numerical only.
- Paired statistical decisions use excess log returns.

---

### Task 1: Lock the residual gamma contract

**Files:**
- Modify: `trade_rl/rl/training.py`
- Modify: `trade_rl/cli/app.py`
- Modify: `tests/rl/test_ensemble_training.py`
- Modify: `tests/cli/test_cli.py`

**Interfaces:**
- Produces constants `DEFAULT_RESIDUAL_GAMMA = 0.99` and `MIN_RESIDUAL_GAMMA = 0.95`.
- Extends `ResidualTrainingConfig` with `allow_low_gamma: bool = False`.
- Adds CLI option `--allow-low-gamma`.

- [ ] Add failing tests for the default, minimum, explicit override, and backend propagation.
- [ ] Run the PR CI and confirm RED is caused by the missing contract.
- [ ] Implement the typed validation and CLI v2 output.
- [ ] Run focused tests and commit.

### Task 2: Expose paired state and separate insolvency

**Files:**
- Modify: `trade_rl/rl/observations.py`
- Modify: `trade_rl/rl/rewards.py`
- Modify: `trade_rl/rl/environment.py`
- Modify: `tests/rl/test_environment_timing.py`

**Interfaces:**
- `build_observation` consumes `hybrid` and `shadow` books.
- `ResidualMarketEnvConfig` adds `hybrid_insolvency_penalty: float = 1.0`.
- `relative_interval_reward` consumes `hybrid_insolvent` and `hybrid_insolvency_penalty`.
- Environment `info` adds `hybrid_insolvent`, `shadow_insolvent`, and `rollout_valid`.

- [ ] Add failing tests for paired layout, reset zeros, identity parity, hybrid-only penalty, and shadow-only invalidity without penalty.
- [ ] Confirm RED in CI.
- [ ] Implement paired observation fields and terminal semantics.
- [ ] Run focused tests and commit.

### Task 3: Align paired inference with the reward quantity

**Files:**
- Modify: `trade_rl/evaluation/comparisons.py`
- Modify: `tests/evaluation/test_comparisons.py`

**Interfaces:**
- `mean_period_excess` becomes mean period excess log return.
- `PairedComparison` adds `mean_period_simple_excess` for descriptive arithmetic differences.
- Bootstrap input is the period excess-log-return series.

- [ ] Add a failing large-return test where arithmetic and log differences diverge.
- [ ] Confirm RED in CI.
- [ ] Implement log-difference bootstrap and diagnostic arithmetic mean.
- [ ] Run evaluation tests and commit.

### Task 4: Document and verify the complete contract

**Files:**
- Modify: `README.md`
- Modify: `README.ja.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/RESEARCH_STATUS.md`
- Modify: `tests/test_architecture_contract.py` only if an explicit DSR absence assertion is needed.

- [ ] Document gamma, paired state, insolvency, and log-bootstrap semantics.
- [ ] Verify no direct-action or DSR path exists.
- [ ] Run Ruff, format, mypy, Import Linter, full pytest with branch coverage, and CLI smoke test.
- [ ] Inspect the complete diff against the design.
- [ ] Open a draft PR and require successful GitHub Actions before merge.