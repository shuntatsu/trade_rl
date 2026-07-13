# Residual Reward Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the residual RL core optimize, observe, terminate, and statistically evaluate the same paired excess-log-growth objective.

**Architecture:** Add a strict long-horizon gamma contract at the typed training boundary, enrich only the residual observation with paired shadow-relative state, separate hybrid and shadow insolvency in the environment and reward function, and centralize paired per-period log excess in evaluation. The removed direct-action and DSR paths remain absent.

**Tech Stack:** Python 3.12, dataclasses, NumPy, Gymnasium, Stable-Baselines3, pytest, Hypothesis, Ruff, mypy, Import Linter.

## Global Constraints

- Production remains **NO-GO**.
- Preserve exact zero-action baseline identity.
- Preserve the current OHLCV, self-financing accounting, and next-open execution contracts.
- Do not reintroduce `mars_lite`, direct-action PPO, turnover reward penalties, DSR, or compatibility adapters.
- One action continues to control one complete decision interval.
- Paired statistical decisions use excess log returns.

---

### Task 1: Lock the residual gamma contract

**Files:**
- Modify: `trade_rl/rl/training.py`
- Modify: `trade_rl/cli/app.py`
- Modify: `tests/rl/test_ensemble_training.py`
- Modify: `tests/cli/test_cli.py`

- [ ] Add failing tests for the default, minimum, explicit override, and backend propagation.
- [ ] Confirm RED in GitHub Actions.
- [ ] Implement constants `DEFAULT_RESIDUAL_GAMMA = 0.99`, `MIN_RESIDUAL_GAMMA = 0.95`, and `allow_low_gamma` validation.
- [ ] Implement CLI default and v2 resolved output.
- [ ] Run focused tests and commit.

### Task 2: Expose paired state and separate insolvency

**Files:**
- Modify: `trade_rl/rl/observations.py`
- Modify: `trade_rl/rl/rewards.py`
- Modify: `trade_rl/rl/environment.py`
- Modify: `tests/rl/test_environment_timing.py`

- [ ] Add failing tests for paired layout, reset zeros, identity parity, hybrid-only penalty, and shadow-only invalidity without penalty.
- [ ] Confirm RED in GitHub Actions.
- [ ] Make observations consume both hybrid and shadow books.
- [ ] Add `hybrid_insolvency_penalty` and separate terminal flags.
- [ ] Run focused tests and commit.

### Task 3: Align paired inference with the reward quantity

**Files:**
- Modify: `trade_rl/evaluation/comparisons.py`
- Modify: `tests/evaluation/test_comparisons.py`

- [ ] Add a failing large-return test where arithmetic and log differences diverge.
- [ ] Confirm RED in GitHub Actions.
- [ ] Bootstrap period excess log returns and retain simple differences only as diagnostics.
- [ ] Run evaluation tests and commit.

### Task 4: Document and verify the complete contract

**Files:**
- Modify: `README.md`
- Modify: `README.ja.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/RESEARCH_STATUS.md`
- Modify: `tests/test_architecture_contract.py`

- [ ] Document gamma, paired state, insolvency, and log-bootstrap semantics.
- [ ] Assert no direct-action or DSR path exists.
- [ ] Run Ruff, format, mypy, Import Linter, full pytest with branch coverage, and CLI smoke test.
- [ ] Inspect the complete diff and GitHub Actions before merge.