# Target-Weight Constrained Growth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add fair target-weight PPO/Lagrangian growth profiles whose objective is execution-adjusted net log growth, while keeping hard safety in the environment and validating time-limit truncation semantics.

**Architecture:** Keep the existing `ActionMode.TARGET_WEIGHT`, `PreTradeRisk`, execution/accounting, reward tracker, cost critic and Lagrangian PPO implementations. Add fail-closed reward/profile contracts in `TrainingRunConfig`, reject the unstable zero-tolerance progressive hinge in `RewardConfig`, then add three example profiles and tests that prove all non-experimental fields are identical. Do not change the existing residual profiles.

**Tech Stack:** Python 3.12, dataclasses, pytest, Gymnasium, Stable-Baselines3, JSON example profiles.

## Global Constraints

- Main reward is execution-adjusted net log return only.
- G1 profiles use `gamma = 1.0` and omit `discount_half_life_hours`.
- All target-weight comparison profiles use the same action, environment, risk, execution, encoder, BC, rollout and seed settings.
- Hard safety remains enforced by `PreTradeRisk` and accounting independently of Lagrangian multipliers.
- Baseline, drawdown, terminal equity, margin deficit and projection shaping weights are zero in pure-growth profiles.
- Existing residual growth profiles remain research controls.
- No production code is changed before a failing test is committed.

---

### Task 1: Reward validation

**Files:**
- Modify: `tests/rl/test_reward_validation.py`
- Modify: `trade_rl/rl/rewards.py`

**Interfaces:**
- Produces: `RewardConfig.__post_init__` rejects `baseline_tolerance == 0.0` with `baseline_progressive_power > 1.0`.

- [ ] Add a failing test constructing `RewardConfig(baseline_tolerance=0.0, baseline_progressive_power=2.0)` and asserting a `ValueError` mentioning tolerance and progressive power.
- [ ] Run the targeted test and confirm it fails because the configuration is currently accepted.
- [ ] Add the minimal validation directly after the progressive-power lower-bound check.
- [ ] Run `pytest tests/rl/test_reward_validation.py -q`.
- [ ] Commit the test and implementation.

### Task 2: Pure-growth training contract

**Files:**
- Modify: `tests/workflows/test_training_run_config.py`
- Modify: `trade_rl/workflows/training_run.py`

**Interfaces:**
- Produces: `TrainingRunConfig.is_pure_growth_profile() -> bool`.
- Produces: `TrainingRunConfig.__post_init__` rejects objective mixing for explicit pure-growth runs and rejects `gamma == 1.0` combined with `discount_half_life_hours`.

- [ ] Add helpers in the test module that build a target-weight pure-growth mapping.
- [ ] Add failing tests for non-zero excess, drawdown, baseline, projection, terminal and margin weights in a pure-growth profile.
- [ ] Add a failing test for `gamma = 1.0` plus a configured half-life.
- [ ] Add a passing test proving target-weight pure-growth PPO and Lagrangian configurations are accepted.
- [ ] Implement the smallest named profile contract. Use a new top-level optional `objective` field with allowed values `legacy_shaped` and `pure_net_log_growth`; include it in config identity. Default to `legacy_shaped` for backward compatibility.
- [ ] For `pure_net_log_growth`, require absolute growth weight 1.0 and all shaping/terminal/margin weights 0.0.
- [ ] For `pure_net_log_growth` with `gamma == 1.0`, reject any non-null half-life.
- [ ] Run `pytest tests/workflows/test_training_run_config.py -q`.
- [ ] Commit the contract.

### Task 3: Target-weight comparison profiles

**Files:**
- Create: `examples/binance-multitimeframe/training-target-weight-growth-ppo.json`
- Create: `examples/binance-multitimeframe/training-target-weight-constrained-growth.json`
- Create: `examples/binance-multitimeframe/training-target-weight-constrained-growth-discounted.json`
- Create: `examples/binance-multitimeframe/walk-forward-target-weight-constrained-growth.json`
- Create: `tests/examples/test_target_weight_constrained_growth_profiles.py`

**Interfaces:**
- G1-PPO: target-weight, pure net log growth, gamma 1.0, PPO.
- G1-Lagrangian: same recipe except algorithm and cost/Lagrangian fields.
- D168-Lagrangian: same as G1-Lagrangian except gamma and 168-hour half-life.

- [ ] Add tests that load the three profiles and assert `objective == "pure_net_log_growth"`, direct target weights, explicit zero shaping weights, identical hard-risk and architecture settings, and expected algorithms.
- [ ] Assert G1-PPO and G1-Lagrangian differ only in algorithm-specific training fields.
- [ ] Assert D168 differs from G1-Lagrangian only in `gamma` and `discount_half_life_hours`.
- [ ] Assert walk-forward candidates exactly match the standalone candidate digests and include nominal, joint 2x and joint 3x stress scenarios.
- [ ] Commit tests and confirm CI fails because files do not exist.
- [ ] Create profiles by copying the current target-weight `training-full.json` recipe, replacing reward with explicit pure-growth weights, retaining identical BC and architecture, and importing the existing seven-cost Lagrangian values.
- [ ] Create the walk-forward profile using the repository's existing candidate/stress schema.
- [ ] Run the target example tests.
- [ ] Commit profiles.

### Task 4: Time-limit truncation regression

**Files:**
- Modify: `tests/rl/test_environment_time_config.py` or the closest existing transition test.
- Modify only if required: `trade_rl/rl/environment_transition.py`, `trade_rl/rl/transition.py`, or the SB3 adapter handling `terminal_observation`.

**Interfaces:**
- Time limit with `liquidate_on_end = false` produces `truncated=True`, `terminated=False`, no liquidation return, and preserves the final observation for bootstrap.
- Insolvency or explicit liquidation remains true termination and is not bootstrapped.

- [ ] Add failing integration assertions for time-limit truncation and final-observation propagation.
- [ ] Run the targeted tests and inspect whether existing behavior already satisfies the contract.
- [ ] If behavior already passes, retain the tests as regression coverage and make no production change.
- [ ] If it fails, apply the smallest adapter fix without changing economic accounting.
- [ ] Run all transition and SB3 rollout tests.
- [ ] Commit the regression coverage and any minimal fix.

### Task 5: Documentation and verification

**Files:**
- Modify: `examples/binance-multitimeframe/README.md` or the canonical training guide.
- Modify: design/plan checkboxes as completed.

- [ ] Document G1-PPO as required control, G1-Lagrangian as candidate, D168 as time-preference ablation, and legacy full as non-default research comparison.
- [ ] Run `pytest`, Ruff, MyPy and import-linter in CI.
- [ ] Verify profile digests, checkpoint/resume compatibility and serving identity tests.
- [ ] Open a draft PR and inspect all GitHub Actions checks.
- [ ] Do not merge while any required check is failing.
