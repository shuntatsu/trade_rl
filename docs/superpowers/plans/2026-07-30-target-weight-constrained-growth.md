# Target-Weight Constrained Growth Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Add fair target-weight PPO/Lagrangian growth profiles whose objective is execution-adjusted net log growth, while keeping hard safety in the environment and preserving continuing-task truncation semantics.

**Architecture:** Keep the existing `ActionMode.TARGET_WEIGHT`, `PreTradeRisk`, execution/accounting, reward tracker, cost critic and Lagrangian PPO implementations. Bind pure growth to the canonical `RewardConfig` weight identity, reject the unstable zero-tolerance progressive hinge, add canonical standalone profiles, let walk-forward configurations reference those profiles instead of duplicating them, and evaluate the sealed fold-seed-scenario evidence through a deterministic production gate.

**Tech Stack:** Python 3.12, dataclasses, NumPy, pytest, Gymnasium, Stable-Baselines3, JSON example profiles.

## Global constraints

- Main reward is execution-adjusted net log return only.
- G1 profiles use `gamma = 1.0` and omit `discount_half_life_hours`.
- All target-weight comparison profiles use the same action, environment, risk, execution, encoder, BC, rollout and seed settings.
- Hard safety remains enforced by `PreTradeRisk` and accounting independently of Lagrangian multipliers.
- Baseline, drawdown, terminal equity, margin deficit and projection shaping weights are zero in pure-growth profiles.
- The artificial 720-hour training boundary is hidden from the policy with `finite_horizon_observation = false` and is handled as mark-to-market truncation.
- Production selection uses unseen economic evidence, never episode reward values from different objectives.
- Existing residual growth profiles remain research controls.
- No production code is changed before a failing test is committed.

---

### Task 1: Reward validation

**Files:**
- Create: `tests/rl/test_reward_profile_contracts.py`
- Modify: `trade_rl/rl/rewards.py`

- [x] Add a failing test for `baseline_tolerance = 0` with progressive power greater than one.
- [x] Reject that unstable configuration in `RewardConfig.__post_init__`.
- [x] Add `RewardConfig.is_pure_net_log_growth()` as the canonical reward identity.
- [x] Require terminal- and margin-disabled reward configurations to satisfy the exact pure-net-log-growth weight contract.
- [ ] Confirm the targeted and full reward suites in CI.

### Task 2: Pure-growth training contract

**Files:**
- Create: `tests/workflows/test_pure_growth_training_contract.py`

- [x] Add passing coverage for explicit pure target-weight growth.
- [x] Add failing coverage for excess, drawdown, baseline and projection objective mixing.
- [x] Preserve legacy shaping through the existing terminal and margin defaults.
- [x] Keep the reward identity inside existing checkpoint/resume/serving digests instead of adding a parallel top-level objective field.
- [ ] Confirm workflow configuration and identity suites in CI.

### Task 3: Target-weight comparison profiles

**Files:**
- Create: `examples/binance-multitimeframe/training-target-weight-growth-ppo.json`
- Create: `examples/binance-multitimeframe/training-target-weight-constrained-growth.json`
- Create: `examples/binance-multitimeframe/training-target-weight-constrained-growth-discounted.json`
- Create: `examples/binance-multitimeframe/walk-forward-target-weight-constrained-growth.json`
- Create: `tests/examples/test_target_weight_constrained_growth_profiles.py`

- [x] Add G1-PPO as the gamma-one pure-growth control.
- [x] Add G1-Lagrangian using the same target-weight, BC, encoder, hard-risk and execution recipe.
- [x] Add D168-Lagrangian differing from G1-Lagrangian only by gamma and the real-time half-life.
- [x] Explicitly disable terminal, margin, baseline, drawdown, excess and projection shaping.
- [x] Explicitly set `liquidate_on_end = false`.
- [x] Explicitly set `finite_horizon_observation = false` so the policy cannot exploit the artificial window boundary.
- [x] Add complete profile-equality and cost-schema tests.
- [ ] Confirm all profile tests in CI.

### Task 4: Canonical walk-forward references

**Files:**
- Modify: `trade_rl/workflows/market_walk_forward_config.py`
- Create: `examples/binance-multitimeframe/walk-forward-target-weight-constrained-growth.json`

- [x] Add candidate `run_file` support while preserving the legacy embedded `run` format.
- [x] Require exactly one of `run` or `run_file` per candidate.
- [x] Resolve run files, referenced artifacts and resume paths from their source configuration directories.
- [x] Delegate the expanded payload to the existing canonical walk-forward validator.
- [x] Prove walk-forward candidate digests match the standalone profile digests.
- [x] Configure nominal, joint 2x and joint 3x evidence scenarios.
- [ ] Confirm existing embedded-run walk-forward tests remain green in CI.

### Task 5: Time-limit truncation regression

- [x] Confirm existing environment coverage distinguishes mark-to-market truncation from liquidation termination.
- [x] Confirm `liquidate_on_end = false` adds no liquidation return.
- [x] Confirm structured terminal observations are rehydrated for SB3.
- [x] Confirm Cost-Critic/Lagrangian rollout code bootstraps reward and cost values only for `TimeLimit.truncated`.
- [x] Avoid unnecessary production changes because the canonical adapter already implements the required semantics.
- [x] Hide the remaining-horizon observation from the new continuing-task profiles.
- [ ] Confirm transition, vector-environment and Cost-Critic suites in CI.

### Task 6: Deterministic production gate

**Files:**
- Create: `trade_rl/evaluation/target_weight_growth_gate.py`
- Create: `tests/evaluation/test_target_weight_growth_gate.py`

- [x] Define validated fold-seed-scenario evidence cells with paired baseline growth, catastrophic counts and soft-constraint estimates.
- [x] Require complete 6-fold × 3-seed support for nominal, joint 2x and joint 3x scenarios.
- [x] Implement fold-cluster bootstrap with fixed sample count and seed.
- [x] Require positive nominal growth, positive paired baseline difference, cross-fold and cross-seed stability, zero catastrophic events and verified identity.
- [x] Require every soft-constraint fold estimate and pooled one-sided upper bound to remain within budget.
- [x] Require joint 2x paired growth to remain positive and joint 3x growth to remain non-negative.
- [x] Select Lagrangian over PPO only when the paired Lagrangian-minus-PPO growth lower bound is positive; otherwise prefer PPO.
- [x] Bind inputs and decisions to canonical SHA-256 evidence digests.
- [ ] Confirm gate tests, MyPy and coverage in CI.

### Task 7: Documentation and verification

**Files:**
- Modify: `docs/BINANCE.md`
- Modify: design and plan records

- [x] Document G1-PPO as required control, G1-Lagrangian as candidate, D168 as time-preference ablation, and legacy full as a non-default research comparison.
- [x] Document the production gate, 2x/3x stress requirements and PPO/Lagrangian selection rule.
- [x] Open draft PR #253.
- [x] Remove pre-existing Ruff-format debt so the full CI pipeline can execute.
- [ ] Run Ruff, format, MyPy, import-linter, full pytest and critical coverage in CI.
- [ ] Verify compatibility jobs, PostgreSQL catalog and training-image smoke.
- [ ] Update the PR with final evidence and keep it unmerged until every required check is green.
