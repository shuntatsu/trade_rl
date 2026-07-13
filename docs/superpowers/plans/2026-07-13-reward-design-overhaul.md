# Reward Design Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align reward optimization, decision timing, observability, statistical inference, and release metadata for residual and direct-weight RL.

**Architecture:** Keep baseline-residual paired log growth as the primary objective, but make its horizon, state, termination, and inference internally consistent. Refactor direct-weight execution to use the shared interval execution core and express optional turnover aversion in return units inside the reward scale. Gate DSR behind an explicit research-only contract.

**Tech Stack:** Python 3.11+, Gymnasium, NumPy, Stable-Baselines3 PPO, pytest, GitHub Actions.

## Global Constraints

- Preserve exact identity-action parity between residual hybrid and shadow books.
- Do not alter trend-family, alpha-model, HTF, post-processing, or execution-cost formulas except where required for interval execution reuse.
- Use excess log returns for all residual optimization and paired statistical decisions.
- `reward_scale` must be numerical only and must not change the economic objective.
- Production/release registration remains disabled for residual research runs.

---

### Task 1: Residual gamma contract and metadata

**Files:**
- Modify: `mars_lite/pipeline/cli.py`
- Modify: `mars_lite/pipeline/training_engine.py`
- Modify: `mars_lite/pipeline/residual_pipeline.py`
- Modify: `mars_lite/pipeline/residual_walk_forward.py`
- Test: `tests/test_residual_reward_contract.py`
- Test: `tests/test_residual_wf_config.py`

**Interfaces:**
- Produces: `resolve_training_gamma(action_mode: str, gamma: float, allow_low_residual_gamma: bool) -> float`
- Produces CLI flag: `--allow-low-residual-gamma`

- [ ] Write tests asserting residual default gamma is `0.99`, residual gamma below `0.95` fails without the override, direct gamma remains configurable, and manifests/reports record the effective value.
- [ ] Run the focused tests and confirm they fail for the missing contract.
- [ ] Implement gamma resolution in one shared function and route all residual training and walk-forward configuration through it.
- [ ] Run the focused tests and existing residual configuration tests.
- [ ] Commit.

### Task 2: Residual paired-state observation and termination

**Files:**
- Modify: `mars_lite/env/baseline_residual_env.py`
- Modify: `mars_lite/eval/relative_evaluation.py`
- Test: `tests/test_baseline_residual_env.py`
- Test: `tests/test_residual_reward_contract.py`

**Interfaces:**
- Residual per-symbol observation appends `hybrid_weight - shadow_weight`.
- Residual global observation appends log wealth ratio, shadow drawdown, shadow gross, and turnover excess.
- Environment constructor adds `hybrid_insolvency_penalty: float = 1.0` in unscaled log-return units.
- `info` adds `hybrid_insolvent`, `shadow_insolvent`, and `rollout_valid`.

- [ ] Write tests for observation layout, exact identity parity, hybrid-only penalty, shadow-only no-penalty termination, and fail-closed evaluation.
- [ ] Run tests and confirm expected failures.
- [ ] Add paired state to the residual observation without changing the direct schema.
- [ ] Replace the fixed `-reward_scale` terminal reward with realized paired reward plus hybrid-only penalty.
- [ ] Make relative evaluation reject invalid shadow-insolvent rollouts.
- [ ] Run focused and parity tests.
- [ ] Commit.

### Task 3: Log-return statistical alignment

**Files:**
- Modify: `mars_lite/eval/relative_evaluation.py`
- Modify: `mars_lite/learning/relative_val_selection.py`
- Test: `tests/test_relative_evaluation.py`
- Test: `tests/test_relative_val_selection.py`

**Interfaces:**
- Produces: `excess_log_return_series(hybrid_returns, shadow_returns) -> np.ndarray`
- `paired.mean_base_bar_excess` is retained for compatibility but becomes the mean excess log return and adds `mean_base_bar_simple_excess` as diagnostic-only.

- [ ] Write tests with non-small returns where simple differences and log differences diverge.
- [ ] Confirm checkpoint blocks and bootstrap currently use the wrong quantity.
- [ ] Implement one shared excess-log series helper and use it for validation blocks, bootstrap, and paired means.
- [ ] Run focused tests.
- [ ] Commit.

### Task 4: Direct reward scale invariance

**Files:**
- Modify: `mars_lite/env/portfolio_env.py`
- Modify: `mars_lite/pipeline/cli.py`
- Modify: `mars_lite/pipeline/training_engine.py`
- Test: `tests/test_portfolio_reward_contract.py`
- Test: `tests/test_target_wiring.py`

**Interfaces:**
- `PortfolioTradingEnv(..., turnover_penalty_rate: float = 0.0)`.
- Deprecated alias `lambda_turnover` is accepted only when `turnover_penalty_rate` is not supplied and is converted as `lambda_turnover / reward_scale`.
- CLI adds `--turnover-penalty-rate`; `--lambda-turnover` defaults to `None` and is deprecated.

- [ ] Write tests showing doubling reward scale doubles reward but leaves implied turnover-return penalty unchanged.
- [ ] Write tests for alias conversion and mutually exclusive CLI/config use.
- [ ] Confirm failures.
- [ ] Implement the return-unit penalty inside the reward scale and configuration translation.
- [ ] Run focused wiring and environment tests.
- [ ] Commit.

### Task 5: Direct interval execution semantics

**Files:**
- Modify: `mars_lite/env/portfolio_env.py`
- Reuse: `mars_lite/env/market_execution_core.py`
- Test: `tests/test_portfolio_interval_execution.py`
- Test: `tests/test_portfolio_env.py`

**Interfaces:**
- One `step(action)` advances `min(decision_every, remaining_episode_bars)` base bars.
- `info` adds `bars_advanced`, `interval_net_return`, `interval_log_return`, `interval_cost`, and `decision_step_index`.
- Entry turnover and cost are applied once per decision interval.

- [ ] Write deterministic tests proving full interval advancement, one entry cost, funding on each bar, one reward, and correct truncation.
- [ ] Confirm the current ignored-action behavior fails the tests.
- [ ] Refactor direct state to use `BookState` and `MarketExecutionCore` while preserving existing public metrics and info compatibility.
- [ ] Compute reward from interval log return and interval turnover penalty.
- [ ] Run all direct environment, baseline, and pipeline tests.
- [ ] Commit.

### Task 6: DSR research-only boundary

**Files:**
- Modify: `mars_lite/env/portfolio_env.py`
- Modify: `mars_lite/pipeline/cli.py`
- Modify: `mars_lite/pipeline/training_engine.py`
- Modify: `mars_lite/pipeline/production_pipeline.py`
- Test: `tests/test_dsr_reward_contract.py`
- Test: `tests/test_candidate_eligibility.py`

**Interfaces:**
- CLI flag `--experimental-dsr`.
- Environment options `experimental_dsr: bool = False`, `dsr_clip: float = 10.0`.
- DSR state `A` and `B` is included in the observation only when enabled.
- Release eligibility is false whenever experimental DSR is enabled.

- [ ] Write tests that old implicit DSR activation is rejected, explicit activation enriches observation, clips rewards, and disqualifies registration.
- [ ] Confirm failures.
- [ ] Implement the explicit boundary and remove in-episode reward-mode switching from production paths.
- [ ] Run focused tests.
- [ ] Commit.

### Task 7: Documentation, manifests, and full verification

**Files:**
- Modify: `docs/BASELINE_RESIDUAL_RL.md`
- Modify: `docs/ja/BASELINE_RESIDUAL_RL.md`
- Modify: `README.md`
- Modify: `README_ja.md`
- Modify: relevant manifest/report builders
- Modify: `.github/workflows/ci.yml` only if new focused tests are not already covered by the full suite.

- [ ] Update user-facing formulas, gamma policy, observation state, terminal semantics, direct penalty units, interval transitions, and DSR boundary.
- [ ] Add resolved reward semantics to manifests and reports.
- [ ] Run formatting/lint checks configured by the repository.
- [ ] Run the focused reward contract suite.
- [ ] Run the complete pytest suite.
- [ ] Run or inspect GitHub Actions for the branch/PR and fix every failure.
- [ ] Review the final diff against every requirement in the design spec.
- [ ] Open a draft pull request with validation evidence.