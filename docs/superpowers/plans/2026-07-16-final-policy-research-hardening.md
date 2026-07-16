# Final Policy and Research Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the remaining policy-distribution, terminal-accounting, advanced-risk, indicator-contract, and ablation-evidence gaps after PR #55 Tasks 1–8.

**Architecture:** Preserve the validated causal structured observation, shared per-asset actor, index-backed rollout, and serving contracts. Add bounded target-weight probability semantics at the policy boundary, make advanced portfolio-risk inputs explicit and causal, bind terminal and Ichimoku semantics into evidence, and add an immutable ablation workflow that cannot grant production approval.

**Tech Stack:** Python 3.12, NumPy, PyTorch 2.3.1, Gymnasium 0.29.1, Stable-Baselines3 2.3.2, pytest, GitHub Actions.

## Global Constraints

- Follow red-green-refactor for every behavior change.
- No policy action may be transformed after log-probability calculation without the transformation being represented by the probability distribution.
- No risk statistic may use data after the current decision index or be fitted outside the fold-train range.
- Existing artifacts must fail closed when a new identity-bound contract is absent or mismatched.
- Research ablations may emit evidence but may never change production status from `NO-GO`.
- Live order routing remains outside model and training artifacts.

---

### Task 9: Squashed target-weight PPO and action-space behavior cloning

**Files:**
- Modify: `trade_rl/rl/policies.py`
- Modify: `trade_rl/learning/behavior_cloning.py`
- Modify: `trade_rl/integrations/sb3_training.py`
- Test: `tests/rl/test_sequence_policy_core.py`
- Test: `tests/learning/test_behavior_cloning.py`
- Test: `tests/integrations/test_sb3_training.py`

**Interfaces:**
- Produces: `SharedPerAssetActorCriticPolicy` using `SquashedDiagGaussianDistribution` for target-weight actions.
- Produces: behavior-cloning target comparison against deterministic action-space output, not pre-squash Gaussian location.
- Produces: model architecture evidence field `action_distribution: "squashed_diag_gaussian"`.

- [ ] Write a policy test asserting stochastic and deterministic actions are strictly bounded by the action space and that `evaluate_actions` returns finite log-probabilities for actions near ±1.
- [ ] Run `uv run pytest -q tests/rl/test_sequence_policy_core.py::test_shared_sequence_policy_uses_squashed_target_weight_distribution` and verify it fails because the current policy installs `DiagGaussianDistribution`.
- [ ] Write a behavior-cloning test whose fake distribution exposes different pre-squash mean and deterministic action-space mode; assert cloning uses the bounded mode.
- [ ] Run `uv run pytest -q tests/learning/test_behavior_cloning.py::test_behavior_cloning_uses_deterministic_action_space_output` and verify it fails because `_actor_mean()` reads `distribution.distribution.mean`.
- [ ] Set `self.action_dist = SquashedDiagGaussianDistribution(action_dim)` before the base policy `_build()` call, retain the shared action head, and preserve constructor reload fields.
- [ ] Replace `_actor_mean()` with deterministic `distribution.get_actions(deterministic=True)` and fail closed when the result is not a tensor matching teacher actions.
- [ ] Record the bounded distribution name in `model-architecture.json`.
- [ ] Run focused policy, behavior-cloning, SB3 smoke, checkpoint reload, and serving reload tests.
- [ ] Run Ruff, MyPy, and full CI.

### Task 10: Explicit horizon and terminal accounting contract

**Files:**
- Modify: `trade_rl/rl/environment.py`
- Modify: `trade_rl/rl/environment_config.py`
- Modify: `trade_rl/workflows/training_run.py`
- Test: `tests/rl/test_environment.py`
- Test: `tests/rl/test_environment_time_config.py`

**Interfaces:**
- Produces: terminal evidence fields identifying truncation, economic termination, optional close liquidation, and discarded pending delayed target.
- Produces: environment identity binding for terminal accounting semantics.

- [ ] Add tests proving `liquidate_on_end=False` truncates without artificial closing fills or costs and retains marked open positions in terminal metrics.
- [ ] Add tests proving `liquidate_on_end=True` includes complete closing costs and reports liquidation completeness.
- [ ] Add a delayed-action horizon test proving a pending action is discarded rather than executed after the final decision interval.
- [ ] Verify the tests fail because pending-target disposition and terminal accounting mode are not explicit in evidence.
- [ ] Add terminal info fields `terminal_accounting_mode`, `pending_target_discarded`, and `terminal_liquidation_cost`, and bind the mode into environment artifacts.
- [ ] Run environment, reward, execution, workflow, and full CI tests.

### Task 11: Causal advanced portfolio-risk input provider

**Files:**
- Create: `trade_rl/risk/inputs.py`
- Modify: `trade_rl/risk/portfolio.py`
- Modify: `trade_rl/rl/environment.py`
- Modify: `trade_rl/workflows/training_run.py`
- Test: `tests/risk/test_portfolio_risk_inputs.py`
- Test: `tests/rl/test_environment.py`

**Interfaces:**
- Produces: `PortfolioRiskInputs(covariance, beta, stress_losses, as_of_index, digest)`.
- Produces: `PortfolioRiskInputsProvider.inputs(dataset, index, train_start) -> PortfolioRiskInputs`.
- Produces: default causal provider using only completed returns through `index` and an explicitly configured rolling window.

- [ ] Write no-future-leakage tests by mutating rows after the decision index and asserting identical risk inputs.
- [ ] Write fail-closed tests for configured volatility, beta, or stress limits without available finite inputs.
- [ ] Verify the tests fail because `_constrain_target()` currently passes only market notional.
- [ ] Implement immutable risk inputs and a dataset-derived provider with covariance, BTC-relative beta, and deterministic signed stress shocks.
- [ ] Inject the provider into `ResidualMarketEnv`, pass its arrays into `PortfolioRiskModel.constrain()`, and bind provider configuration/digest into environment identity.
- [ ] Run risk, environment, walk-forward, serving identity, and full CI tests.

### Task 12: Unshifted decision-time Ichimoku contract

**Files:**
- Modify: `trade_rl/data/contracts.py`
- Modify: `trade_rl/data/features.py`
- Modify: `trade_rl/integrations/binance.py`
- Modify: `trade_rl/workflows/training_run.py`
- Test: `tests/data/test_indicator_features.py`
- Test: `tests/integrations/test_binance_multitimeframe.py`

**Interfaces:**
- Produces: feature metadata `alignment="unshifted_decision_time"` for Ichimoku cloud values.
- Produces: dataset/reference evidence that no forward Senkou or backward Chikou plotting shift is consumed.

- [ ] Add tests proving future-bar mutations cannot change any Ichimoku value at the current index.
- [ ] Add metadata tests asserting all four Ichimoku channels declare unshifted decision-time alignment.
- [ ] Verify metadata tests fail because alignment is currently documented but not machine-bound.
- [ ] Add the non-breaking alignment metadata to feature specs/dataset evidence without renaming the existing ordered feature channels.
- [ ] Run feature, Binance, dataset identity, and full CI tests.

### Task 13: Immutable ablation evidence matrix

**Files:**
- Create: `trade_rl/evaluation/ablations.py`
- Create: `trade_rl/workflows/ablation_run.py`
- Create: `examples/binance-multitimeframe/ablation-matrix.json`
- Modify: `examples/binance-multitimeframe/run_full_research.py`
- Test: `tests/evaluation/test_ablations.py`
- Test: `tests/workflows/test_ablation_run.py`

**Interfaces:**
- Produces: `AblationVariant` and `AblationMatrix` with content digests.
- Produces: variants for snapshot-only, each timeframe removal, feature-group removal, reduced network, and BC disabled.
- Produces: an evidence bundle bound to the same folds, seeds, dataset, costs, and gate configuration as the maintained candidate.

- [ ] Write tests rejecting variants that change folds, seeds, data ranges, execution costs, or risk limits.
- [ ] Write tests proving ablation artifacts always report `production_status="NO-GO"`.
- [ ] Verify the tests fail because no executable ablation contract exists.
- [ ] Implement immutable matrix parsing, identity validation, per-variant orchestration, and comparative evidence output.
- [ ] Add a dry-run command that prints the exact experiment matrix without training.
- [ ] Run evaluation, workflow, example, artifact, and full CI tests.
