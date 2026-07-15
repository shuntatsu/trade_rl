# 15-Minute Target-Weight Oracle BC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and compare a causal 15-minute direct-target PPO policy and a DP-oracle behavior-cloned PPO policy through the complete Docker walk-forward profitability gate.

**Architecture:** Add a direct target-weight action contract and one shared hysteresis/no-trade/emergency-risk pipeline, then add train-only cost-aware oracle labels and optional SB3 actor pretraining. Freeze two A/B candidates in the full 15-minute walk-forward configuration and preserve complete teacher, policy, selection, sealed-access, and Docker provenance evidence.

**Tech Stack:** Python 3.12, NumPy, PyTorch, Stable-Baselines3 PPO, Gymnasium, Pytest, Ruff, MyPy, Docker Compose, CUDA.

## Global Constraints

- Use TDD: every production behavior begins with a focused test that fails for the missing behavior.
- Keep public market data and all runtime artifacts inside `trade-rl-training-data`.
- Use four CPU rollout environments and one RTX 4050 GPU; train seeds and candidates sequentially.
- Oracle future data is legal only inside the exact fold train range and never enters student observations.
- Both A/B candidates share action, environment, risk, execution, network, and PPO fine-tuning settings.
- The final gate requires two distinct RL policy digests, mean return above zero, uplift at least zero, and maximum fold drawdown at most 20%.

---

### Task 1: Direct target-weight action contract

**Files:**
- Modify: `trade_rl/rl/actions.py`
- Modify: `trade_rl/rl/environment.py`
- Modify: `trade_rl/workflows/training_run.py`
- Test: `tests/rl/test_target_weight_action.py`
- Test: `tests/rl/test_environment_identity.py`

**Interfaces:**
- Produces: `ActionMode`, `TargetWeightAction`, and `ActionSpec.parse(..., n_symbols=...)` with canonical target action names.
- Consumes: existing gross/per-asset limits and `ResidualComposition` evidence shape.

- [ ] Write tests proving three direct weights parse exactly, dimension and non-finite values fail, action identity changes by mode/symbols, and zero direct action means flat rather than Trend baseline.
- [ ] Run `uv run pytest tests/rl/test_target_weight_action.py tests/rl/test_environment_identity.py -q` and confirm failures are caused by the missing direct-target API.
- [ ] Implement the minimal action types, digest payload, environment action space, parse branch, and direct proposal composition.
- [ ] Run the same tests and existing residual action tests; require green output.
- [ ] Run Ruff and MyPy on the changed action/environment modules.

### Task 2: Hysteresis and no-trade target post-processing

**Files:**
- Create: `trade_rl/risk/rebalancing.py`
- Modify: `trade_rl/risk/pretrade.py`
- Modify: `trade_rl/rl/environment.py`
- Test: `tests/risk/test_target_rebalancing.py`
- Test: `tests/risk/test_pretrade.py`

**Interfaces:**
- Produces: `TargetRebalanceConfig` and `TargetRebalancePolicy.apply(requested, current, emergency)`.
- Consumes: direct requested weights and returns filtered weights plus reasons/suppressed turnover.

- [ ] Write tests for 10% entry, 3% exit, hold region, reversal threshold, per-asset 5% no-trade band, and emergency reduction bypass.
- [ ] Run the focused tests and verify expected missing-module/API failures.
- [ ] Implement the pure rebalance policy and integrate it before ordinary turnover limiting while preserving hard-risk overrides.
- [ ] Run focused risk/environment tests and verify ordinary target increases cannot use the emergency bypass.
- [ ] Run Ruff and MyPy on the risk/environment slice.

### Task 3: Causal emergency risk monitor

**Files:**
- Create: `trade_rl/risk/emergency.py`
- Modify: `trade_rl/rl/environment.py`
- Modify: `trade_rl/workflows/training_run.py`
- Test: `tests/risk/test_emergency_monitor.py`
- Test: `tests/rl/test_emergency_drawdown.py`

**Interfaces:**
- Produces: `EmergencyRiskConfig`, `EmergencyRiskAssessment`, and `EmergencyRiskMonitor.assess(dataset, index, current_weights, drawdown)`.
- Consumes: only data at or before `index`; produces reduction-only targets and typed reasons.

- [ ] Write prefix-causality and behavior tests for signed one-hour stop loss, 15-minute gap, 24-hour/trailing-volatility shock, non-tradable assets, insufficient history, and reduction-only enforcement.
- [ ] Run focused tests and confirm RED failures.
- [ ] Implement the monitor with dataset time helpers rather than hard-coded bar counts, then wire it into the shared rebalance path.
- [ ] Run emergency, timing, causality, and environment tests; require green output.
- [ ] Run Ruff and MyPy on the emergency slice.

### Task 4: Cost-aware DP oracle teacher artifact

**Files:**
- Create: `trade_rl/learning/oracle_teacher.py`
- Create: `trade_rl/learning/teacher_artifact.py`
- Create: `trade_rl/learning/__init__.py`
- Test: `tests/learning/test_oracle_teacher.py`
- Test: `tests/learning/test_teacher_artifact.py`

**Interfaces:**
- Produces: `OracleTeacherConfig`, `oracle_target_path(dataset, train_range, config)`, `SupervisedPolicyDataset`, and content-addressed teacher artifact read/write functions.
- Consumes: exact training range, close returns, execution cost contract, direct action spec, and shared target post-processing.

- [ ] Write tests for deterministic DP paths, cost-induced flat regions, gross/per-asset bounds, prefix/range rejection, no labels beyond train stop, and artifact tamper detection.
- [ ] Run the learning tests and confirm missing APIs fail.
- [ ] Port the historical Viterbi logic using current `MarketDataset`, current execution costs, and canonical artifact codecs.
- [ ] Add deterministic teacher rollout collection through a full-train-range environment and persist observation/action/range digests.
- [ ] Run learning tests, Ruff, and MyPy; require green output.

### Task 5: Behavior cloning before PPO

**Files:**
- Modify: `trade_rl/rl/training.py`
- Modify: `trade_rl/integrations/sb3_training.py`
- Modify: `trade_rl/rl/checkpointing.py`
- Modify: `trade_rl/workflows/training_run.py`
- Test: `tests/integrations/test_sb3_behavior_cloning.py`
- Test: `tests/rl/test_ensemble_training.py`

**Interfaces:**
- Produces: `BehaviorCloningConfig`, optional `SupervisedPolicyDataset` injection into `StableBaselines3Backend`, epoch loss evidence, and `pretraining_digest` in training/ensemble manifests.
- Consumes: direct-action teacher observations/actions whose environment and action identities exactly match the PPO environment.

- [ ] Write tests showing actor MSE falls, value-only parameters are not the optimization target, pure PPO skips BC, mismatched observations/actions fail, and PPO fine-tuning still runs after BC.
- [ ] Run focused integration tests and confirm RED failures.
- [ ] Implement seeded mini-batch actor pretraining after model creation and before `learn`, with finite-loss and identity checks.
- [ ] Persist BC configuration and loss evidence into policy results, checkpoints, and ensemble manifests.
- [ ] Run SB3, checkpoint, ensemble, Ruff, and MyPy validation.

### Task 6: Fold-local A/B candidate orchestration

**Files:**
- Modify: `trade_rl/workflows/market_walk_forward.py`
- Modify: `trade_rl/workflows/market_walk_forward_config.py`
- Modify: `trade_rl/workflows/fold_runner.py`
- Test: `tests/workflows/test_market_walk_forward.py`
- Test: `tests/workflows/test_fold_runner.py`

**Interfaces:**
- Produces: train-range-only teacher generation for BC candidates and complete A/B experiment-plan identity.
- Consumes: candidate `BehaviorCloningConfig`, teacher artifact, and existing top-three-per-seed selection protocol.

- [ ] Write tests proving only BC candidates request a teacher, teacher ranges equal fold train, both candidates share non-training contracts, all finalists are evaluated on selection, and only the winner reaches outer evaluation.
- [ ] Run focused workflow tests and confirm RED failures.
- [ ] Implement fold-local teacher generation/injection and include teacher/BC digests in checkpoint, selection, run, and sealed-plan evidence.
- [ ] Run workflow and artifact invariant tests, Ruff, and MyPy.

### Task 7: Freeze 15-minute full assets and fresh folds

**Files:**
- Modify: `examples/binance-multitimeframe/run_full_research.py`
- Modify: `examples/binance-multitimeframe/training-full.json`
- Modify: `examples/binance-multitimeframe/walk-forward-full.json`
- Modify: `tests/examples/test_binance_multitimeframe_full_assets.py`
- Modify: `docs/operations/docker-gpu-full-training.md`

**Interfaces:**
- Produces: deterministic 55,392-bar 15-minute dataset, four-clock 96-feature contract, 0.25-hour decisions, and two predeclared A/B candidates over fresh June outer windows.
- Consumes: direct action, rebalance/emergency configs, BC config, `[256,256]` network, four environments, and top-three checkpoint selection.

- [ ] Write asset tests for base/native clocks, expected bars, exact fold boundaries, gamma half-life identity, direct target action, A/B-only difference, thresholds, seeds, network, GPU, and outer gates.
- [ ] Run example tests and confirm failures against the current hourly assets.
- [ ] Update runner/config/docs, compute gamma through the maintained helper, and preserve Docker generation/cache isolation.
- [ ] Run example, Docker contract, configuration, Ruff, and MyPy tests.

### Task 8: Repository and GPU verification

**Files:**
- Verify only; fix the owning task if a failure appears.

**Interfaces:**
- Produces: a clean, provenance-locked commit and CUDA image ready for the final experiment.
- Consumes: every preceding task.

- [ ] Run `uv run ruff format --check trade_rl tests examples` and `uv run ruff check trade_rl tests examples`.
- [ ] Run `uv run mypy trade_rl examples/binance-multitimeframe`.
- [ ] Run import contracts and `uv run pytest --cov=trade_rl --cov-report=term-missing --cov-fail-under=70`.
- [ ] Run `git diff --check`, commit the frozen protocol, and build with exact clean Git provenance.
- [ ] Run CUDA preflight and an 8,192-step direct-target BC smoke test in the Docker named volume.

### Task 9: Complete final Docker A/B training

**Files:**
- Runtime output: `/workspace/var/runs/<generation>/`

**Interfaces:**
- Produces: full training ensemble, two-fold A/B walk-forward artifacts, selected policies, teacher/BC evidence, and final research gate.
- Consumes: the immutable image and `trade-rl-training-data` volume.

- [ ] Start one unique detached generation and monitor dataset A/B identity, 55,392 bars, 96 raw features, approximately 1,252 policy observation values, CUDA device, four rollout workers, policies, and checkpoints.
- [ ] Let both candidates and all seeds finish; do not tune from the fresh outer results.
- [ ] Verify exactly two different selected RL policy digests and their train/checkpoint/selection/outer range evidence.
- [ ] Verify cost-adjusted selected mean return is positive, uplift over Trend is non-negative, and every fold drawdown is at most 20%.
- [ ] If the frozen gate fails, preserve the generation as NO-GO and use only pre-June development validation for any next protocol; never relabel an opened outer window as sealed.
