# Baseline-Anchored Residual RL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the release-capable direct-weight PPO path with a baseline-anchored residual policy whose identity action exactly reproduces an absolute-time trend baseline and whose value is evaluated relative to that baseline.

**Architecture:** Add pure trend-family, residual composer, HTF proposal constraint, shared interval execution, shadow-book relative reward, relative checkpoint selection, explicit residual/baseline-only gates, and action-schema-aware serving. Keep direct mode research-compatible while requiring `baseline_residual_v1` or `baseline_only` for release.

**Tech Stack:** Python 3.12, NumPy, Gymnasium, Stable-Baselines3 PPO, pytest, existing ServingBundle/Registry/Control Plane.

## Global Constraints

- Identity action is exactly `[0.0, 0.0]` and must match `base_trend_v2` within weight `atol=1e-8` and cost/PnL `atol=1e-10`.
- Trend targets depend only on FeatureSet price history and UTC timestamps, never current portfolio weights or slice-relative indices.
- `decision_every > 1` produces one transition per decision interval; no ignored-action transitions.
- HTF modifies desired proposal before stateful post-processing.
- Baseline-residual initial settings use `no_trade_band=0.0` and `lambda_turnover=0.0`; real execution costs remain enabled.
- `oracle_dp` and `oracle_ic*` are diagnostic-only and never mandatory gates.
- Holdout and Serving never refit the residual alpha model.
- Release candidates require action schema `baseline_residual_v1` with `ppo_residual_ensemble` or `baseline_only` policy mode.

---

### Task 1: Mandatory gate contracts

**Files:**
- Create: `mars_lite/pipeline/gates.py`
- Modify: `mars_lite/pipeline/evaluator.py`
- Modify: `mars_lite/pipeline/production_pipeline.py`
- Test: `tests/test_release_gates.py`

**Interfaces:**
- Produces `evaluate_residual_gate2(agent_metrics, shadow_metrics, paired_p_value) -> dict`.
- Produces `evaluate_baseline_only_gate(dev_gate, holdout_metrics, cost2x_metrics, positive_p_value, max_dd_limit) -> dict`.
- Produces `diagnostic_baseline(name: str) -> bool`.

- [ ] Write failing tests proving oracle names are diagnostic, residual Gate 2 compares only hybrid/flat/shadow, and baseline-only does not require beating itself.
- [ ] Run `uv run pytest tests/test_release_gates.py -q`; expect failures because `mars_lite.pipeline.gates` does not exist.
- [ ] Implement pure gate functions with explicit boolean details and finite-value validation.
- [ ] Replace evaluator's loop-over-all-baselines Gate 2 with the new residual gate; preserve all baseline metrics as diagnostics.
- [ ] Route `baseline_only` through its separate production eligibility path.
- [ ] Run focused tests and existing release-eligibility tests.
- [ ] Commit `fix: separate residual and baseline-only release gates`.

### Task 2: Absolute-time TrendFamily

**Files:**
- Create: `mars_lite/trading/trend_family.py`
- Test: `tests/test_trend_family.py`

**Interfaces:**
- `TrendFamilyConfig(fast_lookback=24, base_lookback=48, slow_lookback=96, rebalance_every=24, base_timeframe='1h')`.
- `TrendTargets(fast, base, slow)`.
- `TrendFamily.targets(fs: FeatureSet, t: int) -> TrendTargets`.

- [ ] Write failing tests for portfolio-state independence, same-timestamp/sliced-FeatureSet equality, endpoint targets, finite output, and gross <= 1.
- [ ] Run focused tests; expect import failure.
- [ ] Implement UTC-slot calculation from `fs.timestamps[t]`, locate the absolute rebalance timestamp with `searchsorted`, calculate causal time-series momentum, tanh-scale, and project gross.
- [ ] Add clear errors for non-monotonic timestamps, insufficient history, non-finite close, and unsupported timeframe.
- [ ] Run focused tests.
- [ ] Commit `feat: add absolute-time trend family`.

### Task 3: Residual composer and HTF proposal constraint

**Files:**
- Create: `mars_lite/trading/baseline_residual.py`
- Create: `mars_lite/trading/htf_constraint.py`
- Modify: `mars_lite/trading/pipeline.py`
- Test: `tests/test_baseline_residual.py`
- Test: `tests/test_htf_constraint.py`

**Interfaces:**
- `BaselineResidualComposer.compose(action, trends, alpha, alpha_enabled=True) -> CompositionResult`.
- `HTFProposalConstraint.apply(proposal, htf_trend) -> HTFConstraintResult`.
- `DecisionPipeline.process_proposal(proposal, state, market) -> (target, diagnostics)`.

- [ ] Write failing tests for identity equality, fast/slow endpoints, alpha budget ±30%, alpha-disabled zero budget, gross bound, action shape/non-finite rejection, and HTF idempotence.
- [ ] Run focused tests; expect failures.
- [ ] Implement dataclasses and pure functions.
- [ ] Refactor `DecisionPipeline` so HTF is applied before `PortfolioPostProcessor.process`; retain `target_weights` as a compatibility wrapper.
- [ ] Run focused and existing pipeline/post-processor tests.
- [ ] Commit `feat: compose baseline residual proposals before HTF and post-processing`.

### Task 4: Shared interval execution and shadow relative environment

**Files:**
- Create: `mars_lite/env/market_execution_core.py`
- Create: `mars_lite/env/baseline_residual_env.py`
- Modify: `mars_lite/env/portfolio_env.py`
- Test: `tests/test_interval_execution.py`
- Test: `tests/test_baseline_residual_env.py`

**Interfaces:**
- `BookState` holds weights/value/peak/drawdown/cost/funding/turnover.
- `MarketExecutionCore.execute_interval(book, target, start_t, bars) -> IntervalExecution`.
- `BaselineResidualTradingEnv` exposes Gymnasium action shape `(2,)` and relative reward.

- [ ] Write failing tests that one action advances exactly N bars, cost is charged once, tail intervals advance remaining bars, and identity policy equity exactly matches shadow.
- [ ] Run focused tests; expect failures.
- [ ] Extract one-book execution math from `PortfolioTradingEnv` into `MarketExecutionCore` without changing direct-mode numerical behavior.
- [ ] Implement independent hybrid/shadow books and reward `reward_scale * (hybrid_log_return - shadow_log_return)`.
- [ ] Ensure observations contain only serving-reproducible trend/alpha features, never shadow performance.
- [ ] Run focused tests plus direct environment regression tests.
- [ ] Commit `feat: add decision-interval residual environment with shadow reward`.

### Task 5: Frozen residual alpha artifact

**Files:**
- Create: `mars_lite/trading/residual_alpha.py`
- Modify: `mars_lite/features/gbm_forecaster.py`
- Modify: `mars_lite/serving/candidate.py`
- Test: `tests/test_residual_alpha_artifact.py`

**Interfaces:**
- `FrozenResidualAlpha.fit(fs, horizon, target='cs_demean', model='gbm')`.
- `predict_at(fs, t) -> np.ndarray`.
- `save(path)` / `load(path)` with feature order, fit cutoff, dataset identity, and gate result.

- [ ] Write failing tests for fit cutoff `development_end-horizon`, no refit on predict/load, feature-order mismatch rejection, market-neutral output, and gross <= 1.
- [ ] Run focused tests.
- [ ] Implement serializable artifact; use delayed LightGBM import and deterministic zero provider when gate is disabled.
- [ ] Add candidate bundle file/digest validation for the artifact.
- [ ] Run focused and bundle validation tests.
- [ ] Commit `feat: freeze residual alpha for holdout and serving`.

### Task 6: Residual PPO initialization, run tiers, and relative checkpoint selection

**Files:**
- Create: `mars_lite/learning/relative_val_selection.py`
- Create: `mars_lite/learning/residual_ensemble.py`
- Modify: `mars_lite/pipeline/training_engine.py`
- Modify: `mars_lite/pipeline/cli.py`
- Test: `tests/test_relative_val_selection.py`
- Test: `tests/test_residual_policy_init.py`
- Test: `tests/test_run_tiers.py`

**Interfaces:**
- `zero_initialize_action_head(agent) -> None`.
- `RelativeValSelectionCallback` evaluates >=10 rollout-aligned checkpoints.
- `validate_run_tier(run_tier, timesteps, n_envs, n_steps, n_seeds)`.
- `ResidualActionEnsemble.predict(obs, deterministic=True)` averages actions before composition.

- [ ] Write failing tests for exact zero initial deterministic action, identity snapshot restore, block-median excess selection, fallback, rollout-aligned frequency, tier minimums, and action averaging.
- [ ] Run focused tests.
- [ ] Implement residual-specific PPO path with BC disabled and zero-initialized action head.
- [ ] Implement relative checkpoint callback and fallback metadata.
- [ ] Add `--action-mode`, `--run-tier`, and release seed-count validation.
- [ ] Run focused and existing training tests.
- [ ] Commit `feat: train and select residual policies relative to baseline`.

### Task 7: Evaluation, reports, and A/B/C/D research matrix

**Files:**
- Create: `mars_lite/eval/relative_evaluation.py`
- Modify: `mars_lite/pipeline/evaluator.py`
- Modify: `mars_lite/eval/walk_forward.py`
- Test: `tests/test_relative_evaluation.py`
- Test: `tests/test_train_report_contract.py`

**Interfaces:**
- `evaluate_relative_agent(agent, fs, env_kwargs) -> RelativeEvaluationResult`.
- Report includes action distribution, every weight stage, shadow metrics, excess metrics, checkpoint reasons, gate identities, base bars advanced, and diagnostic baseline flags.

- [ ] Write failing tests for report fields, paired excess calculations, annualization metadata, and oracle diagnostic flags.
- [ ] Run focused tests.
- [ ] Implement paired evaluation and moving-block bootstrap using existing bootstrap utilities.
- [ ] Add A/B/C/D matrix on development data and freeze selected configuration before holdout.
- [ ] Preserve direct-mode reports while adding explicit action schema/mode.
- [ ] Run focused and walk-forward tests.
- [ ] Commit `feat: add relative evaluation and residual research matrix`.

### Task 8: ServingBundle and runtime parity

**Files:**
- Modify: `mars_lite/serving/candidate.py`
- Modify: `mars_lite/serving/runtime.py`
- Modify: `mars_lite/server/signal_server.py`
- Modify: `mars_lite/pipeline/production_pipeline.py`
- Test: `tests/test_serving_residual_action_schema.py`
- Test: `tests/test_train_eval_serve_parity.py`

**Interfaces:**
- Bundle action schema `baseline_residual_v1`.
- Policy mode `ppo_residual_ensemble` or `baseline_only`.
- Runtime reconstructs TrendFamily, frozen alpha, composer, HTF, and post-processor before returning final target weights.

- [ ] Write failing tests for 2D action composition, baseline-only inference without PPO, old schema rejection in release mode, and train/eval/serve target equality.
- [ ] Run focused tests.
- [ ] Extend bundle metadata/file set and fail-closed validation.
- [ ] Implement runtime action dispatch and structured audit fields.
- [ ] Restrict release candidate construction to residual/baseline-only modes.
- [ ] Run focused serving and registry tests.
- [ ] Commit `feat: serve baseline-anchored residual bundles`.

### Task 9: Documentation, migration, and full verification

**Files:**
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/ja/ARCHITECTURE.md`
- Modify: `docs/TESTING.md`
- Modify: `docs/ja/TESTING.md`
- Modify: `docs/MODEL_LIFECYCLE.md`
- Modify: `docs/ja/MODEL_LIFECYCLE.md`
- Modify: `README.md`
- Modify: `README.ja.md`

- [ ] Document action schemas, residual/baseline-only gates, frozen alpha, interval transitions, and migration from direct mode.
- [ ] Run `uv run ruff format .` and `uv run ruff check .`.
- [ ] Run `uv run mypy mars_lite`.
- [ ] Run `uv run pytest --cov=mars_lite --cov-fail-under=70 tests/`.
- [ ] Verify report/schema docs tests.
- [ ] Commit `docs: document baseline-anchored residual architecture`.

### Task 10: Pull request and evidence

- [ ] Push the implementation branch and open a draft PR against `main`.
- [ ] Record exact head SHA, CI run ID, lint, format, mypy, pytest, coverage, and untested external integrations in the PR body.
- [ ] Inspect every failed GitHub Actions job and fix root causes with test-first patches.
- [ ] Re-run the full workflow until all required checks pass.
- [ ] Mark the PR ready only after verification evidence is attached.
