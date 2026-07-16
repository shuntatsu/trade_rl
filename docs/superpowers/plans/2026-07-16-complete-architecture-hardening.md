# Complete Architecture Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the remaining research, execution, model, evaluation, and serving architecture gaps identified after PR #51.

**Architecture:** Preserve the immutable causal dataset and structured multi-timeframe observation contract, while making execution latency explicit, aligning the approximate portfolio teacher with the real executor, producing per-asset actor outputs through shared weights, strengthening train-only state scaling and statistical gates, and adding a structured-policy runtime contract. Every incompatible contract receives a schema-version bump and digest binding.

**Tech Stack:** Python 3.12, NumPy, PyTorch, Gymnasium, Stable-Baselines3, pytest, GitHub Actions.

## Global Constraints

- No future feature values or incomplete native bars.
- All fitted statistics remain fold-train-only.
- Existing artifacts fail closed on schema mismatch.
- Direct target-weight actions remain bounded by hard pre-trade and portfolio risk.
- Production status remains NO-GO until a fully unused confirmation range and live paper-trading evidence exist.
- Follow red-green-refactor for each behavior change.

---

### Task 1: Delayed decision execution

**Files:**
- Modify: `trade_rl/rl/environment_config.py`
- Modify: `trade_rl/rl/environment.py`
- Modify: `trade_rl/rl/observations.py`
- Test: `tests/rl/test_sequence_environment_config.py`
- Test: `tests/rl/test_environment.py`

**Interfaces:**
- Produces: `ResidualMarketEnvConfig.signal_delay_decisions: int`
- Produces: delayed target/action queue reset and digest binding.

- [ ] Write tests proving a target submitted at decision `t` cannot affect the `t+1` open when `signal_delay_decisions=1`, and that queued targets are reset between episodes.
- [ ] Run the focused tests and verify failure against the old immediate-execution contract.
- [ ] Implement a FIFO decision queue. Execute the matured target; enqueue the current proposal; expose queued action state in the observation; include the delay in the environment digest.
- [ ] Run focused and environment test suites.

### Task 2: Executor-aligned approximate portfolio teacher

**Files:**
- Modify: `trade_rl/learning/oracle_teacher.py`
- Modify: `trade_rl/learning/__init__.py`
- Modify: `trade_rl/integrations/sb3_training.py`
- Test: `tests/learning/test_oracle_teacher.py`

**Interfaces:**
- Produces: `ApproximatePortfolioTeacherConfig`
- Produces: `approximate_teacher_target_path(...)`

- [ ] Write tests proving below-minimum orders become executable no-ops and capacity excess becomes partial fill rather than an invalid transition.
- [ ] Verify the tests fail with the current transition-invalidating implementation.
- [ ] Apply tick/lot/minimum-notional/capacity semantics compatible with `MarketExecutor`, retain multiple economically distinct predecessor states per discrete target through a bounded beam, and rename the schema to state that the teacher is approximate.
- [ ] Add teacher approximation metadata to BC artifacts and model architecture evidence.
- [ ] Run teacher, execution, BC, and artifact tests.

### Task 3: Shared per-asset actor policy

**Files:**
- Modify: `trade_rl/rl/policies.py`
- Modify: `trade_rl/integrations/sb3_training.py`
- Test: `tests/rl/test_sequence_policy_core.py`
- Test: `tests/integrations/test_sb3_training.py`

**Interfaces:**
- Produces: `SharedPerAssetActorCriticPolicy`
- Produces: one shared actor head applied independently to each contextual asset token; critic remains portfolio pooled.

- [ ] Write permutation/equivariance and parameter-sharing tests.
- [ ] Verify failure with the flattened generic SB3 actor MLP.
- [ ] Implement the custom SB3 policy and use it for structured PPO/BC.
- [ ] Record actor-head type and symbol-order contract in model architecture metadata.
- [ ] Run policy, BC, PPO smoke, and checkpoint reload tests.

### Task 4: Semantic state normalization and undefined-feature masks

**Files:**
- Modify: `trade_rl/rl/normalization.py`
- Modify: `trade_rl/rl/observations.py`
- Modify: `trade_rl/data/cross_asset_features.py`
- Modify: `trade_rl/rl/sequence_normalization.py`
- Test: `tests/rl/test_sequence_normalization.py`
- Test: `tests/data/test_cross_asset_features.py`

**Interfaces:**
- Produces: fixed semantic transforms for bounded state values and sample-count evidence per sequence channel.

- [ ] Write tests for position age, portfolio-relative value, weights/actions, and undefined correlation/beta.
- [ ] Verify failure with zero-action trajectory scaling and valid-zero degenerate correlations.
- [ ] Separate fitted market statistics from deterministic state transforms; mark zero-variance correlations unavailable; fail closed when required channels lack train samples.
- [ ] Run data, normalization, observation, and walk-forward tests.

### Task 5: Index-backed PPO rollout reconstruction

**Files:**
- Modify: `trade_rl/integrations/compact_rollout_buffer.py`
- Modify: `trade_rl/rl/rollout_memory.py`
- Modify: `trade_rl/integrations/sb3_training.py`
- Test: `tests/rl/test_rollout_memory.py`
- Test: `tests/integrations/test_sb3_training.py`

**Interfaces:**
- Produces: compact buffer storing decision indices plus non-overlapping current state and reconstructing sequence mini-batches from the immutable dataset.

- [ ] Write tests proving sequence arrays are not allocated per rollout step and reconstructed batches equal environment observations.
- [ ] Verify failure with the dtype-compressed overlapping Dict buffer.
- [ ] Implement dataset-bound reconstruction and digest checks.
- [ ] Update memory estimates to reflect actual allocations.
- [ ] Run rollout, PPO smoke, and memory-ceiling tests.

### Task 6: Statistical walk-forward and confirmation protocol

**Files:**
- Modify: `trade_rl/evaluation/research_gate.py`
- Modify: `trade_rl/workflows/market_walk_forward_config.py`
- Modify: `trade_rl/workflows/market_walk_forward.py`
- Modify: `trade_rl/workflows/fold_runner.py`
- Modify: `examples/binance-multitimeframe/walk-forward-full.json`
- Modify: `examples/binance-multitimeframe/run_full_research.py`
- Test: `tests/evaluation/test_research_gate.py`
- Test: `tests/workflows/test_market_walk_forward.py`

**Interfaces:**
- Produces: configurable minimum fold count, OOS duration, bootstrap lower confidence bound, and untouched confirmation range.

- [ ] Write fail-closed tests for too few folds, insufficient OOS duration, non-positive confidence lower bound, unstable seeds, and missing confirmation evidence.
- [ ] Verify failure with the fixed two-fold positive-mean gate.
- [ ] Implement at least six folds and at least 180 OOS days in the maintained preset, reserve a separate confirmation interval, and stop selecting a lucky seed; final training uses a multi-seed ensemble.
- [ ] Run fold, gate, runner, and example tests.

### Task 7: Structured sequence serving runtime

**Files:**
- Create: `trade_rl/serving/sequence_runtime.py`
- Modify: `trade_rl/workflows/training_run.py`
- Modify: `trade_rl/serving/__init__.py`
- Test: `tests/serving/test_sequence_runtime.py`
- Test: `tests/e2e/test_research_to_serving_v2.py`

**Interfaces:**
- Produces: `SequencePolicyRuntime.load(...)` and `predict_target(...)` using rolling causal market state.

- [ ] Write artifact reload and offline parity tests.
- [ ] Verify failure because sequence serving is currently declared unsupported.
- [ ] Publish a structured loader sidecar, restore normalizers/policy/environment contracts, rebuild observations from a bounded rolling dataset, and fail closed on stale/incomplete bars.
- [ ] Keep live order routing outside the model artifact and production status NO-GO.
- [ ] Run serving, artifact, and e2e tests.

### Task 8: Long-run verification, recovery, and documentation

**Files:**
- Modify: `.github/workflows/ci.yml`
- Create: `.github/workflows/gpu-nightly.yml`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/MULTITIMEFRAME_RESEARCH.md`
- Modify: `docs/operations/docker-gpu-full-training.md`

**Interfaces:**
- Produces: checkpoint-resume smoke, structured reload smoke, GPU nightly contract, and accurate NO-GO boundaries.

- [ ] Add CPU checkpoint/resume and structured loader smoke tests to CI.
- [ ] Add a manually dispatchable/self-hosted CUDA workflow that measures peak VRAM and throughput and verifies resume after interruption.
- [ ] Remove claims of exact Oracle optimality and already-implemented shared actor/rollout reconstruction until corresponding tests pass.
- [ ] Run Ruff, format, MyPy, import contracts, Vulture, full pytest with branch coverage, critical coverage, CLI smoke, and training image build.
