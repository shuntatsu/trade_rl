# Causal Sequence Feature Encoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-snapshot market encoder with leak-safe native-timeframe sequences, expand point-in-time features, remove the per-decision turnover throttle, and resize the PPO/BC policy for the richer observation contract.

**Architecture:** Keep market events on their native 15m/1h/4h/1d clocks, expose only completed observations whose `available_at` is not later than the decision timestamp, and build independent causal sequence tensors per timeframe. Encode each sequence with a causal residual TCN, fuse per-asset state through cross-asset attention, and retain separate current-snapshot, execution-state, actor, and critic paths. Dataset, normalizer, teacher, policy, and experiment-plan identities bind ordered feature names, sequence lengths, availability rules, and network topology.

**Tech Stack:** Python 3.12, NumPy, PyTorch, Gymnasium, Stable-Baselines3 PPO, pytest, Hypothesis, Ruff, MyPy, Docker CUDA.

## Global Constraints

- Decision clock remains 15 minutes.
- Native clocks are 15m, 1h, 4h, and 1d; 1m data is not introduced.
- Only completed native bars may contribute; no backward fill, centered windows, future-shifted Ichimoku spans, or outer-range fitting.
- Train-only fitting applies to normalizers, feature filters, BC validation, Oracle construction, checkpoint selection, and all hyperparameter choices.
- The already-opened July range is development validation only; production status remains `NO-GO`.
- The per-decision `max_turnover=0.02` throttle is removed from the maintained 15m preset. Gross, concentration, liquidity, margin, tradability, and emergency-deleveraging constraints remain.
- Sequence tensors are not flattened into one giant MLP input.
- TDD is required for every production change.

---

### Task 1: Freeze the structured causal observation contract

**Files:**
- Create: `trade_rl/rl/sequence_observations.py`
- Modify: `trade_rl/data/market.py`
- Modify: `trade_rl/rl/observations.py`
- Test: `tests/rl/test_sequence_observation_contract.py`
- Test: `tests/rl/test_sequence_observation_causality.py`

**Interfaces:**
- Produces `SequenceWindowSpec`, `StructuredObservationLayout`, and a builder returning ordered native-timeframe tensors plus snapshot, execution, global, availability, and staleness arrays.
- Sequence lengths: 15m=96, 1h=168, 4h=120, 1d=60.

- [ ] Write failing layout, boundary, digest, prefix-causality, incomplete-bar, and symbol/feature-order rejection tests.
- [ ] Verify RED with focused pytest.
- [ ] Implement native-clock index mapping and structured observation construction without future access.
- [ ] Verify GREEN and run existing observation causality tests.
- [ ] Commit.

### Task 2: Add point-in-time feature families

**Files:**
- Modify: `trade_rl/data/contracts.py`
- Modify: `trade_rl/data/features.py`
- Modify: `trade_rl/integrations/binance.py`
- Create: `trade_rl/data/cross_asset_features.py`
- Test: `tests/data/test_extended_indicator_features.py`
- Test: `tests/data/test_cross_asset_features.py`
- Test: `tests/data/test_extended_prefix_causality.py`

**Interfaces:**
- Adds candle geometry, Parkinson and Garman-Klass volatility, upside/downside volatility, volatility-of-volatility, range expansion, ATR change, +DI/-DI/DI spread, EMA distance/slope, rolling regression slope/R², MFI, CMF, VWAP distance, price-volume correlation, OBV change/acceleration, relative volume, funding change/z-score, basis change, rolling BTC correlation/beta, relative return, momentum rank, and dispersion.
- Renames the maintained Bollinger position label to `bollinger_percent_b_centered` while preserving its centered-%B mathematics.

- [ ] Write numerical fixture tests and prefix-mutation causality tests first.
- [ ] Verify RED.
- [ ] Implement each feature with trailing windows only and explicit source-start/availability metadata.
- [ ] Define ordered timeframe-specific feature presets rather than blindly duplicating every channel.
- [ ] Verify GREEN and deterministic dataset identity.
- [ ] Commit.

### Task 3: Remove the per-decision turnover throttle from the maintained preset

**Files:**
- Modify: `trade_rl/risk/pretrade.py`
- Modify: `examples/binance-multitimeframe/training-full.json`
- Modify: `examples/binance-multitimeframe/walk-forward-full.json`
- Test: `tests/risk/test_pretrade.py`
- Test: `tests/rl/test_target_weight_action.py`

**Interfaces:**
- Makes turnover throttling explicitly optional for direct target-weight policies; disabled means the requested target is not sliced across decisions.
- Preserves all hard portfolio, execution, and emergency constraints.

- [ ] Write failing tests proving 0→40% can be requested in one decision when turnover throttling is disabled, while risk and liquidity still constrain realized fills.
- [ ] Verify RED.
- [ ] Implement optional throttle semantics and update maintained presets.
- [ ] Verify GREEN and property tests.
- [ ] Commit.

### Task 4: Implement native-timeframe causal TCN encoders

**Files:**
- Create: `trade_rl/integrations/sb3_sequence_policy.py`
- Modify: `trade_rl/integrations/sb3_training.py`
- Modify: `trade_rl/rl/training.py`
- Test: `tests/integrations/test_sb3_sequence_policy.py`
- Test: `tests/integrations/test_sb3_training.py`

**Interfaces:**
- Causal residual TCNs use dilation 1/2/4/8/16, GroupNorm or LayerNorm, SiLU/GELU, and dropout no greater than 0.05.
- Initial latent sizes: 15m=192, 1h=192, 4h=160, 1d=128; current snapshot=256; execution state=96.
- Asset fusion is 640→384→320, cross-asset attention is 2 layers with 8 heads and `d_model=320`, actor is 384→256→128, critic is 512→384→256.
- Total parameters must be measured and remain between approximately 6M and 10M, with a hard rejection above 12M.

- [ ] Write failing shape, strict-causality, parameter-budget, mask, shared-BC/PPO-encoder, and deterministic inference tests.
- [ ] Verify RED.
- [ ] Implement the feature extractor and separate actor/critic heads.
- [ ] Verify GREEN on CPU.
- [ ] Commit.

### Task 5: Bind sequence identity through BC, PPO, serving, and walk-forward

**Files:**
- Modify: `trade_rl/learning/teacher_artifact.py`
- Modify: `trade_rl/learning/behavior_cloning.py`
- Modify: `trade_rl/learning/oracle_teacher.py`
- Modify: `trade_rl/workflows/training_run.py`
- Modify: `trade_rl/workflows/fold_runner.py`
- Modify: `trade_rl/workflows/market_walk_forward.py`
- Modify: serving normalizer/policy adapter files discovered during implementation
- Test: `tests/learning/test_teacher_artifact.py`
- Test: `tests/workflows/test_market_walk_forward.py`
- Test: `tests/serving/test_shared_observation_builder.py`

**Interfaces:**
- Teacher and policy artifacts bind the exact train range, dataset ID, native feature availability, ordered channel names, window lengths, action spec, and architecture digest.
- BC early stopping uses a chronological validation slice contained wholly inside the fold train range.
- Outer ranges cannot be requested by Oracle, normalizer, BC, checkpoint selection, or feature-filter code.

- [ ] Write failing artifact-tamper, range-overlap, order-mismatch, and train-only-fit tests.
- [ ] Verify RED.
- [ ] Implement digest propagation and fail-closed range checks.
- [ ] Verify GREEN.
- [ ] Commit.

### Task 6: Add memory-safe rollout handling and maintained configurations

**Files:**
- Modify: `trade_rl/rl/training.py`
- Modify: `trade_rl/integrations/sb3_training.py`
- Modify: `examples/binance-multitimeframe/training-full.json`
- Modify: `examples/binance-multitimeframe/walk-forward-full.json`
- Test: `tests/rl/test_algorithm_configs.py`
- Test: `tests/integrations/test_sb3_sequence_policy.py`

**Interfaces:**
- Initial PPO settings: 4 envs, `n_steps=1024`, `batch_size=128`, learning rate in `[1e-4, 1.5e-4]`, gradient clipping 0.5, target KL `[0.015, 0.02]`.
- Rollout memory is measured. Any compressed storage must be lossless with respect to policy input after float32 reconstruction and must not bypass causality or identity checks.

- [ ] Write failing configuration and memory-budget tests.
- [ ] Verify RED.
- [ ] Implement maintained config and bounded storage path.
- [ ] Verify GREEN.
- [ ] Commit.

### Task 7: Verification, ablation assets, and research-only documentation

**Files:**
- Create/modify ablation example configs under `examples/binance-multitimeframe/`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/RESEARCH_STATUS.md`
- Modify: `docs/operations/docker-gpu-full-training.md`

**Interfaces:**
- Supports comparisons: snapshot MLP; existing features + TCN; extended features + TCN; extended features + larger network; 262,144 vs 524,288 steps; pure PPO vs Oracle BC→PPO.

- [ ] Run focused tests after every task.
- [ ] Run Ruff format/check, MyPy, import contracts, full pytest/coverage, and `git diff --check`.
- [ ] Build clean provenance Docker image.
- [ ] Run CPU smoke and CUDA sequence/BC/PPO smoke on 4 environments.
- [ ] Report parameter count, rollout memory, feature counts by timeframe, and any remaining leakage or sample-size limitations.
- [ ] Keep production status `NO-GO`; do not reuse the opened July range as confirmatory evidence.
