# Phase A and Model Evolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** After a passing C3 gate, distill causal value/regret evidence into the maintained residual policy and then evaluate the remaining model improvements as isolated, attributable ablations.

**Architecture:** Phase A introduces a new teacher artifact derived only from frozen C1/C2/C3 predictions. It trains the existing actor with value-weighted behavior cloning and then continues PPO on the same model. Subsequent model changes are one-at-a-time opt-in profiles with unchanged controls and separate artifacts.

**Tech Stack:** Python 3.12, NumPy, PyTorch, Stable-Baselines3, existing `SequencePolicyPlane`, teacher artifact conventions, C3 gate/report artifacts, deterministic walk-forward, Pytest.

## Global Constraints

- Phase A cannot start unless a loaded `PhaseAEntryGateEvidence` has `passed=true`.
- Perfect-Information actions, historical DP actions, and realized query futures are prohibited labels.
- Teacher queries and values are generated from fold train ranges only using the frozen C3 configuration.
- Checkpoint, selection, purge, test, outer, and fresh-confirmation rows never enter teacher fitting or normalizers.
- Preserve the maintained residual action contract and full 4070 Ti SUPER model.
- Pure PPO and BC→PPO use identical folds, seeds, timesteps, architecture, reward, constraints, AUM, and execution.
- Every later model improvement is a separate ablation; no bundled architecture jump.
- Production remains `NO-GO`.

---

### Task 1: Add the Phase A authorization boundary

**Files:**
- Create: `trade_rl/learning/phase_a_authorization.py`
- Test: `tests/learning/test_phase_a_authorization.py`
- Create: `tests/architecture/test_phase_a_gate_boundary.py`

**Interfaces:**
- Produces `PhaseATeacherAuthorization` and `authorize_phase_a_teacher(gate, experiment_plan)`.
- Consumes an immutable C3 gate artifact and a predeclared Phase A experiment plan.

- [ ] **Step 1: Write RED gate tests**

Reject authorization when `passed=false`, the gate digest is unknown, required C3 config/report artifacts are missing, source identities differ, or the Phase A plan changes C3 horizon, scenario count, candidate generator, score, or thresholds.

- [ ] **Step 2: Implement immutable authorization**

Bind:

```text
c3_gate_digest
c3_report_digest
c3_config_digest
dataset_plan_digest
fold_plan_digest
teacher_config_digest
student_config_digest
source_commit
authorized_at
schema_version
```

Authorization does not approve Serving, sealed testing, release, or production.

- [ ] **Step 3: Add architecture rules**

Only `trade_rl.learning` and evaluation/workflow adapters may consume the authorization. Serving, release, promotion, and direct execution must not import it.

- [ ] **Step 4: Run GREEN and commit**

```bash
uv run pytest tests/learning/test_phase_a_authorization.py tests/architecture/test_phase_a_gate_boundary.py -q
```

Commit: `feat: gate Phase A teacher development`.

### Task 2: Build a causal value/regret teacher artifact

**Files:**
- Create: `trade_rl/learning/causal_value_teacher.py`
- Create: `trade_rl/learning/causal_value_teacher_artifact.py`
- Modify: `trade_rl/learning/teacher_artifact.py`
- Test: `tests/learning/test_causal_value_teacher.py`
- Test: `tests/learning/test_causal_value_teacher_artifact.py`

**Interfaces:**
- Produces `CausalValueTeacherConfig`, `CausalValueTeacherBatch`, `CausalValueTeacherArtifact`, builder, writer, and loader.
- Consumes train-only observations, C3 predicted candidate values/regrets, persisted decisions, and Phase A authorization.

- [ ] **Step 1: Write RED train-only tests**

Mutate every row outside the fold train range and prove sample observations, candidate values, selected actions, weights, and artifact digest remain unchanged. Any query whose complete causal snapshot or 96-decision scenario history is unavailable is excluded with an explicit reason.

- [ ] **Step 2: Define the sample contract**

Each sample stores:

```text
observation
selected_raw_residual
selected_submitted_target
selected_predicted_advantage
selected_downside_cvar
selected_score
second_best_score
regret_margin
bootstrap_lower_bound
candidate_count
zero_candidate_score
query_identity
```

The training weight is deterministic:

```python
positive_confidence = max(bootstrap_lower_bound, 0.0)
margin = max(regret_margin, 0.0)
weight = min(max(positive_confidence * margin, 0.0), config.max_weight)
```

Samples with `weight == 0` remain diagnostic but do not contribute actor loss.

- [ ] **Step 3: Add quality and support summaries**

Record fold/query counts, positive-weight fraction, action distribution, zero-residual frequency, confidence/margin quantiles, scenario-neighbor quality, and digest-bound exclusion counts. Do not choose thresholds from selection or outer results.

- [ ] **Step 4: Implement deterministic exact-file artifact**

Use JSON plus NPZ closure, immutable arrays, atomic writes, idempotent identical publication, and fail-closed loading. Recompute weights and summaries on load.

- [ ] **Step 5: Run GREEN and commit**

```bash
uv run pytest tests/learning/test_causal_value_teacher.py tests/learning/test_causal_value_teacher_artifact.py -q
```

Commit: `feat: build causal value teacher artifacts`.

### Task 3: Add value-weighted actor pretraining

**Files:**
- Create: `trade_rl/learning/value_weighted_bc.py`
- Modify: `trade_rl/integrations/sb3_training.py`
- Modify: `trade_rl/rl/algorithm_configs.py`
- Test: `tests/learning/test_value_weighted_bc.py`
- Test: `tests/integrations/test_phase_a_sb3_training.py`

**Interfaces:**
- Produces `ValueWeightedBCConfig`, `ValueWeightedBCEvidence`, and `pretrain_residual_actor_from_causal_values`.

- [ ] **Step 1: Write RED loss tests**

The actor loss is:

```python
per_sample = ((predicted_action - teacher_action) ** 2).mean(dim=-1)
loss = (per_sample * normalized_weight).sum()
```

Normalize weights only across positive-weight samples. Reject an all-zero training batch instead of dividing by zero. Do not update reward critic, Cost Critics, dual state, observation normalizers, or environment state.

- [ ] **Step 2: Add deterministic split and early stopping**

Split by chronological query blocks inside train, never random row shuffling across the validation boundary. Bind split ranges, batch size, epochs, optimizer, seed, patience, and minimum improvement into evidence.

- [ ] **Step 3: Preserve PPO continuation semantics**

Pretrain `model.policy`, then call the existing PPO/Lagrangian learning path on the same model. Do not export/import ONNX between BC and PPO. Checkpoint and final artifacts remain SB3 `policy.zip` plus full identity evidence.

- [ ] **Step 4: Add zero-effect controls**

A configuration with zero BC epochs or no authorized teacher artifact must use the exact pure-PPO path. State tensors and optimizer construction must match the ordinary control after identical RNG reset.

- [ ] **Step 5: Run GREEN and commit**

```bash
uv run pytest tests/learning/test_value_weighted_bc.py tests/integrations/test_phase_a_sb3_training.py -q
```

Commit: `feat: add causal value weighted behavior cloning`.

### Task 4: Add Phase A walk-forward comparison

**Files:**
- Create: `examples/binance-multitimeframe/training-causal-value-bc-ppo.json`
- Create: `examples/binance-multitimeframe/walk-forward-causal-value-bc-ppo.json`
- Modify: `trade_rl/workflows/fold_runner.py`
- Modify: `trade_rl/workflows/market_walk_forward.py`
- Create: `trade_rl/evaluation/phase_a_comparison.py`
- Test: `tests/examples/test_phase_a_profiles.py`
- Test: `tests/workflows/test_phase_a_walk_forward.py`
- Test: `tests/evaluation/test_phase_a_comparison.py`

**Interfaces:**
- Compares pure PPO and causal-value-BC→PPO across six folds and three seeds.

- [ ] **Step 1: Freeze experiment equality**

The only allowed difference is the authorized pretraining phase and teacher artifact. Model, optimizer after PPO starts, timesteps, schedules, folds, seeds, constraints, AUM, and execution must match.

- [ ] **Step 2: Report teacher and policy gaps**

Store C3 Scenario-Oracle uplift, teacher imitation error, selected-action agreement, pure-PPO return, BC→PPO return, baseline uplift, drawdown, turnover, economic costs, constraints, seed stability, and the remaining `ScenarioOracle - BC→PPO` approximation gap.

- [ ] **Step 3: Define continuation rule**

Phase A is useful only when BC→PPO passes all existing eligibility gates and improves the paired selection-period result over pure PPO with a strictly positive predeclared confidence margin. Otherwise keep pure PPO or baseline fallback and preserve failed evidence.

- [ ] **Step 4: Run GREEN and commit**

```bash
uv run pytest tests/examples/test_phase_a_profiles.py tests/workflows/test_phase_a_walk_forward.py tests/evaluation/test_phase_a_comparison.py -q
```

Commit: `feat: compare causal value BC and pure PPO`.

### Task 5: State-dependent exploration ablation

**Files:**
- Create: `trade_rl/rl/state_dependent_exploration.py`
- Modify: `trade_rl/integrations/sb3_training.py`
- Modify: `trade_rl/rl/algorithm_configs.py`
- Test: `tests/rl/test_state_dependent_exploration.py`
- Test: `tests/integrations/test_state_dependent_exploration.py`

**Interfaces:**
- Adds an opt-in exploration-standard-deviation head conditioned on causal pooled policy features.

- [ ] **Step 1: Add RED bounds and determinism tests**

The head outputs log standard deviation clipped to configured finite bounds. Deterministic evaluation ignores sampling and returns the mean action. Ordinary profiles keep the existing global log-std parameters unchanged.

- [ ] **Step 2: Bind architecture and checkpoints**

Include head type, hidden widths, bounds, and parameter digest in policy/checkpoint identity. Reject legacy resume when enabled.

- [ ] **Step 3: Compare against unchanged control**

Use identical data, folds, seeds, model trunk, timesteps, and selection gates. Record action entropy by regime and raw-to-filled distortion.

- [ ] **Step 4: Verify and commit**

Commit: `feat: add state dependent policy exploration`.

### Task 6: PopArt for reward and separate cost critics

**Files:**
- Create: `trade_rl/rl/popart.py`
- Modify: `trade_rl/integrations/cost_critics.py`
- Modify: `trade_rl/integrations/lagrangian_ppo.py`
- Test: `tests/rl/test_popart.py`
- Test: `tests/integrations/test_popart_cost_critics.py`

**Interfaces:**
- Adds independent PopArt statistics for reward value and every enabled cost head; actor advantages remain in corrected raw economic/cost units before final combined normalization.

- [ ] **Step 1: Prove output-preserving statistic updates**

When running mean/scale changes, adjust final value-head weight and bias so denormalized predictions remain equal within strict tolerance.

- [ ] **Step 2: Keep statistics independent**

No shared reward/cost normalization state. Rare-event heads may disable PopArt by explicit config; never infer behavior from observed sparsity.

- [ ] **Step 3: Persist and resume exactly**

Checkpoint all counts, means, second moments, scales, and schema identity. Resume must reproduce the next update.

- [ ] **Step 4: Compare against unchanged control and commit**

Commit: `feat: add PopArt value normalization`.

### Task 7: Regime-balanced sampling and execution domain randomization

**Files:**
- Create: `trade_rl/rl/regime_sampling.py`
- Create: `trade_rl/rl/execution_domain_randomization.py`
- Modify: `trade_rl/workflows/fold_runner.py`
- Test: `tests/rl/test_regime_sampling.py`
- Test: `tests/rl/test_execution_domain_randomization.py`

**Interfaces:**
- Regime sampling selects only causal episode start indices from the current train capability.
- Domain randomization draws only from predeclared train-time execution profiles and never changes evaluation profiles.

- [ ] **Step 1: Define causal regime bins**

Use train-only volatility, Trend sign/strength, correlation, and liquidity quantile bins. Fit edges only on train and bind them to fold identity.

- [ ] **Step 2: Balance without future leakage**

Sample start indices by deterministic seeded strata. The sampler cannot inspect future episode outcome or selection/test metrics.

- [ ] **Step 3: Define execution distributions**

Predeclare bounded fee, spread, impact, latency, participation, reject, and borrow/funding stress multipliers. Store the sampled profile per episode; evaluation remains fixed nominal and adverse.

- [ ] **Step 4: Run separate comparisons**

Regime balancing and domain randomization are two PRs and two empirical comparisons, not one combined candidate.

### Task 8: Residual VALUE/RISK architecture and auxiliary heads

**Files:**
- Create: `trade_rl/rl/value_risk_tokens.py`
- Create: `trade_rl/rl/residual_critic_adapter.py`
- Create: `trade_rl/rl/auxiliary_risk_heads.py`
- Modify: `trade_rl/rl/sequence_observations.py`
- Modify: `trade_rl/integrations/sb3_training.py`
- Test: `tests/rl/test_value_risk_tokens.py`
- Test: `tests/rl/test_residual_critic_adapter.py`
- Test: `tests/rl/test_auxiliary_risk_heads.py`

**Interfaces:**
- VALUE token summarizes critic-specific pooled context.
- RISK token summarizes current causal account/risk/execution state.
- Residual adapters alter critic pathways without changing actor observation closure.
- Auxiliary heads predict predeclared return quantiles and economic hazard events for representation learning only.

- [ ] **Step 1: Add token identity and no-future tests**

Token inputs must come from already causal observation/account fields. Changing processing-bar or future rows cannot change tokens.

- [ ] **Step 2: Preserve actor control**

The first PR adds critic adapters only. A later separate PR may expose shared token features to the actor after evidence.

- [ ] **Step 3: Add auxiliary loss isolation**

Auxiliary losses update the shared encoder and their own heads with explicit coefficients. They do not change reward, cost, constraint, or selection labels.

- [ ] **Step 4: Compare adapters and auxiliary heads separately**

Commit independently:

```text
feat: add residual value risk critic adapters
feat: add quantile and hazard auxiliary heads
```

### Task 9: SSL pretraining, recurrent memory, and capacity ablation

**Files:**
- Create: `trade_rl/learning/sequence_ssl.py`
- Create: `trade_rl/rl/recurrent_policy_memory.py`
- Modify: `trade_rl/rl/sequence_observations.py`
- Modify: `trade_rl/integrations/sb3_training.py`
- Test: `tests/learning/test_sequence_ssl.py`
- Test: `tests/rl/test_recurrent_policy_memory.py`
- Test: `tests/rl/test_policy_capacity_profiles.py`

- [ ] **Step 1: Add train-only SSL artifact**

Pretrain masked/reconstruction or contrastive sequence objectives exclusively on fold train data. Bind feature schema, masks, ranges, augmentations, encoder architecture, and source commit. Selection/outer rows never enter pretraining.

- [ ] **Step 2: Transfer encoder without ONNX**

Load verified PyTorch state directly into the policy encoder before PPO. Record missing/unexpected keys and reject partial silent loads.

- [ ] **Step 3: Add recurrent memory as a separate algorithm profile**

Memory state resets on true episode boundaries and independently reset folds. Checkpoint/resume persists recurrent state only where the rollout contract permits; Serving snapshot identity includes required memory state.

- [ ] **Step 4: Add large-model capacity profiles last**

Keep the full current model as control. Predeclare larger widths/layers and parameter/VRAM ceilings; do not tune size from sealed results. Compare throughput, overfitting, seed stability, and OOS evidence.

- [ ] **Step 5: Run full exact-head verification for every PR**

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run lint-imports
uv run pytest -q
uv run python tools/check_critical_coverage.py
```

Require Ubuntu, Windows, training-image, and PostgreSQL workflows where applicable. Each final diff removes temporary model files, event logs, datasets, and workflows.
