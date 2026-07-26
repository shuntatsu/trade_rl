# Constrained PPO PR B Verification Record

Date: 2026-07-26  
Branch: `agent/constrained-ppo-cost-critics`  
Stacked base: PR #188 / `agent/constrained-ppo-design` at `f1b5ced121da8ef3c557b4589560d6c7f522f232`

## Scope completed

PR B adds an opt-in `cost_critic_ppo` algorithm that learns seven independent constraint-cost values while preserving ordinary PPO and the exact all-cost net-log-growth reward.

Implemented contracts:

- typed canonical cost schema in fixed head order;
- independent per-cost gamma, GAE lambda, returns, and advantages;
- true-termination versus time-limit-truncation bootstrapping;
- separate continuous and rare-event Cost Critic adapters;
- one cumulative value head per cost and optional auxiliary event logits;
- actor- and reward-value isolation from Cost Critic optimization;
- compact-info collection across vector environments;
- Brier score, zero-only baseline, positive support, calibration bins, explained variance, and family-gradient diagnostics;
- opt-in training configuration that fails closed when cost settings are supplied to ordinary PPO;
- exact Cost Critic rollout-memory accounting in the existing memory ceiling;
- Cost Critic parameters and optimizer state in SB3 save/load;
- checkpoint identity for algorithm, cost ordering, cost schema, architecture, and rollout schema;
- deterministic save/load round-trip with Cost Critic weight equality;
- machine-readable compute and rare-event-support evidence.

PR B intentionally does not add Lagrange multipliers, dual EMA state, or any cost term to the actor objective. Those remain PR C work.

## Production-profile resource contract

The maintained growth-optimal profile uses three assets and `sequence_d_model=336`. The shared sequence feature extractor therefore emits:

```text
3 * 336 asset-token values
+ 336 pooled-asset values
+ 128 global values
+ 3 active flags
= 1,475 Cost Critic input features
```

With the default family-separated adapters `(128, 64)`, five continuous value heads, two event value heads, and auxiliary event logits disabled:

- additional Cost Critic parameters: `395,591`;
- FP32 parameter payload: `1,582,364` bytes (`1.509 MiB`);
- cost rollout dimensions: `n_steps=256`, `n_envs=4`, `n_costs=7`;
- transitions per rollout: `1,024`;
- exact NumPy cost-rollout payload: `145,408` bytes (`142 KiB`).

The rollout calculation is:

```text
1,024 transitions * 7 costs * 5 float32 arrays * 4 bytes
+ 1,024 transitions * 2 boolean arrays * 1 byte
= 145,408 bytes
```

These values are locked by `test_growth_optimal_profile_locks_cost_critic_resource_contract`.

`build_cost_critic_compute_evidence` additionally records actual parameter bytes, initialized Adam state bytes, rollout bytes, update count, event-positive support, runtime throughput, and peak device memory when supplied by a real training run.

## Rare-event promotion boundary

An event head is not promotion-eligible merely because its MSE is small. Promotion requires:

- positive event support;
- a finite Brier score;
- Brier performance better than the zero-only predictor.

A zero-positive rollout explicitly reports `eligible_for_promotion=False`. Selection or sealed-test data must not be used to tune this boundary.

## Checkpoint and resume evidence

Tests cover:

- schema, head-order, architecture, and rollout-identity mismatch rejection;
- ordinary PPO versus Cost Critic state incompatibility;
- actual SB3 `save()` then `CostCriticPPO.load()` round-trip;
- restoration of Cost Critic parameters and optimizer state;
- normalization of restored hidden-dimension containers;
- PPO-family rejection of replay-buffer resume.

## CI evidence before the resource-contract documentation commit

Exact code head `48cb820226c45d50d7c09109f3a600aa90356797` was verified by CI run `#3152`:

- `1,656 passed`, `2 skipped`, `11 warnings`;
- total coverage `85.75%`;
- critical branch coverage passed;
- Ruff passed;
- format check passed;
- Mypy passed;
- import architecture passed;
- dead-code report passed;
- recovery and structured-serving smoke passed;
- CLI smoke passed;
- Ubuntu compatibility passed;
- Windows compatibility passed;
- complete non-root training image build and probe passed.

A fresh full CI run is required for the final documentation and resource-contract-test head before PR B is marked ready.

## Explicitly not claimed

GitHub-hosted CI does not provide the target RTX GPU. Therefore this PR does not claim a measured CUDA peak-memory value or production steps-per-second value. The evidence schema accepts and validates those measurements, but they must be populated by a later representative GPU smoke or full training run.

The deterministic parameter and rollout-memory values above are exact allocation contracts, not estimates of total PyTorch CUDA allocator peak usage.
