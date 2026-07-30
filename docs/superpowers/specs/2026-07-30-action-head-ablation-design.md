# Action-Head Ablation Design

## Context

The maintained production candidate uses a continuous target-weight action contract with a `hierarchical_sequence_v2` observation encoder and the `hierarchical_gate_target_v1` actor head. The policy first composes a deterministic target from current weights, a continuous Gate, and a proposed target, then applies one shared masked tanh-squashed Gaussian exploration distribution. The direct shared target head already exists in the policy module, but the training configuration and policy identity currently prevent it from being selected for the sequence encoder.

This change makes the direct head a first-class experimental candidate without changing the external action contract, execution simulator, risk controls, reward, PPO objective, observation encoder, or production serving boundary.

## Decision

Support exactly two sequence actor heads:

- `hierarchical_gate_target_v1`: `current + gate * (target - current)` before Gaussian exploration.
- `shared_target_v1`: one direct target-weight mean per active asset before the same Gaussian exploration.

The hierarchical head remains the default. The direct head is an ablation candidate, not a replacement and not an automatic promotion.

## Goals

1. Run a controlled Gate-versus-direct actor comparison with the same dataset, folds, seeds, encoder, critic, PPO settings, reward, risk, execution assumptions, Oracle teacher data, and causal holdout.
2. Bind the chosen actor head and its exact exploration coupling into checkpoint and serving identity.
3. Emit comparable action-path telemetry for both heads.
4. Apply mandatory causal non-collapse checks to Oracle BC for both heads.
5. Supply canonical paired training profiles and one walk-forward profile that cannot silently drift apart.

## Non-goals

- No discrete Buy/Sell/Hold action space.
- No SAC, TD3, TQC, or algorithm comparison.
- No signed-simplex or budget-aware output head in this change.
- No change to `target_weight:*` action names or their ordering.
- No change to no-trade-band, risk projection, execution, reward, or liquidation semantics.
- No automatic model promotion and no live exchange routing.

## Actor Contract

Both heads consume the same per-asset context produced by `SharedAssetActorCriticExtractor`, including the current effective weight and active mask. Both use the same `MaskedSharedSquashedDiagGaussianDistribution`, one shared scalar `log_std`, tanh squashing, and disabled gSDE.

The policy exposes a head-independent action-stage API:

```python
@dataclass(frozen=True, slots=True)
class ActionStageOutputs:
    current_weights: torch.Tensor
    deterministic_actions: torch.Tensor
    active_mask: torch.Tensor
    change_intensity: torch.Tensor | None
```

`change_intensity` is populated only for `hierarchical_gate_target_v1`. `hierarchical_actor_outputs()` remains available exclusively for hierarchical BC and keeps its existing semantics.

## Configuration Contract

For `observation_encoder=hierarchical_sequence_v2`, `policy_actor_head` accepts only:

- `hierarchical_gate_target_v1`
- `shared_target_v1`

The default remains `hierarchical_gate_target_v1`. `hierarchical_gate_temperature` remains required by `training_run_config_v3`; it must be positive for the hierarchical head and exactly `1.0` for the direct head because it is inactive there. Non-sequence encoders continue to require `standard_continuous_v1`.

## Identity Contract

Upgrade newly bound policies to `sb3_policy_identity_v4`. Version v2 is rejected. Existing v3 hierarchical identities remain readable for checkpoint and serving migration, but no new v3 identity is produced and the direct head is valid only under v4.

The v4 sequence identity records:

- ordered symbols and action names;
- sequence architecture and digest;
- actor head;
- current-weight observation identity;
- masked shared squashed-Gaussian distribution;
- shared scalar `log_std`;
- disabled gSDE;
- head-specific mean/exploration coupling;
- gate temperature only when the hierarchical head is active;
- a policy-architecture digest over the complete contract.

Head-specific coupling values are:

- hierarchical: `post_composition_gate_independent_v1`;
- direct: `direct_target_mean_v1`.

A checkpoint or structured export from one head must fail to load as the other head.

## Behavior-Cloning Contract

Both candidates use the same immutable Oracle episode teacher and the same causal held-out episodes.

The hierarchical candidate keeps Gate, Target, and composed reconstruction losses and its existing reconstruction diagnostics. The direct candidate keeps ordinary action-space MSE because it has no Gate decomposition.

For Oracle BC, both candidates must pass the same causal non-collapse group:

- minimum executed target changes;
- at least one submitted target change;
- non-constant submitted actions;
- causal regret upper-confidence bound;
- catastrophic after-cost regret limit.

Teacher change support is derived from the same chronological teacher labels for both candidates. A direct-head candidate may not bypass the economic holdout merely because it lacks `hierarchical_actor_outputs()`.

Trend-baseline BC retains its existing MSE-only behavior because it does not currently produce the Oracle episode holdout artifact.

## Telemetry Contract

TensorBoard computes the same action stages for both heads:

- deterministic change from current weights;
- Gaussian exploration displacement;
- sampled change from current weights;
- submission displacement;
- effective filled displacement.

Only the hierarchical head records `change_intensity_mean`. Existing fallback telemetry remains available for policies that do not implement the common action-stage API.

## Experiment Profiles

Add two canonical training files derived from `training-target-weight-growth-ppo.json`:

- `training-action-head-ablation-gate.json`
- `training-action-head-ablation-direct.json`

They differ only in `training.policy_actor_head`. Add `walk-forward-action-head-ablation.json` with exactly these two `run_file` candidates and the maintained six-fold, three-seed, execution-stress selection contract.

A regression test canonicalizes both training mappings, removes `training.policy_actor_head`, and requires the remaining payloads to be byte-for-byte equal. This prevents accidental differences in folds, seeds, costs, reward, network size, BC teacher, or risk settings.

## Decision Evidence

The experiment report must compare at least:

- net return and baseline uplift;
- maximum drawdown;
- turnover and execution cost;
- deterministic-change L1;
- exploration L1;
- sampled-change L1;
- submission and effective-action L1;
- seed dispersion and worst-seed uplift.

The hierarchical head remains preferred only if it preserves return or baseline uplift while reducing turnover/cost or improving seed stability. A budget-aware head is a later, separate experiment and is considered only when maintained projection-distance evidence is materially large.

## Testing

Use TDD and verify:

1. sequence configuration accepts both supported heads and rejects all others;
2. direct-head gate temperature is fail-closed when non-default;
3. both heads expose common deterministic action-stage outputs;
4. inactive action dimensions remain exactly zero;
5. v4 identity round-trips for both heads and rejects cross-head loading;
6. v2 identities fail and existing hierarchical v3 identities remain readable for migration;
7. direct Oracle BC enforces the causal non-collapse gate;
8. TensorBoard emits comparable stage tags for both heads and Gate telemetry only for the hierarchical head;
9. paired profiles differ only in actor head and the walk-forward profile resolves both files;
10. structured export and native serving fixtures use v4 identity.

## Safety Boundary

This work produces research and serving-compatible artifacts only. It does not authorize direct exchange execution. Any future live-routing change requires an independent design, risk review, and explicit user authorization.