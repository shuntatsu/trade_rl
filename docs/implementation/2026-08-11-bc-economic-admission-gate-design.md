# Behavior-Cloning Economic Admission Gate

## Decision

Oracle behavior cloning is admitted to PPO only when its causal, after-cost holdout evidence satisfies both of these conditions:

1. the holdout contains enough complete validation episodes;
2. a deterministic one-sided bootstrap lower confidence bound for the policy's episode net returns is not below an authored floor.

The maintained Binance target-weight growth profiles require at least five complete holdout episodes and a 95% lower confidence bound of at least `-0.05` net return.

This is an admission floor, not a profitability claim. Nested walk-forward selection, sealed outer tests, execution stress, and production gates remain authoritative.

## Scope

This change is limited to behavior-cloning evidence and admission. It does not change:

- PPO or Lagrangian PPO updates;
- reward or episode-boundary semantics;
- action composition or execution simulation;
- checkpoint selection, serving, or live-order behavior;
- an active training generation.

It also does not modify any file changed by open PR #385 or PR #387.

## Configuration

`ResidualTrainingConfig` adds two backward-compatible fields:

```python
behavior_cloning_min_causal_holdout_episodes: int = 1
behavior_cloning_min_causal_holdout_net_return_lower_bound: float = -1.0
```

Validation is fail-closed:

- the episode minimum must be a positive, non-boolean integer;
- the net-return floor must be finite and at least `-1.0`.

Historical `training_run_config_v4` documents that omit these fields retain their prior behavior and digest payload through the defaults. When either field is non-default, both fields are included in the training configuration digest, so strengthened profiles cannot silently reuse checkpoints created under weaker admission semantics.

## Statistical Evidence

`deterministic_bootstrap_lower_bound()` accepts a non-empty finite vector, including losses, and computes a reproducible lower bound for its mean:

```python
lower = quantile(
    bootstrap_episode_means,
    1.0 - confidence_level,
    method="lower",
)
```

The helper uses the same content-derived deterministic RNG contract as the existing upper-bound helper and requires:

- confidence strictly between `0.5` and `1.0`;
- at least 1,000 bootstrap resamples;
- non-empty seed material.

## Episode Holdout Evidence

`EpisodeBehaviorCloningHoldoutEvaluation` records:

```python
causal_net_return_lower_confidence_bound: float
```

The value is computed from complete validation-episode causal policy net returns after execution costs. It is persisted in:

- `behavior-cloning-holdout.json`;
- the Oracle audit payload;
- the artifact content digest.

The episode holdout schema advances to `episode_oracle_bc_evaluation_v2` because its immutable evidence payload changes.

## Mandatory Gate

`BehaviorCloningGateThresholds` adds:

```python
minimum_causal_holdout_net_return_lower_bound: float = -1.0
```

The causal gate adds a mandatory metric:

```text
causal_net_return_lower_confidence_bound
```

The metric passes only when:

- complete episode support is at least `minimum_causal_holdout_episodes`; and
- the observed lower confidence bound is at least `minimum_causal_holdout_net_return_lower_bound`.

For historical single-path holdouts without episode records, the observed causal policy net return is used as the bound and support is one. This preserves the previous default behavior while allowing maintained profiles to opt into the stronger contract.

The behavior-cloning gate schema advances to `behavior_cloning_gate_evaluation_v2` because its mandatory metric set changes.

Existing reconstruction, activity, trade-support, collapse, catastrophic-regret, and regret-upper-bound checks remain mandatory.

## Maintained Profiles

These profiles explicitly declare the stronger admission contract:

- `training-target-weight-growth-ppo.json`;
- `training-target-weight-constrained-growth.json`;
- `training-target-weight-constrained-growth-discounted.json`.

Each contains:

```json
{
  "behavior_cloning_min_causal_holdout_episodes": 5,
  "behavior_cloning_min_causal_holdout_net_return_lower_bound": -0.05
}
```

## Verification Contract

Regression coverage proves:

- the lower-bound estimator is deterministic, one-sided, and accepts losses;
- insufficient complete-episode support fails closed;
- a lower bound below the authored floor fails closed;
- historical single-path defaults remain compatible;
- invalid configuration values are rejected;
- real episode holdout evidence is persisted under schema v2;
- all maintained target-weight profiles declare the stronger thresholds.

The implementation was developed with an observed RED exact head before production code and is accepted only after exact-head static checks, compatibility tests, full pytest, branch-aware coverage, architecture checks, package identity, and training-image verification succeed.

Production remains `NO-GO`.
