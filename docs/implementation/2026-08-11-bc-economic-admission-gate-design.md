# Behavior-Cloning Economic Admission Gate Design

## Decision

Strengthen Oracle behavior-cloning admission without changing PPO, the action space, reward semantics, episode routing, or execution simulation.

A BC warm start must satisfy both of these causal after-cost conditions before PPO begins:

1. the holdout contains a configured minimum number of complete validation episodes;
2. a deterministic one-sided bootstrap lower confidence bound for the causal policy's episode net returns is not below a configured floor.

The maintained Binance target-weight growth profiles will require at least five held-out episodes and a 95% lower confidence bound of at least `-0.05` net return. Existing catastrophic-regret, trade-support, collapse, and reconstruction checks remain mandatory.

## Context

The current BC gate correctly separates hindsight Oracle agreement from causal policy evidence. It already rejects zero-trade collapse, constant actions, insufficient teacher support, and catastrophic after-cost loss. It also computes a one-sided upper confidence bound for cash-baseline regret.

Two gaps remain:

- `minimum_causal_holdout_episodes` is hard-coded to one in the SB3 integration, so a single favorable episode can satisfy the statistical gate;
- the current catastrophic cash-regret threshold permits a materially losing BC policy to pass, because it is an emergency bound rather than a non-inferiority bound.

These gaps concern BC admission only. They are independent from the open universal-instrument artifact and episode-router work.

## Approaches Considered

### A. Tighten the existing point-regret threshold only

This is the smallest change, but it treats one realized path as sufficient evidence and does not distinguish a stable policy from a noisy favorable episode.

### B. Require episode support and a deterministic lower confidence bound

This is the selected approach. It uses the complete episode records already produced by the causal holdout evaluator, adds no new market-data access, and preserves the current separation between Oracle diagnostics and causal evidence.

### C. Add a separate trend-baseline rollout and require paired non-inferiority

This would provide a stronger economic comparison, but it requires a new baseline-policy evaluation path and additional accounting identity. It is a separate follow-up because it is larger and would mix baseline-policy semantics into this focused admission change.

## Contract

### Configuration

Add two backward-compatible `ResidualTrainingConfig` fields:

```python
behavior_cloning_min_causal_holdout_episodes: int = 1
behavior_cloning_min_causal_holdout_net_return_lower_bound: float = -1.0
```

Validation rules:

- episode minimum must be a positive, non-boolean integer;
- the lower-bound floor must be finite and at least `-1.0`;
- historical `training_run_config_v4` documents that omit the fields retain their old effective behavior through the defaults;
- maintained target-weight profiles declare both fields explicitly.

### Statistical evidence

Add:

```python
def deterministic_bootstrap_lower_bound(
    values: object,
    *,
    confidence_level: float,
    resamples: int,
    seed_material: str,
) -> float
```

The helper:

- accepts a non-empty finite rank-one vector, including negative values;
- uses the same content-derived deterministic RNG contract as the existing upper-bound helper;
- samples episode returns with replacement;
- returns the `(1 - confidence_level)` quantile using NumPy's `lower` method;
- requires at least 1,000 bootstrap resamples and confidence strictly between 0.5 and 1.0.

### Episode holdout evidence

`EpisodeBehaviorCloningHoldoutEvaluation` records:

```python
causal_net_return_lower_confidence_bound: float
```

It is computed from the per-episode causal policy net returns and persisted in `behavior-cloning-holdout.json` and the Oracle audit payload. The episode holdout schema advances from v1 to v2 because the immutable evidence payload changes.

### Mandatory gate

`BehaviorCloningGateThresholds` gains:

```python
minimum_causal_holdout_net_return_lower_bound: float = -1.0
```

The mandatory causal gate gains a metric named:

```text
causal_net_return_lower_confidence_bound
```

It passes only when:

- complete holdout episode support is at least `minimum_causal_holdout_episodes`; and
- the observed lower confidence bound is at least `minimum_causal_holdout_net_return_lower_bound`.

For legacy single-path holdouts without episode records, the causal policy net return is used as the observed bound and support is one. This preserves existing direct-BC tests and semantics.

The behavior-cloning gate schema advances from v1 to v2 because the mandatory metric set changes.

## Data Flow

```text
complete validation episodes
  -> causal policy rollouts after costs
  -> per-episode net returns
  -> deterministic one-sided bootstrap lower bound
  -> immutable holdout evidence v2
  -> mandatory causal BC gate v2
  -> BC candidate saved and PPO allowed only after gate success
```

Oracle action agreement, Oracle regret, and normalized Oracle regret remain hindsight diagnostics. They are not used as production-generalization evidence.

## Failure Behavior

The implementation fails closed when:

- the holdout has fewer complete episodes than configured;
- the lower confidence bound is unavailable or non-finite;
- the lower confidence bound is below the configured floor;
- bootstrap confidence, resample count, or seed material is invalid;
- existing reconstruction, activity, collapse, trade-support, or catastrophic-regret checks fail.

The error identifies the first failing mandatory metric through the existing `require_passed()` path.

## Maintained Profile Policy

The following maintained target-weight profiles declare:

```json
{
  "behavior_cloning_min_causal_holdout_episodes": 5,
  "behavior_cloning_min_causal_holdout_net_return_lower_bound": -0.05
}
```

- `training-target-weight-growth-ppo.json`
- `training-target-weight-constrained-growth.json`
- `training-target-weight-constrained-growth-discounted.json`

The threshold is an admission floor, not a profitability claim. Walk-forward and sealed outer-test gates remain authoritative for model selection and production status.

## Files and Responsibilities

- `trade_rl/learning/evaluation.py`: deterministic lower-bound helper and mandatory gate metric.
- `trade_rl/learning/episode_oracle_bc.py`: per-episode causal return evidence and schema v2 payload.
- `trade_rl/rl/training.py`: authored training fields and validation.
- `trade_rl/integrations/sb3_behavior_cloning.py`: map authored fields into gate thresholds.
- `examples/binance-multitimeframe/*.json`: explicit maintained thresholds.
- `tests/learning/`: statistical, evidence, and gate regressions.
- `tests/rl/` and `tests/workflows/`: configuration parsing and validation regressions.

## Isolation from Parallel Work

This change does not modify any file changed by open PR #385 or PR #387. In particular, it does not touch universal instrument artifacts, PostgreSQL universal materialization, universal episode bindings, universal router tests, or the PostgreSQL workflow.

## Non-Goals

- no learner-state dataset aggregation or DAgger;
- no trend-baseline economic rollout;
- no BC-versus-no-BC duplicate PPO experiment;
- no PPO, Lagrangian, reward, execution, episode-boundary, action-head, checkpoint, serving, or live-order change;
- no modification of an active training generation;
- no profitability or Production authorization claim.

## Testing Strategy

1. Prove the lower-bound helper is deterministic, one-sided, and accepts negative values.
2. Prove the episode holdout persists the lower bound and binds it into the artifact digest.
3. Prove the mandatory gate rejects insufficient episode support.
4. Prove the mandatory gate rejects a lower bound below the configured floor even when action and trade-collapse checks pass.
5. Prove backward-compatible defaults preserve legacy single-path behavior.
6. Prove configuration rejects invalid episode minima and invalid lower-bound floors.
7. Prove the three maintained target-weight profiles declare the stronger contract.
8. Run focused tests, Ruff, Ruff format, MyPy, architecture checks, and exact-head repository CI.