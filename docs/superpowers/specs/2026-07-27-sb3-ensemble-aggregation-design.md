# SB3 Deterministic Ensemble Aggregation Boundary Design

## Problem

Stable-Baselines3 deterministic ensemble inference is implemented independently in two production paths:

- `trade_rl/integrations/sb3_serving.py` validates each action and averages members for flat and structured Serving policies;
- `trade_rl/workflows/_market_walk_forward_core.py` repeats the same prediction, finite-value, bound, shape, and mean logic for deployable walk-forward evaluation.

The implementations currently agree in intent but differ in error wrapping and shape validation. A future correction to one path can therefore change evaluation/Serving parity without a type or dependency failure.

## Considered approaches

### A. Keep both implementations and add parity tests

This detects some drift but retains duplicated runtime behavior and requires every future correction to be applied twice.

### B. Move aggregation into Serving

Walk-forward evaluation could import a Serving-private helper, but that would make evaluation depend on bundle loading, dataset validation, and Serving policy classes.

### C. Extract a pure SB3 integration helper

Create a small integration-level function that owns only deterministic member prediction, action validation, and float64 mean aggregation. Serving and walk-forward retain their observation and wrapper-specific contracts. This is selected.

## Decision

Create `trade_rl/integrations/sb3_ensemble.py` with:

```python
predict_deterministic_mean_action(
    models,
    observation,
    *,
    action_size=None,
    context="SB3 ensemble",
) -> np.ndarray
```

The helper will:

1. reject an empty ensemble;
2. call every member with `deterministic=True`;
3. wrap member prediction failures with the member index;
4. flatten each action to one dimension as `float32`;
5. reject non-finite and out-of-range values;
6. enforce an exact action size when supplied, otherwise require member shapes to agree;
7. compute the mean in `float64`;
8. reject a non-finite mean and return `float32`.

## Ownership

`sb3_ensemble.py` may depend on NumPy and structural runtime objects only. It must not depend on Serving bundles, walk-forward workflows, market datasets, training, or artifact publication.

`sb3_serving.py` keeps observation validation, structured reconstruction, bundle loading, and the public policy wrappers.

`_market_walk_forward_core.py` keeps the SB3-compatible `(action, state)` wrapper and continues to reject `deterministic=False`.

## Compatibility

- Flat and structured Serving outputs remain value-equivalent.
- Walk-forward evaluation continues to return `(mean_action, None)`.
- Member actions remain bounded to `[-1, 1]` and averaged in `float64` before conversion to `float32`.
- Serving continues to enforce the bundle-declared action size.
- Walk-forward continues to infer the action shape and reject disagreement.
- No training, reward, environment, selection, or release behavior changes.

## Testing

Regression tests will prove deterministic member calls, mean precision and dtype, empty ensembles, prediction failures, finite/bound validation, exact-size validation, inferred-shape disagreement, Serving/walk-forward parity, and an AST dependency ratchet that prevents either wrapper from reimplementing NumPy mean aggregation.
