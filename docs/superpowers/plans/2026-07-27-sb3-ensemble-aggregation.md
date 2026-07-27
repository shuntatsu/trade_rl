# SB3 Deterministic Ensemble Aggregation Implementation Plan

## Goal

Make evaluation and Serving use one fail-closed deterministic ensemble aggregation implementation without changing observations, policy loading, training, or artifact contracts.

## Task 1: Lock the pure aggregation contract

Add `tests/integrations/test_sb3_ensemble.py` covering:

- deterministic mean output and `float32` result;
- `deterministic=True` propagation;
- empty ensemble rejection;
- member prediction failure wrapping;
- non-finite and out-of-range action rejection;
- exact action-size mismatch;
- inferred member-shape disagreement;
- non-finite mean rejection.

Run the focused test and confirm RED because `trade_rl.integrations.sb3_ensemble` does not exist.

## Task 2: Lock wrapper parity and dependency direction

Add an AST-based boundary test proving:

- Serving and walk-forward import and call `predict_deterministic_mean_action`;
- neither wrapper owns NumPy mean/stack aggregation;
- the helper does not import Serving or workflow modules.

Add a focused walk-forward wrapper test preserving `deterministic=False` rejection and `(action, None)` output.

## Task 3: Implement the pure helper

Create `trade_rl/integrations/sb3_ensemble.py` with one public function and private validation helpers. Keep the interface independent of bundle, dataset, and workflow types.

Run focused helper tests until GREEN.

## Task 4: Route Serving through the helper

Update flat and structured ensemble policies in `sb3_serving.py` to call the helper with the manifest action size. Remove the duplicated action and mean helpers.

Run Serving loader and structured-observation tests.

## Task 5: Route walk-forward evaluation through the helper

Update `_DeterministicMeanPolicy.predict()` to preserve its deterministic flag guard, call the helper without an explicit size, and return `(action, None)`.

Run walk-forward and wrapper parity tests.

## Task 6: Full verification and integration

Run Ruff, format, Mypy, import architecture, dead-code, recovery/Serving smoke, Ubuntu and Windows compatibility, Training image, full Pytest and coverage, critical branch coverage, and CLI smoke at the exact PR head. Remove temporary scaffolding, update the PR evidence, and squash merge only when every required check is green.
