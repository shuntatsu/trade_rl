# Constrained Growth PPO Phase 1 — PR A Implementation Plan

> **Execution rule:** Implement with strict RED/GREEN tests. Preserve scalar reward, PPO behavior, and execution behavior.

**Goal:** Add auditable action-path diagnostics and canonical non-negative constraint-cost signals to every market-environment transition before introducing cost critics or Lagrangian policy updates.

**Architecture:** A focused `environment_constraints.py` module owns immutable action-path and cost contracts. Risk results preserve copied pipeline stages and constraint metadata. `EnvironmentInfoBuilder` derives causal transition telemetry and exposes compact objects plus flat scalar keys. No PR A output is consumed by reward or policy optimization.

**Tech stack:** Python 3.12, NumPy, Gymnasium environment services, Stable-Baselines3 integration, pytest, Ruff, Mypy, import-linter, GitHub Actions.

## Global invariants

- Keep `r_t = 100 * log(V_net[t+1] / V_net[t])` unchanged.
- Do not alter ordinary PPO, BC-to-PPO, checkpoint schemas, or model architecture.
- Costs must be causal, finite, non-negative, and independently named.
- Routine end-of-episode flattening is not a forced-liquidation event.
- Shadow-only failure never creates hybrid constraint events.
- Existing hard risk and execution controls remain authoritative.
- Preserve existing public info keys and add only observational telemetry.

## Task 1 — Immutable action-path contract

**Files:**
- `trade_rl/rl/environment_constraints.py`
- `tests/rl/test_environment_constraints.py`

**Status:** Implemented and tested.

The canonical path is:

1. `policy_target` — current policy proposal;
2. `execution_intent_target` — target released from the signal-delay queue;
3. `pretrade_target` — result after emergency, hysteresis, no-trade, and turnover controls;
4. `feasible_target` — result after hard and portfolio feasibility projection;
5. `submitted_order_target` — target passed to execution;
6. `filled_weight` — realized weight after fills and same-transition liquidation.

Required diagnostics include adjacent-stage L1 and maximum-absolute distances, `execution_intent_to_filled`, `policy_to_filled`, and stage-specific change flags. Legacy aggregate `policy_to_pretrade` remains available but includes signal-delay displacement when delay is enabled.

All vectors are copied, converted to finite `float64`, shape-checked, and stored read-only.

## Task 2 — Canonical constraint-cost vector

**Files:**
- `trade_rl/rl/environment_constraints.py`
- `tests/rl/test_environment_constraints.py`
- `tests/rl/test_environment_constraint_denominators.py`

**Status:** Implemented and tested.

Seven optimization-eligible costs are emitted independently:

```text
drawdown_excess
drawdown_stop_event
margin_deficit_fraction
forced_liquidation_event
gross_exposure_request_excess
daily_turnover
execution_cost_fraction
```

Positive funding is separate non-constraint telemetry: `funding_credit_fraction`.

Canonical formulas:

```python
drawdown_excess = max(0.0, drawdown - drawdown_budget)
margin_deficit_fraction = margin_deficit / max(initial_capital, eps)
gross_exposure_request_excess = max(
    0.0,
    abs(execution_intent_target).sum() - max_gross,
)
daily_turnover = filled_turnover * 24.0 / decision_hours
execution_cost_fraction = (
    max(0.0, interval_cost)
    + max(0.0, -interval_funding)
    + interval_borrow_cost
) / max(previous_equity, eps)
funding_credit_fraction = max(0.0, interval_funding) / max(previous_equity, eps)
```

Routine terminal flattening and completed drawdown-stop deleveraging are excluded from forced-liquidation events. Same-transition ordinary and liquidation execution metrics are combined exactly once.

## Task 3 — Preserve risk-pipeline stages and limits

**Files:**
- `trade_rl/risk/pretrade.py`
- `trade_rl/rl/environment_risk.py`
- related risk tests

**Status:** Implemented and tested.

`RiskConstrainedTarget` preserves defensive copies of:

- original risk proposal;
- final pre-trade target;
- final feasible weights.

It also carries the authoritative `max_gross` and drawdown budget needed for causal cost derivation. Existing risk calculation order and numerical behavior remain unchanged. Invalid metadata or mismatched stage shapes fail closed.

## Task 4 — Attach transition telemetry

**Files:**
- `trade_rl/rl/environment_info.py`
- `trade_rl/rl/environment_runtime_services.py`
- `tests/rl/test_environment_info_service.py`
- `tests/rl/test_environment_constraint_derivation.py`
- `tests/rl/test_environment_info_fail_closed.py`
- `tests/rl/test_target_weight_action.py`

**Status:** Implemented and tested.

`EnvironmentInfoBuilder`:

- distinguishes current policy target from delayed execution intent;
- validates execution intent against the proposal received by the risk projector;
- derives six-stage action diagnostics from causal transition state;
- receives configured initial capital explicitly;
- reconstructs previous equity from final equity and the transition log return;
- uses the dataset clock for actual transition duration;
- includes regular and same-transition liquidation cost, funding, borrow, and turnover exactly once;
- emits immutable objects and flat scalar keys;
- leaves scalar reward unchanged.

Fail-closed tests cover invalid initial capital, invalid transition duration, non-finite targets and liquidation metrics, missing stage metadata, target disagreement, missing constraint limits, and non-finite previous-equity reconstruction.

## Task 5 — Preserve compact training information

**Files:**
- `tests/integrations/test_sb3_constraint_info.py`
- existing `_compact_training_info` behavior

**Status:** Implemented and tested without an algorithm change.

Compact training info retains action/cost objects and scalar telemetry while removing history-bearing execution and liquidation payloads.

## Task 6 — Full verification and PR evidence

**Status:** Full CI succeeded on implementation head `486ea00b486c49f98e3e9e6a25e60cc789636cd5`, including Ruff, format, Mypy, import architecture, dead-code report, recovery/serving smoke, complete pytest with branch coverage, critical coverage ratchet, CLI smoke, Windows compatibility, Ubuntu compatibility, and training-image validation.

Finalization requires fresh verification on the documentation-cleanup head before the PR is marked ready.

## Out of scope

PR A does not add:

- cost-aware rollout buffers;
- cost critics or cost GAE;
- Lagrange multipliers or policy penalties;
- PPO hyperparameter changes;
- GRN, residual adapters, or model-capacity changes.

Those remain isolated in PR B through PR D and later architecture ablations.
