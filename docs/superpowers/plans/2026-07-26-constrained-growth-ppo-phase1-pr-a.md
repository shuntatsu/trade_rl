# Constrained Growth PPO Phase 1 PR A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add auditable action-path diagnostics and canonical non-negative constraint-cost signals to every market-environment transition without changing PPO, scalar rewards, or execution behavior.

**Architecture:** Introduce one focused `environment_constraints.py` module that owns immutable action-path and cost-vector contracts plus pure calculators. Extend the existing risk result with the proposal and pre-trade stages, then attach the diagnostics and costs to `EnvironmentStepInfoRequest`. All values are observational only in PR A; no reward or optimizer consumes them.

**Tech Stack:** Python 3.12, NumPy, Gymnasium environment contracts, dataclasses, Pytest, Ruff, Mypy.

## Global Constraints

- Keep `r_t = 100 * log(V_net[t+1] / V_net[t])` unchanged.
- Do not alter ordinary PPO, BC-to-PPO, checkpoint schemas, or model architecture.
- Costs must be causal, finite, non-negative, and independently named.
- Routine end-of-episode flattening is not a forced-liquidation event.
- Shadow-only failure never creates hybrid constraint events.
- Existing hard pre-trade, portfolio-risk, margin, and emergency controls remain authoritative.
- Preserve all existing public keys and add only new info keys.

---

### Task 1: Immutable action-path contract

**Files:**
- Create: `tests/rl/test_environment_constraints.py`
- Create: `trade_rl/rl/environment_constraints.py`

**Interfaces:**
- Produces: `ActionPathDiagnostics.from_stages(*, policy_target, pretrade_target, feasible_target, submitted_order_target, filled_weight) -> ActionPathDiagnostics`
- Produces scalar fields: `policy_to_pretrade_l1`, `pretrade_to_feasible_l1`, `feasible_to_submitted_l1`, `submitted_to_filled_l1`, `policy_to_filled_l1`, matching maximum-absolute deviations, and boolean stage-change flags.

- [ ] **Step 1: Write failing tests**

Create tests that pass five two-asset vectors and assert exact L1 distances, maximum absolute deviations, flags, defensive copies, shape validation, and rejection of non-finite values.

- [ ] **Step 2: Verify RED**

Run `pytest tests/rl/test_environment_constraints.py -q` in CI. Expected failure: `ModuleNotFoundError: trade_rl.rl.environment_constraints`.

- [ ] **Step 3: Implement the minimal contract**

Use a frozen slotted dataclass. Convert every vector to a copied `float64` one-dimensional array, require one common non-zero shape, require finite values, and make stored arrays read-only. Compute diagnostics only from adjacent maintained stages plus end-to-end policy-to-filled distance.

- [ ] **Step 4: Verify GREEN**

Run `pytest tests/rl/test_environment_constraints.py -q`. Expected: all action-path tests pass.

- [ ] **Step 5: Commit**

Commit message: `feat: add action path diagnostic contract`.

### Task 2: Canonical constraint-cost vector

**Files:**
- Modify: `tests/rl/test_environment_constraints.py`
- Modify: `trade_rl/rl/environment_constraints.py`

**Interfaces:**
- Produces: `ConstraintCostRequest`
- Produces: `ConstraintCostVector`
- Produces: `calculate_constraint_costs(request: ConstraintCostRequest) -> ConstraintCostVector`
- Required execution protocol fields: `filled_turnover`, `interval_cost`, `interval_funding`, `interval_borrow_cost`.

- [ ] **Step 1: Write failing tests**

Add exact tests for:

```python
costs = calculate_constraint_costs(
    ConstraintCostRequest(
        policy_target=np.array([0.8, -0.6]),
        max_gross=1.0,
        decision_hours=0.25,
        drawdown=0.14,
        drawdown_budget=0.10,
        margin_deficit=250.0,
        previous_equity=100_000.0,
        filled_turnover=0.125,
        interval_cost=20.0,
        interval_funding=-5.0,
        interval_borrow_cost=2.0,
        termination_reason="drawdown_stop",
        emergency_deleverage=True,
        liquidation_terminal=False,
        liquidation_complete=True,
    )
)
```

Assert:
- drawdown excess `0.04`;
- drawdown-stop event `1.0`;
- margin-deficit fraction `0.0025`;
- gross-exposure request excess `0.4`;
- daily turnover `12.0`;
- execution-cost fraction `0.00027`;
- funding-credit fraction `0.0`;
- forced-liquidation event `0.0` for a completed drawdown-stop deleverage;
- forced-liquidation event `1.0` for `margin_call`, `liquidation`, `insolvency`, `minimum_equity`, `execution_cost_exhaustion`, or incomplete emergency liquidation;
- routine terminal flattening remains zero;
- positive funding is reported as a credit and cannot reduce the non-negative execution cost;
- invalid negative/non-finite inputs fail closed.

- [ ] **Step 2: Verify RED**

Run the focused file and confirm missing cost types/functions cause the failure.

- [ ] **Step 3: Implement the pure calculator**

Use these formulas:

```python
drawdown_excess = max(0.0, drawdown - drawdown_budget)
margin_deficit_fraction = margin_deficit / max(previous_equity, eps)
gross_exposure_request_excess = max(0.0, abs(policy_target).sum() - max_gross)
daily_turnover = filled_turnover * 24.0 / decision_hours
execution_cost_fraction = (
    max(0.0, interval_cost)
    + max(0.0, -interval_funding)
    + max(0.0, interval_borrow_cost)
) / max(previous_equity, eps)
funding_credit_fraction = max(0.0, interval_funding) / max(previous_equity, eps)
```

Forced liquidation excludes routine `liquidation_terminal` flattening and completed `drawdown_stop` deleveraging.

- [ ] **Step 4: Verify GREEN**

Run the focused tests and ensure all exact formulas pass.

- [ ] **Step 5: Commit**

Commit message: `feat: add canonical environment constraint costs`.

### Task 3: Preserve risk-pipeline stages

**Files:**
- Modify: `trade_rl/risk/pretrade.py`
- Modify: `trade_rl/rl/environment_risk.py`
- Test: `tests/risk/test_pretrade.py`
- Test: `tests/rl/test_environment_risk_service.py`

**Interfaces:**
- Extend `RiskConstrainedTarget` with optional copied arrays `proposal_weights` and `pretrade_weights`.
- `PreTradeRisk.constrain` records the original finite proposal and its own final pre-trade output.
- `EnvironmentRiskProjector.project` records the original proposal, pre-trade output, and final portfolio-feasible `weights`.

- [ ] **Step 1: Write failing tests**

Assert that pre-trade output exposes the original proposal and final pre-trade target, and that portfolio projection retains both while `weights` contains the final feasible target. Mutating caller-owned arrays after the call must not alter the stored stages.

- [ ] **Step 2: Verify RED**

Run the two focused test files and confirm missing attributes fail.

- [ ] **Step 3: Implement copied stage fields**

Add default-`None` fields for backward-compatible construction. Fill both fields with independent copies in maintained constructors.

- [ ] **Step 4: Verify GREEN**

Run the focused risk suites.

- [ ] **Step 5: Commit**

Commit message: `feat: preserve risk projection stages`.

### Task 4: Attach diagnostics and costs to environment info

**Files:**
- Modify: `trade_rl/rl/environment.py`
- Modify: `trade_rl/rl/environment_info.py`
- Modify: `trade_rl/integrations/sb3_training.py`
- Test: `tests/rl/test_environment_info_service.py`
- Create: `tests/rl/test_environment_constraint_info.py`
- Test: `tests/integrations/test_sb3_training.py`

**Interfaces:**
- Add `action_path: ActionPathDiagnostics` and `constraint_costs: ConstraintCostVector` to `EnvironmentStepInfoRequest`.
- Step info exposes both objects and flat keys prefixed `action_path_` and `constraint_cost_`.
- `_compact_training_info` preserves the small immutable objects and flat scalar keys while continuing to remove history-bearing execution objects.

- [ ] **Step 1: Write failing tests**

Extend the info-builder unit test with deterministic diagnostic and cost objects. Add one environment integration test that steps a small target-weight environment and verifies:
- scalar reward still equals `reward_total_scaled`;
- action-path arrays and scalar distances are present;
- all constraint costs are finite and non-negative;
- `constraint_cost_execution_fraction` reflects actual execution economics;
- compact training info retains new compact fields and removes heavy execution objects.

- [ ] **Step 2: Verify RED**

Run the three focused test files. Expected failures are missing request fields and info keys.

- [ ] **Step 3: Integrate in `ResidualMarketEnv.step`**

Capture `previous_hybrid_equity` before execution. After transition accounting:
- build `ActionPathDiagnostics` from the delayed executed proposal, stored pre-trade target, final risk target, actual submitted target, and final book weights;
- calculate costs using final drawdown/margin state and the hybrid execution interval;
- use `termination_reason`, `emergency_deleverage`, `liquidation_terminal`, and `liquidation_complete` to classify events;
- pass both objects to the info builder.

Set the maintained drawdown budget to the existing `pre_trade_risk.config.drawdown_start`; do not introduce a new configuration field in PR A.

- [ ] **Step 4: Verify GREEN**

Run all focused tests, then `pytest tests/rl tests/risk tests/integrations/test_sb3_training.py -q`.

- [ ] **Step 5: Commit**

Commit message: `feat: expose action and constraint telemetry`.

### Task 5: Full verification and PR evidence

**Files:**
- Modify: `docs/superpowers/specs/2026-07-26-constrained-growth-ppo-design.md` only if implementation terminology differs from the maintained risk-pipeline order.

**Interfaces:**
- No new runtime behavior.

- [ ] **Step 1: Run formatting and static checks**

Run:

```bash
ruff format --check .
ruff check .
mypy .
```

- [ ] **Step 2: Run complete tests**

Run `pytest -q` with the repository coverage gates.

- [ ] **Step 3: Review invariants**

Confirm no production code reads `constraint_costs` to modify rewards or policy updates; verify ordinary PPO files have no algorithmic change; verify all costs are causal and non-negative.

- [ ] **Step 4: Open PR A**

Title: `feat: add action and constraint cost contracts`

The PR body must list exact formulas, state that scalar rewards and PPO are unchanged, include focused/full CI evidence, and identify PR B as the next dependency.
