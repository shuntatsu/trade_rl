# Execution Robustness Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend sealed walk-forward execution sensitivity from exchange-rule rounding only to deterministic cost, liquidity, latency, tail-slippage, and borrow stress without changing training, BC, reward, episode-boundary, or selection semantics.

**Architecture:** Add a simulation-layer `ExecutionEnvironmentStress` subtype of the existing `ExecutionRuleStress`, so the existing evaluation transport can carry both exchange-rule and execution-cost assumptions without changing the environment facade or walk-forward core. The reward/execution resource builder applies the stress to a copied `ExecutionCostConfig` before constructing both hybrid and shadow executors. The maintained configuration wrapper parses the additional fields while retaining the existing v1 rule pack and requiring all non-standard scenarios to remain report-only.

**Tech Stack:** Python 3.12, frozen dataclasses, Gymnasium environment resources, pytest, JSON walk-forward profiles, GitHub Actions.

## Global Constraints

- Do not modify files changed by open PR #369 (reward and episode-boundary contract).
- Do not modify files changed by open PR #372 (behavior-cloning temporal correctness).
- Keep the existing `execution_sensitivity_config_v1` standard scenario pack and `required_scenario=joint_2x` unchanged.
- Additional execution-environment scenarios must remain `report_only=true` under configuration v1.
- Do not change PPO, Lagrangian PPO, BC, reward, action, risk, checkpoint, serving, or live-order behavior.
- Apply identical stress assumptions to selected and baseline evaluation paths.
- Preserve deterministic scenario identity in the existing experiment-plan and scenario-result digests.
- Production remains `NO-GO`.

---

### Task 1: Add the execution-environment stress contract

**Files:**
- Create: `trade_rl/simulation/execution_stress.py`
- Modify: `trade_rl/workflows/market_walk_forward_config.py`
- Test: `tests/workflows/test_execution_robustness_config.py`

**Interfaces:**
- Produces: `ExecutionEnvironmentStress(ExecutionRuleStress)` with `apply(base: ExecutionCostConfig) -> ExecutionCostConfig`.
- Produces: `apply_execution_environment_stress(base, stress) -> ExecutionCostConfig`.
- Produces: extended `ExecutionSensitivityScenario.stress() -> ExecutionEnvironmentStress`.

- [ ] **Step 1: Write the failing dataclass-field contract test**

```python
from dataclasses import fields

from trade_rl.workflows.market_walk_forward_config import ExecutionSensitivityScenario


def test_execution_sensitivity_scenario_declares_environment_cost_stress_fields() -> None:
    field_names = {field.name for field in fields(ExecutionSensitivityScenario)}
    assert {
        "fee_multiplier",
        "spread_multiplier",
        "impact_multiplier",
        "slippage_std_multiplier",
        "participation_fraction",
        "minimum_order_latency_bars",
        "tail_slippage_probability_floor",
        "tail_slippage_multiplier_floor",
        "borrow_rate_multiplier",
    } <= field_names
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `pytest tests/workflows/test_execution_robustness_config.py -q`

Expected: one assertion failure because the current scenario exposes only tick, lot, minimum-notional, rounding, and report-only fields.

- [ ] **Step 3: Add the simulation-layer stress object**

Implement a frozen, slotted subclass with identity defaults. Validate multipliers as finite and at least one, `participation_fraction` in `(0, 1]`, latency as a non-negative integer, probability in `[0, 1]`, and tail multiplier floor at least one.

`apply()` must use `dataclasses.replace` and set:

```python
fee_rate = base.fee_rate * fee_multiplier
maker_fee_rate = base.maker_fee_rate * fee_multiplier
taker_fee_rate = base.taker_fee_rate * fee_multiplier
spread_rate = base.spread_rate * spread_multiplier
impact_rate = base.impact_rate * impact_multiplier
slippage_std = base.slippage_std * slippage_std_multiplier
max_participation_rate = base.max_participation_rate * participation_fraction
order_latency_bars = max(base.order_latency_bars, minimum_order_latency_bars)
tail_slippage_probability = max(base.tail_slippage_probability, tail_slippage_probability_floor)
tail_slippage_multiplier = max(base.tail_slippage_multiplier, tail_slippage_multiplier_floor)
borrow_rate_multiplier = base.borrow_rate_multiplier * borrow_rate_multiplier
```

- [ ] **Step 4: Extend the maintained walk-forward scenario parser**

Replace the current alias with a frozen, slotted subclass of the base scenario. Parse all standard and report-only extension rows into the extended class, preserve standard rule values resolved by the base parser, and include every execution-stress field in `digest_payload()` through `stress().digest_payload()`.

- [ ] **Step 5: Run focused tests and commit GREEN**

Run:

```bash
pytest tests/workflows/test_execution_robustness_config.py -q
ruff check trade_rl/simulation/execution_stress.py trade_rl/workflows/market_walk_forward_config.py tests/workflows/test_execution_robustness_config.py
```

Expected: all focused checks pass.

---

### Task 2: Apply the stress to both accounting paths

**Files:**
- Modify: `trade_rl/rl/environment_reward_execution_resources.py`
- Modify: `tests/rl/test_environment_reward_execution_resources.py`

**Interfaces:**
- Consumes: `apply_execution_environment_stress(base, stress)`.
- Produces: hybrid and shadow `MarketExecutor` instances using equal stressed `ExecutionCostConfig` values while retaining independent executor state.

- [ ] **Step 1: Write the failing resource-builder test**

Build a nonzero base cost, apply a joint stress, and assert both executors receive the multiplied fee, spread, impact, slippage, reduced participation, latency floor, tail-slippage floors, and borrow multiplier. Also assert the original config remains unchanged.

- [ ] **Step 2: Run the focused test and verify RED**

Run: `pytest tests/rl/test_environment_reward_execution_resources.py -q`

Expected: the new test fails because both executors still receive the original cost object.

- [ ] **Step 3: Apply the stress before executor construction**

Resolve one immutable stressed cost in `EnvironmentRewardExecutionResourcesBuilder.build()` and pass it to both `MarketExecutor` constructors. Continue passing the same stress object for tick, lot, minimum-notional, and adverse-rounding behavior.

- [ ] **Step 4: Run focused tests and commit GREEN**

Run:

```bash
pytest tests/rl/test_environment_reward_execution_resources.py tests/workflows/test_execution_robustness_config.py -q
ruff check trade_rl/rl/environment_reward_execution_resources.py tests/rl/test_environment_reward_execution_resources.py
```

Expected: all focused checks pass.

---

### Task 3: Add a maintained report-only robustness profile

**Files:**
- Create: `examples/binance-multitimeframe/walk-forward-target-weight-execution-robustness.json`
- Create: `docs/EXECUTION_ROBUSTNESS.md`
- Create: `tests/examples/test_execution_robustness_profile.py`

**Interfaces:**
- Consumes: the existing three target-weight `run_file` profiles.
- Produces: one six-fold robustness workflow retaining `joint_2x` as the required production gate and adding report-only fee/spread, impact, capacity, latency, tail-slippage, borrow, and joint-adverse scenarios.

- [ ] **Step 1: Write the failing profile contract test**

Assert the new file exists, loads through `MarketWalkForwardConfig.from_json`, references the existing candidate names in order, retains `required_scenario == "joint_2x"`, and exposes the exact extended scenario assumptions through `scenario.stress()`.

- [ ] **Step 2: Run the focused test and verify RED**

Run: `pytest tests/examples/test_execution_robustness_profile.py -q`

Expected: failure because the profile does not yet exist.

- [ ] **Step 3: Add the profile and operations document**

Include these report-only scenarios:

- `fee_spread_2x`
- `impact_2x`
- `capacity_half`
- `latency_1bar`
- `tail_slippage_adverse`
- `borrow_2x`
- `joint_execution_adverse`

The joint scenario uses rule factors 2x, fee 1.5x, spread and impact 2x, slippage standard deviation 2x, participation 0.5x, one-bar minimum latency, one-percent tail probability, tail multiplier floor 5x, and borrow 2x.

- [ ] **Step 4: Run profile and parser tests and commit GREEN**

Run:

```bash
pytest tests/examples/test_execution_robustness_profile.py tests/workflows/test_execution_robustness_config.py -q
python -m json.tool examples/binance-multitimeframe/walk-forward-target-weight-execution-robustness.json >/dev/null
```

Expected: all focused checks pass.

---

### Task 4: Review and full verification

**Files:**
- Review every changed file and the complete branch diff.

- [ ] **Step 1: Verify no overlap with active work**

Compare the branch changed-file set against PR #369 and PR #372. The intersection must be empty.

- [ ] **Step 2: Run repository verification**

Run the repository CI suite at one exact head, including pytest with branch coverage, Ruff, formatting, MyPy, import architecture, dead-code, compatibility, frontend, package identity, and training-image checks.

- [ ] **Step 3: Self-review**

Confirm backward compatibility for existing v1 JSON, exact selected/baseline stress parity, immutable base costs, deterministic digest coverage, no unused abstractions, and no production-gate widening.

- [ ] **Step 4: Publish a Draft PR**

The PR must list What, Why, design decisions, non-goals, RED/GREEN evidence, exact-head CI, overlap audit, and remaining risk. Keep Production `NO-GO`.
