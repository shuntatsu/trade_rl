# Execution Robustness Environment Implementation Plan

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

## Task 1: Add the execution-environment stress contract

Files:

- Create `trade_rl/simulation/execution_stress.py`.
- Modify `trade_rl/workflows/market_walk_forward_config.py`.
- Test in `tests/workflows/test_execution_robustness_config.py`.

The maintained scenario type declares fee, spread, impact, slippage, capacity, latency, tail-slippage, and borrow fields. A frozen `ExecutionEnvironmentStress` applies those values to an immutable `ExecutionCostConfig` while retaining the inherited tick, lot, minimum-notional, and adverse-rounding rule stress.

## Task 2: Apply the stress to both accounting paths

Files:

- Modify `trade_rl/rl/environment_reward_execution_resources.py`.
- Extend `tests/rl/test_environment_reward_execution_resources.py`.

Resolve one immutable stressed cost inside `EnvironmentRewardExecutionResourcesBuilder.build()` and pass it to both hybrid and shadow `MarketExecutor` instances. Base `ExecutionRuleStress` objects preserve the original cost object and historical behavior.

## Task 3: Add a maintained report-only robustness profile

Files:

- Create `examples/binance-multitimeframe/walk-forward-target-weight-execution-robustness.json`.
- Create `docs/EXECUTION_ROBUSTNESS.md`.
- Create `tests/examples/test_execution_robustness_profile.py`.

The profile retains the existing three candidate profiles and `joint_2x` required gate, then adds report-only fee/spread, impact, capacity, latency, tail-slippage, borrow, and joint-adverse scenarios.

## Task 4: Review and full verification

- Compare the final changed-file set with open PR #369 and PR #372; the intersection must be empty.
- Run pytest with coverage, Ruff, formatting, MyPy, import architecture, dead-code, compatibility, frontend, package identity, and training-image checks at one exact head.
- Review backward compatibility, selected/baseline stress parity, immutable base costs, digest coverage, production-gate isolation, and unused code.
- Publish and update Draft PR #377; do not merge without explicit authorization.
