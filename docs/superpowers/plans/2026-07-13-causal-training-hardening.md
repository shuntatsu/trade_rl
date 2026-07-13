# Causal Training Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make episode endings, observation inputs, execution capacity, and PPO training identity causal, explicit, and reproducible.

**Architecture:** Preserve the existing residual environment boundaries. Tighten semantics at the environment/execution interfaces, introduce a stable observation schema constant, and pass one typed PPO configuration through training orchestration into the SB3 adapter and policy artifact identity.

**Tech Stack:** Python 3.12, Gymnasium, Stable-Baselines3 PPO, NumPy, pytest, Ruff, mypy.

## Global Constraints

- Keep `baseline_residual_v1` action semantics unchanged.
- Training default must not force liquidation at synthetic episode boundaries.
- Future tradability and future completed-bar volume must not be policy-visible decision inputs.
- Incomplete explicit liquidation must fail closed.
- Do not enlarge the model merely to raise GPU utilization.
- All production code changes require a failing regression test first.

---

### Task 1: Correct episode-end semantics

**Files:**
- Modify: `tests/rl/test_environment_timing.py`
- Modify: `tests/rl/test_environment_time_config.py`
- Modify: `trade_rl/rl/environment.py`

**Interfaces:**
- Produces: `ResidualMarketEnvConfig.liquidate_on_end: bool = False`
- Produces: mutually exclusive terminal flags for time-limit continuation and explicit liquidation.

- [ ] Add a failing test that the default time-limit path returns `truncated=True`, `terminated=False`, and no liquidation entries.
- [ ] Add a failing test that `liquidate_on_end=True` returns `terminated=True`, `truncated=False`, and flat books.
- [ ] Add a failing test that incomplete explicit liquidation raises a clear error.
- [ ] Implement the minimal terminal-flag and liquidation checks.
- [ ] Run targeted environment tests.

### Task 2: Remove training-only and future observation inputs

**Files:**
- Modify: `tests/rl/test_observation_v2.py`
- Modify: `tests/rl/test_environment_timing.py`
- Modify: `trade_rl/rl/observations.py`
- Modify: `trade_rl/rl/environment.py`
- Modify: `trade_rl/rl/__init__.py`

**Interfaces:**
- Produces: `OBSERVATION_SCHEMA = "baseline_residual_observation_v2"`
- Produces: observation layout with nine global account/risk fields and no progress field.

- [ ] Add a failing test that observation tradability equals `tradable[index]`, not `tradable[index + 1]`.
- [ ] Add a failing test that changing only `end_index` does not change the observation.
- [ ] Add a failing environment test showing future non-tradability is handled by execution, not pre-trade target rewriting.
- [ ] Remove episode progress and future tradability from observation construction.
- [ ] Remove future tradability filtering from `_constrain_target`.
- [ ] Export and test the observation schema constant.
- [ ] Run targeted observation and environment tests.

### Task 3: Make next-open liquidity capacity causal

**Files:**
- Modify: `tests/simulation/test_execution_v2.py`
- Modify: `trade_rl/simulation/execution.py`

**Interfaces:**
- `MarketExecutor.execute_interval()` uses `volume[close_index]` to estimate capacity for a fill at `open[close_index + 1]`.
- `MarketExecutor.liquidate_at_close()` still uses current close-bar volume and returns unfilled turnover.

- [ ] Add a failing test where `volume[t]` and `volume[t+1]` imply different capacities and assert the prior bar controls the next-open fill.
- [ ] Implement the capacity-volume argument separately from fill prices and actual tradability.
- [ ] Keep impact participation based on the same causal capacity volume.
- [ ] Run execution and environment identity tests.

### Task 4: Make PPO settings explicit and record actual work

**Files:**
- Modify: `tests/rl/test_ensemble_training.py`
- Modify: `tests/cli/test_cli.py`
- Modify: `trade_rl/rl/training.py`
- Modify: `trade_rl/domain/policies.py`
- Modify: `trade_rl/cli/app.py`

**Interfaces:**
- Produces: expanded `ResidualTrainingConfig` with explicit PPO defaults.
- Produces: `ResidualTrainingResult` from backend with checkpoint path, actual timesteps, and resolved device.
- Produces: `PolicyEnsembleManifest.training_config_digest`, `observation_schema`, `requested_timesteps`, `actual_timesteps`, and `resolved_device`.

- [ ] Add failing configuration validation tests for rollout length, batch divisibility, coefficients, and device.
- [ ] Add a failing test that `rounded_timesteps` is the smallest multiple of `n_steps` not below `timesteps`.
- [ ] Update the fake backend test contract to accept the full config and return run metadata.
- [ ] Implement typed backend result and explicit PPO fields.
- [ ] Forward all fields to `stable_baselines3.PPO` and verify the model-reported timestep count.
- [ ] Include canonical training configuration and observation schema in the ensemble digest.
- [ ] Extend CLI `train config` output and arguments.
- [ ] Run training, domain, and CLI tests.

### Task 5: Document GPU expectations and verify the branch

**Files:**
- Modify: `README.md`
- Modify: `docs/RESEARCH_STATUS.md`

**Interfaces:**
- Documents: low GPU utilization for a small single-environment MLP PPO is expected; throughput and OOS quality are the decision metrics.

- [ ] Document explicit `--device`, rollout, batch, and epoch configuration.
- [ ] Document why GPU utilization can remain low and when vectorized environments/larger batches are appropriate.
- [ ] Run Ruff, format check, mypy, Import Linter, pytest with branch coverage, and CLI smoke test.
- [ ] Open a draft PR and use GitHub Actions as the authoritative verification because this session cannot execute the repository locally.
