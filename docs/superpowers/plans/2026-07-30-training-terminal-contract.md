# Training Terminal Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make standalone and nested walk-forward training preserve the configured mark-to-market truncation contract while keeping range evaluation liquidation explicit.

**Architecture:** Training configuration must not be rewritten from `liquidate_on_end=false` to forced-close termination. A shared training-environment resolver will preserve mark-to-market semantics, set only fold/run-local episode bounds and reward pre-roll requirements, and leave evaluation construction responsible for explicitly requesting terminal liquidation.

**Tech Stack:** Python 3.12, dataclasses, Gymnasium 0.29.1, pytest, Stable-Baselines3 2.3.2.

## Global Constraints

- The maintained target-weight growth profiles remain `finite_horizon_observation=false` and `liquidate_on_end=false`.
- Artificial 720-hour training boundaries are Gymnasium truncations, not economic terminations.
- Reward and seven cost critics retain terminal-observation bootstrap on truncation.
- Out-of-sample range evaluation continues to request `liquidate_on_end=true` explicitly.
- No reward, action, risk, network, seed, fold, or execution-cost setting changes in this PR.

---

### Task 1: Lock the training terminal contract with failing tests

**Files:**
- Create: `tests/workflows/test_training_terminal_contract.py`

**Interfaces:**
- Consumes: `TrainingRunConfig.from_json`, `normalize_training_run_config`, `_maintained_training_environment`.
- Produces: regression tests requiring standalone and walk-forward training to preserve `mark_to_market` terminal accounting.

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path

from trade_rl.workflows._market_walk_forward_core import (
    _maintained_training_environment,
)
from trade_rl.workflows.training_run import (
    TrainingRunConfig,
    normalize_training_run_config,
)

ROOT = Path(__file__).resolve().parents[2]
PROFILE = ROOT / "examples/binance-multitimeframe/training-target-weight-constrained-growth.json"


def test_standalone_training_preserves_mark_to_market_truncation() -> None:
    config = TrainingRunConfig.from_json(PROFILE)

    normalized = normalize_training_run_config(config)

    assert normalized.environment.liquidate_on_end is False
    assert normalized.environment.terminal_accounting_mode == "mark_to_market"
    assert normalized.environment.finite_horizon_observation is False


def test_walk_forward_training_preserves_mark_to_market_truncation() -> None:
    config = TrainingRunConfig.from_json(PROFILE)

    environment = _maintained_training_environment(
        config.environment,
        episode_bars=1_024,
    )

    assert environment.episode_bars == 1_024
    assert environment.episode_hour_choices == ()
    assert environment.liquidate_on_end is False
    assert environment.terminal_accounting_mode == "mark_to_market"
    assert environment.require_full_reward_preroll is True
```

- [ ] **Step 2: Run CI and verify RED**

Run through the branch CI. Expected: both tests fail because current workflow normalization sets `liquidate_on_end=True`.

### Task 2: Centralize training-environment resolution

**Files:**
- Create: `trade_rl/workflows/training_environment.py`
- Modify: `trade_rl/workflows/training_run.py`
- Modify: `trade_rl/workflows/_market_walk_forward_core.py`
- Test: `tests/workflows/test_training_terminal_contract.py`

**Interfaces:**
- Produces: `resolve_training_environment(config: ResidualMarketEnvConfig, *, episode_bars: int | None = None) -> ResidualMarketEnvConfig`.
- Consumes: the configured `liquidate_on_end` value and rejects training recipes that request forced-close termination without finite-horizon observation.

- [ ] **Step 1: Implement the shared resolver**

```python
def resolve_training_environment(
    config: ResidualMarketEnvConfig,
    *,
    episode_bars: int | None = None,
) -> ResidualMarketEnvConfig:
    if config.liquidate_on_end:
        raise ValueError(
            "training environments must use mark-to-market truncation; "
            "terminal liquidation belongs to explicit evaluation ranges"
        )
    changes: dict[str, object] = {
        "episode_hour_choices": (),
        "require_full_reward_preroll": True,
    }
    if episode_bars is not None:
        changes["episode_bars"] = episode_bars
    return replace(config, **changes)
```

- [ ] **Step 2: Route standalone training through the resolver**

Replace the old `normalize_training_run_config` forced-liquidation rewrite with a dataclass replacement that uses `resolve_training_environment(config.environment)`.

- [ ] **Step 3: Route walk-forward training through the resolver**

Replace `_maintained_training_environment` forced-liquidation rewrite with `resolve_training_environment(config, episode_bars=episode_bars)` and retain `fail_on_incomplete_emergency_liquidation=False` only as an explicit emergency-liquidation behavior.

- [ ] **Step 4: Run targeted tests and verify GREEN**

Run:

```bash
pytest -q tests/workflows/test_training_terminal_contract.py
pytest -q tests/examples/test_target_weight_constrained_growth_profiles.py
pytest -q tests/workflows/test_training_run_config.py
```

Expected: all pass.

### Task 3: Verify environment and evaluation boundaries

**Files:**
- Test: `tests/rl/test_environment_time_config.py`
- Test: `tests/integrations/test_cost_critic_ppo.py`
- Test: `tests/workflows/test_training_terminal_contract.py`

**Interfaces:**
- Confirms training time limits remain truncations with reward/cost bootstrap and explicit range evaluation remains liquidating.

- [ ] **Step 1: Run terminal-semantics tests**

```bash
pytest -q tests/rl/test_environment_time_config.py
pytest -q tests/integrations/test_cost_critic_ppo.py
pytest -q tests/workflows/test_training_terminal_contract.py
```

- [ ] **Step 2: Run repository quality gates**

```bash
ruff check .
ruff format --check .
mypy .
lint-imports
pytest --cov=trade_rl --cov-branch
```

- [ ] **Step 3: Commit and open a focused PR**

Use a PR limited to the training terminal contract. Do not mix README or BC schema changes into this PR.
