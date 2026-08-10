# Reward / Episode-Boundary Reconciliation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reapply the completed PR #369 reward/episode-boundary contract onto the current `main`, while preserving the already-merged PR #368 telemetry semantics and PR #370 three-candidate default workflow.

**Architecture:** Build a conflict-resolved merge commit whose first parent is current `main` and whose second parent is PR #369’s final head. Use the current `main` tree as the base, overlay only the semantic boundary/profile/document changes, and intentionally exclude stale telemetry formatting plus PR #369’s obsolete two-candidate workflow state.

**Tech Stack:** Python 3.12, Gymnasium, Stable-Baselines3, frozen dataclasses, canonical JSON/content digests, pytest, Ruff, MyPy, Import Linter, GitHub Actions.

## Global Constraints

- Pure-growth reward remains `scale * log(net_equity_after / net_equity_before)`.
- Do not mix drawdown, margin, turnover, execution cost, terminal equity, or baseline shaping into maintained pure-growth profiles.
- Maintained `gamma = 1.0` profiles use `finite_horizon_termination`, `finite_horizon_observation = true`, and `liquidate_on_end = false`.
- Discounted continuing profiles use `external_truncation`, `finite_horizon_observation = false`, and `liquidate_on_end = false`.
- Default environment behavior remains `external_truncation`; historical default environment digests stay unchanged.
- Only non-default finite-horizon mode is serialized into environment identity.
- `terminated` and `truncated` remain mutually exclusive.
- Finite-horizon time-limit termination does not set `hybrid_terminated` and does not trigger terminal-equity shaping.
- Preserve PR #368 exploration and telemetry compatibility behavior exactly.
- Preserve PR #370’s default candidate order exactly: PPO gamma-one, Lagrangian gamma-one, discounted Lagrangian.
- Do not include Universal-policy, symbol routing, catalog, normalization, BC, architecture, Docker, or serving changes.
- Do not modify the active legacy BTC generation.
- Production remains `NO-GO`.

---

### Task 1: Pin the boundary contract with RED tests

**Files:**
- Modify: `tests/rl/test_environment_time_config.py`
- Modify: `tests/rl/test_transition_shadow_failure.py`
- Modify: `tests/workflows/test_training_terminal_contract.py`
- Create: `tests/examples/test_reward_objective_boundary_profiles.py`

**Interfaces:**
- Consumes: current `ResidualMarketEnvConfig`, `classify_economic_transition()`, environment identity, and maintained JSON profiles.
- Produces: regression tests for enum normalization, finite-horizon validation, Gymnasium flags, identity compatibility, and profile semantics.

- [ ] **Step 1: Add config tests**

```python
from trade_rl.rl.environment_config import EpisodeBoundaryMode, ResidualMarketEnvConfig


def test_finite_horizon_boundary_requires_time_to_go() -> None:
    with pytest.raises(
        ValueError,
        match="finite_horizon_termination requires finite_horizon_observation",
    ):
        ResidualMarketEnvConfig(
            initial_capital=100_000.0,
            episode_boundary_mode="finite_horizon_termination",
            finite_horizon_observation=False,
        )


def test_finite_horizon_boundary_normalizes_to_enum() -> None:
    config = ResidualMarketEnvConfig(
        initial_capital=100_000.0,
        episode_boundary_mode="finite_horizon_termination",
        finite_horizon_observation=True,
    )
    assert config.episode_boundary_mode is EpisodeBoundaryMode.FINITE_HORIZON_TERMINATION
    assert config.time_limit_terminates is True
```

- [ ] **Step 2: Add pure transition tests**

```python
result = classify_economic_transition(
    hybrid=hybrid,
    shadow=shadow,
    time_limit_reached=True,
    liquidation_terminal=False,
    liquidation_complete=True,
    time_limit_terminates=True,
)
assert result.terminated is True
assert result.truncated is False
assert result.reason == "finite_horizon"
```

Retain a separate assertion that default time limits produce `terminated=False`, `truncated=True`.

- [ ] **Step 3: Add identity tests**

```python
assert default_external.environment_digest == explicit_external.environment_digest
assert finite_horizon.environment_digest != explicit_external.environment_digest
assert finite_horizon.observation_contract_digest == explicit_external.observation_contract_digest
```

- [ ] **Step 4: Add canonical profile tests**

For gamma-one profiles:

```python
assert payload["training"]["gamma"] == 1.0
assert payload["environment"]["episode_boundary_mode"] == "finite_horizon_termination"
assert payload["environment"]["finite_horizon_observation"] is True
assert payload["environment"]["liquidate_on_end"] is False
```

For discounted profiles:

```python
assert payload["training"]["gamma"] < 1.0
assert payload["environment"].get("episode_boundary_mode", "external_truncation") == "external_truncation"
assert payload["environment"]["finite_horizon_observation"] is False
assert payload["environment"]["liquidate_on_end"] is False
```

- [ ] **Step 5: Verify RED**

```bash
pytest \
  tests/rl/test_environment_time_config.py \
  tests/rl/test_transition_shadow_failure.py \
  tests/workflows/test_training_terminal_contract.py \
  tests/examples/test_reward_objective_boundary_profiles.py -q
```

Expected: failures for missing boundary enum/config/transition support and unmigrated profiles.

- [ ] **Step 6: Commit RED tests**

```bash
git add tests/rl tests/workflows/test_training_terminal_contract.py tests/examples/test_reward_objective_boundary_profiles.py
git commit -m "test: pin finite-horizon growth boundary contract"
```

---

### Task 2: Implement explicit environment boundary semantics

**Files:**
- Modify: `trade_rl/rl/environment_config.py`
- Modify: `trade_rl/rl/transition.py`
- Modify: `trade_rl/rl/environment_transition.py`
- Modify: `trade_rl/rl/environment.py`

**Interfaces:**
- Produces: `EpisodeBoundaryMode`, `ResidualMarketEnvConfig.time_limit_terminates`, and `classify_economic_transition(..., time_limit_terminates: bool = False)`.
- Consumes: existing environment persistence and content-digest contracts.

- [ ] **Step 1: Add the enum and config field**

```python
from enum import Enum


class EpisodeBoundaryMode(str, Enum):
    EXTERNAL_TRUNCATION = "external_truncation"
    FINITE_HORIZON_TERMINATION = "finite_horizon_termination"
```

```python
episode_boundary_mode: EpisodeBoundaryMode | str = (
    EpisodeBoundaryMode.EXTERNAL_TRUNCATION
)
```

- [ ] **Step 2: Normalize and validate the mode**

```python
try:
    boundary_mode = EpisodeBoundaryMode(self.episode_boundary_mode)
except ValueError as error:
    raise ValueError("episode_boundary_mode is not supported") from error
if (
    boundary_mode is EpisodeBoundaryMode.FINITE_HORIZON_TERMINATION
    and not self.finite_horizon_observation
):
    raise ValueError(
        "finite_horizon_termination requires finite_horizon_observation"
    )
object.__setattr__(self, "episode_boundary_mode", boundary_mode)
```

```python
@property
def time_limit_terminates(self) -> bool:
    return (
        self.episode_boundary_mode
        is EpisodeBoundaryMode.FINITE_HORIZON_TERMINATION
    )
```

- [ ] **Step 3: Extend transition classification**

```python
finite_horizon_terminal = time_limit_reached and time_limit_terminates
terminated = hybrid.insolvent or liquidation_terminal or finite_horizon_terminal
truncated = time_limit_reached and not terminated
```

Return reason `"finite_horizon"` only after economic and liquidation reasons have been considered.

- [ ] **Step 4: Route the mode through termination coordination**

Pass:

```python
time_limit_terminates=self.config.time_limit_terminates
```

Keep:

```python
hybrid_terminated = hybrid.insolvent
shadow_terminated = shadow.insolvent
```

- [ ] **Step 5: Preserve historical environment identity**

Inside the environment digest payload, add only the non-default mode:

```python
**(
    {
        "episode_boundary_mode": EpisodeBoundaryMode(
            self.config.episode_boundary_mode
        ).value
    }
    if self.config.time_limit_terminates
    else {}
),
```

- [ ] **Step 6: Verify focused GREEN**

```bash
pytest \
  tests/rl/test_environment_time_config.py \
  tests/rl/test_transition_shadow_failure.py \
  tests/workflows/test_training_terminal_contract.py -q
ruff check trade_rl/rl/environment_config.py trade_rl/rl/transition.py trade_rl/rl/environment_transition.py trade_rl/rl/environment.py
ruff format --check trade_rl/rl/environment_config.py trade_rl/rl/transition.py trade_rl/rl/environment_transition.py trade_rl/rl/environment.py
mypy trade_rl/rl/environment_config.py trade_rl/rl/transition.py trade_rl/rl/environment_transition.py trade_rl/rl/environment.py
```

Expected: all pass.

- [ ] **Step 7: Commit implementation**

```bash
git add trade_rl/rl tests/rl tests/workflows/test_training_terminal_contract.py
git commit -m "feat: define finite-horizon environment boundaries"
```

---

### Task 3: Migrate only maintained training profiles

**Files:**
- Modify: `examples/binance-multitimeframe/training-action-head-ablation-direct.json`
- Modify: `examples/binance-multitimeframe/training-action-head-ablation-gate.json`
- Modify: `examples/binance-multitimeframe/training-constrained-growth-control.json`
- Modify: `examples/binance-multitimeframe/training-constrained-growth-discounted.json`
- Modify: `examples/binance-multitimeframe/training-constrained-growth-gae097.json`
- Modify: `examples/binance-multitimeframe/training-constrained-growth.json`
- Modify: `examples/binance-multitimeframe/training-growth-optimal.json`
- Modify: `examples/binance-multitimeframe/training-target-weight-constrained-growth-discounted.json`
- Modify: `examples/binance-multitimeframe/training-target-weight-constrained-growth.json`
- Modify: `examples/binance-multitimeframe/training-target-weight-growth-ppo.json`
- Modify: `tests/examples/test_constrained_growth_profiles.py`
- Modify: `tests/examples/test_growth_optimal_reward_profiles.py`
- Modify: `tests/examples/test_target_weight_constrained_growth_profiles.py`
- Test: `tests/examples/test_reward_objective_boundary_profiles.py`

**Interfaces:**
- Consumes: Task 2 boundary field.
- Produces: profile-level reward/boundary semantics without changing any walk-forward candidate catalog.

- [ ] **Step 1: Update gamma-one profiles**

```json
"episode_boundary_mode": "finite_horizon_termination",
"finite_horizon_observation": true,
"liquidate_on_end": false
```

- [ ] **Step 2: Update discounted profiles**

```json
"episode_boundary_mode": "external_truncation",
"finite_horizon_observation": false,
"liquidate_on_end": false
```

- [ ] **Step 3: Do not copy stale PR #369 walk-forward files**

Keep current `main` versions of every `walk-forward-*.json`. In particular, preserve `walk-forward-target-weight-constrained-growth.json` with exactly:

```json
[
  "training-target-weight-growth-ppo.json",
  "training-target-weight-constrained-growth.json",
  "training-target-weight-constrained-growth-discounted.json"
]
```

Do not add PR #369’s standalone discounted walk-forward files in this reconciliation PR.

- [ ] **Step 4: Verify profiles and PR #370 default**

```bash
pytest \
  tests/examples/test_constrained_growth_profiles.py \
  tests/examples/test_growth_optimal_reward_profiles.py \
  tests/examples/test_target_weight_constrained_growth_profiles.py \
  tests/examples/test_reward_objective_boundary_profiles.py \
  tests/examples/test_full_research_default_workflow.py -q
```

Expected: all pass and the default candidate order remains unchanged.

- [ ] **Step 5: Commit profiles**

```bash
git add examples/binance-multitimeframe/training-*.json tests/examples
git commit -m "config: align growth profiles with finite-horizon semantics"
```

---

### Task 4: Reconcile documentation without reverting merged behavior

**Files:**
- Create: `docs/REWARD_OBJECTIVE.md`
- Modify: `docs/BINANCE.md`
- Modify: `docs/SINGLE_SYMBOL.md`
- Modify: `tests/architecture/test_maintained_single_symbol_boundary.py`

**Interfaces:**
- Consumes: Tasks 2–3.
- Produces: durable reward/boundary documentation that also preserves PR #370’s runtime default description.

- [ ] **Step 1: Add the durable objective document**

Document:

```text
reward_t = scale * log(net_equity_after / net_equity_before)
```

Define both boundary modes, independent cost channels, hard risk, finite-horizon telescoping, compatibility, and fail-closed conditions.

- [ ] **Step 2: Merge BINANCE and SINGLE_SYMBOL text**

Both documents must state:

```text
No explicit training template -> three target-weight profiles
training-full.json -> explicit legacy comparison only
```

Add the gamma-one finite-horizon and discounted external-truncation distinction without reverting that statement.

- [ ] **Step 3: Exclude stale telemetry-only formatting**

Do not modify `trade_rl/rl/tensorboard_logging.py` or `tests/rl/test_tensorboard_logging.py`; PR #369’s diff there was formatter-only and predates PR #368.

- [ ] **Step 4: Verify documentation contracts**

```bash
pytest \
  tests/architecture/test_maintained_single_symbol_boundary.py \
  tests/examples/test_full_research_default_workflow.py \
  tests/rl/test_tensorboard_logging.py -q
ruff check docs tests/architecture/test_maintained_single_symbol_boundary.py
```

Expected: all pass.

- [ ] **Step 5: Commit documentation**

```bash
git add docs tests/architecture/test_maintained_single_symbol_boundary.py
git commit -m "docs: define maintained growth and boundary contract"
```

---

### Task 5: Create a conflict-resolved merge commit and finish PR #369

**Files:**
- Modify: PR #369 metadata/body after verification.

**Interfaces:**
- Consumes: Tasks 1–4 plus current `main` and PR #369 final head.
- Produces: PR #369 as a descendant of both histories, mergeable against current `main`, with exact-head evidence.

- [ ] **Step 1: Create the resolved merge commit**

Use current `main` as first parent and `89703977d445abaa627f70b0081e2a1bb6d464e6` as second parent. The resolved tree must contain only the files listed in Tasks 1–4 on top of current `main`.

- [ ] **Step 2: Update the PR branch by fast-forward**

Move `agent/reward-objective-contract` to the merge commit without force. Because the old PR head is a parent, the update must be a fast-forward.

- [ ] **Step 3: Review the exact diff**

Confirm:

```text
No Universal-policy files
No active-run files
No stale telemetry replacement
No two-candidate default regression
No unrelated JSON compaction
```

- [ ] **Step 4: Run complete exact-head CI**

Require on one head:

```text
full pytest and branch coverage
Ruff and format
MyPy
Import Linter
Dead-code report
frontend tests/typecheck/build/layout
critical coverage
Windows/Ubuntu compatibility
Training image and non-root probe
PostgreSQL Catalog
Nautilus capability
structured serving/recovery smoke
```

- [ ] **Step 5: Self-review semantic invariants**

```text
external_truncation -> terminated=false, truncated=true
finite_horizon_termination -> terminated=true, truncated=false
finite-horizon boundary does not set hybrid_terminated
finite-horizon boundary does not trigger terminal-equity shaping
default external digest is unchanged
observation digest is mode-independent
PR #368 telemetry remains
PR #370 default order remains
```

- [ ] **Step 6: Update PR body and mark Ready**

Record RED/GREEN evidence, final SHA, all CI results, compatibility, active-run isolation, remaining risks, and `Production NO-GO`.

- [ ] **Step 7: Do not merge without explicit user authorization**

Leave PR #369 open and Ready after all verification succeeds.
