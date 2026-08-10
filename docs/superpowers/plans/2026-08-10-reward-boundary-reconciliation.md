# Reward / Episode-Boundary Reconciliation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reapply the completed PR #369 reward/episode-boundary contract onto the current `main` that already contains PR #368 and PR #370, without mixing Universal-policy implementation into the change.

**Architecture:** Start from the latest `main`, add an explicit `EpisodeBoundaryMode` to the environment contract, route it through transition classification and environment identity, then migrate only the maintained gamma-one and discounted profiles. Preserve legacy default semantics by omitting the default `external_truncation` field from the environment digest, retain PR #368 telemetry behavior, and retain PR #370’s target-weight default workflow.

**Tech Stack:** Python 3.12, Gymnasium, Stable-Baselines3, frozen dataclasses, canonical JSON/content digests, pytest, Ruff, MyPy, Import Linter, GitHub Actions.

## Global Constraints

- The maintained economic scalar reward remains actual after-cost interval net log growth; do not add drawdown, margin, turnover, execution-cost, terminal-equity, or baseline penalties to the pure-growth profiles.
- `gamma = 1.0` maintained growth profiles must use `finite_horizon_termination`, `finite_horizon_observation = true`, and `liquidate_on_end = false`.
- Discounted continuing ablations must use `external_truncation`, `finite_horizon_observation = false`, and `liquidate_on_end = false`.
- Default environment behavior remains `external_truncation`; existing configs and artifacts retain their historical meaning.
- Only non-default `finite_horizon_termination` is added to environment identity, so explicit/default external truncation produce the historical digest.
- `terminated` and `truncated` must remain mutually exclusive.
- A finite-horizon time limit is not an economic insolvency and must not set `hybrid_terminated` or trigger terminal-equity shaping.
- Preserve PR #368’s smooth exploration telemetry and legacy `terminated`/`truncated` normalization.
- Preserve PR #370’s no-override default workflow and three-candidate order.
- Do not modify Universal-policy, symbol routing, catalog, normalization, BC, model architecture, Docker, or serving behavior in this PR.
- Do not stop, rewrite, resume, or migrate the active legacy BTC generation.
- Production remains `NO-GO`.

---

### Task 1: Reproduce the boundary-contract failures on current `main`

**Files:**
- Modify: `tests/rl/test_environment_time_config.py`
- Modify: `tests/rl/test_transition_shadow_failure.py`
- Modify: `tests/workflows/test_training_terminal_contract.py`
- Create: `tests/examples/test_reward_objective_boundary_profiles.py`

**Interfaces:**
- Consumes: existing `ResidualMarketEnvConfig`, `classify_economic_transition()`, training JSON loader, and profile fixtures.
- Produces: failing contract tests for `EpisodeBoundaryMode`, finite-horizon Gymnasium flags, environment identity, and canonical profile alignment.

- [ ] **Step 1: Add failing config and transition tests**

Add tests equivalent to the following contracts:

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

Add pure transition tests:

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

Retain the external-truncation assertion:

```python
assert classify_economic_transition(
    hybrid=hybrid,
    shadow=shadow,
    time_limit_reached=True,
    liquidation_terminal=False,
    liquidation_complete=True,
).truncated is True
```

- [ ] **Step 2: Add failing identity and profile tests**

Pin these contracts:

```python
assert default_external.environment_digest == explicit_external.environment_digest
assert finite_horizon.environment_digest != explicit_external.environment_digest
assert finite_horizon.observation_contract_digest == explicit_external.observation_contract_digest
```

For every maintained gamma-one profile assert:

```python
assert payload["training"]["gamma"] == 1.0
assert payload["environment"]["episode_boundary_mode"] == "finite_horizon_termination"
assert payload["environment"]["finite_horizon_observation"] is True
assert payload["environment"]["liquidate_on_end"] is False
```

For every discounted profile assert:

```python
assert payload["training"]["gamma"] < 1.0
assert payload["environment"].get("episode_boundary_mode", "external_truncation") == "external_truncation"
assert payload["environment"]["finite_horizon_observation"] is False
assert payload["environment"]["liquidate_on_end"] is False
```

- [ ] **Step 3: Run focused tests and verify RED**

Run:

```bash
pytest \
  tests/rl/test_environment_time_config.py \
  tests/rl/test_transition_shadow_failure.py \
  tests/workflows/test_training_terminal_contract.py \
  tests/examples/test_reward_objective_boundary_profiles.py -q
```

Expected: failures caused by missing `EpisodeBoundaryMode`, missing `episode_boundary_mode`, missing `time_limit_terminates`, and unchanged gamma-one profile boundary fields. Existing unrelated tests must remain green.

- [ ] **Step 4: Commit the RED tests**

```bash
git add \
  tests/rl/test_environment_time_config.py \
  tests/rl/test_transition_shadow_failure.py \
  tests/workflows/test_training_terminal_contract.py \
  tests/examples/test_reward_objective_boundary_profiles.py
git commit -m "test: pin finite-horizon growth boundary contract"
```

---

### Task 2: Implement explicit environment boundary semantics

**Files:**
- Modify: `trade_rl/rl/environment_config.py`
- Modify: `trade_rl/rl/transition.py`
- Modify: `trade_rl/rl/environment_transition.py`
- Modify: `trade_rl/rl/environment.py`
- Modify: `trade_rl/rl/environment_transition.py`
- Test: `tests/rl/test_environment_time_config.py`
- Test: `tests/rl/test_transition_shadow_failure.py`
- Test: `tests/workflows/test_training_terminal_contract.py`

**Interfaces:**
- Produces: `EpisodeBoundaryMode`, `ResidualMarketEnvConfig.time_limit_terminates`, and `classify_economic_transition(..., time_limit_terminates: bool = False)`.
- Consumes: existing environment config persistence, transition coordinator, and content-digest contracts.

- [ ] **Step 1: Add the enum and validated config field**

Implement in `trade_rl/rl/environment_config.py`:

```python
from enum import Enum


class EpisodeBoundaryMode(str, Enum):
    EXTERNAL_TRUNCATION = "external_truncation"
    FINITE_HORIZON_TERMINATION = "finite_horizon_termination"
```

Add to `ResidualMarketEnvConfig` after terminal-accounting fields:

```python
episode_boundary_mode: EpisodeBoundaryMode | str = (
    EpisodeBoundaryMode.EXTERNAL_TRUNCATION
)
```

Normalize and validate in `__post_init__`:

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

Expose:

```python
@property
def time_limit_terminates(self) -> bool:
    return (
        self.episode_boundary_mode
        is EpisodeBoundaryMode.FINITE_HORIZON_TERMINATION
    )
```

- [ ] **Step 2: Extend the pure transition classifier**

Change the signature in `trade_rl/rl/transition.py`:

```python
def classify_economic_transition(
    *,
    hybrid: BookState,
    shadow: BookState,
    time_limit_reached: bool,
    liquidation_terminal: bool,
    liquidation_complete: bool,
    time_limit_terminates: bool = False,
) -> EconomicTransition:
```

Implement exclusive classification:

```python
finite_horizon_terminal = time_limit_reached and time_limit_terminates
terminated = hybrid.insolvent or liquidation_terminal or finite_horizon_terminal
truncated = time_limit_reached and not terminated
```

After economic and liquidation reasons, return `"finite_horizon"` for the intrinsic time boundary. Do not mutate either book and do not mark `hybrid_terminated` true for this reason.

- [ ] **Step 3: Route the mode through the coordinator**

In `EnvironmentTerminationCoordinator.resolve()` pass:

```python
time_limit_terminates=self.config.time_limit_terminates
```

Keep:

```python
hybrid_terminated = hybrid.insolvent
shadow_terminated = shadow.insolvent
```

This preserves reward/economic termination semantics while changing only Gymnasium boundary flags.

- [ ] **Step 4: Bind only the non-default mode into environment identity**

Import `EpisodeBoundaryMode` in `trade_rl/rl/environment.py` and add this conditional entry inside `environment_config`:

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

Do not add `external_truncation` to the historical default digest payload.

- [ ] **Step 5: Run focused tests**

```bash
pytest \
  tests/rl/test_environment_time_config.py \
  tests/rl/test_transition_shadow_failure.py \
  tests/workflows/test_training_terminal_contract.py -q
```

Expected: PASS.

- [ ] **Step 6: Run static checks for changed modules**

```bash
ruff check \
  trade_rl/rl/environment_config.py \
  trade_rl/rl/transition.py \
  trade_rl/rl/environment_transition.py \
  trade_rl/rl/environment.py \
  tests/rl/test_environment_time_config.py \
  tests/rl/test_transition_shadow_failure.py \
  tests/workflows/test_training_terminal_contract.py
ruff format --check \
  trade_rl/rl/environment_config.py \
  trade_rl/rl/transition.py \
  trade_rl/rl/environment_transition.py \
  trade_rl/rl/environment.py
mypy \
  trade_rl/rl/environment_config.py \
  trade_rl/rl/transition.py \
  trade_rl/rl/environment_transition.py \
  trade_rl/rl/environment.py
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add trade_rl/rl tests/rl tests/workflows/test_training_terminal_contract.py
git commit -m "feat: define finite-horizon environment boundaries"
```

---

### Task 3: Migrate maintained profiles without changing reward meaning

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
- Create: `examples/binance-multitimeframe/walk-forward-constrained-growth-discounted.json`
- Modify: `examples/binance-multitimeframe/walk-forward-constrained-growth.json`
- Modify: `examples/binance-multitimeframe/walk-forward-growth-optimal.json`
- Create: `examples/binance-multitimeframe/walk-forward-target-weight-constrained-growth-discounted.json`
- Modify: `examples/binance-multitimeframe/walk-forward-target-weight-constrained-growth.json`
- Modify: `tests/examples/test_constrained_growth_profiles.py`
- Modify: `tests/examples/test_growth_optimal_reward_profiles.py`
- Modify: `tests/examples/test_target_weight_constrained_growth_profiles.py`
- Test: `tests/examples/test_reward_objective_boundary_profiles.py`

**Interfaces:**
- Consumes: Task 2’s boundary config field.
- Produces: canonical gamma-one and discounted profile families whose only reward/time-boundary differences are explicit and test-pinned.

- [ ] **Step 1: Migrate every gamma-one maintained profile**

Add to each gamma-one profile environment:

```json
"episode_boundary_mode": "finite_horizon_termination",
"finite_horizon_observation": true,
"liquidate_on_end": false
```

Do not change its pure-growth reward block.

- [ ] **Step 2: Pin discounted continuing profiles**

Set or preserve:

```json
"episode_boundary_mode": "external_truncation",
"finite_horizon_observation": false,
"liquidate_on_end": false
```

Do not change the configured discount half-life or Lagrangian budget values.

- [ ] **Step 3: Preserve PR #370’s default workflow**

`walk-forward-target-weight-constrained-growth.json` must still expose the exact no-override default candidate order:

```json
[
  "training-target-weight-growth-ppo.json",
  "training-target-weight-constrained-growth.json",
  "training-target-weight-constrained-growth-discounted.json"
]
```

The new standalone discounted walk-forward files are explicit ablations; they must not replace or reorder the PR #370 default catalog.

- [ ] **Step 4: Run all profile contract tests**

```bash
pytest \
  tests/examples/test_constrained_growth_profiles.py \
  tests/examples/test_growth_optimal_reward_profiles.py \
  tests/examples/test_target_weight_constrained_growth_profiles.py \
  tests/examples/test_reward_objective_boundary_profiles.py \
  tests/examples/test_full_research_default_workflow.py -q
```

Expected: PASS, including the default three-candidate order.

- [ ] **Step 5: Commit**

```bash
git add examples/binance-multitimeframe tests/examples
git commit -m "config: align growth profiles with finite-horizon semantics"
```

---

### Task 4: Reconcile documentation and telemetry-era formatting

**Files:**
- Create: `docs/REWARD_OBJECTIVE.md`
- Modify: `docs/BINANCE.md`
- Modify: `docs/SINGLE_SYMBOL.md`
- Modify only if formatting requires it: `trade_rl/rl/tensorboard_logging.py`
- Modify only if formatting requires it: `tests/rl/test_tensorboard_logging.py`
- Modify: `tests/architecture/test_maintained_single_symbol_boundary.py`

**Interfaces:**
- Consumes: Tasks 2–3’s final runtime/profile behavior.
- Produces: one documented reward/boundary contract that does not regress PR #368 telemetry or PR #370 defaults.

- [ ] **Step 1: Add the durable reward-objective document**

Document exactly:

```text
reward_t = scale * log(net_equity_after / net_equity_before)
```

State that safety is managed through hard risk, seven independent cost channels, and walk-forward gates. Define both boundary modes and explain that gamma-one pure growth requires finite-horizon termination and time-to-go observation.

- [ ] **Step 2: Merge BINANCE and SINGLE_SYMBOL documentation**

Preserve the following PR #370 statement:

```text
No explicit training template -> target-weight three-profile workflow
training-full.json -> explicit legacy comparison only
```

Add PR #369’s boundary distinction without reverting that default.

- [ ] **Step 3: Preserve PR #368 telemetry semantics**

Do not replace the current telemetry implementation with the stale PR #369 branch copy. If Ruff formatting differs, apply only the formatter-equivalent line wrapping. Verify tests still cover:

```text
sampled action - smooth deterministic mean
terminated = hybrid_terminated or (done and not truncated)
legacy dual-flag rows normalized on read
```

- [ ] **Step 4: Run docs/architecture/telemetry tests**

```bash
pytest \
  tests/architecture/test_maintained_single_symbol_boundary.py \
  tests/rl/test_tensorboard_logging.py \
  tests/examples/test_full_research_default_workflow.py -q
ruff check docs trade_rl/rl/tensorboard_logging.py tests/rl/test_tensorboard_logging.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add docs tests/architecture trade_rl/rl/tensorboard_logging.py tests/rl/test_tensorboard_logging.py
git commit -m "docs: define maintained growth and boundary contract"
```

---

### Task 5: Exact-head verification and PR completion

**Files:**
- Modify: PR body only after verification evidence exists.

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces: one mergeable Ready PR based on current `main`, with no Universal-policy code and complete exact-head evidence.

- [ ] **Step 1: Review the complete diff**

Verify:

```bash
git diff --check main...HEAD
git status --short
git diff --stat main...HEAD
```

Reject unrelated files, generated artifacts, Docker state, secrets, or Universal-policy changes.

- [ ] **Step 2: Run focused and RL test suites**

```bash
pytest tests/rl tests/examples tests/workflows/test_training_terminal_contract.py -q
```

Expected: all pass.

- [ ] **Step 3: Run complete repository verification**

Run the repository’s canonical commands for:

```text
full pytest with branch coverage
Ruff
Ruff format check
MyPy
Import Linter
Dead-code report
frontend tests/typecheck/build/layout checks
critical coverage
Windows compatibility
Ubuntu compatibility
Training image and non-root probe
PostgreSQL Catalog integration
structured serving/recovery smoke
```

Every required job must pass on the same final head.

- [ ] **Step 4: Self-review semantic invariants**

Confirm from code and tests:

```text
external_truncation -> terminated=false, truncated=true, SB3 bootstraps
finite_horizon_termination -> terminated=true, truncated=false, no bootstrap
finite-horizon boundary does not set hybrid_terminated
finite-horizon boundary does not trigger terminal-equity shaping
default external environment digest is unchanged
observation digest does not change solely because boundary mode changes
PR #368 telemetry remains present
PR #370 default candidate order remains present
```

- [ ] **Step 5: Update the PR and mark Ready**

Record:

```text
What
Why
Architecture
Compatibility
RED evidence
GREEN evidence
exact final head
all CI results
active-run isolation
remaining risks
Production NO-GO
```

Change Draft to Ready only after exact-head CI and self-review pass.

- [ ] **Step 6: Do not merge without explicit user authorization**

Leave the PR open and mergeable. Report the PR number, final SHA, changed files, tests, unverified items, and residual risks.
