# Absolute Growth Reward v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the baseline-relative reward with cost-adjusted absolute log growth plus incremental rolling baseline-shortfall and drawdown penalties, while exposing reward state and terminating safely at the drawdown stop.

**Architecture:** Keep reward mathematics pure and immutable in `trade_rl/rl/rewards.py`. Let `ResidualMarketEnv` resolve time windows, snapshot before/after state, execute emergency liquidation, and publish diagnostics. Extend the observation contract with compact reward-state summaries and bind every parameter to the environment digest.

**Tech Stack:** Python 3.12, NumPy, Gymnasium, Stable-Baselines3, pytest, Ruff, mypy, Import Linter.

## Global Constraints

- Absolute cost-adjusted log wealth growth is the primary objective.
- Baseline underperformance is a one-sided 30-day rolling hinge with 1.5% tolerance.
- Drawdown penalty has a 5% free zone and slopes 1/3/8 at 10% and 15%.
- Penalties apply only to newly worsening state; no recovery or outperformance bonus.
- Drawdown 20% triggers actual emergency liquidation before termination.
- No fixed terminal reward.
- Reward state must be observable and included in schema identity.
- All implementation changes are completed before the full test suite is run, per user instruction.

---

### Task 1: Pure reward model

**Files:**
- Modify: `trade_rl/rl/rewards.py`
- Create: `tests/rl/test_absolute_growth_reward.py`

**Interfaces:**
- Produces: `AbsoluteGrowthRewardConfig`, `RewardState`, `RewardBreakdown`, `build_reward_state(...)`, `absolute_growth_reward(...)`, `drawdown_severity(...)`.

- [ ] Replace the legacy relative reward with validated immutable configuration and state records.
- [ ] Implement base-bar rolling log-growth calculation, scaled tolerance warm-up, one-sided baseline shortfall, continuous piecewise drawdown severity, and positive-increment-only penalties.
- [ ] Add focused unit tests for all boundaries and invalid inputs.

### Task 2: Observation schema v3

**Files:**
- Modify: `trade_rl/rl/observations.py`
- Modify: `tests/rl/test_observation_v2.py`

**Interfaces:**
- Consumes: `RewardState` from Task 1.
- Produces: `OBSERVATION_SCHEMA="baseline_residual_observation_v3"`; `build_observation(..., reward_state=RewardState)`.

- [ ] Increase global observation width by five.
- [ ] Append policy rolling growth, baseline rolling growth, their gap, shortfall, and tolerance.
- [ ] Validate finite values and preserve the no-episode-progress contract.
- [ ] Update schema and layout assertions.

### Task 3: Environment integration and emergency stop

**Files:**
- Modify: `trade_rl/rl/environment.py`
- Modify: `tests/rl/test_environment_timing.py`
- Modify: `tests/rl/test_environment_identity.py`

**Interfaces:**
- Consumes: Task 1 reward functions and Task 2 observation state.
- Produces: resolved reward window properties, absolute-growth reward diagnostics, and `termination_reason`.

- [ ] Replace legacy downside/excess-drawdown configuration with the approved reward parameters.
- [ ] Resolve 720-hour and 168-hour windows to base bars and include them in environment identity.
- [ ] Snapshot reward state before execution and after all execution/liquidation economics.
- [ ] On 20% policy drawdown, liquidate the policy book at the current close, require a complete flatten, include liquidation return, and terminate as `drawdown_stop`.
- [ ] Keep time-limit truncation and sealed evaluation liquidation semantics unchanged.
- [ ] Publish decomposed reward and wealth diagnostics in `info`.
- [ ] Update zero-action tests to expect absolute growth while preserving exact shadow identity.
- [ ] Add digest-sensitivity and emergency-stop regression tests.

### Task 4: Documentation and repository verification

**Files:**
- Modify: `README.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/RESEARCH_STATUS.md`

- [ ] Document the new objective, rolling hinge, drawdown schedule, observation schema, and zero-action reward semantics.
- [ ] Run `uv run ruff check .`.
- [ ] Run `uv run ruff format --check --diff .`.
- [ ] Run `uv run mypy trade_rl`.
- [ ] Run `uv run lint-imports`.
- [ ] Run `uv run pytest --cov=trade_rl --cov-branch --cov-report=term-missing`.
- [ ] Run `uv run trade-rl --version`.
- [ ] Inspect the final diff and CI evidence before claiming completion.