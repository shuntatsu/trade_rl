# Realistic Environment Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the rebuilt residual environment to a regular-time, next-open, quantity-accounted, liquidity-aware and risk-constrained simulation with explicit observation and OOS identities.

**Architecture:** Extend the immutable `MarketDataset` contract first, then replace book/execution internals behind their existing module boundaries. Integrate the shared pre-trade risk contract into the residual environment, upgrade observations and reward termination, move public time configuration to hours, and finally make walk-forward stitching mode explicit. Legacy `mars_lite` code remains deleted.

**Tech Stack:** Python 3.12, NumPy, Gymnasium, Stable-Baselines3, pytest, Hypothesis, Ruff, mypy, Import Linter.

## Global Constraints

- Work directly on `main` as explicitly authorized.
- Do not restore `mars_lite` or direct-action policy code.
- Preserve `baseline_residual_v1` and exact zero-action baseline identity.
- Use TDD: commit failing behavior tests before production changes and verify the expected CI failure.
- All public time settings are hours; all internal execution remains exact integer bars.
- All simulation arrays are finite and shape-checked.
- Final verification is the complete CI command set from `.github/workflows/ci.yml`.

---

### Task 1: Add failing v2 market-contract tests

**Files:**
- Create: `tests/data/test_market_dataset_v2.py`
- Modify: `tests/rl/test_environment_timing.py`

**Interfaces:**
- Produces the required `MarketDataset` fields `open`, `high`, `low`, `volume`, `tradable`, and `feature_available`.
- Requires properties `bar_hours` and `bars_for_hours(hours)`.

- [ ] Add tests that reject irregular timestamps, invalid OHLC, mismatched periods-per-year and invalid mask shapes.
- [ ] Update the shared environment fixture to construct complete OHLCV and availability arrays.
- [ ] Run CI and confirm failure because the new constructor/properties do not exist.

### Task 2: Implement the v2 market contract

**Files:**
- Modify: `trade_rl/data/market.py`
- Modify: `trade_rl/domain/datasets.py`

**Interfaces:**
- `MarketDataset.bar_hours -> float`
- `MarketDataset.bars_for_hours(hours: float) -> int`
- Timestamps are exact regular bar-close times.

- [ ] Implement immutable OHLCV, tradability and feature-availability arrays.
- [ ] Validate exact cadence and periods-per-year compatibility.
- [ ] Add dataset manifest schema v2 and bar-duration metadata without importing NumPy into `domain`.
- [ ] Run data and existing tests until green.

### Task 3: Add failing self-financing accounting tests

**Files:**
- Create: `tests/simulation/test_accounting_v2.py`
- Create: `tests/simulation/test_execution_v2.py`

**Interfaces:**
- `BookState.zero(n_symbols, initial_capital, initial_prices)`
- Derived `weights` and `portfolio_value`.
- `MarketExecutor.execute_interval(...)` reports requested, filled and unfilled turnover.

- [ ] Test quantity/cash reconciliation and drift from marks.
- [ ] Test decision at close `t` fills at open `t+1`.
- [ ] Test participation-limited partial fill and non-tradable no-fill behavior.
- [ ] Test fill count and rebalance-event count separately.
- [ ] Run CI and confirm expected failures.

### Task 4: Implement quantity accounting and liquidity-aware next-open execution

**Files:**
- Replace internals: `trade_rl/simulation/accounting.py`
- Replace internals: `trade_rl/simulation/execution.py`
- Modify exports: `trade_rl/simulation/__init__.py`

**Interfaces:**
- `BookState.execute(fill_prices, target_quantities, cost_amount, turnover)` updates cash and quantities.
- `BookState.mark_to_market(mark_prices, funding_amount)` records one base-bar return.
- `ExecutionCostConfig` adds `max_participation_rate`, `slippage_std`, `tail_slippage_probability`, `tail_slippage_multiplier`, and `random_seed`.

- [ ] Implement signed quantities, cash, marks and derived weights/equity.
- [ ] Split each base bar into close-to-open gap, next-open fill, and open-to-close mark.
- [ ] Compute market notional from open times volume and cap fills by participation.
- [ ] Apply symbol-level fee, spread, nonlinear impact and optional seeded slippage.
- [ ] Preserve exact zero-cost deterministic behavior when all cost parameters are zero.
- [ ] Run simulation tests and full tests.

### Task 5: Add failing risk/observation/reward tests

**Files:**
- Modify: `tests/rl/test_environment_timing.py`
- Create: `tests/rl/test_observation_v2.py`
- Create: `tests/rl/test_reward_v2.py`

**Interfaces:**
- `ResidualMarketEnv` accepts `PreTradeRisk`.
- Observation contains hybrid and shadow state plus risk scales.
- Reward distinguishes hybrid failure from shadow failure.

- [ ] Test constrained zero action remains exactly equal to shadow.
- [ ] Test a 40% symbol cap and 10–20% drawdown deleveraging are applied before execution.
- [ ] Test shadow weights, relative NAV, both drawdowns and risk scales are observable.
- [ ] Test hybrid termination is negative while shadow-only termination is not.
- [ ] Run CI and confirm expected failures.

### Task 6: Integrate risk and observation schema v2

**Files:**
- Modify: `trade_rl/risk/pretrade.py`
- Modify: `trade_rl/rl/observations.py`
- Modify: `trade_rl/rl/rewards.py`
- Modify: `trade_rl/rl/environment.py`

**Interfaces:**
- `PreTradeRisk.risk_scale(drawdown: float) -> float`
- `build_observation(..., hybrid, shadow, hybrid_risk_scale, shadow_risk_scale)`
- `relative_interval_reward(..., hybrid_terminated, shadow_terminated, downside_penalty, excess_drawdown_penalty)`

- [ ] Change pre-trade defaults to gross 1.0, absolute weight 0.40, turnover 1.0, drawdown 0.10–0.20.
- [ ] Apply tradability mask and risk constraints to hybrid and shadow targets.
- [ ] Upgrade the stable observation layout with masks and both books.
- [ ] Return valid terminal observations and separate termination causes.
- [ ] Add optional downside and excess-drawdown reward penalties defaulting to zero.
- [ ] Run RL, risk and full tests.

### Task 7: Add failing time-normalization and liquidation tests

**Files:**
- Create: `tests/strategies/test_trend_time_config.py`
- Modify: `tests/rl/test_environment_timing.py`
- Modify: `tests/rl/test_training.py`
- Modify: `tests/cli/test_cli.py`

**Interfaces:**
- `TrendConfig(fast_hours, base_hours, slow_hours)`.
- `ResidualMarketEnvConfig(episode_hours, decision_hours, liquidate_on_end)`.
- `gamma_from_half_life(decision_hours, half_life_hours)`.

- [ ] Test equal real-time lookbacks across hourly and four-hour datasets.
- [ ] Test environment duration and decision cadence are resolved from hours.
- [ ] Test final liquidation charges normal costs.
- [ ] Test half-life conversion and CLI output.
- [ ] Run CI and confirm expected failures.

### Task 8: Implement time-normalized strategy, environment and training

**Files:**
- Modify: `trade_rl/strategies/trend.py`
- Modify: `trade_rl/rl/environment.py`
- Modify: `trade_rl/rl/training.py`
- Modify: `trade_rl/cli/app.py`
- Modify: `README.md`

**Interfaces:**
- `gamma_from_half_life(decision_hours: float, half_life_hours: float) -> float`.
- CLI `trade-rl train config` accepts either explicit `--gamma` or both `--decision-hours` and `--discount-half-life-hours`.

- [ ] Resolve trend lookbacks through `MarketDataset.bars_for_hours`.
- [ ] Resolve episode and decision hours at environment construction.
- [ ] Add close-of-episode liquidation using the execution cost model.
- [ ] Implement half-life discount conversion and validation.
- [ ] Update documentation examples away from gamma 0.5.
- [ ] Run strategy, environment, training, CLI and full tests.

### Task 9: Add failing walk-forward identity tests

**Files:**
- Modify: `tests/evaluation/test_stitching.py`
- Modify: `tests/workflows/test_walk_forward.py`

**Interfaces:**
- `StitchMode.INDEPENDENT_FOLDS`
- `StitchMode.CONTINUOUS_ACCOUNT`
- Optional opening/closing state digests on `FoldOOSResult`.

- [ ] Test independent folds record gaps and permit resets.
- [ ] Test continuous mode rejects gaps and broken state-digest chains.
- [ ] Test result metrics retain stitch mode.
- [ ] Run CI and confirm expected failures.

### Task 10: Implement explicit independent/continuous OOS modes

**Files:**
- Modify: `trade_rl/evaluation/walk_forward/stitching.py`
- Modify: `trade_rl/workflows/walk_forward.py`
- Modify: `trade_rl/evaluation/walk_forward/__init__.py`

**Interfaces:**
- `stitch_oos(results, *, mode=StitchMode.INDEPENDENT_FOLDS)`.
- Continuous mode requires contiguous ranges and matching state digest chains.

- [ ] Add enum and provenance fields.
- [ ] Preserve independent-fold default while making the identity explicit.
- [ ] Enforce continuous-account invariants.
- [ ] Run evaluation, workflow and full tests.

### Task 11: Final architecture, formatting and documentation verification

**Files:**
- Modify as required: `docs/ARCHITECTURE.md`, `docs/RESEARCH_STATUS.md`
- Modify as required: `.github/workflows/ci.yml`

**Interfaces:**
- CI remains fail-fast and covers all maintained `trade_rl` modules.

- [ ] Document market timing, masks, quantity accounting, partial fills, risk parity and OOS modes.
- [ ] Run `uv run ruff check .`.
- [ ] Run `uv run ruff format --check --diff .`.
- [ ] Run `uv run mypy trade_rl`.
- [ ] Run `uv run lint-imports`.
- [ ] Run `uv run pytest --cov=trade_rl --cov-branch --cov-report=term-missing`.
- [ ] Run `uv run trade-rl --version`.
- [ ] Inspect the final main diff and CI status before declaring completion.
