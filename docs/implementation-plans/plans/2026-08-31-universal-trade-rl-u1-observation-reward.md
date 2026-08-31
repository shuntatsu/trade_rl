# Universal Trade RL U1 Observation / Action / Reward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a one-symbol Universal Trade RL U1 environment with causal strategy-prior-free observations, a scalar fixed-semantic target-exposure action, U0 Train-only equal-symbol market normalization from verified published artifacts, and reward exactly reconciled to realized after-cost wealth.

**Architecture:** `ResidualMarketEnv` remains the single Risk / Execution / Accounting authority. U1 adds a read-only runtime snapshot, a U1 market normalizer, a U1 observation builder, and a Gym wrapper that validates fixed V1 runtime semantics and delegates every economic transition to the maintained environment exactly once. `normalizer.json` and `u1_contract.json` bind the frozen U0 generation and materialize atomically.

**Tech Stack:** Python 3.12, NumPy, Gymnasium, dataclasses, existing `MarketDataset`, published market artifacts, `SequenceObservationBuilder`, `ResidualMarketEnv`, PreTradeRisk, execution/accounting, canonical SHA-256 digests, pytest, Hypothesis, Ruff, MyPy, Import Linter.

**Spec:** `docs/implementation-plans/specs/2026-08-31-universal-trade-rl-u1-observation-reward-design.md`

**Pinned design base:** `b3a4cf0fd98f459ceb2262a4a759af83f9b1df3c` (PR #426 U0 head at plan finalization). Before production code is written, fetch PR #426 again. If its intended head differs, synchronize this branch and amend the pinned comparator first.

## Global Constraints

- Production status is always `NO-GO` in U1.
- No RL training, Admission opening, profitability claim, or Production promotion.
- U1 wrapper requires exactly one symbol.
- Existing U3-U6 and Causal Alpha V9/V10/V11 economics remain unchanged.
- `ResidualMarketEnv` is the only Risk / Execution / Accounting engine.
- U1 V1 requires `ActionMode.TARGET_WEIGHT`, target count 1, `signal_delay_decisions == 1`, structured sequence observation, exact sequence windows, external truncation, `finite_horizon_observation == False`, `liquidate_on_end == False`, and sampled `initial_state_modes == ("cash",)`.
- Policy tensors contain no symbol/dataset IDs, raw absolute OHLC, raw nominal volume/cash/quantity, `TrendTargets`, alpha output, factor priors, shadow/baseline state, ownership latch, or remaining-horizon fraction.
- Windows are exactly `15m×96`, `1h×168`, `4h×120`, `1d×60`.
- Market FeatureKinds are the frozen spec allowlist; cross-asset/BTC-reference kinds are rejected.
- Every sequence source row is causal.
- Missing value, availability, and staleness are distinct.
- `policy_requested_weight`, `pending_target_weight` + `pending_target_active`, `risk_projected_weight`, and `current_weight` are separate meanings.
- Signal-delay pending target is not pending-order lifecycle.
- Existing `ObservationExecutionState.requested_weights` remains post-risk target; U1 reads it as `risk_projected_weight` without changing legacy semantics.
- Existing `ObservationExecutionState.execution_cost` is already `cost_by_symbol / initial_capital`; U1 exposes it as `execution_cost_rate` unchanged.
- U0 normalization authorization runs before artifact inspection/data load.
- Train source identity is proven by inspecting published market artifact manifests; caller-supplied digest strings are not trusted.
- Market normalizer fits only unique available feature events at or before the knowledge cutoff, then aggregates first/second moments with equal symbol weight.
- Policy/endogenous state uses deterministic versioned transforms and is not statistically fitted.
- Reward is exactly `100 * log(W_after / W_before)` over realized hybrid wealth, with no extra shaping/cost penalty.
- Non-positive/non-finite wealth fails closed in U1 reward.
- Sample end is truncation with no forced liquidation/terminal bonus.
- No merge to `main` without explicit user permission.

---

## File Map

**Create production modules:**

- `trade_rl/rl/universal_trade_contract.py` — frozen schemas, FeatureKind allowlist, policy contract/digest.
- `trade_rl/rl/universal_trade_action.py` — strict scalar action parser.
- `trade_rl/rl/universal_trade_runtime.py` — immutable named runtime snapshot.
- `trade_rl/rl/universal_trade_normalization.py` — published-source validation, unique-event/equal-symbol market statistics, codec.
- `trade_rl/rl/universal_trade_observation.py` — U1 Dict observation and deterministic state transforms.
- `trade_rl/rl/universal_trade_reward.py` — pure wealth reward and reconciliation oracle.
- `trade_rl/rl/universal_trade_environment.py` — U1 Gym wrapper.
- `trade_rl/workflows/universal_trade_rl_u1_contract.py` — U0-bound U1 contract artifact.
- `trade_rl/workflows/universal_trade_rl_u1_runner.py` — atomic `normalizer.json` + `u1_contract.json` publication.

**Modify production/docs:**

- `trade_rl/rl/environment.py` — add one read-only `universal_trade_runtime_snapshot()` accessor; no step/economic semantic change.
- `docs/UNIVERSAL_TRADE_RL.md` — maintained U1 contract and U2 gate.

**Create test support/tests:**

- `tests/rl/universal_trade_test_support.py`
- `tests/rl/test_universal_trade_contract.py`
- `tests/rl/test_universal_trade_action.py`
- `tests/rl/test_universal_trade_runtime.py`
- `tests/rl/test_universal_trade_normalization.py`
- `tests/rl/test_universal_trade_observation.py`
- `tests/rl/test_universal_trade_reward.py`
- `tests/rl/test_universal_trade_environment.py`
- `tests/rl/test_universal_trade_falsification.py`
- `tests/workflows/test_universal_trade_rl_u1_contract.py`
- `tests/workflows/test_universal_trade_rl_u1_runner.py`

---

### Task 1: Freeze policy contract and deterministic test fixtures

**Files:**
- Create: `trade_rl/rl/universal_trade_contract.py`
- Create: `tests/rl/universal_trade_test_support.py`
- Create: `tests/rl/test_universal_trade_contract.py`

**Interfaces produced:**

- `UNIVERSAL_TRADE_OBSERVATION_SCHEMA = "universal_trade_observation_v1"`
- `UNIVERSAL_TRADE_ACTION_SCHEMA = "normalized_target_exposure_v1"`
- `UNIVERSAL_TRADE_REWARD_SCHEMA = "universal_net_log_growth_reward_v1"`
- `UNIVERSAL_TRADE_STATE_LAYOUT_SCHEMA = "universal_trade_policy_state_v1"`
- `UNIVERSAL_TRADE_SEQUENCE_WINDOWS = (("15m", 96), ("1h", 168), ("4h", 120), ("1d", 60))`
- `UNIVERSAL_TRADE_ALLOWED_FEATURE_KINDS: frozenset[FeatureKind]`
- `UniversalTradePolicyContract(feature_specs: tuple[FeatureSpec, ...], policy_weight_scale: float = 1.0, reward_scale: float = 100.0)` with `digest`.
- Test support `make_u1_feature_specs()`, `make_u1_market()`, `make_u1_base_env()`.

- [ ] **Step 1: Create deterministic single-symbol test support**

Use this exact fixture shape in `tests/rl/universal_trade_test_support.py` so later tasks share one oracle:

```python
from __future__ import annotations

from dataclasses import replace

import numpy as np

from trade_rl.data.contracts import FeatureKind, FeatureSpec
from trade_rl.data.market import MarketDataset
from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig
from trade_rl.rl.actions import ActionMode, ActionSpec, ActionValidationMode
from trade_rl.rl.environment import ResidualMarketEnv
from trade_rl.rl.environment_config import EpisodeBoundaryMode, ResidualMarketEnvConfig
from trade_rl.rl.rewards import RewardConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.trend import TrendConfig, TrendStrategy


def make_u1_feature_specs() -> tuple[FeatureSpec, ...]:
    return (
        FeatureSpec(name="15m__ret", kind=FeatureKind.LOG_RETURN, lookback=1),
        FeatureSpec(name="1h__ret", kind=FeatureKind.LOG_RETURN, lookback=1, timeframe="1h"),
        FeatureSpec(name="4h__ret", kind=FeatureKind.LOG_RETURN, lookback=1, timeframe="4h"),
        FeatureSpec(name="1d__ret", kind=FeatureKind.LOG_RETURN, lookback=1, timeframe="1d"),
    )


def make_u1_market(
    *,
    symbol: str = "BTCUSDT",
    n_bars: int = 6000,
    price_scale: float = 1.0,
    feature_level: float = 0.0,
    volume: float = 1_000_000.0,
) -> MarketDataset:
    periods = np.asarray([1, 4, 16, 96], dtype=np.int64)
    rows = np.arange(n_bars, dtype=np.int64)
    features = np.empty((n_bars, 1, 4), dtype=np.float32)
    staleness_hours = np.empty((n_bars, 1, 4), dtype=np.float64)
    for column, period in enumerate(periods):
        source = (rows // period) * period
        features[:, 0, column] = feature_level + source.astype(np.float32) * 1e-4
        staleness_hours[:, 0, column] = (rows - source) * 0.25
    normalized_staleness = np.minimum(staleness_hours / 24.0, 1.0)
    timestamps = np.datetime64("2025-01-01T00:00:00", "ns") + rows * np.timedelta64(15, "m")
    close = price_scale * (100.0 + rows.astype(np.float64) * 1e-4)
    close_2d = close[:, None]
    dataset = MarketDataset(
        dataset_id="0" * 64,
        symbols=(symbol,),
        timestamps=timestamps,
        features=features,
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=close_2d.copy(),
        high=close_2d * 1.001,
        low=close_2d * 0.999,
        close=close_2d,
        volume=np.full((n_bars, 1), volume, dtype=np.float64),
        funding_rate=np.zeros((n_bars, 1), dtype=np.float64),
        tradable=np.ones((n_bars, 1), dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, 4), dtype=np.bool_),
        feature_names=tuple(spec.name for spec in make_u1_feature_specs()),
        global_feature_names=("market",),
        periods_per_year=35_040,
        feature_staleness_hours=staleness_hours,
        feature_staleness=normalized_staleness,
        mark_price=close_2d,
        index_price=close_2d,
    )
    return dataset.with_content_identity({"fixture": "universal_trade_u1_test_v1"})


def pure_growth_reward_config() -> RewardConfig:
    return RewardConfig(
        scale=100.0,
        absolute_growth_weight=1.0,
        excess_growth_weight=0.0,
        incremental_drawdown_weight=0.0,
        baseline_underperformance_weight=0.0,
        projection_penalty_weight=0.0,
        terminal_equity_weight=0.0,
        margin_deficit_weight=0.0,
    )


def make_u1_base_env(
    *,
    dataset: MarketDataset | None = None,
    max_abs_weight: float = 1.0,
    execution_cost: ExecutionCostConfig | None = None,
) -> ResidualMarketEnv:
    market = make_u1_market() if dataset is None else dataset
    return ResidualMarketEnv(
        market,
        trend_strategy=TrendStrategy(TrendConfig(fast_lookback=2, base_lookback=4, slow_lookback=8)),
        action_spec=ActionSpec(
            mode=ActionMode.TARGET_WEIGHT,
            alpha_enabled=False,
            risk_tilt_enabled=False,
            target_weight_count=1,
            validation_mode=ActionValidationMode.STRICT,
        ),
        pre_trade_risk=PreTradeRisk(
            PreTradeRiskConfig(
                max_gross=max_abs_weight,
                max_abs_weight=max_abs_weight,
                max_turnover=None,
            )
        ),
        config=ResidualMarketEnvConfig(
            initial_capital=100_000.0,
            episode_bars=64,
            decision_every=1,
            signal_delay_decisions=1,
            reward_config=pure_growth_reward_config(),
            episode_boundary_mode=EpisodeBoundaryMode.EXTERNAL_TRUNCATION,
            finite_horizon_observation=False,
            structured_sequence_observation=True,
            sequence_windows=(("15m", 96), ("1h", 168), ("4h", 120), ("1d", 60)),
            liquidate_on_end=False,
            initial_state_modes=("cash",),
            execution_cost=ExecutionCostConfig.zero() if execution_cost is None else execution_cost,
        ),
    )
```

Run only this helper import after writing it:

```bash
uv run python -c "from tests.rl.universal_trade_test_support import make_u1_market; assert make_u1_market().n_symbols == 1"
```

Expected: PASS.

- [ ] **Step 2: Write contract RED tests**

Create `tests/rl/test_universal_trade_contract.py` with:

```python
from dataclasses import replace

import pytest

from trade_rl.data.contracts import FeatureKind, FeatureSpec
from trade_rl.rl.universal_trade_contract import (
    UNIVERSAL_TRADE_SEQUENCE_WINDOWS,
    UniversalTradePolicyContract,
)
from tests.rl.universal_trade_test_support import make_u1_feature_specs


def test_u1_contract_freezes_windows_and_digest() -> None:
    contract = UniversalTradePolicyContract(feature_specs=make_u1_feature_specs())
    assert UNIVERSAL_TRADE_SEQUENCE_WINDOWS == (
        ("15m", 96), ("1h", 168), ("4h", 120), ("1d", 60)
    )
    assert len(contract.digest) == 64


@pytest.mark.parametrize(
    "kind",
    (
        FeatureKind.RELATIVE_RETURN_TO_BTC,
        FeatureKind.ROLLING_CORRELATION_TO_BTC,
        FeatureKind.ROLLING_BETA_TO_BTC,
        FeatureKind.CROSS_SECTIONAL_MOMENTUM_RANK,
        FeatureKind.CROSS_ASSET_DISPERSION,
    ),
)
def test_u1_contract_rejects_cross_asset_kinds(kind: FeatureKind) -> None:
    with pytest.raises(ValueError, match="U1 feature"):
        UniversalTradePolicyContract(
            feature_specs=(FeatureSpec(name="forbidden", kind=kind),)
        )


def test_u1_contract_binds_feature_order() -> None:
    specs = make_u1_feature_specs()
    assert UniversalTradePolicyContract(feature_specs=specs).digest != UniversalTradePolicyContract(
        feature_specs=tuple(reversed(specs))
    ).digest


@pytest.mark.parametrize("scale", (0.0, -1.0, float("inf"), float("nan")))
def test_u1_contract_rejects_invalid_policy_scale(scale: float) -> None:
    with pytest.raises(ValueError, match="policy_weight_scale"):
        UniversalTradePolicyContract(
            feature_specs=make_u1_feature_specs(), policy_weight_scale=scale
        )
```

- [ ] **Step 3: Run contract tests and confirm RED**

```bash
uv run pytest tests/rl/test_universal_trade_contract.py -q
```

Expected: import failure because `trade_rl.rl.universal_trade_contract` does not exist.

- [ ] **Step 4: Implement minimal immutable contract**

Implement the exact allowlist from the spec and validate:

```python
UNIVERSAL_TRADE_ALLOWED_FEATURE_KINDS = frozenset(
    {
        FeatureKind.LOG_RETURN,
        FeatureKind.BODY_RETURN,
        FeatureKind.HIGH_LOW_RANGE,
        FeatureKind.GAP_RETURN,
        FeatureKind.REALIZED_VOLATILITY,
        FeatureKind.DOWNSIDE_VOLATILITY,
        FeatureKind.UPSIDE_VOLATILITY,
        FeatureKind.VOLATILITY_OF_VOLATILITY,
        FeatureKind.ATR_PCT,
        FeatureKind.ATR_CHANGE,
        FeatureKind.EMA_DISTANCE,
        FeatureKind.EMA_SLOPE,
        FeatureKind.LINEAR_REGRESSION_SLOPE,
        FeatureKind.TREND_R2,
        FeatureKind.VOLUME_ZSCORE,
        FeatureKind.VOLUME_LOG_CHANGE,
        FeatureKind.RELATIVE_VOLUME,
        FeatureKind.FUNDING_BPS,
        FeatureKind.FUNDING_CHANGE,
        FeatureKind.FUNDING_ZSCORE,
    }
)
```

`UniversalTradePolicyContract.digest_payload()` must include ordered `FeatureSpec.canonical_payload()`, all three schema strings, state-layout schema, exact windows, `policy_weight_scale`, `reward_scale`, `signal_delay_decisions=1`, external-truncation semantic, finite-horizon false, liquidate-on-end false, and cash-only reset semantic.

- [ ] **Step 5: Verify contract and helper**

```bash
uv run pytest tests/rl/test_universal_trade_contract.py -q
uv run ruff check trade_rl/rl/universal_trade_contract.py tests/rl/universal_trade_test_support.py tests/rl/test_universal_trade_contract.py
uv run mypy trade_rl/rl/universal_trade_contract.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add trade_rl/rl/universal_trade_contract.py tests/rl/universal_trade_test_support.py tests/rl/test_universal_trade_contract.py
git commit -m "feat: define Universal Trade RL U1 contract"
```

---

### Task 2: Implement strict scalar target-exposure action

**Files:**
- Create: `trade_rl/rl/universal_trade_action.py`
- Create: `tests/rl/test_universal_trade_action.py`

**Interfaces produced:**

- `NormalizedTargetExposureAction(normalized: float, policy_requested_weight: float)`
- `parse_normalized_target_exposure(value: np.ndarray, *, policy_weight_scale: float) -> NormalizedTargetExposureAction`

- [ ] **Step 1: Write action RED tests**

```python
import numpy as np
import pytest

from trade_rl.rl.universal_trade_action import parse_normalized_target_exposure


@pytest.mark.parametrize(
    ("raw", "scale", "expected"),
    ((-1.0, 1.0, -1.0), (-0.5, 1.0, -0.5), (0.0, 1.0, 0.0), (0.5, 0.4, 0.2), (1.0, 0.4, 0.4)),
)
def test_action_mapping_is_linear_and_static(raw: float, scale: float, expected: float) -> None:
    parsed = parse_normalized_target_exposure(
        np.asarray([raw], dtype=np.float32), policy_weight_scale=scale
    )
    assert parsed.normalized == pytest.approx(raw)
    assert parsed.policy_requested_weight == pytest.approx(expected)


@pytest.mark.parametrize(
    "value",
    (
        np.asarray([1.01]),
        np.asarray([-1.01]),
        np.asarray([np.nan]),
        np.asarray([np.inf]),
        np.asarray([0.0, 0.0]),
    ),
)
def test_action_rejects_invalid_policy_output(value: np.ndarray) -> None:
    with pytest.raises(ValueError):
        parse_normalized_target_exposure(value, policy_weight_scale=1.0)


@pytest.mark.parametrize("scale", (0.0, -0.1, 1.01, np.inf, np.nan))
def test_action_rejects_invalid_static_scale(scale: float) -> None:
    with pytest.raises(ValueError, match="policy_weight_scale"):
        parse_normalized_target_exposure(np.asarray([0.5]), policy_weight_scale=scale)
```

- [ ] **Step 2: Run and confirm RED**

```bash
uv run pytest tests/rl/test_universal_trade_action.py -q
```

Expected: module import failure.

- [ ] **Step 3: Implement strict parser**

Implementation must parse exactly one finite scalar, reject values outside `[-1,1]`, validate `0 < policy_weight_scale <= 1`, and return `normalized * policy_weight_scale`. Do not call risk projection and do not `np.clip` the action.

Core implementation body:

```python
vector = np.asarray(value, dtype=np.float64).reshape(-1)
if vector.shape != (1,) or not np.isfinite(vector).all():
    raise ValueError("Universal Trade RL action must be one finite scalar")
normalized = float(vector[0])
if not -1.0 <= normalized <= 1.0:
    raise ValueError("Universal Trade RL action must be within [-1, 1]")
if not np.isfinite(policy_weight_scale) or not 0.0 < policy_weight_scale <= 1.0:
    raise ValueError("policy_weight_scale must be within (0, 1]")
return NormalizedTargetExposureAction(
    normalized=normalized,
    policy_requested_weight=normalized * policy_weight_scale,
)
```

- [ ] **Step 4: Verify**

```bash
uv run pytest tests/rl/test_universal_trade_action.py -q
uv run ruff check trade_rl/rl/universal_trade_action.py tests/rl/test_universal_trade_action.py
uv run mypy trade_rl/rl/universal_trade_action.py
```

- [ ] **Step 5: Commit**

```bash
git add trade_rl/rl/universal_trade_action.py tests/rl/test_universal_trade_action.py
git commit -m "feat: add normalized target exposure action"
```

---

### Task 3: Expose audited runtime state without changing environment economics

**Files:**
- Create: `trade_rl/rl/universal_trade_runtime.py`
- Modify: `trade_rl/rl/environment.py`
- Modify: `tests/rl/universal_trade_test_support.py`
- Create: `tests/rl/test_universal_trade_runtime.py`

**Interfaces produced:**

`UniversalTradeRuntimeSnapshot` contains exactly the endogenous fields specified by the U1 spec, including the four exposure-stage fields and all pending-order fields. `ResidualMarketEnv.universal_trade_runtime_snapshot()` returns it read-only.

- [ ] **Step 1: Add a test helper for runtime snapshots**

After the production dataclass is declared in the failing branch, add this helper to `tests/rl/universal_trade_test_support.py` for later tests:

```python
from dataclasses import replace
from trade_rl.rl.universal_trade_runtime import UniversalTradeRuntimeSnapshot


def make_runtime_snapshot(**overrides: object) -> UniversalTradeRuntimeSnapshot:
    base = UniversalTradeRuntimeSnapshot(
        policy_requested_weight=0.0,
        pending_target_weight=0.0,
        pending_target_active=False,
        risk_projected_weight=0.0,
        current_weight=0.0,
        previous_action=0.0,
        fill_ratio=1.0,
        unfilled_turnover_ratio=0.0,
        participation_ratio=0.0,
        execution_cost_rate=0.0,
        position_age_hours=0.0,
        pending_notional_ratio=0.0,
        pending_order_type_code=0.0,
        pending_order_status_code=0.0,
        pending_order_age_hours=0.0,
        pending_order_eligible_delay_hours=0.0,
        pending_order_triggered=0.0,
        pending_order_expiry_distance_hours=0.0,
        asset_active=1.0,
        tradable=1.0,
        borrow_available=1.0,
        borrow_rate=0.0,
        mark_index_basis=0.0,
        current_drawdown=0.0,
        current_gross_exposure=0.0,
        current_net_exposure=0.0,
        cash_weight=1.0,
        risk_scale=1.0,
        margin_utilization=0.0,
    )
    return replace(base, **overrides)
```

- [ ] **Step 2: Write runtime RED tests using real signal delay/risk/liquidity**

```python
import numpy as np
import pytest

from trade_rl.simulation.execution import ExecutionCostConfig
from tests.rl.universal_trade_test_support import make_u1_base_env, make_u1_market


def test_runtime_separates_submitted_pending_risk_and_realized_weights() -> None:
    market = make_u1_market(volume=100.0)
    env = make_u1_base_env(
        dataset=market,
        max_abs_weight=0.35,
        execution_cost=ExecutionCostConfig(
            fee_rate=0.0,
            spread_rate=0.0,
            impact_rate=0.0,
            max_participation_rate=0.01,
        ),
    )
    env.reset(options={"start_idx": 5700, "initial_state_mode": "cash"})
    env.step(np.asarray([0.60], dtype=np.float32))
    env.step(np.asarray([0.80], dtype=np.float32))

    snapshot = env.universal_trade_runtime_snapshot()
    assert snapshot.policy_requested_weight == pytest.approx(0.80)
    assert snapshot.pending_target_active is True
    assert snapshot.pending_target_weight == pytest.approx(0.80)
    assert snapshot.risk_projected_weight == pytest.approx(0.35)
    assert abs(snapshot.current_weight) < abs(snapshot.risk_projected_weight)
    assert 0.0 <= snapshot.fill_ratio < 1.0


def test_runtime_distinguishes_pending_flat_from_no_pending_target() -> None:
    env = make_u1_base_env()
    env.reset(options={"start_idx": 5700, "initial_state_mode": "cash"})
    before = env.universal_trade_runtime_snapshot()
    assert before.pending_target_active is False
    env.step(np.asarray([0.0], dtype=np.float32))
    after = env.universal_trade_runtime_snapshot()
    assert after.pending_target_active is True
    assert after.pending_target_weight == pytest.approx(0.0)
```

- [ ] **Step 3: Run and confirm RED**

```bash
uv run pytest tests/rl/test_universal_trade_runtime.py -q
```

Expected: missing runtime module/accessor.

- [ ] **Step 4: Implement immutable snapshot and accessor**

Accessor rules:

```text
policy_requested_weight = float(self._previous_action[0])
pending_target_active   = self._pending_hybrid_target is not None
pending_target_weight   = 0.0 if None else float(self._pending_hybrid_target[0])
risk_projected_weight   = float(self._execution_state.requested_weights[0])
current_weight          = float(self.hybrid.weights[0])
execution_cost_rate     = float(self._execution_state.execution_cost[0])
position_age_hours      = float(self._execution_state.position_age[0] * self.dataset.bar_hours)
```

Use `_pending_order_observation_state()` for pending-order values. Convert pending-order bar ages/delays/distances to hours with `dataset.bar_hours`. Derive current drawdown/gross/net/cash/risk/margin from existing hybrid/Risk state. Require one symbol and target-weight mode. Return scalar copies; do not expose mutable arrays.

Do not add `_last_risk_projected_target`; the existing execution state is the audited oracle.

- [ ] **Step 5: Verify targeted + environment regressions**

```bash
uv run pytest tests/rl/test_universal_trade_runtime.py -q
uv run pytest tests/rl/test_environment_reduce_only_integration.py tests/learning/test_rollout_execution_lifecycle.py -q
uv run ruff check trade_rl/rl/environment.py trade_rl/rl/universal_trade_runtime.py tests/rl/universal_trade_test_support.py tests/rl/test_universal_trade_runtime.py
uv run mypy trade_rl/rl/environment.py trade_rl/rl/universal_trade_runtime.py
```

- [ ] **Step 6: Commit**

```bash
git add trade_rl/rl/environment.py trade_rl/rl/universal_trade_runtime.py tests/rl/universal_trade_test_support.py tests/rl/test_universal_trade_runtime.py
git commit -m "feat: expose Universal Trade RL runtime state"
```

---

### Task 4: Fit Train-only market normalizer from verified published artifacts

**Files:**
- Create: `trade_rl/rl/universal_trade_normalization.py`
- Create: `tests/rl/test_universal_trade_normalization.py`
- Reuse: U0 provenance/access/manifest tests.

**Interfaces produced:**

- `UniversalTradePublishedSource(symbol: str, artifact_root: Path)`
- `UniversalTradeChannelStatistics(timeframe: str, feature_names: tuple[str, ...], mean: np.ndarray, scale: np.ndarray, per_symbol_sample_counts: tuple[tuple[str, tuple[int, ...]], ...])`
- `UniversalTradeSequenceNormalizer` with `statistics_digest`, U0-bound `digest`, `provenance_digest`, `to_payload()`, `from_payload()`, and `transform(observation: dict[str, np.ndarray])`.
- `fit_universal_trade_sequence_normalizer(*, manifest, access, sources, contract, knowledge_cutoff_ns, clip=10.0, epsilon=1e-8) -> UniversalTradeSequenceNormalizer`.

- [ ] **Step 1: Write source-publication helpers inside the test module**

```python
from pathlib import Path

from trade_rl.data.artifact import publish_market_dataset_artifact
from trade_rl.domain.universal_trade_rl_universe import UniversalTradeRLUniverseConfig
from trade_rl.workflows.universal_trade_rl_universe_access import UniversalTradeRLAccessPhase, UniversalTradeRLUniverseAccess
from trade_rl.workflows.universal_trade_rl_universe_config import UniversalTradeRLSymbolSource
from trade_rl.workflows.universal_trade_rl_universe_manifest import build_universal_trade_rl_universe_manifest


def _source(symbol: str, artifact, dataset) -> UniversalTradeRLSymbolSource:
    timestamps = dataset.timestamps.astype("datetime64[ns]").astype("int64")
    return UniversalTradeRLSymbolSource(
        symbol=symbol,
        dataset_digest=artifact.artifact_digest,
        first_timestamp_ns=int(timestamps[0]),
        last_timestamp_ns=int(timestamps[-1]),
        row_count=dataset.n_bars,
    )


def _manifest(train_rows, *, admission_digest: str = "a" * 64):
    config = UniversalTradeRLUniverseConfig(
        train_symbols=("BTCUSDT", "ETHUSDT"),
        development_symbols=("LINKUSDT",),
        admission_symbols=("AVAXUSDT",),
    )
    sources = tuple(sorted(
        train_rows
        + [
            UniversalTradeRLSymbolSource("AVAXUSDT", admission_digest, 1, 2, 2),
            UniversalTradeRLSymbolSource("LINKUSDT", "d" * 64, 1, 2, 2),
        ],
        key=lambda item: item.symbol,
    ))
    return build_universal_trade_rl_universe_manifest(config=config, sources=sources)
```

Use keyword arguments if `UniversalTradeRLSymbolSource` rejects positional construction; do not change its production API.

- [ ] **Step 2: Write firewall-before-filesystem-access RED test**

```python
from pathlib import Path
import pytest

from trade_rl.rl.universal_trade_contract import UniversalTradePolicyContract
from trade_rl.rl.universal_trade_normalization import UniversalTradePublishedSource, fit_universal_trade_sequence_normalizer
from trade_rl.workflows.universal_trade_rl_universe_access import UniversalTradeRLAccessPhase, UniversalTradeRLUniverseAccess
from tests.rl.universal_trade_test_support import make_u1_feature_specs


def test_normalization_scope_fails_before_artifact_path_is_touched(manifest) -> None:
    access = UniversalTradeRLUniverseAccess.for_phase(
        manifest=manifest, phase=UniversalTradeRLAccessPhase.DEVELOPMENT
    )
    sources = (
        UniversalTradePublishedSource("BTCUSDT", Path("/definitely/missing/btc")),
        UniversalTradePublishedSource("ETHUSDT", Path("/definitely/missing/eth")),
    )
    with pytest.raises(PermissionError, match="normalization|Train"):
        fit_universal_trade_sequence_normalizer(
            manifest=manifest,
            access=access,
            sources=sources,
            contract=UniversalTradePolicyContract(feature_specs=make_u1_feature_specs()),
            knowledge_cutoff_ns=10_000,
        )
```

The test must fail with `PermissionError`, never `FileNotFoundError`.

- [ ] **Step 3: Write artifact-digest spoof RED test**

Publish a BTC artifact A, build the manifest from A, then pass artifact B for the same symbol:

```python
def test_normalizer_rejects_published_artifact_not_bound_to_manifest(tmp_path) -> None:
    btc_a = make_u1_market(symbol="BTCUSDT", feature_level=0.0)
    btc_b = make_u1_market(symbol="BTCUSDT", feature_level=5.0)
    eth = make_u1_market(symbol="ETHUSDT", feature_level=10.0)
    art_a = publish_market_dataset_artifact(tmp_path / "btc-a", btc_a)
    art_b = publish_market_dataset_artifact(tmp_path / "btc-b", btc_b)
    art_eth = publish_market_dataset_artifact(tmp_path / "eth", eth)
    manifest = _manifest([_source("BTCUSDT", art_a, btc_a), _source("ETHUSDT", art_eth, eth)])
    access = UniversalTradeRLUniverseAccess.for_phase(manifest=manifest, phase=UniversalTradeRLAccessPhase.TRAIN)
    with pytest.raises(ValueError, match="artifact|dataset"):
        fit_universal_trade_sequence_normalizer(
            manifest=manifest,
            access=access,
            sources=(
                UniversalTradePublishedSource("BTCUSDT", art_b.root),
                UniversalTradePublishedSource("ETHUSDT", art_eth.root),
            ),
            contract=UniversalTradePolicyContract(feature_specs=make_u1_feature_specs()),
            knowledge_cutoff_ns=int(btc_a.timestamps[-1].astype("datetime64[ns]").astype("int64")),
        )
```

- [ ] **Step 4: Write exact equal-symbol moment RED test**

Use unequal history lengths so row-concatenation cannot accidentally pass:

```python
def test_normalizer_uses_equal_symbol_moments_not_equal_rows(tmp_path) -> None:
    btc = make_u1_market(symbol="BTCUSDT", n_bars=5800, feature_level=0.0)
    eth = make_u1_market(symbol="ETHUSDT", n_bars=6000, feature_level=10.0)
    art_btc = publish_market_dataset_artifact(tmp_path / "btc", btc)
    art_eth = publish_market_dataset_artifact(tmp_path / "eth", eth)
    manifest = _manifest([_source("BTCUSDT", art_btc, btc), _source("ETHUSDT", art_eth, eth)])
    access = UniversalTradeRLUniverseAccess.for_phase(manifest=manifest, phase=UniversalTradeRLAccessPhase.TRAIN)
    normalizer = fit_universal_trade_sequence_normalizer(
        manifest=manifest,
        access=access,
        sources=(UniversalTradePublishedSource("BTCUSDT", art_btc.root), UniversalTradePublishedSource("ETHUSDT", art_eth.root)),
        contract=UniversalTradePolicyContract(feature_specs=make_u1_feature_specs()),
        knowledge_cutoff_ns=min(
            int(btc.timestamps[-1].astype("datetime64[ns]").astype("int64")),
            int(eth.timestamps[-1].astype("datetime64[ns]").astype("int64")),
        ),
    )
    stats = normalizer.statistics_for("15m")
    # The same deterministic slope exists in both symbols; ETH is exactly +10.
    # Therefore the equal-symbol aggregate mean is the average of each symbol's
    # independently computed mean, not the concatenated-row mean.
    btc_events = btc.features[:, 0, 0].astype(np.float64)
    eth_events = eth.features[: btc.n_bars, 0, 0].astype(np.float64)
    expected = 0.5 * (float(np.mean(btc_events)) + float(np.mean(eth_events)))
    assert stats.mean[0] == pytest.approx(expected)
```

The implementation test should compute second-moment expected scale with the same independent formula and assert `stats.scale[0]` exactly within floating tolerance.

- [ ] **Step 5: Write carried-event de-duplication RED test**

For the 4h column, fixture source event changes every 16 base rows. Assert sample count equals unique recovered source event timestamps, not available base rows:

```python
def test_normalizer_counts_each_carried_feature_event_once(tmp_path) -> None:
    btc = make_u1_market(symbol="BTCUSDT", n_bars=5800)
    eth = make_u1_market(symbol="ETHUSDT", n_bars=5800)
    art_btc = publish_market_dataset_artifact(tmp_path / "btc", btc)
    art_eth = publish_market_dataset_artifact(tmp_path / "eth", eth)
    manifest = _manifest([_source("BTCUSDT", art_btc, btc), _source("ETHUSDT", art_eth, eth)])
    normalizer = fit_universal_trade_sequence_normalizer(
        manifest=manifest,
        access=UniversalTradeRLUniverseAccess.for_phase(manifest=manifest, phase=UniversalTradeRLAccessPhase.TRAIN),
        sources=(UniversalTradePublishedSource("BTCUSDT", art_btc.root), UniversalTradePublishedSource("ETHUSDT", art_eth.root)),
        contract=UniversalTradePolicyContract(feature_specs=make_u1_feature_specs()),
        knowledge_cutoff_ns=int(btc.timestamps[-1].astype("datetime64[ns]").astype("int64")),
    )
    stats = normalizer.statistics_for("4h")
    expected_unique_events = (btc.n_bars - 1) // 16 + 1
    assert dict(stats.per_symbol_sample_counts)["BTCUSDT"][0] == expected_unique_events
    assert expected_unique_events < btc.n_bars
```

- [ ] **Step 6: Write statistics-vs-generation identity RED test**

```python
def test_admission_only_generation_change_preserves_statistics_not_artifact_digest(tmp_path) -> None:
    btc = make_u1_market(symbol="BTCUSDT")
    eth = make_u1_market(symbol="ETHUSDT")
    art_btc = publish_market_dataset_artifact(tmp_path / "btc", btc)
    art_eth = publish_market_dataset_artifact(tmp_path / "eth", eth)
    train_rows = [_source("BTCUSDT", art_btc, btc), _source("ETHUSDT", art_eth, eth)]
    manifest_a = _manifest(train_rows, admission_digest="a" * 64)
    manifest_b = _manifest(train_rows, admission_digest="f" * 64)
    contract = UniversalTradePolicyContract(feature_specs=make_u1_feature_specs())
    sources = (UniversalTradePublishedSource("BTCUSDT", art_btc.root), UniversalTradePublishedSource("ETHUSDT", art_eth.root))
    cutoff = int(btc.timestamps[-1].astype("datetime64[ns]").astype("int64"))
    normalizer_a = fit_universal_trade_sequence_normalizer(
        manifest=manifest_a,
        access=UniversalTradeRLUniverseAccess.for_phase(manifest=manifest_a, phase=UniversalTradeRLAccessPhase.TRAIN),
        sources=sources,
        contract=contract,
        knowledge_cutoff_ns=cutoff,
    )
    normalizer_b = fit_universal_trade_sequence_normalizer(
        manifest=manifest_b,
        access=UniversalTradeRLUniverseAccess.for_phase(manifest=manifest_b, phase=UniversalTradeRLAccessPhase.TRAIN),
        sources=sources,
        contract=contract,
        knowledge_cutoff_ns=cutoff,
    )
    assert normalizer_a.statistics_digest == normalizer_b.statistics_digest
    assert normalizer_a.digest != normalizer_b.digest
```

- [ ] **Step 7: Run normalization tests and confirm RED**

```bash
uv run pytest tests/rl/test_universal_trade_normalization.py -q
```

Expected: missing normalization module.

- [ ] **Step 8: Implement fit pipeline in exact order**

Implementation order is part of the contract:

```text
canonicalize/sort source symbols
-> access.require_normalization_scope(symbols)
-> build FEATURE_NORMALIZATION provenance
-> inspect published artifact manifest
-> compare artifact_digest with U0 manifest entry
-> load verified MarketDataset artifact
-> require one expected symbol and exact feature layout
-> select available rows <= knowledge_cutoff_ns
-> recover source_event_time_ns from feature_staleness_hours
-> deduplicate (feature, source_event_time_ns)
-> per-symbol mean and mean(square)
-> equal-symbol aggregate
-> scale floor
-> statistics_digest
-> U0/provenance-bound artifact digest
```

Source-event recovery must use:

```python
_NS_PER_HOUR = 3_600_000_000_000
source_event_time_ns = timestamp_ns - np.rint(age_hours * _NS_PER_HOUR).astype(np.int64)
```

Reject recovered event times after the bar timestamp or outside the published dataset range. Do not dedupe by feature value.

- [ ] **Step 9: Implement strict codec/transform and verify**

`to_payload()` / `from_payload()` use exact keys and verify content digests. `transform()` standardizes only `sequence_<tf>_values`, using the ordered feature names for that timeframe; it must return availability/staleness/policy state unchanged.

Run:

```bash
uv run pytest tests/rl/test_universal_trade_normalization.py tests/workflows/test_universal_trade_rl_data_provenance.py -q
uv run ruff check trade_rl/rl/universal_trade_normalization.py tests/rl/test_universal_trade_normalization.py
uv run mypy trade_rl/rl/universal_trade_normalization.py
```

Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add trade_rl/rl/universal_trade_normalization.py tests/rl/test_universal_trade_normalization.py
git commit -m "feat: add Train-only universal market normalizer"
```

---

### Task 5: Build the causal strategy-prior-free U1 observation

**Files:**
- Create: `trade_rl/rl/universal_trade_observation.py`
- Create: `tests/rl/test_universal_trade_observation.py`

**Interfaces produced:**

- `UNIVERSAL_TRADE_POLICY_STATE_NAMES: tuple[str, ...]`
- `UniversalTradeObservationBuilder(contract: UniversalTradePolicyContract)`
- `build(dataset: MarketDataset, *, index: int, runtime: UniversalTradeRuntimeSnapshot, normalizer: UniversalTradeSequenceNormalizer | None = None) -> dict[str, np.ndarray]`
- `observation_space(dataset: MarketDataset) -> gym.spaces.Dict`
- `schema_digest(dataset: MarketDataset) -> str`

- [ ] **Step 1: Write exact layout RED test**

```python
from trade_rl.rl.universal_trade_contract import UniversalTradePolicyContract
from trade_rl.rl.universal_trade_observation import UNIVERSAL_TRADE_POLICY_STATE_NAMES, UniversalTradeObservationBuilder
from tests.rl.universal_trade_test_support import make_runtime_snapshot, make_u1_feature_specs, make_u1_market


def test_observation_has_only_market_sequences_and_named_policy_state() -> None:
    dataset = make_u1_market()
    builder = UniversalTradeObservationBuilder(
        UniversalTradePolicyContract(feature_specs=make_u1_feature_specs())
    )
    obs = builder.build(dataset, index=5700, runtime=make_runtime_snapshot())
    assert set(obs) == {
        "sequence_15m_values", "sequence_15m_available", "sequence_15m_staleness",
        "sequence_1h_values", "sequence_1h_available", "sequence_1h_staleness",
        "sequence_4h_values", "sequence_4h_available", "sequence_4h_staleness",
        "sequence_1d_values", "sequence_1d_available", "sequence_1d_staleness",
        "policy_state",
    }
    forbidden = ("trend", "alpha", "shadow", "baseline", "remaining", "symbol", "dataset")
    assert not any(token in name for name in UNIVERSAL_TRADE_POLICY_STATE_NAMES for token in forbidden)
    assert builder.observation_space(dataset).contains(obs)
```

- [ ] **Step 2: Write deterministic transform RED test**

```python
def test_policy_state_uses_versioned_deterministic_transforms() -> None:
    dataset = make_u1_market()
    builder = UniversalTradeObservationBuilder(UniversalTradePolicyContract(feature_specs=make_u1_feature_specs()))
    runtime = make_runtime_snapshot(
        position_age_hours=24.0,
        pending_order_age_hours=48.0,
        pending_order_eligible_delay_hours=24.0,
        pending_order_expiry_distance_hours=72.0,
        mark_index_basis=0.01,
        borrow_rate=0.02,
    )
    obs = builder.build(dataset, index=5700, runtime=runtime)
    state = dict(zip(UNIVERSAL_TRADE_POLICY_STATE_NAMES, obs["policy_state"], strict=True))
    assert state["position_age_days"] == pytest.approx(np.log1p(1.0))
    assert state["pending_order_age_days"] == pytest.approx(np.log1p(2.0))
    assert state["mark_index_basis"] == pytest.approx(np.tanh(1.0))
    assert state["borrow_rate"] == pytest.approx(np.tanh(0.02))
```

- [ ] **Step 3: Write future-mutation RED test**

```python
from dataclasses import replace


def test_future_market_mutation_cannot_change_observation_at_t() -> None:
    dataset = make_u1_market()
    t = 5700
    mutated_features = dataset.features.copy()
    mutated_features[t + 1 :] = 999.0
    mutated_close = dataset.close.copy()
    mutated_close[t + 1 :] *= 50.0
    future_changed = replace(
        dataset,
        dataset_id="f" * 64,
        identity_payload_json=None,
        features=mutated_features,
        close=mutated_close,
        open=mutated_close,
        high=mutated_close * 1.001,
        low=mutated_close * 0.999,
        mark_price=mutated_close,
        index_price=mutated_close,
    )
    builder = UniversalTradeObservationBuilder(UniversalTradePolicyContract(feature_specs=make_u1_feature_specs()))
    runtime = make_runtime_snapshot()
    a = builder.build(dataset, index=t, runtime=runtime)
    b = builder.build(future_changed, index=t, runtime=runtime)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
```

- [ ] **Step 4: Write symbol/price-unit invariance RED tests**

```python
def test_symbol_text_is_not_a_policy_tensor() -> None:
    dataset = make_u1_market(symbol="BTCUSDT")
    renamed = replace(dataset, dataset_id="e" * 64, identity_payload_json=None, symbols=("FOOUSDT",))
    builder = UniversalTradeObservationBuilder(UniversalTradePolicyContract(feature_specs=make_u1_feature_specs()))
    runtime = make_runtime_snapshot()
    a = builder.build(dataset, index=5700, runtime=runtime)
    b = builder.build(renamed, index=5700, runtime=runtime)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])


def test_price_unit_scaling_does_not_enter_policy_tensor() -> None:
    a_dataset = make_u1_market(price_scale=1.0)
    b_dataset = make_u1_market(price_scale=1000.0)
    builder = UniversalTradeObservationBuilder(UniversalTradePolicyContract(feature_specs=make_u1_feature_specs()))
    runtime = make_runtime_snapshot(mark_index_basis=0.0)
    a = builder.build(a_dataset, index=5700, runtime=runtime)
    b = builder.build(b_dataset, index=5700, runtime=runtime)
    for key in a:
        np.testing.assert_allclose(a[key], b[key], rtol=0.0, atol=1e-7)
```

- [ ] **Step 5: Run and confirm RED**

```bash
uv run pytest tests/rl/test_universal_trade_observation.py -q
```

- [ ] **Step 6: Implement builder from named sources, never legacy vector offsets**

Call `SequenceObservationBuilder` directly. Never slice or parse `baseline_residual_observation_v5`. Policy-state field order is a module constant and includes transformed age field names (`position_age_days`, `pending_order_age_days`, `pending_order_eligible_delay_days`, `pending_order_expiry_distance_days`) plus the remaining dimensionless runtime fields.

If a normalizer is provided, build raw sequence components first and call its `transform()` once. Normalizer must not alter `policy_state`.

- [ ] **Step 7: Verify**

```bash
uv run pytest tests/rl/test_universal_trade_observation.py -q
uv run ruff check trade_rl/rl/universal_trade_observation.py tests/rl/test_universal_trade_observation.py
uv run mypy trade_rl/rl/universal_trade_observation.py
```

- [ ] **Step 8: Commit**

```bash
git add trade_rl/rl/universal_trade_observation.py tests/rl/test_universal_trade_observation.py
git commit -m "feat: add Universal Trade RL observation"
```

---

### Task 6: Implement pure after-cost wealth reward oracle

**Files:**
- Create: `trade_rl/rl/universal_trade_reward.py`
- Create: `tests/rl/test_universal_trade_reward.py`

**Interfaces produced:**

- `universal_net_log_growth_reward(*, before_value: float, after_value: float, scale: float = 100.0) -> float`
- `reconcile_universal_trade_reward(*, rewards: Sequence[float], initial_value: float, final_value: float, scale: float = 100.0, atol: float = 1e-12) -> None`

- [ ] **Step 1: Write reward RED tests**

```python
import math

import pytest

from trade_rl.rl.universal_trade_reward import reconcile_universal_trade_reward, universal_net_log_growth_reward


def test_reward_telescopes_exactly_to_final_wealth() -> None:
    values = (100.0, 101.0, 99.5, 103.25)
    rewards = tuple(
        universal_net_log_growth_reward(before_value=a, after_value=b)
        for a, b in zip(values, values[1:], strict=True)
    )
    assert sum(rewards) / 100.0 == pytest.approx(math.log(values[-1] / values[0]))
    reconcile_universal_trade_reward(
        rewards=rewards,
        initial_value=values[0],
        final_value=values[-1],
    )


@pytest.mark.parametrize("value", (0.0, -1.0, float("inf"), float("nan")))
def test_reward_rejects_invalid_wealth(value: float) -> None:
    with pytest.raises(ValueError):
        universal_net_log_growth_reward(before_value=value, after_value=100.0)
    with pytest.raises(ValueError):
        universal_net_log_growth_reward(before_value=100.0, after_value=value)
```

- [ ] **Step 2: Run and confirm RED**

```bash
uv run pytest tests/rl/test_universal_trade_reward.py -q
```

- [ ] **Step 3: Implement minimal formula/reconciliation**

After explicit finite/positive validation:

```python
reward = scale * math.log(after_value / before_value)
```

Reconciliation compares `sum(rewards)/scale` to `math.log(final_value/initial_value)` with `rel_tol=0.0` and caller-supplied absolute tolerance; mismatch raises `ValueError`.

- [ ] **Step 4: Verify and commit**

```bash
uv run pytest tests/rl/test_universal_trade_reward.py -q
uv run ruff check trade_rl/rl/universal_trade_reward.py tests/rl/test_universal_trade_reward.py
uv run mypy trade_rl/rl/universal_trade_reward.py
git add trade_rl/rl/universal_trade_reward.py tests/rl/test_universal_trade_reward.py
git commit -m "feat: add Universal Trade RL wealth reward"
```

---

### Task 7: Wrap maintained environment with U1 semantics

**Files:**
- Create: `trade_rl/rl/universal_trade_environment.py`
- Create: `tests/rl/test_universal_trade_environment.py`

**Interface produced:** `UniversalTradeMarketEnv(gym.Wrapper)`.

- [ ] **Step 1: Write constructor contract RED test**

```python
from dataclasses import replace

import pytest

from trade_rl.rl.environment_config import EpisodeBoundaryMode
from trade_rl.rl.universal_trade_contract import UniversalTradePolicyContract
from trade_rl.rl.universal_trade_environment import UniversalTradeMarketEnv
from tests.rl.universal_trade_test_support import make_u1_base_env, make_u1_feature_specs


def _wrap(env):
    return UniversalTradeMarketEnv(
        env,
        contract=UniversalTradePolicyContract(feature_specs=make_u1_feature_specs()),
    )


def test_valid_base_environment_is_accepted() -> None:
    wrapper = _wrap(make_u1_base_env())
    assert wrapper.action_space.shape == (1,)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("signal_delay_decisions", 0),
        ("finite_horizon_observation", True),
        ("liquidate_on_end", True),
        ("initial_state_modes", ("baseline",)),
        ("episode_boundary_mode", EpisodeBoundaryMode.FINITE_HORIZON_TERMINATION),
    ),
)
def test_wrapper_rejects_runtime_contract_drift(field: str, value: object) -> None:
    env = make_u1_base_env()
    object.__setattr__(env, "config", replace(env.config, **{field: value}))
    with pytest.raises(ValueError):
        _wrap(env)
```

For finite-horizon termination, create a valid paired config with `finite_horizon_observation=True` before assigning it so dataclass validation itself does not hide the wrapper oracle.

- [ ] **Step 2: Write reset/action-delay integration RED test**

```python
import numpy as np
import pytest


def test_wrapper_exposes_current_submission_and_executes_previous_pending_target() -> None:
    wrapper = _wrap(make_u1_base_env(max_abs_weight=0.4))
    obs, _ = wrapper.reset(options={"start_idx": 5700, "initial_state_mode": "cash"})
    assert wrapper.observation_space.contains(obs)
    wrapper.step(np.asarray([0.60], dtype=np.float32))
    obs, _, _, _, _ = wrapper.step(np.asarray([0.80], dtype=np.float32))
    state = wrapper.policy_state_dict(obs)
    assert state["policy_requested_weight"] == pytest.approx(0.80)
    assert state["pending_target_weight"] == pytest.approx(0.80)
    assert state["pending_target_active"] == pytest.approx(1.0)
    assert state["risk_projected_weight"] == pytest.approx(0.40)
```

- [ ] **Step 3: Write reward reconciliation RED test**

```python
import math


def test_wrapper_reward_is_realized_base_wealth_log_growth() -> None:
    wrapper = _wrap(make_u1_base_env())
    wrapper.reset(options={"start_idx": 5700, "initial_state_mode": "cash"})
    before = wrapper.unwrapped.hybrid.portfolio_value
    _, reward, _, _, _ = wrapper.step(np.asarray([0.5], dtype=np.float32))
    after = wrapper.unwrapped.hybrid.portfolio_value
    assert reward == pytest.approx(100.0 * math.log(after / before), abs=1e-10)
```

- [ ] **Step 4: Write truncation/cash-only RED tests**

```python
def test_wrapper_rejects_non_cash_reset_request() -> None:
    wrapper = _wrap(make_u1_base_env())
    with pytest.raises(ValueError, match="cash"):
        wrapper.reset(options={"start_idx": 5700, "initial_state_mode": "baseline"})


def test_time_limit_is_truncation_without_forced_liquidation() -> None:
    wrapper = _wrap(make_u1_base_env())
    wrapper.reset(options={"start_idx": 5700, "initial_state_mode": "cash"})
    terminated = truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, _ = wrapper.step(np.asarray([0.2], dtype=np.float32))
    assert terminated is False
    assert truncated is True
    assert wrapper.unwrapped.config.liquidate_on_end is False
```

- [ ] **Step 5: Run and confirm RED**

```bash
uv run pytest tests/rl/test_universal_trade_environment.py -q
```

- [ ] **Step 6: Implement wrapper**

Constructor validates every fixed U1 runtime property plus feature order/windows and pure base `RewardConfig.is_pure_net_log_growth()`.

`reset()` delegates once to base env, rejects requested sampled mode other than cash, ignores base observation value, obtains `universal_trade_runtime_snapshot()`, and builds U1 observation.

`step()`:

```text
strict action parse
-> before wealth
-> base_env.step([policy_requested_weight]) exactly once
-> after wealth
-> U1 reward calculation
-> require abs(base_reward - u1_reward) <= 1e-10
-> runtime snapshot
-> U1 observation build/normalize
-> return U1 observation/reward with base terminated/truncated/info
```

Never call risk/execution/accounting directly.

- [ ] **Step 7: Verify targeted + compatibility**

```bash
uv run pytest tests/rl/test_universal_trade_environment.py tests/rl/test_universal_trade_runtime.py -q
uv run pytest tests/rl/test_environment_reduce_only_integration.py tests/learning/test_rollout_execution_lifecycle.py -q
uv run ruff check trade_rl/rl/universal_trade_environment.py tests/rl/test_universal_trade_environment.py
uv run mypy trade_rl/rl/universal_trade_environment.py
```

- [ ] **Step 8: Commit**

```bash
git add trade_rl/rl/universal_trade_environment.py tests/rl/test_universal_trade_environment.py
git commit -m "feat: add Universal Trade RL environment wrapper"
```

---

### Task 8: Bind U1 contract identity to frozen U0 and runtime economics

**Files:**
- Create: `trade_rl/workflows/universal_trade_rl_u1_contract.py`
- Create: `tests/workflows/test_universal_trade_rl_u1_contract.py`

**Interfaces produced:**

- `U1_CONTRACT_SCHEMA = "universal_trade_rl_u1_contract_v1"`
- `UniversalTradeRLU1Contract` with universe/U0/policy/observation/normalizer/state/runtime/execution/risk digests, `production_status="NO-GO"`, `digest`, strict `to_payload()` / `from_payload()`.
- `build_universal_trade_rl_u1_contract(*, manifest, u0_identity, policy_contract, normalizer, base_env, observation_schema_digest, state_layout_digest) -> UniversalTradeRLU1Contract`.

- [ ] **Step 1: Write identity drift RED test**

```python
from dataclasses import replace


def test_u1_contract_digest_changes_for_every_runtime_semantic(u1_contract) -> None:
    fields = (
        "policy_contract_digest",
        "normalizer_digest",
        "observation_schema_digest",
        "state_layout_digest",
        "runtime_config_digest",
        "execution_policy_digest",
        "pretrade_risk_digest",
        "portfolio_risk_digest",
    )
    for field in fields:
        changed = replace(u1_contract, **{field: "f" * 64}, digest="")
        assert changed.digest != u1_contract.digest
```

- [ ] **Step 2: Write U0/provenance mismatch RED tests**

```python
def test_u1_contract_rejects_normalizer_from_another_universe(manifest_a, manifest_b, normalizer_a, u0_identity_b, base_env) -> None:
    with pytest.raises(ValueError, match="universe|normalizer"):
        build_universal_trade_rl_u1_contract(
            manifest=manifest_b,
            u0_identity=u0_identity_b,
            policy_contract=UniversalTradePolicyContract(feature_specs=make_u1_feature_specs()),
            normalizer=normalizer_a,
            base_env=base_env,
            observation_schema_digest="1" * 64,
            state_layout_digest="2" * 64,
        )
```

The test fixture creates `normalizer_a` with `FEATURE_NORMALIZATION` provenance from `manifest_a`; do not mutate provenance to make the test pass.

- [ ] **Step 3: Write payload tamper RED test**

```python
def test_u1_contract_payload_rejects_tampered_digest(u1_contract) -> None:
    payload = u1_contract.to_payload()
    payload["runtime_config_digest"] = "f" * 64
    with pytest.raises(ValueError, match="digest"):
        UniversalTradeRLU1Contract.from_payload(payload)
```

- [ ] **Step 4: Run and confirm RED**

```bash
uv run pytest tests/workflows/test_universal_trade_rl_u1_contract.py -q
```

- [ ] **Step 5: Implement strict builder/codec**

Builder validates:

```text
u0_identity.stage == UNIVERSE_MATERIALIZATION
u0_identity.universe_manifest_digest == manifest.digest
normalizer.universe_manifest_digest == manifest.digest
normalizer provenance purpose == FEATURE_NORMALIZATION
normalizer provenance passes require_universal_trade_rl_train_only_provenance
base env satisfies same fixed U1 runtime contract as wrapper
```

Compute `runtime_config_digest` from exact decision/episode hours, signal delay, boundary mode, finite horizon flag, liquidate-on-end, sampled initial-state modes, and sequence windows. Compute risk config digests from canonical `asdict()` payloads; never include symbol text. Require `production_status == "NO-GO"`.

- [ ] **Step 6: Verify and commit**

```bash
uv run pytest tests/workflows/test_universal_trade_rl_u1_contract.py tests/workflows/test_universal_trade_rl_run_identity.py tests/workflows/test_universal_trade_rl_data_provenance.py -q
uv run ruff check trade_rl/workflows/universal_trade_rl_u1_contract.py tests/workflows/test_universal_trade_rl_u1_contract.py
uv run mypy trade_rl/workflows/universal_trade_rl_u1_contract.py
git add trade_rl/workflows/universal_trade_rl_u1_contract.py tests/workflows/test_universal_trade_rl_u1_contract.py
git commit -m "feat: bind Universal Trade RL U1 contract identity"
```

---

### Task 9: Materialize U1 artifacts atomically and canonically

**Files:**
- Create: `trade_rl/workflows/universal_trade_rl_u1_runner.py`
- Create: `tests/workflows/test_universal_trade_rl_u1_runner.py`

**Interface produced:**

`materialize_universal_trade_rl_u1(*, contract: UniversalTradeRLU1Contract, normalizer: UniversalTradeSequenceNormalizer, output_root: str | Path) -> tuple[UniversalTradeRLU1Contract, UniversalTradeSequenceNormalizer]`.

- [ ] **Step 1: Write exact-output/idempotency RED test**

```python
import json


def test_u1_materialization_is_exact_and_idempotent(tmp_path, u1_contract, normalizer) -> None:
    output = tmp_path / "u1"
    first = materialize_universal_trade_rl_u1(
        contract=u1_contract, normalizer=normalizer, output_root=output
    )
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    second = materialize_universal_trade_rl_u1(
        contract=u1_contract, normalizer=normalizer, output_root=output
    )
    after = {path.name: path.read_bytes() for path in output.iterdir()}
    assert set(before) == {"normalizer.json", "u1_contract.json"}
    assert before == after
    assert first[0].digest == second[0].digest
    for content in after.values():
        assert content.endswith(b"\n")
```

- [ ] **Step 2: Write drift/extra-file RED tests**

```python
def test_u1_materialization_rejects_existing_drift(tmp_path, u1_contract, normalizer) -> None:
    output = tmp_path / "u1"
    materialize_universal_trade_rl_u1(contract=u1_contract, normalizer=normalizer, output_root=output)
    (output / "normalizer.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="drift|existing"):
        materialize_universal_trade_rl_u1(contract=u1_contract, normalizer=normalizer, output_root=output)


def test_u1_materialization_rejects_extra_file(tmp_path, u1_contract, normalizer) -> None:
    output = tmp_path / "u1"
    materialize_universal_trade_rl_u1(contract=u1_contract, normalizer=normalizer, output_root=output)
    (output / "extra.txt").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError, match="extra|existing"):
        materialize_universal_trade_rl_u1(contract=u1_contract, normalizer=normalizer, output_root=output)
```

- [ ] **Step 3: Write partial-publication failure RED test**

Monkeypatch the runner's second-file write helper to raise after `normalizer.json` is written in staging:

```python
def test_u1_materialization_failure_never_publishes_partial_final_directory(tmp_path, monkeypatch, u1_contract, normalizer) -> None:
    output = tmp_path / "u1"
    original = runner._write_canonical_json
    calls = 0

    def fail_second(path, payload):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected write failure")
        return original(path, payload)

    monkeypatch.setattr(runner, "_write_canonical_json", fail_second)
    with pytest.raises(OSError, match="injected"):
        runner.materialize_universal_trade_rl_u1(contract=u1_contract, normalizer=normalizer, output_root=output)
    assert not output.exists()
    assert not tuple(tmp_path.glob(".u1.staging-*"))
```

- [ ] **Step 4: Run and confirm RED**

```bash
uv run pytest tests/workflows/test_universal_trade_rl_u1_runner.py -q
```

- [ ] **Step 5: Implement U0-style atomic publication**

Before writing, require contract/normalizer digest and provenance digest equality. Canonical JSON uses sorted keys, compact deterministic separators already used by repository conventions, UTF-8, exactly one trailing newline. Write both files in a staging directory, flush+`os.fsync` each file, fsync staging directory where supported, then publish the directory with one `os.replace`. Existing final output succeeds only if its exact two filenames and canonical bytes match; otherwise raise without repair.

- [ ] **Step 6: Verify and commit**

```bash
uv run pytest tests/workflows/test_universal_trade_rl_u1_runner.py tests/workflows/test_universal_trade_rl_universe_runner.py -q
uv run ruff check trade_rl/workflows/universal_trade_rl_u1_runner.py tests/workflows/test_universal_trade_rl_u1_runner.py
uv run mypy trade_rl/workflows/universal_trade_rl_u1_runner.py
git add trade_rl/workflows/universal_trade_rl_u1_runner.py tests/workflows/test_universal_trade_rl_u1_runner.py
git commit -m "feat: materialize Universal Trade RL U1 artifacts"
```

---

### Task 10: Add adversarial/falsification regression tests

**Files:**
- Create: `tests/rl/test_universal_trade_falsification.py`

This task does not add production APIs unless a falsification test exposes a missing observable contract.

- [ ] **Step 1: Add flat-market cost double-counting test**

Use nonzero fee/spread/impact, flat prices, and enough volume to fill. Run `flat -> long -> flat` through `UniversalTradeMarketEnv` and collect rewards:

```python
def test_flat_market_round_trip_reward_equals_accounting_loss_only() -> None:
    execution = ExecutionCostConfig(
        fee_rate=0.0005,
        spread_rate=0.0002,
        impact_rate=0.0002,
        max_participation_rate=1.0,
    )
    market = make_u1_market(volume=1_000_000_000.0)
    wrapper = _wrap(make_u1_base_env(dataset=market, execution_cost=execution))
    wrapper.reset(options={"start_idx": 5700, "initial_state_mode": "cash"})
    initial = wrapper.unwrapped.hybrid.portfolio_value
    rewards = []
    for action in (0.8, 0.0, 0.0):
        _, reward, _, _, _ = wrapper.step(np.asarray([action], dtype=np.float32))
        rewards.append(reward)
    final = wrapper.unwrapped.hybrid.portfolio_value
    assert final < initial
    assert sum(rewards) / 100.0 == pytest.approx(math.log(final / initial), abs=1e-10)
```

Account for one-decision signal delay: the final `0.0` decision executes the previous flatten target. Do not shorten this sequence to two actions.

- [ ] **Step 2: Add reset state-leak test**

```python
def test_cash_reset_clears_prior_episode_policy_execution_state() -> None:
    wrapper = _wrap(make_u1_base_env())
    wrapper.reset(options={"start_idx": 5700, "initial_state_mode": "cash"})
    wrapper.step(np.asarray([0.7], dtype=np.float32))
    wrapper.step(np.asarray([-0.4], dtype=np.float32))
    obs, _ = wrapper.reset(options={"start_idx": 5720, "initial_state_mode": "cash"})
    state = wrapper.policy_state_dict(obs)
    assert state["current_weight"] == pytest.approx(0.0)
    assert state["policy_requested_weight"] == pytest.approx(0.0)
    assert state["pending_target_active"] == pytest.approx(0.0)
    assert state["pending_notional_ratio"] == pytest.approx(0.0)
    assert state["position_age_days"] == pytest.approx(0.0)
```

- [ ] **Step 3: Add missing-vs-zero observation test**

```python
def test_true_zero_and_unavailable_zero_are_distinct() -> None:
    dataset = make_u1_market()
    unavailable = dataset.feature_available.copy()
    unavailable[5700, 0, 0] = False
    missing = replace(dataset, dataset_id="9" * 64, identity_payload_json=None, feature_available=unavailable)
    builder = UniversalTradeObservationBuilder(UniversalTradePolicyContract(feature_specs=make_u1_feature_specs()))
    runtime = make_runtime_snapshot()
    present_obs = builder.build(dataset, index=5700, runtime=runtime)
    missing_obs = builder.build(missing, index=5700, runtime=runtime)
    assert present_obs["sequence_15m_available"][0, -1, 0] == 1
    assert missing_obs["sequence_15m_available"][0, -1, 0] == 0
```

- [ ] **Step 4: Add combined four-stage/pending-order falsification**

Use the Task 3 low-volume/risk-cap environment. After two decisions assert:

```python
snapshot = wrapper.unwrapped.universal_trade_runtime_snapshot()
assert snapshot.policy_requested_weight == pytest.approx(0.80)
assert snapshot.pending_target_weight == pytest.approx(0.80)
assert snapshot.pending_target_active is True
assert snapshot.risk_projected_weight == pytest.approx(0.35)
assert abs(snapshot.current_weight) < 0.35
assert snapshot.pending_notional_ratio >= 0.0
```

Then assert a pending flat action has `pending_target_active=True` even with `pending_target_weight==0`, proving it cannot alias no-pending state.

- [ ] **Step 5: Run falsification suite**

```bash
uv run pytest tests/rl/test_universal_trade_falsification.py -q
```

If any test reveals a production defect, fix the smallest responsible task module, run its targeted tests first, then rerun this suite. Do not weaken assertions or skip the failing falsification.

- [ ] **Step 6: Commit**

```bash
git add tests/rl/test_universal_trade_falsification.py
git commit -m "test: falsify Universal Trade RL U1 boundaries"
```

---

### Task 11: Documentation, full verification, independent review, and exact-HEAD CI

**Files:**
- Modify: `docs/UNIVERSAL_TRADE_RL.md`
- Review: every U1 production/test file above.
- Modify `tests/test_architecture_contract.py` only if import/package architecture checks require an explicit new allowlist entry.

- [ ] **Step 1: Update maintained documentation with exact U1 semantics**

Add these concrete sections to `docs/UNIVERSAL_TRADE_RL.md`:

```text
U1 Production status: NO-GO
U0 prerequisite and published source identity verification
one-symbol deployment contract
allowed/forbidden feature surface
policy_requested -> signal-delay pending -> risk_projected -> current exposure flow
pending-order state is separate
cash-only sampled reset
external truncation / no horizon feature / no forced liquidation
unique_feature_event_time_v1 normalizer sampling
equal-symbol first/second moment aggregation
pure after-cost reward equation and telescoping oracle
normalizer.json + u1_contract.json materialization
U1 Quality Gate
U2 BASE_TRAINING handoff; Development evaluation-only; Admission unopened
```

- [ ] **Step 2: Run all U1/U0 targeted tests**

```bash
uv run pytest \
  tests/rl/test_universal_trade_contract.py \
  tests/rl/test_universal_trade_action.py \
  tests/rl/test_universal_trade_runtime.py \
  tests/rl/test_universal_trade_normalization.py \
  tests/rl/test_universal_trade_observation.py \
  tests/rl/test_universal_trade_reward.py \
  tests/rl/test_universal_trade_environment.py \
  tests/rl/test_universal_trade_falsification.py \
  tests/workflows/test_universal_trade_rl_u1_contract.py \
  tests/workflows/test_universal_trade_rl_u1_runner.py \
  tests/workflows/test_universal_trade_rl_data_provenance.py \
  tests/workflows/test_universal_trade_rl_universe_access.py \
  tests/workflows/test_universal_trade_rl_run_identity.py \
  -q
```

Record the exact result counts.

- [ ] **Step 3: Run maintained compatibility tests around changed boundaries**

```bash
uv run pytest \
  tests/rl/test_environment_reduce_only_integration.py \
  tests/learning/test_rollout_execution_lifecycle.py \
  tests/learning/test_causal_alpha_v10_closed_loop.py \
  tests/learning/test_causal_alpha_v11_policy.py \
  -q
```

Expected: no unexplained regression.

- [ ] **Step 4: Run static/architecture checks**

```bash
uv run ruff check trade_rl tests
uv run ruff format --check trade_rl tests
uv run mypy trade_rl
uv run lint-imports
```

Also execute the exact MyPy/import/build commands used by current repository CI if they differ from these invocations. A partial affected-file check is not evidence of full static success.

- [ ] **Step 5: Run full suite and package build**

```bash
uv run pytest -q
uv build
```

Record exact `passed / skipped / failed` counts and produced sdist/wheel names. Do not commit generated distributions unless repository policy explicitly requires them.

- [ ] **Step 6: Self-review exact diff/status/history**

```bash
git diff b3a4cf0fd98f459ceb2262a4a759af83f9b1df3c...HEAD
git status --short
git log --oneline --decorate -n 40
```

If PR #426 head no longer equals the pinned base, stop this comparator review, synchronize the U1 branch to the intended U0 head, update the pinned SHA in this plan, rerun targeted/full verification, and only then continue.

Review every diff for:

```text
manual strategy prior leakage
symbol/dataset/raw nominal leakage
untrusted source digest path
Development/Admission normalization leakage
row-count weighted normalization
carried feature event double counting
signal-delay target vs pending-order semantic confusion
post-risk vs realized exposure aliasing
reward cost double counting
truncation/terminal leakage
non-cash sampled reset leakage
duplicated risk/execution/accounting logic
unrelated refactor/debug code/temp workflow/generated artifact/secrets
```

- [ ] **Step 7: Independent/falsification review from the original spec**

Without using implementation conclusions as premises, answer each with concrete code/test evidence:

```text
Can future data change Observation(t)?
Can explicit symbol/dataset identity reach a tensor?
Can a different published dataset be passed while claiming the manifest digest?
Can Development or Admission values affect fitted statistics?
Can a long-history Train symbol get a larger normalizer vote?
Can a carried 4h/1d event be counted repeatedly?
Can pending flat target alias no pending target?
Can risk_projected_weight alias current_weight under partial fill?
Can accounting cost be charged twice by reward?
Can sample-end knowledge enter policy state or create free liquidation?
Can TrendStrategy enter U1 through reset state?
Did any U3-U6/Causal Alpha economic path change?
```

If a substantive gap is found, fix it and rerun from the nearest targeted test through Steps 2-6.

- [ ] **Step 8: Commit documentation/architecture adjustment**

```bash
git add docs/UNIVERSAL_TRADE_RL.md
if ! git diff --quiet -- tests/test_architecture_contract.py; then git add tests/test_architecture_contract.py; fi
git commit -m "docs: document Universal Trade RL U1 contract"
```

Do not create an empty commit if only the documentation commit was already included in a prior corrective commit.

- [ ] **Step 9: Invoke verification-before-completion before any completion claim**

Load `superpowers:verification-before-completion` and re-check final HEAD, diff, status, targeted/full tests, static checks, build, and remaining risks. “All tests green” alone is not the Quality Gate.

- [ ] **Step 10: Push and open/update a Draft PR; verify exact final HEAD CI**

The PR body must include What / Why / Acceptance Criteria / Design decisions / Scope / Non-goals / Tests / Verification / Risks / Remaining limitations / Follow-up.

Record:

```text
final HEAD SHA
U0 comparator SHA
CI workflow/run IDs on final HEAD
PostgreSQL Catalog conclusion on final HEAD
Nautilus Capability conclusion on final HEAD
required static/training checks on final HEAD
full-suite exact counts
coverage evidence if produced
unverified economic claims
remaining risks
```

Do not mark Ready until every U1 Quality Gate item below is satisfied. Do not merge without explicit user permission.

---

## U1 Quality Gate

U1 may be described as implementation-complete only when all are evidenced on the exact final HEAD:

1. exactly-one-symbol wrapper and fixed V1 runtime config are enforced;
2. Policy tensors contain no ID/raw nominal/manual prior/horizon fraction;
3. future-mutation causality test passes;
4. missing/availability/staleness remain distinct;
5. submitted policy target, signal-delay pending+active, post-risk target, realized weight, and pending-order lifecycle are separately observable;
6. scalar action semantic is static and independent of dynamic risk cap;
7. cash-only sampled reset prevents TrendStrategy prior injection;
8. U0 normalization firewall executes before filesystem/source inspection;
9. published artifact digest is independently inspected and matched to U0 manifest before numeric data use;
10. source feature events are deduplicated by recovered source event time, not by value and not by base-row position;
11. equal-symbol first/second-moment oracle passes with unequal source history lengths;
12. unavailable placeholder values do not enter fit;
13. Admission-only generation drift keeps Train statistics digest equal and U0-bound artifact digest different when Train artifacts/cutoff are identical;
14. pure reward telescopes to realized after-cost wealth on maintained normal/cost/funding-or-borrow/partial-fill paths that exist in the repository test fixtures;
15. non-positive/non-finite wealth fails closed;
16. external truncation causes no forced liquidation/terminal reward and remaining horizon is absent from policy input;
17. `normalizer.json` + `u1_contract.json` publish atomically/canonically/idempotently and reject drift/partial output;
18. existing U3-U6/Causal Alpha compatibility tests show no intended economic behavior change;
19. targeted/property/integration/falsification tests, Ruff, format, required MyPy/static analysis, import architecture, full suite, and package build pass;
20. self-review plus independent/falsification review has no unresolved substantive finding;
21. required CI is green on exact final HEAD, not an earlier commit;
22. final report explicitly says U1 does not prove RL learnability, zero-shot economics, profitability, Admission performance, real-market execution fidelity, or Production readiness.

## U2 Handoff

U2 begins only after a real production-candidate U0 generation and U1 artifacts are frozen. U2 must build `UniversalTradeRLFitPurpose.RL_TRAINING` provenance from U0 Train symbols and bind the frozen U1 contract digest into the existing `UniversalTradeRLRunStage.BASE_TRAINING` model/checkpoint identity. Development remains evaluation-only and Admission remains unopened.
