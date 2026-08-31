# Universal Trade RL U1 Observation / Action / Reward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Every production change follows Red → Green → Refactor. Do not weaken tests to obtain Green.

**Goal:** Implement a one-symbol Universal Trade RL U1 environment with causal strategy-prior-free observations, a scalar fixed-semantic target-exposure action, U0 Train-only equal-symbol market normalization from verified published artifacts, and reward exactly reconciled to realized after-cost wealth.

**Architecture:** `ResidualMarketEnv` remains the sole Risk / Execution / Accounting authority. U1 adds a read-only runtime snapshot, a market normalizer, a policy observation builder, and a Gym wrapper that validates fixed V1 runtime semantics and delegates every economic transition to the maintained environment exactly once. `normalizer.json` and `u1_contract.json` bind the frozen U0 generation and are published atomically.

**Spec:** `docs/implementation-plans/specs/2026-08-31-universal-trade-rl-u1-observation-reward-design.md`

**Pinned design base:** `b3a4cf0fd98f459ceb2262a4a759af83f9b1df3c` (PR #426 U0 head at plan finalization). Before Task 1 production code, fetch PR #426. If its intended U0 head differs, synchronize this branch, update this SHA in spec/plan, and re-check the design before coding.

## Fixed U1 V1 Contract

- Production status: `NO-GO`.
- No RL training, Admission opening, profitability claim, or Production promotion.
- Exactly one symbol and one capital budget per environment.
- `ActionMode.TARGET_WEIGHT`, target count `1`, strict action semantics.
- `decision_hours = 0.25`, `decision_every = None`.
- `episode_hours = 720.0`, `episode_bars = None`, `episode_hour_choices = ()`.
- `signal_delay_decisions = 1`.
- `episode_boundary_mode = EXTERNAL_TRUNCATION`.
- `finite_horizon_observation = False`.
- `liquidate_on_end = False`.
- `initial_state_modes = ("cash",)`.
- structured sequence observation enabled with exactly `15m×96`, `1h×168`, `4h×120`, `1d×60`.
- Base reward config must satisfy `RewardConfig.is_pure_net_log_growth()` and scale `100.0`.
- Existing U3-U6 and Causal Alpha V9/V10/V11 economics remain unchanged.
- `ResidualMarketEnv` remains the only Risk / Execution / Accounting implementation.
- Policy tensors contain no symbol/dataset IDs, raw absolute OHLC, raw nominal volume/cash/quantity, `TrendTargets`, alpha output, factor priors, shadow/baseline state, ownership latch, or remaining-horizon fraction.
- Missing value, availability, and staleness are distinct channels.
- Exposure meanings are distinct: current policy request, current signal-delay pending target + active mask, current post-risk execution target, realized current weight.
- Signal-delay pending target is not pending-order lifecycle.
- Existing `ObservationExecutionState.requested_weights` is post-risk; U1 reads it as `risk_projected_weight` without changing legacy semantics.
- Existing `ObservationExecutionState.execution_cost` is `cost_by_symbol / initial_capital`; U1 exposes it as `execution_cost_rate` unchanged.
- U0 normalization authorization runs before filesystem/source inspection.
- Train source identity is proven by inspecting published market artifacts; caller-supplied digest strings are not trusted.
- Normalizer counts unique available feature source-events at or before the knowledge cutoff, computes per-symbol first/second moments, then gives each Train symbol one equal vote.
- Endogenous policy state uses deterministic versioned transforms, never fitted rollout statistics.
- Reward is exactly `100 * log(W_after / W_before)` over realized hybrid `BookState.portfolio_value`; no extra shaping or cost penalty.
- Non-positive/non-finite wealth fails closed.
- Sample end is truncation with no forced liquidation or terminal bonus.
- No merge to `main` without explicit user permission.

## File Map

**Production files to create**

- `trade_rl/rl/universal_trade_contract.py`
- `trade_rl/rl/universal_trade_action.py`
- `trade_rl/rl/universal_trade_runtime.py`
- `trade_rl/rl/universal_trade_normalization.py`
- `trade_rl/rl/universal_trade_observation.py`
- `trade_rl/rl/universal_trade_reward.py`
- `trade_rl/rl/universal_trade_environment.py`
- `trade_rl/workflows/universal_trade_rl_u1_contract.py`
- `trade_rl/workflows/universal_trade_rl_u1_runner.py`

**Production/docs files to modify**

- `trade_rl/rl/environment.py` — read-only U1 runtime accessor only; no economic semantic change.
- `docs/UNIVERSAL_TRADE_RL.md`
- `tests/test_architecture_contract.py` only if the existing architecture allowlist requires an explicit entry.

**Test support/tests to create**

- `tests/rl/universal_trade_test_support.py`
- `tests/workflows/universal_trade_rl_u1_test_support.py`
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

## Task 1: Freeze the U1 policy contract and deterministic test market

**Files:** create `trade_rl/rl/universal_trade_contract.py`, `tests/rl/universal_trade_test_support.py`, `tests/rl/test_universal_trade_contract.py`.

### Step 1 — Create the shared deterministic test market helper

Create `tests/rl/universal_trade_test_support.py` with these interfaces:

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
    n_bars: int = 10_000,
    price_scale: float = 1.0,
    price_drift: float = 1e-4,
    feature_level: float = 0.0,
    volume: float = 1_000_000.0,
    funding_rate_value: float = 0.0,
    funding_due_from: int | None = None,
) -> MarketDataset:
    rows = np.arange(n_bars, dtype=np.int64)
    periods = (1, 4, 16, 96)
    features = np.empty((n_bars, 1, 4), dtype=np.float32)
    staleness_hours = np.empty((n_bars, 1, 4), dtype=np.float64)
    for column, period in enumerate(periods):
        source = (rows // period) * period
        features[:, 0, column] = feature_level + source.astype(np.float32) * 1e-4
        staleness_hours[:, 0, column] = (rows - source) * 0.25
    timestamps = np.datetime64("2025-01-01T00:00:00", "ns") + rows * np.timedelta64(15, "m")
    close = price_scale * (100.0 + rows.astype(np.float64) * price_drift)
    close_2d = close[:, None]
    funding_rate = np.full((n_bars, 1), funding_rate_value, dtype=np.float64)
    funding_due = np.zeros((n_bars, 1), dtype=np.bool_)
    if funding_due_from is not None:
        funding_due[funding_due_from:, 0] = True
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
        funding_rate=funding_rate,
        funding_due=funding_due,
        tradable=np.ones((n_bars, 1), dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, 4), dtype=np.bool_),
        feature_names=tuple(spec.name for spec in make_u1_feature_specs()),
        global_feature_names=("market",),
        periods_per_year=35_040,
        feature_staleness_hours=staleness_hours,
        feature_staleness=np.minimum(staleness_hours / 24.0, 1.0),
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
        trend_strategy=TrendStrategy(
            TrendConfig(fast_lookback=2, base_lookback=4, slow_lookback=8)
        ),
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
            episode_hours=720.0,
            decision_hours=0.25,
            episode_hour_choices=(),
            episode_bars=None,
            decision_every=None,
            signal_delay_decisions=1,
            initial_capital=100_000.0,
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

Run:

```bash
uv run python -c "from tests.rl.universal_trade_test_support import make_u1_base_env; env=make_u1_base_env(); assert env.decision_hours == 0.25 and env.episode_hours == 720.0"
```

Expected: PASS. If this helper fails because an existing constructor/API differs, correct the helper to the current maintained API before continuing; do not change production code to satisfy a test helper.

### Step 2 — Write RED contract tests

Create `tests/rl/test_universal_trade_contract.py`:

```python
import pytest

from trade_rl.data.contracts import FeatureKind, FeatureSpec
from trade_rl.rl.universal_trade_contract import (
    UNIVERSAL_TRADE_SEQUENCE_WINDOWS,
    UniversalTradePolicyContract,
)
from tests.rl.universal_trade_test_support import make_u1_feature_specs


def test_contract_freezes_sequence_windows_and_digest() -> None:
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
def test_contract_rejects_cross_asset_feature(kind: FeatureKind) -> None:
    with pytest.raises(ValueError, match="U1 feature"):
        UniversalTradePolicyContract(
            feature_specs=(FeatureSpec(name="forbidden", kind=kind),)
        )


def test_contract_binds_feature_order() -> None:
    specs = make_u1_feature_specs()
    assert UniversalTradePolicyContract(feature_specs=specs).digest != UniversalTradePolicyContract(
        feature_specs=tuple(reversed(specs))
    ).digest


@pytest.mark.parametrize("scale", (0.0, -1.0, 1.01, float("inf"), float("nan")))
def test_contract_rejects_invalid_policy_scale(scale: float) -> None:
    with pytest.raises(ValueError, match="policy_weight_scale"):
        UniversalTradePolicyContract(
            feature_specs=make_u1_feature_specs(), policy_weight_scale=scale
        )
```

Run:

```bash
uv run pytest tests/rl/test_universal_trade_contract.py -q
```

Expected: RED because `universal_trade_contract` does not exist.

### Step 3 — Implement minimal contract

Create exact schema constants, frozen sequence windows, and the FeatureKind allowlist from the spec. `UniversalTradePolicyContract` validates non-empty/unique ordered feature specs, allowed kinds, `0 < policy_weight_scale <= 1`, positive finite reward scale, and binds the fixed runtime semantics in `digest_payload()`:

```text
decision_hours=0.25
decision_every=None
episode_hours=720.0
episode_bars=None
episode_hour_choices=[]
signal_delay_decisions=1
external_truncation
finite_horizon_observation=false
liquidate_on_end=false
initial_state_modes=[cash]
exact sequence windows
```

### Step 4 — Verify and commit

```bash
uv run pytest tests/rl/test_universal_trade_contract.py -q
uv run ruff check trade_rl/rl/universal_trade_contract.py tests/rl/universal_trade_test_support.py tests/rl/test_universal_trade_contract.py
uv run mypy trade_rl/rl/universal_trade_contract.py
git add trade_rl/rl/universal_trade_contract.py tests/rl/universal_trade_test_support.py tests/rl/test_universal_trade_contract.py
git commit -m "feat: define Universal Trade RL U1 contract"
```

---

## Task 2: Implement strict scalar target-exposure action

**Files:** create `trade_rl/rl/universal_trade_action.py`, `tests/rl/test_universal_trade_action.py`.

### Step 1 — Write RED tests

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
        np.asarray([1.01]), np.asarray([-1.01]), np.asarray([np.nan]),
        np.asarray([np.inf]), np.asarray([0.0, 0.0]),
    ),
)
def test_action_rejects_invalid_output(value: np.ndarray) -> None:
    with pytest.raises(ValueError):
        parse_normalized_target_exposure(value, policy_weight_scale=1.0)
```

Run `uv run pytest tests/rl/test_universal_trade_action.py -q`; expected RED.

### Step 2 — Implement minimal parser

Create immutable `NormalizedTargetExposureAction(normalized, policy_requested_weight)`. Parse exactly one finite scalar, reject outside `[-1,1]`, validate `0 < policy_weight_scale <= 1`, return `normalized * policy_weight_scale`. Do not clip and do not call Risk logic.

### Step 3 — Verify and commit

```bash
uv run pytest tests/rl/test_universal_trade_action.py -q
uv run ruff check trade_rl/rl/universal_trade_action.py tests/rl/test_universal_trade_action.py
uv run mypy trade_rl/rl/universal_trade_action.py
git add trade_rl/rl/universal_trade_action.py tests/rl/test_universal_trade_action.py
git commit -m "feat: add normalized target exposure action"
```

---

## Task 3: Expose audited runtime state without changing economics

**Files:** create `trade_rl/rl/universal_trade_runtime.py`, modify `trade_rl/rl/environment.py`, modify `tests/rl/universal_trade_test_support.py`, create `tests/rl/test_universal_trade_runtime.py`.

### Step 1 — Write RED runtime integration tests

Use reset `start_idx=6000`; the 60-day sequence warmup is satisfied and a 720h episode still fits in the 10,000-bar fixture.

```python
import numpy as np
import pytest

from trade_rl.simulation.execution import ExecutionCostConfig
from tests.rl.universal_trade_test_support import make_u1_base_env, make_u1_market


def test_runtime_separates_submission_pending_risk_and_realized_weight() -> None:
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
    env.reset(options={"start_idx": 6000, "initial_state_mode": "cash"})
    env.step(np.asarray([0.60], dtype=np.float32))
    env.step(np.asarray([0.80], dtype=np.float32))
    snapshot = env.universal_trade_runtime_snapshot()
    assert snapshot.policy_requested_weight == pytest.approx(0.80)
    assert snapshot.pending_target_active is True
    assert snapshot.pending_target_weight == pytest.approx(0.80)
    assert snapshot.risk_projected_weight == pytest.approx(0.35)
    assert abs(snapshot.current_weight) < abs(snapshot.risk_projected_weight)
    assert 0.0 <= snapshot.fill_ratio < 1.0


def test_pending_flat_is_not_no_pending_target() -> None:
    env = make_u1_base_env()
    env.reset(options={"start_idx": 6000, "initial_state_mode": "cash"})
    assert env.universal_trade_runtime_snapshot().pending_target_active is False
    env.step(np.asarray([0.0], dtype=np.float32))
    snapshot = env.universal_trade_runtime_snapshot()
    assert snapshot.pending_target_active is True
    assert snapshot.pending_target_weight == pytest.approx(0.0)
```

Run `uv run pytest tests/rl/test_universal_trade_runtime.py -q`; expected RED.

### Step 2 — Implement snapshot/accessor

`UniversalTradeRuntimeSnapshot` contains the exact endogenous fields from the spec. The `ResidualMarketEnv` accessor requires one symbol and target-weight mode and maps:

```text
policy_requested_weight = _previous_action[0]
pending_target_active   = _pending_hybrid_target is not None
pending_target_weight   = 0.0 if None else _pending_hybrid_target[0]
risk_projected_weight   = _execution_state.requested_weights[0]
current_weight          = hybrid.weights[0]
execution_cost_rate     = _execution_state.execution_cost[0]
position_age_hours      = _execution_state.position_age[0] * dataset.bar_hours
```

Use the existing `_pending_order_observation_state()` for order lifecycle, convert its bar ages/delays/distances to hours, and derive hybrid drawdown/gross/net/cash/risk/margin from maintained state. Return immutable scalar copies. Do not add a second risk-projected state variable.

### Step 3 — Add `make_runtime_snapshot()` to test support

After the production dataclass exists, add a helper creating the neutral runtime state with cash=1, active/tradable/borrow-available=1 and all other fields zero except `fill_ratio=1`, `risk_scale=1`. It accepts keyword overrides through `dataclasses.replace`. This helper is then imported by Tasks 5/10; no test may reference an undefined pytest fixture.

### Step 4 — Verify and commit

```bash
uv run pytest tests/rl/test_universal_trade_runtime.py -q
uv run pytest tests/rl/test_environment_reduce_only_integration.py tests/learning/test_rollout_execution_lifecycle.py -q
uv run ruff check trade_rl/rl/environment.py trade_rl/rl/universal_trade_runtime.py tests/rl/universal_trade_test_support.py tests/rl/test_universal_trade_runtime.py
uv run mypy trade_rl/rl/environment.py trade_rl/rl/universal_trade_runtime.py
git add trade_rl/rl/environment.py trade_rl/rl/universal_trade_runtime.py tests/rl/universal_trade_test_support.py tests/rl/test_universal_trade_runtime.py
git commit -m "feat: expose Universal Trade RL runtime state"
```

---

## Task 4: Fit Train-only equal-symbol market normalizer from verified artifacts

**Files:** create `trade_rl/rl/universal_trade_normalization.py`, `tests/rl/test_universal_trade_normalization.py`.

**Production interfaces:** `UniversalTradePublishedSource(symbol, artifact_root)`, `UniversalTradeChannelStatistics`, `UniversalTradeSequenceNormalizer` (`statistics_digest`, U0-bound `digest`, `provenance_digest`, strict codec/transform), `fit_universal_trade_sequence_normalizer(...)`.

### Step 1 — Define exact local U0 test helpers in the test module

Use keyword-only construction; do not leave constructor choices unresolved:

```python
from trade_rl.domain.universal_trade_rl_universe import UniversalTradeRLUniverseConfig
from trade_rl.workflows.universal_trade_rl_universe_config import UniversalTradeRLSymbolSource
from trade_rl.workflows.universal_trade_rl_universe_manifest import build_universal_trade_rl_universe_manifest


def _source(symbol, artifact, dataset):
    timestamp_ns = dataset.timestamps.astype("datetime64[ns]").astype(np.int64)
    return UniversalTradeRLSymbolSource(
        symbol=symbol,
        dataset_digest=artifact.artifact_digest,
        first_timestamp_ns=int(timestamp_ns[0]),
        last_timestamp_ns=int(timestamp_ns[-1]),
        row_count=dataset.n_bars,
    )


def _manifest(train_sources, *, admission_digest="a" * 64):
    first_ns = min(item.first_timestamp_ns for item in train_sources)
    last_ns = max(item.last_timestamp_ns for item in train_sources)
    row_count = max(item.row_count for item in train_sources)
    sources = tuple(sorted(
        tuple(train_sources)
        + (
            UniversalTradeRLSymbolSource(
                symbol="AVAXUSDT", dataset_digest=admission_digest,
                first_timestamp_ns=first_ns, last_timestamp_ns=last_ns, row_count=row_count,
            ),
            UniversalTradeRLSymbolSource(
                symbol="LINKUSDT", dataset_digest="d" * 64,
                first_timestamp_ns=first_ns, last_timestamp_ns=last_ns, row_count=row_count,
            ),
        ),
        key=lambda item: item.symbol,
    ))
    config = UniversalTradeRLUniverseConfig(
        train_symbols=("BTCUSDT", "ETHUSDT"),
        development_symbols=("LINKUSDT",),
        admission_symbols=("AVAXUSDT",),
    )
    return build_universal_trade_rl_universe_manifest(config=config, sources=sources)
```

### Step 2 — Write firewall-before-filesystem RED test

Construct the manifest inside the test; no `manifest` fixture is assumed:

```python
def test_scope_fails_before_missing_artifact_path_is_touched() -> None:
    dummy_train = (
        UniversalTradeRLSymbolSource(
            symbol="BTCUSDT", dataset_digest="b" * 64,
            first_timestamp_ns=1, last_timestamp_ns=2, row_count=2,
        ),
        UniversalTradeRLSymbolSource(
            symbol="ETHUSDT", dataset_digest="e" * 64,
            first_timestamp_ns=1, last_timestamp_ns=2, row_count=2,
        ),
    )
    manifest = _manifest(dummy_train)
    access = UniversalTradeRLUniverseAccess.for_phase(
        manifest=manifest, phase=UniversalTradeRLAccessPhase.DEVELOPMENT
    )
    with pytest.raises(PermissionError, match="normalization|Train"):
        fit_universal_trade_sequence_normalizer(
            manifest=manifest,
            access=access,
            sources=(
                UniversalTradePublishedSource("BTCUSDT", Path("/missing/btc")),
                UniversalTradePublishedSource("ETHUSDT", Path("/missing/eth")),
            ),
            contract=UniversalTradePolicyContract(feature_specs=make_u1_feature_specs()),
            knowledge_cutoff_ns=2,
        )
```

Expected failure is `PermissionError`, never `FileNotFoundError`.

### Step 3 — Write published-artifact spoof RED test

Publish BTC artifact A and different BTC artifact B. Bind U0 manifest to A, pass B to the normalizer. Expect `ValueError` before B numeric arrays are accepted. Use `publish_market_dataset_artifact()` and compare inspected `artifact_digest` with manifest entry `dataset_digest`.

### Step 4 — Write equal-symbol + unavailable-value RED test

Use BTC `n_bars=5800`, ETH `n_bars=6000`. In BTC, replace row 100 of the 15m feature with `1e9` and set `feature_available[100,0,0]=False`, then reset identity and call `.with_content_identity(...)` before publishing. Use knowledge cutoff equal to ETH's last timestamp, so BTC contributes 5800-1 available events and ETH contributes 6000 events. Compute the expected first and second moments independently:

```python
btc_values = btc.features[:, 0, 0][btc.feature_available[:, 0, 0]].astype(np.float64)
eth_values = eth.features[:, 0, 0][eth.feature_available[:, 0, 0]].astype(np.float64)
mu_btc, mu_eth = float(np.mean(btc_values)), float(np.mean(eth_values))
q_btc, q_eth = float(np.mean(np.square(btc_values))), float(np.mean(np.square(eth_values)))
expected_mean = 0.5 * (mu_btc + mu_eth)
expected_q = 0.5 * (q_btc + q_eth)
expected_scale = math.sqrt(max(expected_q - expected_mean * expected_mean, 0.0))
assert stats.mean[0] == pytest.approx(expected_mean)
assert stats.scale[0] == pytest.approx(expected_scale)
```

This test must fail for row-concatenated weighting and must prove the unavailable `1e9` is excluded.

### Step 5 — Write carried-event de-duplication RED test

For the 4h fixture feature, source event changes every 16 base rows. After fit:

```python
expected_btc_events = (btc.n_bars - 1) // 16 + 1
assert dict(stats.per_symbol_sample_counts)["BTCUSDT"][0] == expected_btc_events
assert expected_btc_events < btc.n_bars
```

The implementation must recover event time from `feature_staleness_hours`, not compare feature values.

### Step 6 — Write knowledge-cutoff RED test

Publish a dataset whose 15m values after row 5000 are made extremely large while remaining available. Fit with cutoff exactly `timestamps[5000]`. Assert the fitted 15m mean/scale equal independent moments from rows `0:5001` only. This proves future published rows do not enter fit even though they exist in the verified artifact.

### Step 7 — Write statistics-vs-generation-identity RED test

Publish identical BTC/ETH Train artifacts once, build two manifests differing only in Admission digest, fit both with identical cutoff and policy contract:

```python
assert normalizer_a.statistics_digest == normalizer_b.statistics_digest
assert normalizer_a.digest != normalizer_b.digest
```

### Step 8 — Run RED suite

```bash
uv run pytest tests/rl/test_universal_trade_normalization.py -q
```

Expected: missing normalization module.

### Step 9 — Implement fit pipeline in this exact order

```text
canonicalize/sort source symbols
-> access.require_normalization_scope(symbols)
-> build FEATURE_NORMALIZATION provenance
-> inspect_published_market_dataset_artifact(root)
-> compare inspected artifact_digest to U0 manifest entry
-> load_market_dataset_artifact(root)
-> require exactly one expected symbol + exact feature layout
-> select available feature events with recovered source_event_time <= knowledge_cutoff
-> de-duplicate by (feature, recovered source_event_time_ns)
-> per-symbol mean and mean(square)
-> equal-symbol aggregate mean/second moment
-> scale floor at epsilon
-> statistics_digest
-> universe/provenance-bound artifact digest
```

Recover event time with:

```python
_NS_PER_HOUR = 3_600_000_000_000
source_event_time_ns = timestamp_ns - np.rint(staleness_hours * _NS_PER_HOUR).astype(np.int64)
```

Reject recovered event times after their bar timestamp or outside the published source interval. Do not de-duplicate by feature value.

`transform()` standardizes only `sequence_<tf>_values`; availability, staleness, and policy state pass through unchanged. Strict payload codec verifies exact keys and digests.

### Step 10 — Verify and commit

```bash
uv run pytest tests/rl/test_universal_trade_normalization.py tests/workflows/test_universal_trade_rl_data_provenance.py -q
uv run ruff check trade_rl/rl/universal_trade_normalization.py tests/rl/test_universal_trade_normalization.py
uv run mypy trade_rl/rl/universal_trade_normalization.py
git add trade_rl/rl/universal_trade_normalization.py tests/rl/test_universal_trade_normalization.py
git commit -m "feat: add Train-only universal market normalizer"
```

---

## Task 5: Build the causal strategy-prior-free U1 observation

**Files:** create `trade_rl/rl/universal_trade_observation.py`, `tests/rl/test_universal_trade_observation.py`.

**Interfaces:** `UNIVERSAL_TRADE_POLICY_STATE_NAMES`, `UniversalTradeObservationBuilder.build()`, `.observation_space()`, `.schema_digest()`, `.state_layout_digest`.

### Step 1 — Write exact layout RED test

Build at `index=6000` with `make_runtime_snapshot()`. Assert the exact 13 observation keys: values/available/staleness for each of 15m/1h/4h/1d plus `policy_state`, `observation_space.contains(obs)`, and no policy-state name contains `trend`, `alpha`, `shadow`, `baseline`, `remaining`, `symbol`, or `dataset`.

### Step 2 — Write deterministic transform RED test

Override runtime state:

```text
position_age_hours=24
pending_order_age_hours=48
pending_order_eligible_delay_hours=24
pending_order_expiry_distance_hours=72
mark_index_basis=0.01
borrow_rate=0.02
```

Assert named policy state equals `log1p(1)`, `log1p(2)`, `log1p(1)`, `log1p(3)`, `tanh(1)`, `tanh(0.02)` respectively. This fixes transform semantics independently of implementation internals.

### Step 3 — Write future-mutation RED test

Create two datasets identical through `t=6000`; use `dataclasses.replace(..., identity_payload_json=None, dataset_id="f"*64)` to mutate all later features/OHLC/mark/index in one. Build observations at `t` and assert `np.testing.assert_array_equal` for every key.

### Step 4 — Write symbol/price-unit invariance RED tests

- Rename only `symbols` and dataset identity; all policy tensors must be equal.
- Build `price_scale=1` and `price_scale=1000` fixtures with identical dimensionless features and same runtime state; all policy tensors must match within `atol=1e-7`.

### Step 5 — Write missing-vs-zero RED test

Make current 15m feature value `0.0` in both datasets but set availability false in only one. Assert `sequence_15m_values` can be equal while `sequence_15m_available` differs at the current sample.

### Step 6 — Run RED suite

```bash
uv run pytest tests/rl/test_universal_trade_observation.py -q
```

### Step 7 — Implement only from named causal sources

Call `SequenceObservationBuilder` directly. Never parse/slice `baseline_residual_observation_v5`. Use the deterministic age/basis/borrow transforms from the spec. If a normalizer is supplied, it may change only sequence value tensors, never policy state, availability, or staleness. `state_layout_digest` binds exact field names/order/transforms; `schema_digest` binds policy contract and sequence layout but not symbol text as a tensor feature.

### Step 8 — Verify and commit

```bash
uv run pytest tests/rl/test_universal_trade_observation.py -q
uv run ruff check trade_rl/rl/universal_trade_observation.py tests/rl/test_universal_trade_observation.py
uv run mypy trade_rl/rl/universal_trade_observation.py
git add trade_rl/rl/universal_trade_observation.py tests/rl/test_universal_trade_observation.py
git commit -m "feat: add Universal Trade RL observation"
```

---

## Task 6: Implement pure realized-wealth reward oracle

**Files:** create `trade_rl/rl/universal_trade_reward.py`, `tests/rl/test_universal_trade_reward.py`.

### Step 1 — Write RED tests

For values `(100, 101, 99.5, 103.25)`, compute per-transition reward and assert:

```python
sum(rewards) / 100.0 == pytest.approx(math.log(103.25 / 100.0))
```

Call `reconcile_universal_trade_reward()` and expect no error. Parameterize `0`, negative, `inf`, `nan` for before/after wealth and require `ValueError`.

### Step 2 — Implement minimal formula

After finite/positive validation:

```python
return scale * math.log(after_value / before_value)
```

Reconciliation compares `sum(rewards)/scale` with `log(final/initial)` using zero relative tolerance and explicit absolute tolerance; mismatch raises.

### Step 3 — Verify and commit

```bash
uv run pytest tests/rl/test_universal_trade_reward.py -q
uv run ruff check trade_rl/rl/universal_trade_reward.py tests/rl/test_universal_trade_reward.py
uv run mypy trade_rl/rl/universal_trade_reward.py
git add trade_rl/rl/universal_trade_reward.py tests/rl/test_universal_trade_reward.py
git commit -m "feat: add Universal Trade RL wealth reward"
```

---

## Task 7: Add the U1 Gym wrapper around maintained execution

**Files:** create `trade_rl/rl/universal_trade_environment.py`, modify `tests/rl/universal_trade_test_support.py`, create `tests/rl/test_universal_trade_environment.py`.

### Step 1 — Write constructor RED tests with valid mutated configs

Create `_wrap(env)` locally in this test module. A valid U1 base env must be accepted. For each drift, replace `env.config` before wrapper construction and expect `ValueError`:

- `signal_delay_decisions=0`
- `decision_every=1` (hidden override)
- `episode_bars=2880` (hidden override)
- `episode_hour_choices=(720.0,)` (hidden override)
- `liquidate_on_end=True`
- `initial_state_modes=("baseline",)`
- `finite_horizon_observation=True`

For finite-horizon termination, build a valid paired replacement:

```python
env.config = replace(
    env.config,
    episode_boundary_mode=EpisodeBoundaryMode.FINITE_HORIZON_TERMINATION,
    finite_horizon_observation=True,
)
```

and expect wrapper rejection.

### Step 2 — Write signal-delay integration RED test

Reset at `start_idx=6000`. Submit action `0.60`, then `0.80`, with max risk weight `0.40`. Returned second observation must have current submission/pending `0.80` and risk-projected target `0.40`. This proves action_t is not being incorrectly treated as same-step executed target.

### Step 3 — Write reward drift-guard RED test

Record base hybrid wealth before/after wrapper step and assert wrapper reward equals `100*log(after/before)` within `1e-10`. The wrapper implementation must also compare the delegated base reward against this oracle and raise if future base reward shaping drifts.

### Step 4 — Write cash-only/truncation RED tests

- Wrapper reset with `initial_state_mode="baseline"` must fail.
- Repeated `0.2` actions until sample horizon must end with `terminated=False`, `truncated=True`, and no liquidation-on-end.

### Step 5 — Implement wrapper

Constructor validates all fixed U1 V1 fields, exact windows/feature layout, one symbol/target, and pure base reward. `reset()` delegates once, rejects sampled non-cash mode, ignores legacy base observation, then builds U1 observation from runtime snapshot. `step()` strictly parses scalar action, records wealth, delegates exactly once to base env with `[policy_requested_weight]`, recomputes U1 reward, requires base reward equality within `1e-10`, builds U1 observation, preserves termination/truncation/info. Never call Risk/Execution/Accounting directly.

### Step 6 — Export reusable `make_u1_wrapper()` from test support

After production wrapper exists, add:

```python
def make_u1_wrapper(*, base_env=None, normalizer=None):
    from trade_rl.rl.universal_trade_contract import UniversalTradePolicyContract
    from trade_rl.rl.universal_trade_environment import UniversalTradeMarketEnv
    env = make_u1_base_env() if base_env is None else base_env
    return UniversalTradeMarketEnv(
        env,
        contract=UniversalTradePolicyContract(feature_specs=make_u1_feature_specs()),
        normalizer=normalizer,
    )
```

Task 10 imports this helper; it must not import a private `_wrap` from another test module.

### Step 7 — Verify and commit

```bash
uv run pytest tests/rl/test_universal_trade_environment.py tests/rl/test_universal_trade_runtime.py -q
uv run pytest tests/rl/test_environment_reduce_only_integration.py tests/learning/test_rollout_execution_lifecycle.py -q
uv run ruff check trade_rl/rl/universal_trade_environment.py tests/rl/universal_trade_test_support.py tests/rl/test_universal_trade_environment.py
uv run mypy trade_rl/rl/universal_trade_environment.py
git add trade_rl/rl/universal_trade_environment.py tests/rl/universal_trade_test_support.py tests/rl/test_universal_trade_environment.py
git commit -m "feat: add Universal Trade RL environment wrapper"
```

---

## Task 8: Bind U1 contract identity to U0 and create explicit workflow test support

**Files:** create `trade_rl/workflows/universal_trade_rl_u1_contract.py`, `tests/workflows/universal_trade_rl_u1_test_support.py`, `tests/workflows/test_universal_trade_rl_u1_contract.py`.

### Step 1 — Define `U1WorkflowFixture` and builder in test support

The helper must create all objects explicitly; Task 8/9 tests use no undefined pytest fixtures:

```python
@dataclass(frozen=True, slots=True)
class U1WorkflowFixture:
    manifest: UniversalTradeRLUniverseManifest
    u0_identity: UniversalTradeRLRunIdentity
    policy_contract: UniversalTradePolicyContract
    normalizer: UniversalTradeSequenceNormalizer
    base_env: ResidualMarketEnv
    observation_schema_digest: str
    state_layout_digest: str
    u1_contract: UniversalTradeRLU1Contract


def build_u1_workflow_fixture(root: Path, *, admission_digest: str = "a" * 64) -> U1WorkflowFixture:
    btc = make_u1_market(symbol="BTCUSDT")
    eth = make_u1_market(symbol="ETHUSDT", feature_level=10.0)
    art_btc = publish_market_dataset_artifact(root / "btc", btc)
    art_eth = publish_market_dataset_artifact(root / "eth", eth)
    timestamps = btc.timestamps.astype("datetime64[ns]").astype(np.int64)
    source_btc = UniversalTradeRLSymbolSource(
        symbol="BTCUSDT", dataset_digest=art_btc.artifact_digest,
        first_timestamp_ns=int(timestamps[0]), last_timestamp_ns=int(timestamps[-1]), row_count=btc.n_bars,
    )
    source_eth = UniversalTradeRLSymbolSource(
        symbol="ETHUSDT", dataset_digest=art_eth.artifact_digest,
        first_timestamp_ns=int(timestamps[0]), last_timestamp_ns=int(timestamps[-1]), row_count=eth.n_bars,
    )
    config = UniversalTradeRLUniverseConfig(
        train_symbols=("BTCUSDT", "ETHUSDT"),
        development_symbols=("LINKUSDT",), admission_symbols=("AVAXUSDT",),
    )
    sources = tuple(sorted((
        source_btc, source_eth,
        UniversalTradeRLSymbolSource(
            symbol="AVAXUSDT", dataset_digest=admission_digest,
            first_timestamp_ns=int(timestamps[0]), last_timestamp_ns=int(timestamps[-1]), row_count=btc.n_bars,
        ),
        UniversalTradeRLSymbolSource(
            symbol="LINKUSDT", dataset_digest="d" * 64,
            first_timestamp_ns=int(timestamps[0]), last_timestamp_ns=int(timestamps[-1]), row_count=btc.n_bars,
        ),
    ), key=lambda item: item.symbol))
    manifest = build_universal_trade_rl_universe_manifest(config=config, sources=sources)
    u0_identity = UniversalTradeRLRunIdentity(
        stage=UniversalTradeRLRunStage.UNIVERSE_MATERIALIZATION,
        universe_manifest_digest=manifest.digest,
        model_config_digest=None,
        fit_provenance_digests=(),
    )
    policy_contract = UniversalTradePolicyContract(feature_specs=make_u1_feature_specs())
    access = UniversalTradeRLUniverseAccess.for_phase(
        manifest=manifest, phase=UniversalTradeRLAccessPhase.TRAIN
    )
    normalizer = fit_universal_trade_sequence_normalizer(
        manifest=manifest,
        access=access,
        sources=(
            UniversalTradePublishedSource("BTCUSDT", art_btc.root),
            UniversalTradePublishedSource("ETHUSDT", art_eth.root),
        ),
        contract=policy_contract,
        knowledge_cutoff_ns=int(timestamps[-1]),
    )
    base_env = make_u1_base_env(dataset=btc)
    observation_builder = UniversalTradeObservationBuilder(policy_contract)
    observation_schema_digest = observation_builder.schema_digest(btc)
    state_layout_digest = observation_builder.state_layout_digest
    u1_contract = build_universal_trade_rl_u1_contract(
        manifest=manifest,
        u0_identity=u0_identity,
        policy_contract=policy_contract,
        normalizer=normalizer,
        base_env=base_env,
        observation_schema_digest=observation_schema_digest,
        state_layout_digest=state_layout_digest,
    )
    return U1WorkflowFixture(
        manifest, u0_identity, policy_contract, normalizer, base_env,
        observation_schema_digest, state_layout_digest, u1_contract,
    )
```

Use explicit imports corresponding to these types/functions; do not use stringly-typed stubs.

### Step 2 — Write RED identity tests

```python
def test_u1_contract_binds_u0_and_runtime_identity(tmp_path) -> None:
    fixture = build_u1_workflow_fixture(tmp_path)
    assert fixture.u1_contract.universe_manifest_digest == fixture.manifest.digest
    assert fixture.u1_contract.u0_identity_digest == fixture.u0_identity.digest
    assert fixture.u1_contract.normalizer_digest == fixture.normalizer.digest
    assert fixture.u1_contract.production_status == "NO-GO"


def test_u1_contract_rejects_normalizer_from_other_generation(tmp_path) -> None:
    a = build_u1_workflow_fixture(tmp_path / "a", admission_digest="a" * 64)
    b = build_u1_workflow_fixture(tmp_path / "b", admission_digest="f" * 64)
    with pytest.raises(ValueError, match="universe|normalizer"):
        build_universal_trade_rl_u1_contract(
            manifest=b.manifest,
            u0_identity=b.u0_identity,
            policy_contract=b.policy_contract,
            normalizer=a.normalizer,
            base_env=b.base_env,
            observation_schema_digest=b.observation_schema_digest,
            state_layout_digest=b.state_layout_digest,
        )
```

Add a tamper test by changing one digest in `to_payload()` and requiring `from_payload()` to reject it. Add a `dataclasses.replace` test showing every runtime/policy/normalizer/risk/execution digest field changes the final contract digest.

### Step 3 — Implement strict U1 contract builder/codec

Validate U0 identity stage is `UNIVERSE_MATERIALIZATION`, U0 identity manifest digest matches, normalizer manifest/provenance matches and purpose is `FEATURE_NORMALIZATION`, and `require_universal_trade_rl_train_only_provenance()` succeeds. Validate the base env satisfies the same fixed runtime contract as the wrapper. `runtime_config_digest` binds exact decision/episode hours plus override knobs (`decision_every=None`, `episode_bars=None`, `episode_hour_choices=()`), signal delay, boundary, horizon flag, end liquidation, reset modes, sequence windows. Bind execution policy and canonical pretrade/portfolio risk config digests. Never bind symbol text. Require `production_status="NO-GO"`.

### Step 4 — Verify and commit

```bash
uv run pytest tests/workflows/test_universal_trade_rl_u1_contract.py tests/workflows/test_universal_trade_rl_run_identity.py tests/workflows/test_universal_trade_rl_data_provenance.py -q
uv run ruff check trade_rl/workflows/universal_trade_rl_u1_contract.py tests/workflows/universal_trade_rl_u1_test_support.py tests/workflows/test_universal_trade_rl_u1_contract.py
uv run mypy trade_rl/workflows/universal_trade_rl_u1_contract.py
git add trade_rl/workflows/universal_trade_rl_u1_contract.py tests/workflows/universal_trade_rl_u1_test_support.py tests/workflows/test_universal_trade_rl_u1_contract.py
git commit -m "feat: bind Universal Trade RL U1 contract identity"
```

---

## Task 9: Materialize `normalizer.json` and `u1_contract.json` atomically

**Files:** create `trade_rl/workflows/universal_trade_rl_u1_runner.py`, `tests/workflows/test_universal_trade_rl_u1_runner.py`.

### Step 1 — Write RED tests using explicit workflow fixture helper

Every test begins with `fixture = build_u1_workflow_fixture(tmp_path / "fixture")`; no `u1_contract` or `normalizer` pytest fixture is assumed.

Idempotency oracle:

```python
output = tmp_path / "u1"
materialize_universal_trade_rl_u1(
    contract=fixture.u1_contract, normalizer=fixture.normalizer, output_root=output
)
before = {path.name: path.read_bytes() for path in output.iterdir()}
materialize_universal_trade_rl_u1(
    contract=fixture.u1_contract, normalizer=fixture.normalizer, output_root=output
)
after = {path.name: path.read_bytes() for path in output.iterdir()}
assert set(before) == {"normalizer.json", "u1_contract.json"}
assert before == after
assert all(content.endswith(b"\n") for content in after.values())
```

Drift oracle: edit `normalizer.json`, rerun, expect `FileExistsError` and no repair. Extra-file oracle: add `extra.txt`, rerun, expect rejection. Partial-publication oracle: monkeypatch the second staging JSON write to raise; final output directory must not exist and `.u1.staging-*` must be removed.

### Step 2 — Run RED suite

```bash
uv run pytest tests/workflows/test_universal_trade_rl_u1_runner.py -q
```

### Step 3 — Implement U0-style atomic publication

Before writing require `contract.normalizer_digest == normalizer.digest` and matching provenance digest. Use exact two filenames, canonical sorted JSON with one trailing newline, file flush+`fsync`, staging directory fsync where supported, then one directory-level `os.replace`. Existing final output succeeds only when exact filenames and canonical bytes match; otherwise fail closed without repair.

### Step 4 — Verify and commit

```bash
uv run pytest tests/workflows/test_universal_trade_rl_u1_runner.py tests/workflows/test_universal_trade_rl_universe_runner.py -q
uv run ruff check trade_rl/workflows/universal_trade_rl_u1_runner.py tests/workflows/test_universal_trade_rl_u1_runner.py
uv run mypy trade_rl/workflows/universal_trade_rl_u1_runner.py
git add trade_rl/workflows/universal_trade_rl_u1_runner.py tests/workflows/test_universal_trade_rl_u1_runner.py
git commit -m "feat: materialize Universal Trade RL U1 artifacts"
```

---

## Task 10: Add adversarial accounting/state falsification tests

**Files:** create `tests/rl/test_universal_trade_falsification.py`.

Import `make_u1_wrapper()` from `tests.rl.universal_trade_test_support`; do not import a private helper from another test module.

### Step 1 — Flat-market cost double-counting oracle

Use `price_drift=0`, high volume, and nonzero fee/spread/impact. Run actions `(0.8, 0.0, 0.0)` because signal delay means the third decision executes the flatten target. Assert final wealth is below initial and:

```python
sum(rewards) / 100.0 == pytest.approx(math.log(final / initial), abs=1e-10)
```

No additional turnover/cost reward penalty may appear.

### Step 2 — Funding inclusion oracle

Use `price_drift=0`, zero execution cost, `funding_rate_value=0.001`, `funding_due_from=6002`, high volume. Reset at 6000 and run four `0.8` actions. Assert `wrapper.unwrapped.hybrid.funding_pnl != 0.0`, final wealth differs from initial in the expected funding direction for the resulting long position, and reward telescopes exactly to final wealth. This specifically proves funding is included through maintained accounting rather than separately shaped in U1.

Keep the existing simulation funding boundary tests in the final compatibility command:

```bash
uv run pytest tests/simulation/test_funding_boundary_evidence.py -q
```

### Step 3 — Cash-reset state leak oracle

After nonzero positive and negative submissions, cash reset at another valid start index. Named policy state must return current weight, current submitted action, pending target active flag, pending notional, and position age to zero/default cash values.

### Step 4 — Combined signal-delay/risk/partial-fill oracle

Reuse low-volume + max-risk `0.35`, submit `0.60` then `0.80`, and assert current submitted/pending target is `0.80`, post-risk execution target `0.35`, realized weight has smaller absolute magnitude, and pending-order state remains separately accessible. Then submit a pending flat target and prove `pending_target_active=True` while `pending_target_weight==0.0`.

### Step 5 — Run and commit

```bash
uv run pytest tests/rl/test_universal_trade_falsification.py -q
```

If a falsification fails, fix the smallest responsible production module and rerun its targeted test before rerunning this suite. Do not skip or weaken the oracle.

```bash
git add tests/rl/test_universal_trade_falsification.py
git commit -m "test: falsify Universal Trade RL U1 boundaries"
```

---

## Task 11: Documentation, full verification, independent review, exact-HEAD CI

**Files:** modify `docs/UNIVERSAL_TRADE_RL.md`; modify `tests/test_architecture_contract.py` only if current architecture rules require it.

### Step 1 — Update maintained documentation

Document: U1 `NO-GO`; U0 published-source verification; one-symbol contract; allowed/forbidden features; policy request → signal-delay pending → risk projection → realized exposure; separate pending-order lifecycle; cash-only sampled reset; 0.25h decisions / 720h sampled horizon with override knobs forbidden; external truncation/no horizon feature/no forced liquidation; `unique_feature_event_time_v1`; equal-symbol moments; pure after-cost reward/telescoping; `normalizer.json` + `u1_contract.json`; Quality Gate; U2 `BASE_TRAINING` handoff with Development evaluation-only and Admission unopened.

### Step 2 — Run all U1/U0 targeted tests

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

Record exact counts.

### Step 3 — Run maintained compatibility/accounting tests

```bash
uv run pytest \
  tests/rl/test_environment_reduce_only_integration.py \
  tests/learning/test_rollout_execution_lifecycle.py \
  tests/learning/test_causal_alpha_v10_closed_loop.py \
  tests/learning/test_causal_alpha_v11_policy.py \
  tests/simulation/test_funding_boundary_evidence.py \
  tests/simulation/test_independent_accounting_oracle.py \
  -q
```

No unexplained regression is acceptable.

### Step 4 — Run static/architecture checks

```bash
uv run ruff check trade_rl tests
uv run ruff format --check trade_rl tests
uv run mypy trade_rl
uv run lint-imports
```

Also run the exact current CI MyPy/import commands if they differ; a partial affected-file invocation is not evidence of full static success.

### Step 5 — Run full suite and build

```bash
uv run pytest -q
uv build
```

Record exact pass/skip/fail counts and built distribution names. Do not commit generated distributions unless repository policy explicitly requires them.

### Step 6 — Self-review final diff/status/history

```bash
git diff b3a4cf0fd98f459ceb2262a4a759af83f9b1df3c...HEAD
git status --short
git log --oneline --decorate -n 40
```

If PR #426 no longer points to the pinned U0 base, this comparator is invalid: synchronize first, update the pinned SHA, and rerun targeted/full verification.

Review for manual-prior leakage, ID/raw nominal leakage, fake source digest path, Development/Admission fit leakage, row-count weighting, carried-event double counting, signal-delay/pending-order confusion, post-risk/realized aliasing, reward double counting, truncation leakage, non-cash reset leakage, duplicated risk/execution/accounting, unrelated refactor, debug code, temp workflow, generated artifact, secret.

### Step 7 — Independent/falsification review from the original spec

Answer each with code + test evidence, not implementation assertions:

```text
Can future data change Observation(t)?
Can explicit symbol/dataset identity reach a policy tensor?
Can a different published artifact be passed while claiming the manifest digest?
Can Development/Admission affect fitted statistics?
Can a longer-history Train symbol receive more normalizer weight?
Can a carried 4h/1d event be counted repeatedly?
Can pending flat target alias no pending target?
Can post-risk target alias realized weight under partial fill?
Can execution/funding cost be charged twice by reward?
Can sampled-end information enter policy state or create free liquidation?
Can TrendStrategy enter via reset state?
Did any U3-U6/Causal Alpha economic path change?
```

Fix every substantive finding and rerun from the smallest affected targeted test through Steps 2-6.

### Step 8 — Commit docs/architecture adjustment

```bash
git add docs/UNIVERSAL_TRADE_RL.md
if ! git diff --quiet -- tests/test_architecture_contract.py; then git add tests/test_architecture_contract.py; fi
git commit -m "docs: document Universal Trade RL U1 contract"
```

Do not create an empty commit.

### Step 9 — Invoke `superpowers:verification-before-completion`

Before any completion claim, re-check final HEAD, exact diff/status, targeted/full tests, static checks, build, Acceptance Criteria, falsification results, and remaining risks. “All tests green” alone is not completion evidence.

### Step 10 — Push Draft PR and verify exact-final-HEAD CI

PR body includes What / Why / Acceptance Criteria / Design decisions / Scope / Non-goals / Tests / Verification / Risks / Remaining limitations / Follow-up. Record:

```text
final HEAD SHA
U0 comparator SHA
CI workflow/run IDs on that exact HEAD
PostgreSQL Catalog result on that exact HEAD
Nautilus Capability result on that exact HEAD
required static/training checks on that exact HEAD
full-suite exact counts
coverage evidence if produced
unverified economic claims
remaining risks
```

Do not mark Ready until the Quality Gate below is satisfied. Do not merge without explicit user permission.

## U1 Quality Gate

U1 may be described as implementation-complete only when all are evidenced on the exact final HEAD:

1. Exactly-one-symbol wrapper and all fixed V1 clock/horizon override constraints are enforced.
2. Policy tensors exclude IDs, raw nominal values, manual priors, and horizon fraction.
3. Future-mutation causality test passes.
4. Missing/availability/staleness remain distinct.
5. Current policy request, signal-delay pending+active, post-risk target, realized weight, and pending-order lifecycle are separately observable.
6. Scalar action semantic is static and independent of dynamic risk cap.
7. Cash-only sampled reset prevents TrendStrategy prior injection.
8. U0 normalization firewall executes before filesystem/source inspection.
9. Published artifact digest is independently inspected and matched to U0 manifest before numeric data use.
10. Feature events are deduplicated by recovered source-event time, not by value/base-row position.
11. Equal-symbol first/second-moment oracle passes with unequal available history lengths.
12. Unavailable placeholder values and post-cutoff future events do not enter fit.
13. Admission-only generation drift preserves Train statistics digest while changing U0-bound artifact digest when Train artifacts/cutoff are identical.
14. Pure reward telescopes to realized after-cost wealth on normal, execution-cost, funding, and partial-fill paths.
15. Non-positive/non-finite wealth fails closed.
16. External truncation causes no forced liquidation/terminal reward and remaining horizon is absent from policy input.
17. `normalizer.json` + `u1_contract.json` publish atomically/canonically/idempotently and reject drift/partial output.
18. Existing U3-U6/Causal Alpha/accounting compatibility tests show no intended economic behavior change.
19. Targeted/property/integration/falsification tests, Ruff, format, required MyPy/static analysis, import architecture, full suite, and package build pass.
20. Self-review plus independent/falsification review has no unresolved substantive finding.
21. Required CI is green on the exact final HEAD, not an earlier commit.
22. Final report states explicitly that U1 does not prove RL learnability, zero-shot economics, profitability, Admission performance, real-market execution fidelity, or Production readiness.

## U2 Handoff

U2 begins only after a real production-candidate U0 generation and U1 artifacts are frozen. U2 must build `UniversalTradeRLFitPurpose.RL_TRAINING` provenance from U0 Train symbols and bind the frozen U1 contract digest into the existing `UniversalTradeRLRunStage.BASE_TRAINING` model/checkpoint identity. Development remains evaluation-only and Admission remains unopened.
