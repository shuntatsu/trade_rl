# Universal Trade RL U1 Observation / Action / Reward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Every production change follows Red → Green → Refactor. Do not weaken, skip, or rewrite an oracle merely to obtain Green.

**Goal:** Implement a one-symbol Universal Trade RL U1 contract with causal strategy-prior-free observations, one fixed-semantic scalar target-exposure action, U0 Train-only source-event-balanced normalization, and reward exactly reconciled to realized after-cost wealth.

**Architecture:** `ResidualMarketEnv` remains the sole Risk / Execution / Accounting authority. U1 adds only a read-only runtime snapshot, a U1 observation adapter, a strict action adapter, a new U1 normalizer class inside the existing `trade_rl.rl.universal_normalization` module, and a thin Gym wrapper. Existing `SymbolBalancedStandardNormalizer` behavior is preserved. Existing `EpisodeRoutedSingleInstrumentEnv` remains the future U2 multi-symbol episode router; U1 does not create a second router, and U2 must route U1 concrete environments with `instrument_context_provider=None` and `v4_context_provider=None` unless a later versioned spec explicitly permits context features.

**Tech Stack:** Python 3.12, NumPy, Gymnasium, dataclasses, existing `MarketDataset`, `SequenceObservationBuilder`, `ResidualMarketEnv`, PreTradeRisk, execution/accounting, published market dataset artifacts, canonical SHA-256 digests, pytest, Hypothesis where useful, Ruff, MyPy, Import Linter.

**Spec:** `docs/implementation-plans/specs/2026-08-31-universal-trade-rl-u1-observation-reward-design.md`

**Pinned design base:** `b3a4cf0fd98f459ceb2262a4a759af83f9b1df3c` (PR #426 U0 head at plan finalization). Before Task 1 production code, fetch PR #426 again. If its intended U0 head differs, synchronize this branch, update the pinned base in spec and plan, and repeat the design-diff review before coding.

## Implementation Quality Contract

**Objective**

- Produce a U1 environment contract in which the same numerical Observation/Action/Reward semantics apply to every concrete symbol.
- Preserve existing execution/accounting and existing Universal/Causal Alpha behavior.
- Bind normalization and U1 artifacts to the frozen U0 generation.

**Non-goals**

- No PPO/SAC/TD3 training.
- No BC/teacher policy.
- No Admission opening.
- No profitability, Sharpe, zero-shot economic, Production, or real-market-fidelity claim.
- No portfolio-level multi-symbol action allocation.
- No redesign of existing U3-U6/Causal Alpha economics.

**Invariants**

- One concrete environment controls exactly one symbol.
- `ResidualMarketEnv` is the only Risk / Execution / Accounting implementation.
- Policy input contains no symbol/dataset identity, raw nominal OHLC/volume/cash/quantity, `TrendTargets`, alpha output, factor prior, shadow/baseline state, ownership latch, or remaining-horizon fraction.
- `policy_requested_weight`, signal-delay `pending_target_weight` + active mask, `risk_projected_weight`, and realized `current_weight` remain distinct.
- Signal-delay pending target is not pending-order lifecycle.
- Reward equals `100 * log(W_after / W_before)` on realized hybrid wealth and adds no shaping.
- Normalization fit uses U0 Train only, equal symbol weighting, unique feature source events, and a frozen knowledge cutoff.

**Fixed U1 V1 runtime contract**

```text
ActionMode.TARGET_WEIGHT
target_weight_count = 1
ActionValidationMode.STRICT
accept_legacy_actions = false
decision_hours = 0.25
decision_every = None
episode_hours = 720.0
episode_bars = None
episode_hour_choices = ()
signal_delay_decisions = 1
episode_boundary_mode = EXTERNAL_TRUNCATION
finite_horizon_observation = false
liquidate_on_end = false
initial_state_modes = ("cash",)
structured_sequence_observation = true
sequence_windows = (("15m",96),("1h",168),("4h",120),("1d",60))
reward = pure net log growth, scale = 100
```

**Primary Failure Modes**

- future leakage;
- Development/Admission fit leakage;
- source artifact spoof/drift;
- row-count-weighted or carried-value-duplicated normalization;
- policy-request / signal-delay / post-risk / realized-state aliasing;
- missing-zero ambiguity;
- reward double-counting of fee/spread/impact/funding/borrow;
- free terminal liquidation or horizon leakage;
- TrendStrategy prior injection through reset;
- compatibility regression in old Universal/Causal Alpha paths.

**Test Oracle**

- exact tensor values/shapes/dtypes and source indices;
- exact runtime state transitions;
- published artifact digest and U0 manifest identity;
- independently calculated normalizer first/second moments;
- `sum(reward)/100 == log(final_wealth/initial_wealth)`;
- final Git diff, static checks, full suite, build, and exact-final-HEAD CI.

**Required Test Layers**

Unit + Property/Falsification + Integration + Compatibility + Static Analysis + Full Suite + Build + exact-HEAD CI.

**Quality Gate**

All checks in the final section of this plan must be evidenced on one final HEAD. Test Green alone is insufficient.

---

## File Map

**Create**

- `trade_rl/rl/universal_trade_contract.py` — U1 schemas, feature allowlist, fixed semantic contract/digest.
- `trade_rl/rl/universal_trade_action.py` — strict normalized scalar action parser.
- `trade_rl/rl/universal_trade_runtime.py` — immutable U1 runtime snapshot.
- `trade_rl/rl/universal_trade_observation.py` — strategy-prior-free U1 Dict observation.
- `trade_rl/rl/universal_trade_reward.py` — pure realized-wealth reward and reconciliation helper.
- `trade_rl/rl/universal_trade_environment.py` — thin U1 concrete environment wrapper.
- `trade_rl/workflows/universal_trade_rl_u1_contract.py` — frozen U1 artifact identity.
- `trade_rl/workflows/universal_trade_rl_u1_runner.py` — atomic `normalizer.json` + `u1_contract.json` publication.
- `tests/rl/universal_trade_test_support.py`
- `tests/workflows/universal_trade_rl_u1_test_support.py`
- U1 test files listed in the tasks below.

**Modify**

- `trade_rl/rl/environment.py` — one read-only U1 runtime accessor; no step behavior change.
- `trade_rl/rl/universal_normalization.py` — add U1 source-event normalizer; do not change `SymbolBalancedStandardNormalizer` semantics.
- `docs/UNIVERSAL_TRADE_RL.md`.
- `tests/test_architecture_contract.py` only if the repository architecture check requires explicit new-module allowlisting.

**Explicitly reuse, do not duplicate**

- `trade_rl/rl/sequence_observations.py::SequenceObservationBuilder`.
- `trade_rl/rl/universal_single_instrument_env.py::EpisodeRoutedSingleInstrumentEnv` for U2 routing only.
- PreTradeRisk / PortfolioRisk / MarketExecutor / stateful order lifecycle / BookState accounting.
- U0 universe access, fit provenance, run identity, and materialization conventions.

---

## Task 1: Freeze U1 policy contract and deterministic test support

**Files**

- Create `trade_rl/rl/universal_trade_contract.py`.
- Create `tests/rl/universal_trade_test_support.py`.
- Create `tests/rl/test_universal_trade_contract.py`.

**Produces**

```python
UNIVERSAL_TRADE_OBSERVATION_SCHEMA = "universal_trade_observation_v1"
UNIVERSAL_TRADE_ACTION_SCHEMA = "normalized_target_exposure_v1"
UNIVERSAL_TRADE_REWARD_SCHEMA = "universal_net_log_growth_reward_v1"
UNIVERSAL_TRADE_STATE_LAYOUT_SCHEMA = "universal_trade_policy_state_v1"
UNIVERSAL_TRADE_SEQUENCE_WINDOWS = (
    ("15m", 96),
    ("1h", 168),
    ("4h", 120),
    ("1d", 60),
)
```

and immutable `UniversalTradePolicyContract`.

### Step 1 — Write deterministic test support

Create `tests/rl/universal_trade_test_support.py` with these exact helper semantics:

```python
from __future__ import annotations

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
    borrow_rate_value: float = 0.0,
    borrow_available_value: bool = True,
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
        funding_rate=np.full((n_bars, 1), funding_rate_value, dtype=np.float64),
        funding_due=funding_due,
        tradable=np.ones((n_bars, 1), dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, 4), dtype=np.bool_),
        feature_names=tuple(spec.name for spec in make_u1_feature_specs()),
        global_feature_names=("market",),
        periods_per_year=35_040,
        feature_staleness_hours=staleness_hours,
        feature_staleness=np.minimum(staleness_hours / 24.0, 1.0),
        borrow_available=np.full((n_bars, 1), borrow_available_value, dtype=np.bool_),
        borrow_rate=np.full((n_bars, 1), borrow_rate_value, dtype=np.float64),
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
            sequence_windows=UNIVERSAL_TRADE_SEQUENCE_WINDOWS,
            liquidate_on_end=False,
            initial_state_modes=("cash",),
            accept_legacy_actions=False,
            action_validation_mode=ActionValidationMode.STRICT,
            execution_cost=(
                ExecutionCostConfig.zero() if execution_cost is None else execution_cost
            ),
        ),
    )
```

Import `UNIVERSAL_TRADE_SEQUENCE_WINDOWS` from the new contract module after it exists. Before that import is available, create the failing tests first and expect RED.

### Step 2 — Write contract RED tests

```python
import pytest

from trade_rl.data.contracts import FeatureKind, FeatureSpec
from trade_rl.rl.universal_trade_contract import (
    UNIVERSAL_TRADE_SEQUENCE_WINDOWS,
    UniversalTradePolicyContract,
)
from tests.rl.universal_trade_test_support import make_u1_feature_specs


def test_contract_freezes_windows() -> None:
    assert UNIVERSAL_TRADE_SEQUENCE_WINDOWS == (
        ("15m", 96), ("1h", 168), ("4h", 120), ("1d", 60)
    )


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
            feature_specs=(FeatureSpec(name="15m__forbidden", kind=kind),)
        )


def test_contract_requires_sequence_prefix_matching_resolved_timeframe() -> None:
    bad = FeatureSpec(name="15m__ret", kind=FeatureKind.LOG_RETURN, timeframe="1h")
    with pytest.raises(ValueError, match="prefix|timeframe"):
        UniversalTradePolicyContract(feature_specs=(bad,))


def test_contract_binds_feature_order() -> None:
    specs = make_u1_feature_specs()
    assert UniversalTradePolicyContract(feature_specs=specs).digest != (
        UniversalTradePolicyContract(feature_specs=tuple(reversed(specs))).digest
    )


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

Use the exact U1 FeatureKind allowlist from the spec. Validate every feature name begins with `<resolved_timeframe>__`, all names are unique, every resolved timeframe is one of the four fixed windows, `0 < policy_weight_scale <= 1`, reward scale is exactly `100.0`, and digest payload binds all fixed V1 runtime semantics above.

### Step 4 — Verify and commit

```bash
uv run pytest tests/rl/test_universal_trade_contract.py -q
uv run ruff check trade_rl/rl/universal_trade_contract.py tests/rl/universal_trade_test_support.py tests/rl/test_universal_trade_contract.py
uv run mypy trade_rl/rl/universal_trade_contract.py
git add trade_rl/rl/universal_trade_contract.py tests/rl/universal_trade_test_support.py tests/rl/test_universal_trade_contract.py
git commit -m "feat: define Universal Trade RL U1 contract"
```

---

## Task 2: Implement strict scalar target exposure

**Files**

- Create `trade_rl/rl/universal_trade_action.py`.
- Create `tests/rl/test_universal_trade_action.py`.

### Step 1 — Write RED tests

```python
import numpy as np
import pytest

from trade_rl.rl.universal_trade_action import parse_normalized_target_exposure


@pytest.mark.parametrize(
    ("raw", "scale", "expected"),
    ((-1.0, 1.0, -1.0), (-0.5, 1.0, -0.5), (0.0, 1.0, 0.0), (0.5, 0.4, 0.2), (1.0, 0.4, 0.4)),
)
def test_action_mapping_is_linear(raw: float, scale: float, expected: float) -> None:
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

Create immutable `NormalizedTargetExposureAction(normalized: float, policy_requested_weight: float)`. Accept exactly one finite scalar, reject values outside `[-1,1]`, validate `0 < policy_weight_scale <= 1`, and compute multiplication only. No clipping and no Risk call.

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

**Files**

- Create `trade_rl/rl/universal_trade_runtime.py`.
- Modify `trade_rl/rl/environment.py`.
- Modify `tests/rl/universal_trade_test_support.py` to add `make_runtime_snapshot()` after the production dataclass exists.
- Create `tests/rl/test_universal_trade_runtime.py`.

### Step 1 — Write RED state-transition tests

```python
import numpy as np
import pytest

from trade_rl.simulation.execution import ExecutionCostConfig
from tests.rl.universal_trade_test_support import make_u1_base_env, make_u1_market


def test_runtime_separates_submission_pending_risk_and_realized_weight() -> None:
    env = make_u1_base_env(
        dataset=make_u1_market(volume=100.0),
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


def test_pending_flat_is_distinct_from_no_pending_target() -> None:
    env = make_u1_base_env()
    env.reset(options={"start_idx": 6000, "initial_state_mode": "cash"})
    assert env.universal_trade_runtime_snapshot().pending_target_active is False
    env.step(np.asarray([0.0], dtype=np.float32))
    snapshot = env.universal_trade_runtime_snapshot()
    assert snapshot.pending_target_active is True
    assert snapshot.pending_target_weight == pytest.approx(0.0)
```

### Step 2 — Run RED

```bash
uv run pytest tests/rl/test_universal_trade_runtime.py -q
```

### Step 3 — Implement immutable snapshot/accessor

Map maintained state exactly:

```text
policy_requested_weight = _previous_action[0]
pending_target_active   = _pending_hybrid_target is not None
pending_target_weight   = 0.0 if None else _pending_hybrid_target[0]
risk_projected_weight   = _execution_state.requested_weights[0]
current_weight          = hybrid.weights[0]
execution_cost_rate     = _execution_state.execution_cost[0]
position_age_hours      = _execution_state.position_age[0] * dataset.bar_hours
```

Use `_pending_order_observation_state()` for pending-order lifecycle. Convert its bar ages/delay/expiry distance to hours. Derive drawdown/gross/net/cash/risk/margin from maintained hybrid/Risk state. Return scalar copies only. Do not add a second risk-projected state variable.

### Step 4 — Add neutral runtime helper

`make_runtime_snapshot(**overrides)` must instantiate every U1 runtime field with deterministic neutral values and use `dataclasses.replace` for overrides. Later tests import this helper rather than inventing incomplete snapshots.

### Step 5 — Verify and commit

```bash
uv run pytest tests/rl/test_universal_trade_runtime.py -q
uv run pytest tests/rl/test_environment_reduce_only_integration.py tests/learning/test_rollout_execution_lifecycle.py -q
uv run ruff check trade_rl/rl/environment.py trade_rl/rl/universal_trade_runtime.py tests/rl/universal_trade_test_support.py tests/rl/test_universal_trade_runtime.py
uv run mypy trade_rl/rl/environment.py trade_rl/rl/universal_trade_runtime.py
git add trade_rl/rl/environment.py trade_rl/rl/universal_trade_runtime.py tests/rl/universal_trade_test_support.py tests/rl/test_universal_trade_runtime.py
git commit -m "feat: expose Universal Trade RL runtime state"
```

---

## Task 4: Add U1 source-event normalization to the existing universal normalizer module

**Files**

- Modify `trade_rl/rl/universal_normalization.py`.
- Create `tests/rl/test_universal_trade_u1_normalization.py`.

**Compatibility rule:** do not alter `SymbolBalancedStandardNormalizer.fit()` or `.transform()` behavior. Add separate U1 types/functions in the same module.

**Produces**

```python
@dataclass(frozen=True, slots=True)
class UniversalTradePublishedSource:
    symbol: str
    artifact_root: Path

@dataclass(frozen=True, slots=True)
class UniversalTradeChannelStatistics:
    timeframe: str
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    per_symbol_sample_counts: tuple[tuple[str, tuple[int, ...]], ...]

@dataclass(frozen=True, slots=True)
class UniversalTradeSequenceNormalizer:
    ...
    statistics_digest: str
    digest: str
    provenance_digest: str
    def transform(
        self,
        timeframe: str,
        values: np.ndarray,
        available: np.ndarray,
        *,
        feature_names: tuple[str, ...],
    ) -> np.ndarray: ...
```

and `fit_universal_trade_sequence_normalizer(...)`.

### Step 1 — Write local U0 source helpers

Inside `tests/rl/test_universal_trade_u1_normalization.py`, define `_source(...)` and `_manifest(...)` using keyword construction of `UniversalTradeRLSymbolSource`. Manifest roles are Train `(BTCUSDT, ETHUSDT)`, Development `(LINKUSDT,)`, Admission `(AVAXUSDT,)`.

### Step 2 — Write firewall-before-filesystem RED test

```python
def test_scope_fails_before_missing_artifact_path_is_touched() -> None:
    train = (
        UniversalTradeRLSymbolSource(
            symbol="BTCUSDT", dataset_digest="b" * 64,
            first_timestamp_ns=1, last_timestamp_ns=2, row_count=2,
        ),
        UniversalTradeRLSymbolSource(
            symbol="ETHUSDT", dataset_digest="e" * 64,
            first_timestamp_ns=1, last_timestamp_ns=2, row_count=2,
        ),
    )
    manifest = _manifest(train)
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

Publish BTC artifact A, different BTC artifact B, and ETH. Build the U0 manifest from A+ETH, then pass B+ETH to fit. Require `ValueError` because the independently inspected `PublishedDatasetArtifact.artifact_digest` for B does not equal the manifest entry `dataset_digest`.

### Step 4 — Write equal-symbol/unavailable-value RED test

Use BTC with 5800 rows and ETH with 6000 rows. Make one BTC 15m value `1e9` but set that exact feature availability false before re-binding content identity and publishing. Independently compute, for each symbol, mean and mean(square) over available feature source events; then assert:

```python
expected_mean = 0.5 * (mu_btc + mu_eth)
expected_q = 0.5 * (q_btc + q_eth)
expected_scale = math.sqrt(max(expected_q - expected_mean * expected_mean, 0.0))
assert stats.mean[0] == pytest.approx(expected_mean)
assert stats.scale[0] == pytest.approx(expected_scale)
```

The oracle must differ from concatenated-row weighting.

### Step 5 — Write carried-event de-duplication RED test

For fixture `4h__ret`, source event changes every 16 base rows. Fit through the last row and assert:

```python
expected = (btc.n_bars - 1) // 16 + 1
counts = dict(normalizer.statistics_for("4h").per_symbol_sample_counts)
assert counts["BTCUSDT"][0] == expected
assert expected < btc.n_bars
```

### Step 6 — Write knowledge-cutoff RED test

Make post-row-5000 15m values extremely large but available; fit with cutoff at row 5000. Assert statistics equal independent moments using recovered source events at or before that cutoff only.

### Step 7 — Write generation identity RED test

Two manifests share identical Train artifacts/cutoff but differ only in Admission digest:

```python
assert normalizer_a.statistics_digest == normalizer_b.statistics_digest
assert normalizer_a.digest != normalizer_b.digest
```

### Step 8 — Run RED

```bash
uv run pytest tests/rl/test_universal_trade_u1_normalization.py -q
```

### Step 9 — Implement fit pipeline

Exact order:

```text
canonicalize source symbols
-> access.require_normalization_scope(symbols)
-> build FEATURE_NORMALIZATION provenance
-> inspect_published_market_dataset_artifact(root)
-> compare inspected artifact_digest to U0 manifest entry
-> load_market_dataset_artifact(root)
-> require one expected symbol and exact U1 feature order
-> for each feature, select available rows and recover source_event_time_ns
-> require source_event_time_ns <= knowledge_cutoff_ns
-> de-duplicate by (feature index, source_event_time_ns), never by value
-> compute per-symbol mean and mean(square)
-> equal-weight symbols
-> variance=max(second_mean-mean^2,0), scale floor
-> statistics_digest
-> U0/provenance-bound artifact digest
```

Recover event time as:

```python
_NS_PER_HOUR = 3_600_000_000_000
source_event_time_ns = timestamp_ns - np.rint(
    staleness_hours * _NS_PER_HOUR
).astype(np.int64)
```

The new class implements the existing `SequenceNormalizerProtocol` transform signature. `available=False` outputs zero after transform; staleness is not normalized by this class.

### Step 10 — Verify legacy + U1 behavior and commit

```bash
uv run pytest tests/rl/test_universal_trade_u1_normalization.py tests/workflows/test_universal_trade_rl_data_provenance.py -q
uv run pytest tests/rl -q -k "universal and normaliz"
uv run ruff check trade_rl/rl/universal_normalization.py tests/rl/test_universal_trade_u1_normalization.py
uv run mypy trade_rl/rl/universal_normalization.py
git add trade_rl/rl/universal_normalization.py tests/rl/test_universal_trade_u1_normalization.py
git commit -m "feat: add U1 source-event universal normalizer"
```

---

## Task 5: Build strategy-prior-free causal U1 observation

**Files**

- Create `trade_rl/rl/universal_trade_observation.py`.
- Create `tests/rl/test_universal_trade_observation.py`.

### Step 1 — Write exact-layout RED test

Use `SequenceObservationBuilder` directly as the market source. At index 6000, assert exact keys:

```text
sequence_15m_values / available / staleness
sequence_1h_values / available / staleness
sequence_4h_values / available / staleness
sequence_1d_values / available / staleness
policy_state
```

Assert `source_indices <= 6000`, `observation_space.contains(obs)`, and no policy-state field name contains `trend`, `alpha`, `shadow`, `baseline`, `remaining`, `symbol`, or `dataset`.

### Step 2 — Write deterministic state-transform RED test

With runtime values `position_age_hours=24`, pending order ages `48/24/72`, `mark_index_basis=0.01`, `borrow_rate=0.02`, assert named state contains:

```python
assert state["position_age_days"] == pytest.approx(np.log1p(1.0))
assert state["pending_order_age_days"] == pytest.approx(np.log1p(2.0))
assert state["pending_order_eligible_delay_days"] == pytest.approx(np.log1p(1.0))
assert state["pending_order_expiry_distance_days"] == pytest.approx(np.log1p(3.0))
assert state["mark_index_basis"] == pytest.approx(np.tanh(1.0))
assert state["borrow_rate"] == pytest.approx(np.tanh(0.02))
```

### Step 3 — Write future-mutation RED test

Create two datasets identical through `t=6000`, mutate all later features/OHLC/volume/funding/mark/index in one, and assert every U1 tensor at `t` is exactly equal.

### Step 4 — Write symbol and price-unit invariance RED tests

- Rename dataset symbol only: tensors equal.
- `price_scale=1` vs `price_scale=1000` with identical dimensionless feature/runtime inputs: tensors equal within `atol=1e-7`.

### Step 5 — Write true-zero vs unavailable-zero RED test

Set the current 15m feature value to `0.0` in both datasets but set availability false in one. Assert value tensor may match while availability tensor differs.

### Step 6 — Implement builder

Use `SequenceObservationBuilder.build()`; never slice legacy `ObservationBuilder`. The U1 schema digest must **not** reuse `SequenceObservationBuilder.schema_digest()` because that digest includes concrete `dataset_id`, and must not expose the builder layout’s concrete symbol text as a policy identity. Build a U1 digest from generic one-symbol axis semantics, ordered contract FeatureSpecs/windows, output dtypes/shapes, and exact policy-state field/transforms.

If a `UniversalTradeSequenceNormalizer` is supplied, call its `transform(timeframe, values, available, feature_names=...)` for sequence values only. Availability, staleness, source causality, and policy state remain unchanged.

### Step 7 — Verify and commit

```bash
uv run pytest tests/rl/test_universal_trade_observation.py -q
uv run ruff check trade_rl/rl/universal_trade_observation.py tests/rl/test_universal_trade_observation.py
uv run mypy trade_rl/rl/universal_trade_observation.py
git add trade_rl/rl/universal_trade_observation.py tests/rl/test_universal_trade_observation.py
git commit -m "feat: add Universal Trade RL U1 observation"
```

---

## Task 6: Implement pure realized-wealth reward oracle

**Files**

- Create `trade_rl/rl/universal_trade_reward.py`.
- Create `tests/rl/test_universal_trade_reward.py`.

### Step 1 — Write RED tests

```python
import math
import pytest

from trade_rl.rl.universal_trade_reward import (
    reconcile_universal_trade_reward,
    universal_net_log_growth_reward,
)


def test_reward_telescopes_to_final_wealth() -> None:
    values = (100.0, 101.0, 99.5, 103.25)
    rewards = tuple(
        universal_net_log_growth_reward(before_value=a, after_value=b)
        for a, b in zip(values, values[1:], strict=True)
    )
    assert sum(rewards) / 100.0 == pytest.approx(math.log(values[-1] / values[0]))
    reconcile_universal_trade_reward(
        rewards=rewards, initial_value=values[0], final_value=values[-1]
    )


@pytest.mark.parametrize("bad", (0.0, -1.0, float("inf"), float("nan")))
def test_reward_rejects_invalid_wealth(bad: float) -> None:
    with pytest.raises(ValueError):
        universal_net_log_growth_reward(before_value=bad, after_value=100.0)
    with pytest.raises(ValueError):
        universal_net_log_growth_reward(before_value=100.0, after_value=bad)
```

### Step 2 — Run RED, implement, verify

After finite/positive validation:

```python
return scale * math.log(after_value / before_value)
```

`reconcile_universal_trade_reward` compares `sum(rewards)/scale` to `log(final/initial)` with `rel_tol=0.0` and explicit `atol`, raising on mismatch.

```bash
uv run pytest tests/rl/test_universal_trade_reward.py -q
uv run ruff check trade_rl/rl/universal_trade_reward.py tests/rl/test_universal_trade_reward.py
uv run mypy trade_rl/rl/universal_trade_reward.py
git add trade_rl/rl/universal_trade_reward.py tests/rl/test_universal_trade_reward.py
git commit -m "feat: add Universal Trade RL wealth reward"
```

---

## Task 7: Add thin U1 concrete environment wrapper

**Files**

- Create `trade_rl/rl/universal_trade_environment.py`.
- Modify `tests/rl/universal_trade_test_support.py` to add `make_u1_wrapper()`.
- Create `tests/rl/test_universal_trade_environment.py`.

### Step 1 — Write constructor RED tests

A valid `make_u1_base_env()` must be accepted. Create valid mutated `ResidualMarketEnvConfig` instances and require wrapper rejection for each drift:

```text
accept_legacy_actions = true
action_validation_mode = CLIP
signal_delay_decisions = 0
decision_hours != 0.25
decision_every = 1
episode_hours != 720
episode_bars = 2880
episode_hour_choices = (720.0,)
finite_horizon_observation = true
liquidate_on_end = true
initial_state_modes = ("baseline",)
wrong sequence windows
FINITE_HORIZON_TERMINATION (+ finite_horizon_observation=true)
non-pure RewardConfig
n_symbols != 1
```

Use `dataclasses.replace(env.config, ...)` only with combinations that pass `ResidualMarketEnvConfig.__post_init__`; the wrapper, not dataclass construction, is the intended oracle.

### Step 2 — Write signal-delay integration RED test

With max risk 0.40, submit `0.60`, then `0.80`. The returned second observation must show current policy request/pending target `0.80` while the executed post-risk target is `0.40`.

### Step 3 — Write reward drift-guard RED test

Before each delegated step capture `base.hybrid.portfolio_value`; after it capture again; wrapper reward must equal `100*log(after/before)` within `1e-10`. The wrapper implementation must independently compare delegated base reward with the same oracle and fail if future base reward shaping drifts.

### Step 4 — Write cash-only/truncation RED tests

- Explicit reset request `initial_state_mode="baseline"` fails.
- Repeated actions to horizon end yield `terminated=False`, `truncated=True`, no end liquidation, and no horizon field in policy state.

### Step 5 — Implement wrapper

`reset()` delegates once, allows cash only, ignores legacy base observation, obtains U1 runtime snapshot, builds U1 observation. `step()`:

```text
strict U1 scalar parse
-> record realized hybrid wealth before
-> base_env.step([policy_requested_weight]) exactly once
-> record wealth after
-> compute U1 reward
-> assert delegated base reward matches U1 reward
-> build runtime snapshot and U1 observation
-> return base terminated/truncated/info unchanged
```

Never call Risk/Execution/Accounting directly.

### Step 6 — Add reusable test helper and U2 router compatibility probe

Add `make_u1_wrapper(base_env=None, normalizer=None)` to test support. Add a test that constructs `EpisodeRoutedSingleInstrumentEnv` with U1 concrete-env factory, `instrument_context_provider=None`, `v4_context_provider=None`, and two bindings; assert routed observation/action spaces are compatible and no `instrument_context`, V4 context, or concrete symbol key appears in policy observation. This proves U2 can reuse the existing router without a new routing implementation.

### Step 7 — Verify and commit

```bash
uv run pytest tests/rl/test_universal_trade_environment.py tests/rl/test_universal_trade_runtime.py -q
uv run pytest tests/rl/test_environment_reduce_only_integration.py tests/learning/test_rollout_execution_lifecycle.py -q
uv run ruff check trade_rl/rl/universal_trade_environment.py tests/rl/universal_trade_test_support.py tests/rl/test_universal_trade_environment.py
uv run mypy trade_rl/rl/universal_trade_environment.py
git add trade_rl/rl/universal_trade_environment.py tests/rl/universal_trade_test_support.py tests/rl/test_universal_trade_environment.py
git commit -m "feat: add Universal Trade RL U1 environment"
```

---

## Task 8: Bind U1 artifact identity to U0 and runtime economics

**Files**

- Create `trade_rl/workflows/universal_trade_rl_u1_contract.py`.
- Create `tests/workflows/universal_trade_rl_u1_test_support.py`.
- Create `tests/workflows/test_universal_trade_rl_u1_contract.py`.

### Step 1 — Create explicit workflow fixture builder

`build_u1_workflow_fixture(root: Path, *, admission_digest: str = "a"*64)` must publish BTC/ETH test artifacts, build a U0 manifest + `UNIVERSE_MATERIALIZATION` identity, fit a U1 normalizer, build a valid base env/observation schema, then call `build_universal_trade_rl_u1_contract`. Return a frozen `U1WorkflowFixture` with every object as a named field. No Task 8/9 test may rely on undefined pytest fixtures.

### Step 2 — Write U0/generation RED tests

```python
def test_u1_contract_binds_u0_and_normalizer(tmp_path) -> None:
    f = build_u1_workflow_fixture(tmp_path)
    assert f.u1_contract.universe_manifest_digest == f.manifest.digest
    assert f.u1_contract.u0_identity_digest == f.u0_identity.digest
    assert f.u1_contract.normalizer_digest == f.normalizer.digest
    assert f.u1_contract.production_status == "NO-GO"


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

### Step 3 — Write payload-tamper RED test

```python
def test_u1_contract_rejects_tampered_payload(tmp_path) -> None:
    f = build_u1_workflow_fixture(tmp_path)
    payload = f.u1_contract.to_payload()
    payload["runtime_config_digest"] = "f" * 64
    with pytest.raises(ValueError, match="digest"):
        UniversalTradeRLU1Contract.from_payload(payload)
```

### Step 4 — Write identity-field sensitivity RED test

```python
@pytest.mark.parametrize(
    "field",
    (
        "policy_contract_digest",
        "normalizer_digest",
        "normalizer_provenance_digest",
        "observation_schema_digest",
        "state_layout_digest",
        "runtime_config_digest",
        "execution_policy_digest",
        "pretrade_risk_digest",
        "portfolio_risk_digest",
    ),
)
def test_every_semantic_digest_changes_u1_identity(tmp_path, field: str) -> None:
    f = build_u1_workflow_fixture(tmp_path)
    changed = dataclasses.replace(f.u1_contract, **{field: "f" * 64}, digest="")
    assert changed.digest != f.u1_contract.digest
```

### Step 5 — Implement strict builder/codec

Validate U0 stage, U0 manifest identity, Train-only normalizer provenance, base U1 fixed runtime contract, and `production_status="NO-GO"`. Runtime digest binds all clock/horizon override knobs, action strictness, signal delay, boundary semantics, reset mode, and sequence windows. Bind execution policy and canonical pretrade/portfolio risk config digests. Do not bind concrete symbol text as policy semantics.

### Step 6 — Verify and commit

```bash
uv run pytest tests/workflows/test_universal_trade_rl_u1_contract.py tests/workflows/test_universal_trade_rl_run_identity.py tests/workflows/test_universal_trade_rl_data_provenance.py -q
uv run ruff check trade_rl/workflows/universal_trade_rl_u1_contract.py tests/workflows/universal_trade_rl_u1_test_support.py tests/workflows/test_universal_trade_rl_u1_contract.py
uv run mypy trade_rl/workflows/universal_trade_rl_u1_contract.py
git add trade_rl/workflows/universal_trade_rl_u1_contract.py tests/workflows/universal_trade_rl_u1_test_support.py tests/workflows/test_universal_trade_rl_u1_contract.py
git commit -m "feat: bind Universal Trade RL U1 identity"
```

---

## Task 9: Materialize U1 artifacts atomically

**Files**

- Create `trade_rl/workflows/universal_trade_rl_u1_runner.py`.
- Create `tests/workflows/test_universal_trade_rl_u1_runner.py`.

### Step 1 — Write exact-output/idempotency RED test

```python
def test_materialization_is_exact_and_idempotent(tmp_path) -> None:
    f = build_u1_workflow_fixture(tmp_path / "fixture")
    output = tmp_path / "u1"
    materialize_universal_trade_rl_u1(
        contract=f.u1_contract, normalizer=f.normalizer, output_root=output
    )
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    materialize_universal_trade_rl_u1(
        contract=f.u1_contract, normalizer=f.normalizer, output_root=output
    )
    after = {path.name: path.read_bytes() for path in output.iterdir()}
    assert set(before) == {"normalizer.json", "u1_contract.json"}
    assert before == after
    assert all(content.endswith(b"\n") for content in after.values())
```

### Step 2 — Write drift/extra-file RED tests

```python
def test_materialization_rejects_existing_drift(tmp_path) -> None:
    f = build_u1_workflow_fixture(tmp_path / "fixture")
    output = tmp_path / "u1"
    materialize_universal_trade_rl_u1(
        contract=f.u1_contract, normalizer=f.normalizer, output_root=output
    )
    (output / "normalizer.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="drift|existing"):
        materialize_universal_trade_rl_u1(
            contract=f.u1_contract, normalizer=f.normalizer, output_root=output
        )


def test_materialization_rejects_extra_file(tmp_path) -> None:
    f = build_u1_workflow_fixture(tmp_path / "fixture")
    output = tmp_path / "u1"
    materialize_universal_trade_rl_u1(
        contract=f.u1_contract, normalizer=f.normalizer, output_root=output
    )
    (output / "extra.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="drift|existing"):
        materialize_universal_trade_rl_u1(
            contract=f.u1_contract, normalizer=f.normalizer, output_root=output
        )
```

### Step 3 — Write partial-publication RED test

```python
def test_write_failure_never_publishes_partial_final_directory(
    tmp_path, monkeypatch
) -> None:
    f = build_u1_workflow_fixture(tmp_path / "fixture")
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
        runner.materialize_universal_trade_rl_u1(
            contract=f.u1_contract, normalizer=f.normalizer, output_root=output
        )
    assert not output.exists()
    assert not tuple(tmp_path.glob(".u1.staging-*"))
```

### Step 4 — Implement U0-style publication

Require contract normalizer/provenance digests to match the normalizer before writing. Canonical JSON: sorted keys, compact separators, UTF-8, one trailing newline. Write both files under one staging directory, flush/fsync each, fsync directory where supported, publish directory once with `os.replace`. Existing output succeeds only when exact two filenames and canonical bytes match; otherwise fail closed without repair.

### Step 5 — Verify and commit

```bash
uv run pytest tests/workflows/test_universal_trade_rl_u1_runner.py tests/workflows/test_universal_trade_rl_universe_runner.py -q
uv run ruff check trade_rl/workflows/universal_trade_rl_u1_runner.py tests/workflows/test_universal_trade_rl_u1_runner.py
uv run mypy trade_rl/workflows/universal_trade_rl_u1_runner.py
git add trade_rl/workflows/universal_trade_rl_u1_runner.py tests/workflows/test_universal_trade_rl_u1_runner.py
git commit -m "feat: materialize Universal Trade RL U1 artifacts"
```

---

## Task 10: Adversarial accounting and state falsification

**Files**

- Create `tests/rl/test_universal_trade_falsification.py`.

Import `make_u1_wrapper`, `make_u1_base_env`, `make_u1_market`, and `make_runtime_snapshot` from shared test support.

### Step 1 — Execution-cost double-counting oracle

```python
def test_flat_market_cost_reward_equals_realized_wealth_only() -> None:
    execution = ExecutionCostConfig(
        fee_rate=0.0005,
        spread_rate=0.0002,
        impact_rate=0.0002,
        max_participation_rate=1.0,
    )
    wrapper = make_u1_wrapper(
        base_env=make_u1_base_env(
            dataset=make_u1_market(price_drift=0.0, volume=1_000_000_000.0),
            execution_cost=execution,
        )
    )
    wrapper.reset(options={"start_idx": 6000, "initial_state_mode": "cash"})
    initial = wrapper.unwrapped.hybrid.portfolio_value
    rewards = []
    for action in (0.8, 0.0, 0.0):
        _, reward, _, _, _ = wrapper.step(np.asarray([action], dtype=np.float32))
        rewards.append(reward)
    final = wrapper.unwrapped.hybrid.portfolio_value
    assert final < initial
    assert sum(rewards) / 100.0 == pytest.approx(
        math.log(final / initial), abs=1e-10
    )
```

### Step 2 — Funding oracle

```python
def test_funding_is_accounted_once_in_wealth_reward() -> None:
    wrapper = make_u1_wrapper(
        base_env=make_u1_base_env(
            dataset=make_u1_market(
                price_drift=0.0,
                volume=1_000_000_000.0,
                funding_rate_value=0.001,
                funding_due_from=6002,
            ),
            execution_cost=ExecutionCostConfig.zero(),
        )
    )
    wrapper.reset(options={"start_idx": 6000, "initial_state_mode": "cash"})
    initial = wrapper.unwrapped.hybrid.portfolio_value
    rewards = []
    for _ in range(4):
        _, reward, _, _, _ = wrapper.step(np.asarray([0.8], dtype=np.float32))
        rewards.append(reward)
    final = wrapper.unwrapped.hybrid.portfolio_value
    assert wrapper.unwrapped.hybrid.funding_pnl < 0.0
    assert final < initial
    assert sum(rewards) / 100.0 == pytest.approx(
        math.log(final / initial), abs=1e-10
    )
```

### Step 3 — Borrow oracle

```python
def test_short_borrow_is_accounted_once_in_wealth_reward() -> None:
    execution = dataclasses.replace(
        ExecutionCostConfig.zero(), borrow_rate_multiplier=1.0
    )
    wrapper = make_u1_wrapper(
        base_env=make_u1_base_env(
            dataset=make_u1_market(
                price_drift=0.0,
                volume=1_000_000_000.0,
                borrow_rate_value=0.365,
                borrow_available_value=True,
            ),
            execution_cost=execution,
        )
    )
    wrapper.reset(options={"start_idx": 6000, "initial_state_mode": "cash"})
    initial = wrapper.unwrapped.hybrid.portfolio_value
    rewards = []
    for _ in range(4):
        _, reward, _, _, _ = wrapper.step(np.asarray([-0.8], dtype=np.float32))
        rewards.append(reward)
    final = wrapper.unwrapped.hybrid.portfolio_value
    assert wrapper.unwrapped.hybrid.borrow_cost > 0.0
    assert final < initial
    assert sum(rewards) / 100.0 == pytest.approx(
        math.log(final / initial), abs=1e-10
    )
```

### Step 4 — Reset-state leakage oracle

```python
def test_cash_reset_clears_previous_episode_policy_and_execution_state() -> None:
    wrapper = make_u1_wrapper()
    wrapper.reset(options={"start_idx": 6000, "initial_state_mode": "cash"})
    wrapper.step(np.asarray([0.7], dtype=np.float32))
    wrapper.step(np.asarray([-0.4], dtype=np.float32))
    obs, _ = wrapper.reset(options={"start_idx": 6100, "initial_state_mode": "cash"})
    state = wrapper.policy_state_dict(obs)
    assert state["current_weight"] == pytest.approx(0.0)
    assert state["policy_requested_weight"] == pytest.approx(0.0)
    assert state["pending_target_active"] == pytest.approx(0.0)
    assert state["pending_notional_ratio"] == pytest.approx(0.0)
    assert state["position_age_days"] == pytest.approx(0.0)
```

### Step 5 — Four-stage exposure/pending-flat oracle

Use low-volume market + max risk `0.35`, submit `0.60` then `0.80`, and assert current request/pending is `0.80`, post-risk is `0.35`, realized absolute weight is smaller, and pending-order state remains separately available. Submit `0.0` next and assert `pending_target_active=True` with `pending_target_weight==0.0`.

### Step 6 — Run and commit

```bash
uv run pytest tests/rl/test_universal_trade_falsification.py -q
```

Do not skip or weaken a failing falsification. Fix the smallest responsible module, rerun its targeted tests, then rerun this suite.

```bash
git add tests/rl/test_universal_trade_falsification.py
git commit -m "test: falsify Universal Trade RL U1 boundaries"
```

---

## Task 11: Documentation, full verification, independent review, exact-HEAD CI

**Files**

- Modify `docs/UNIVERSAL_TRADE_RL.md`.
- Modify `tests/test_architecture_contract.py` only if current architecture rules require it.

### Step 1 — Update maintained U1 documentation

Document U1 `NO-GO`, U0 published-source verification, one-symbol semantics, allowed/forbidden features, four exposure stages, separate pending-order state, 15m decisions / 720h sampled horizon and forbidden override knobs, cash-only reset, external truncation, source-event/equal-symbol normalization, pure after-cost reward including fee/spread/impact/funding/borrow, U1 artifacts, Quality Gate, and U2 handoff through existing `EpisodeRoutedSingleInstrumentEnv` with context providers disabled for U1 V1.

### Step 2 — Run targeted U1/U0 suite

```bash
uv run pytest \
  tests/rl/test_universal_trade_contract.py \
  tests/rl/test_universal_trade_action.py \
  tests/rl/test_universal_trade_runtime.py \
  tests/rl/test_universal_trade_u1_normalization.py \
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

Record exact passed/skipped/failed counts.

### Step 3 — Run compatibility/accounting suite

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

Additionally run any existing tests that import `SymbolBalancedStandardNormalizer` or `EpisodeRoutedSingleInstrumentEnv`; their old behavior must remain unchanged.

### Step 4 — Run static/architecture checks

```bash
uv run ruff check trade_rl tests
uv run ruff format --check trade_rl tests
uv run mypy trade_rl
uv run lint-imports
```

Also execute the exact current CI MyPy/import/static commands if the repository workflow uses a different invocation.

### Step 5 — Run full suite and build

```bash
uv run pytest -q
uv build
```

Record exact result counts and built distribution names. Do not commit generated distributions unless repository policy requires them.

### Step 6 — Self-review exact diff/status/history

```bash
git diff b3a4cf0fd98f459ceb2262a4a759af83f9b1df3c...HEAD
git status --short
git log --oneline --decorate -n 40
```

If PR #426 no longer points to the pinned U0 head, stop using this comparator, synchronize first, update the pinned SHA, and rerun verification.

Review explicitly for manual-prior leakage, identity/raw nominal leakage, untrusted source digest, Development/Admission fit leakage, row-count weighting, carried-event duplication, signal-delay/pending-order confusion, post-risk/realized aliasing, fee/funding/borrow double counting, truncation leakage, non-cash reset leakage, duplicated execution/accounting, unrelated refactor, debug/temp/generated files, and secrets.

### Step 7 — Independent/falsification review

Rebuild the oracle from the spec and answer with concrete code/test evidence:

```text
Can future data change Observation(t)?
Can symbol/dataset identity reach a tensor?
Can a different published dataset be passed while claiming the U0 manifest identity?
Can Development/Admission affect fitted statistics?
Can a longer-history Train symbol receive a larger normalizer vote?
Can carried 4h/1d events be counted repeatedly?
Can pending flat target alias no pending target?
Can risk_projected_weight alias current_weight under partial fill?
Can fee/spread/impact/funding/borrow be charged twice by reward?
Can sampled-end information enter policy state or create free liquidation?
Can TrendStrategy enter through reset state?
Did existing SymbolBalancedStandardNormalizer, EpisodeRoutedSingleInstrumentEnv, U3-U6, or Causal Alpha economics change?
```

Fix substantive findings and rerun from the smallest targeted test through full verification.

### Step 8 — Commit documentation/architecture adjustment

```bash
git add docs/UNIVERSAL_TRADE_RL.md
if ! git diff --quiet -- tests/test_architecture_contract.py; then git add tests/test_architecture_contract.py; fi
git commit -m "docs: document Universal Trade RL U1 contract"
```

Do not create an empty commit.

### Step 9 — Invoke verification-before-completion

Load `superpowers:verification-before-completion` before any completion claim. Re-check final HEAD, diff, status, Acceptance Criteria, targeted/full tests, static analysis, build, falsification review, and remaining risks.

### Step 10 — Push Draft PR and verify exact final HEAD CI

PR body must include What / Why / Acceptance Criteria / Design decisions / Scope / Non-goals / Tests / Verification / Risks / Remaining limitations / Follow-up. Record final HEAD SHA, U0 comparator SHA, exact-head CI workflow/run IDs, PostgreSQL Catalog, Nautilus Capability, required training/static jobs, full-suite counts, coverage evidence if produced, unverified economic claims, and remaining risks. Do not mark Ready before the Quality Gate; do not merge without explicit user permission.

---

## U1 Quality Gate

U1 may be described as implementation-complete only when all are evidenced on the exact final HEAD:

1. Exactly-one-symbol wrapper and all fixed action/clock/horizon override constraints are enforced.
2. Policy tensors exclude IDs, raw nominal values, manual priors, and horizon fraction.
3. Future-mutation causality test passes.
4. Missing value / availability / staleness remain distinct.
5. Current policy request, signal-delay pending+active, post-risk target, realized weight, and pending-order lifecycle are separately observable.
6. Scalar action semantic is static and independent of dynamic risk cap.
7. Cash-only reset prevents TrendStrategy prior injection.
8. U0 normalization firewall executes before filesystem/source inspection.
9. Published artifact digest is independently inspected and matched to U0 manifest before numeric data use.
10. Feature events are deduplicated by recovered source-event time, not by value or base-row position.
11. Equal-symbol first/second-moment oracle passes with unequal available history lengths.
12. Unavailable placeholders and post-cutoff future events do not enter fit.
13. Admission-only generation drift preserves Train statistics digest while changing U0-bound artifact digest when Train artifacts/cutoff are identical.
14. Pure reward telescopes to realized after-cost wealth on normal, execution-cost, funding, borrow, and partial-fill paths.
15. Non-positive/non-finite wealth fails closed.
16. External truncation causes no forced liquidation/terminal reward and remaining horizon is absent from policy input.
17. `normalizer.json` + `u1_contract.json` publish atomically/canonically/idempotently and reject drift/partial output.
18. Existing `SymbolBalancedStandardNormalizer`, `EpisodeRoutedSingleInstrumentEnv`, U3-U6, Causal Alpha, and accounting compatibility tests show no intended behavior change.
19. Targeted/property/integration/falsification tests, Ruff, format, required MyPy/static analysis, import architecture, full suite, and package build pass.
20. Self-review and independent/falsification review have no unresolved substantive finding.
21. Required CI is green on the exact final HEAD, not an earlier commit.
22. Final report states explicitly that U1 does not prove RL learnability, zero-shot economics, profitability, Admission performance, real-market execution fidelity, or Production readiness.

## U2 Handoff

U2 begins only after a real production-candidate U0 generation and U1 artifacts are frozen. U2 must create `UniversalTradeRLFitPurpose.RL_TRAINING` provenance from U0 Train symbols, bind the frozen U1 contract digest into `UniversalTradeRLRunStage.BASE_TRAINING`, and reuse `EpisodeRoutedSingleInstrumentEnv` for balanced Train-symbol episodes. U1 V1 U2 routing must keep `instrument_context_provider=None` and `v4_context_provider=None`; Development stays evaluation-only and Admission remains unopened.
