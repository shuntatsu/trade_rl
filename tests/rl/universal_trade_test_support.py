from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from trade_rl.data.contracts import FeatureKind, FeatureSpec
from trade_rl.data.market import MarketDataset
from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig
from trade_rl.rl.actions import ActionMode, ActionSpec, ActionValidationMode
from trade_rl.rl.environment_config import EpisodeBoundaryMode, ResidualMarketEnvConfig
from trade_rl.rl.rewards import RewardConfig
from trade_rl.rl.universal_trade_contract import UNIVERSAL_TRADE_SEQUENCE_WINDOWS
from trade_rl.rl.universal_trade_environment import UniversalTradeMarketEnv
from trade_rl.rl.universal_trade_runtime import UniversalTradeRuntimeSnapshot
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.trend import TrendConfig, TrendStrategy


def make_u1_feature_specs() -> tuple[FeatureSpec, ...]:
    return (
        FeatureSpec(name="15m__ret", kind=FeatureKind.LOG_RETURN, lookback=1),
        FeatureSpec(
            name="1h__ret",
            kind=FeatureKind.LOG_RETURN,
            lookback=1,
            timeframe="1h",
        ),
        FeatureSpec(
            name="4h__ret",
            kind=FeatureKind.LOG_RETURN,
            lookback=1,
            timeframe="4h",
        ),
        FeatureSpec(
            name="1d__ret",
            kind=FeatureKind.LOG_RETURN,
            lookback=1,
            timeframe="1d",
        ),
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

    timestamps = np.datetime64("2025-01-01T00:00:00", "ns") + rows * np.timedelta64(
        15, "m"
    )
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
) -> UniversalTradeMarketEnv:
    market = make_u1_market() if dataset is None else dataset
    return UniversalTradeMarketEnv(
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
            random_initial_gross=min(0.25, max_abs_weight),
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


def make_runtime_snapshot(**overrides: Any) -> UniversalTradeRuntimeSnapshot:
    snapshot = UniversalTradeRuntimeSnapshot(
        policy_requested_weight=0.0,
        pending_target_weight=0.0,
        pending_target_active=False,
        risk_projected_weight=0.0,
        current_weight=0.0,
        previous_action=0.0,
        fill_ratio=0.0,
        unfilled_turnover_ratio=0.0,
        participation_ratio=0.0,
        execution_cost_rate=0.0,
        position_age_hours=0.0,
        pending_notional_ratio=0.0,
        pending_order_type_code=0.0,
        pending_order_status_code=0.0,
        pending_order_age_hours=0.0,
        pending_order_eligible_delay_hours=0.0,
        pending_order_triggered=False,
        pending_order_expiry_distance_hours=0.0,
        asset_active=True,
        tradable=True,
        borrow_available=True,
        borrow_rate=0.0,
        mark_index_basis=0.0,
        current_drawdown=0.0,
        current_gross_exposure=0.0,
        current_net_exposure=0.0,
        cash_weight=1.0,
        risk_scale=1.0,
        margin_utilization=0.0,
    )
    return replace(snapshot, **overrides)
