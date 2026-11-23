"""Synthetic contract tests for opt-in cloud persistence and divergence exits."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from trade_rl.data.features.divergence_mtf import DivergenceCloudMtfPack
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.replay import run_shared_portfolio_replay
from trade_rl.simulation import ExecutionCostConfig
from trade_rl.strategies.divergence_cloud_follow import (
    CloudFollowConfig,
    CloudFollowPolicy,
)
from trade_rl.strategies.portfolio import (
    PortfolioAccountSnapshot,
    PortfolioDecisionSnapshot,
    PortfolioRiskLimits,
)


def _signals(
    cloud: list[int], *, bear_4h: tuple[int, ...] = ()
) -> DivergenceCloudMtfPack:
    length = len(cloud)
    negative = np.zeros(length, dtype=np.int16)
    negative[list(bear_4h)] = 1
    zero = np.zeros(length, dtype=np.int16)
    return DivergenceCloudMtfPack(
        timestamps=(
            np.datetime64("2022-01-01T00:00", "ns")
            + np.arange(length) * np.timedelta64(1, "h")
        ),
        bull_1h=zero,
        bear_1h=zero,
        bull_4h=zero,
        bear_4h=negative,
        cloud_state=np.asarray(cloud, dtype=np.int8),
        cloud_ready=np.asarray([value != 0 for value in cloud], dtype=np.bool_),
        events_1h=(),
        events_4h=(),
    )


def _snapshot(
    signals: DivergenceCloudMtfPack,
    index: int,
    *,
    quantity: float = 0.0,
    weight: float = 0.0,
) -> PortfolioDecisionSnapshot:
    account = PortfolioAccountSnapshot(
        cash=10_000.0,
        portfolio_value=10_000.0,
        peak_value=10_000.0,
        max_drawdown=0.0,
        exact_quantities=(str(quantity),),
        quantities=(quantity,),
        weights=(weight,),
        mark_prices=(100.0,),
        contract_multipliers=(1.0,),
        turnover_total=0.0,
        total_cost=0.0,
        funding_pnl=0.0,
        borrow_cost=0.0,
        margin_used=0.0,
        maintenance_margin=0.25,
        maintenance_requirement=0.0,
        margin_deficit=0.0,
        insolvent=False,
        termination_reason=None,
    )
    risk = PortfolioRiskLimits(
        max_gross=1.0,
        max_abs_weight=1.0,
        max_turnover=None,
        drawdown_start=1.0,
        drawdown_stop=1.0,
        emergency_turnover_override=False,
        fail_closed_tolerance=1e-9,
    )
    return PortfolioDecisionSnapshot(
        index=index,
        timestamp_ns=int(
            signals.timestamps[index].astype("datetime64[ns]").astype("int64")
        ),
        agent_stop_index=len(signals.timestamps),
        remaining_decisions=len(signals.timestamps) - index,
        symbols=("BTCUSDT",),
        feature_names=(),
        global_feature_names=(),
        features=((),),
        feature_available=((),),
        feature_staleness=((),),
        global_features=(),
        global_feature_available=(),
        account=account,
        active_orders=(),
        risk=risk,
    )


def test_enter_after_six_confirmed_hours_only() -> None:
    signals = _signals([1] * 12)
    policy = CloudFollowPolicy(
        symbol="BTCUSDT",
        signals=signals,
        config=CloudFollowConfig(mode="confirmed_cloud"),
    )
    assert policy.decide_portfolio(_snapshot(signals, 4)).target_weights == (0.0,)
    assert policy.decide_portfolio(_snapshot(signals, 5)).target_weights == (0.25,)


def test_exact_hold_then_cloud_exit() -> None:
    signals = _signals([1] * 8 + [0] * 3)
    policy = CloudFollowPolicy(
        symbol="BTCUSDT",
        signals=signals,
        config=CloudFollowConfig(mode="confirmed_cloud"),
    )
    hold = policy.decide_portfolio(_snapshot(signals, 6, quantity=10.0, weight=0.3))
    assert hold.target_weights == (0.3,)
    assert hold.hold_current_quantity_symbols == (0,)
    assert policy.decide_portfolio(
        _snapshot(signals, 8, quantity=10.0, weight=0.3)
    ).target_weights == (0.0,)


def test_divergence_exit_and_24_hour_cooldown() -> None:
    signals = _signals([1] * 50, bear_4h=(12,))
    policy = CloudFollowPolicy(
        symbol="BTCUSDT",
        signals=signals,
        config=CloudFollowConfig(mode="divergence_exit_cooldown"),
    )
    assert policy.decide_portfolio(
        _snapshot(signals, 12, quantity=10.0, weight=0.25)
    ).target_weights == (0.0,)
    assert policy.decide_portfolio(_snapshot(signals, 13)).target_weights == (0.0,)
    assert policy.decide_portfolio(_snapshot(signals, 35)).target_weights == (0.0,)
    assert policy.decide_portfolio(_snapshot(signals, 36)).target_weights == (0.25,)


@pytest.mark.parametrize(
    "invalid",
    [
        {"mode": "not-supported"},
        {"target_abs_weight": float("nan")},
        {"target_abs_weight": 1.01},
        {"entry_confirmation_hours": 0},
        {"divergence_exit_cooldown_hours": -1},
    ],
)
def test_bad_configuration_rejected(invalid: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        CloudFollowConfig(**({"mode": "confirmed_cloud"} | invalid))


def test_snapshot_clock_and_index_must_match() -> None:
    signals = _signals([1] * 12)
    policy = CloudFollowPolicy(
        symbol="BTCUSDT",
        signals=signals,
        config=CloudFollowConfig(mode="confirmed_cloud"),
    )
    with pytest.raises(ValueError, match="timestamp"):
        policy.decide_portfolio(replace(_snapshot(signals, 5), timestamp_ns=0))
    assert policy.decide_portfolio(_snapshot(signals, 5)).target_weights == (0.25,)
    with pytest.raises(ValueError, match="increase"):
        policy.decide_portfolio(_snapshot(signals, 5))


def _native_market() -> MarketDataset:
    n = 55
    close = np.asarray(100 + np.sin(np.arange(n) / 5) * 4, dtype=float)
    op = np.r_[close[0], close[:-1]]
    return MarketDataset(
        dataset_id="d" * 64,
        symbols=("BTCUSDT",),
        timestamps=(
            np.datetime64("2022-01-01T00:00", "ns")
            + np.arange(n) * np.timedelta64(1, "h")
        ),
        features=np.zeros((n, 1, 1), dtype=np.float32),
        global_features=np.zeros((n, 1), dtype=np.float32),
        open=op[:, None],
        high=(np.maximum(op, close) + 0.5)[:, None],
        low=(np.minimum(op, close) - 0.5)[:, None],
        close=close[:, None],
        volume=np.full((n, 1), 100_000.0),
        funding_rate=np.zeros((n, 1)),
        tradable=np.ones((n, 1), dtype=np.bool_),
        feature_available=np.ones((n, 1, 1), dtype=np.bool_),
        feature_names=("dummy",),
        global_feature_names=("global",),
        periods_per_year=8760,
    )


@pytest.mark.parametrize("mode", ["confirmed_cloud", "divergence_exit_cooldown"])
def test_native_account_fills_and_costs(mode: str) -> None:
    market = _native_market()
    states = [1] * 25 + [-1] * 30
    signals = _signals(states, bear_4h=(12, 40))
    policy = CloudFollowPolicy(
        symbol="BTCUSDT",
        signals=signals,
        config=CloudFollowConfig(mode=mode),
    )
    run = run_shared_portfolio_replay(
        market,
        policy,
        start_index=0,
        stop_index=54,
        initial_capital=10_000.0,
        execution_cost=ExecutionCostConfig(),
    )
    assert np.isfinite(run.book.portfolio_value)
    assert run.book.total_cost > 0.0
    assert run.book.fill_count > 0
    assert any(d.requested_hold_symbols == (0,) for d in run.decisions)


def test_future_divergence_mutation_does_not_change_past_decisions() -> None:
    import dataclasses

    before = _signals([1] * 55)
    after = dataclasses.replace(
        before,
        bear_4h=np.asarray([int(i == 45) for i in range(55)], dtype=np.int16),
    )
    outputs: list[tuple[tuple[float, ...], ...]] = []
    for signals in (before, after):
        policy = CloudFollowPolicy(
            symbol="BTCUSDT",
            signals=signals,
            config=CloudFollowConfig(mode="divergence_exit_cooldown"),
        )
        proposals = tuple(
            policy.decide_portfolio(_snapshot(signals, i)).target_weights
            for i in range(30)
        )
        outputs.append(proposals)
    assert outputs[0] == outputs[1]
