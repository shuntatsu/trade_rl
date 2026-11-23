"""Synthetic-only native replay of the opt-in divergence/cloud MTF policy."""

from __future__ import annotations

import numpy as np
import pytest

from trade_rl.data.features.divergence_mtf import DivergenceCloudMtfPack
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.replay import run_shared_portfolio_replay
from trade_rl.risk import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation import ExecutionCostConfig
from trade_rl.strategies.divergence_cloud_mtf import (
    DivergenceCloudMtfConfig,
    DivergenceCloudMtfPolicy,
)


def _test_market() -> MarketDataset:
    c = np.asarray([100.0, 102.0, 105.0, 104.0, 101.0, 99.0], dtype=float)[:, None]
    o = np.vstack((c[0], c[:-1]))
    return MarketDataset(
        dataset_id="d" * 64,
        symbols=("BTCUSDT",),
        timestamps=np.datetime64("2026-01-01T01:00", "ns")
        + np.arange(6) * np.timedelta64(1, "h"),
        features=np.zeros((6, 1, 1), dtype=np.float32),
        global_features=np.zeros((6, 1), dtype=np.float32),
        open=o,
        high=np.maximum(o, c) + 0.5,
        low=np.minimum(o, c) - 0.5,
        close=c,
        volume=np.full((6, 1), 1_000_000.0),
        funding_rate=np.zeros((6, 1)),
        tradable=np.ones((6, 1), dtype=bool),
        feature_available=np.ones((6, 1, 1), dtype=bool),
        feature_names=("dummy",),
        global_feature_names=("global",),
        periods_per_year=8760,
    )


def _test_signals(dataset: MarketDataset) -> DivergenceCloudMtfPack:
    bull1 = np.zeros(6, dtype=np.int16)
    bull4 = np.zeros(6, dtype=np.int16)
    bull1[0] = 2
    bull4[0] = 2  # Explicitly injected synthetic signal, NOT a calculated 4h bar.
    return DivergenceCloudMtfPack(
        timestamps=dataset.timestamps,
        bull_1h=bull1,
        bull_4h=bull4,
        bear_1h=np.zeros(6, dtype=np.int16),
        bear_4h=np.zeros(6, dtype=np.int16),
        cloud_state=np.array([1, 1, -1, -1, -1, -1], dtype=np.int8),
        cloud_ready=np.ones(6, dtype=bool),
        events_1h=(),
        events_4h=(),
    )


def test_portfolio_policy_native_replay_pays_fills_and_holds_exact_quantity() -> None:
    dataset = _test_market()
    policy = DivergenceCloudMtfPolicy(
        symbol="BTCUSDT",
        signals=_test_signals(dataset),
    )
    risk = PreTradeRisk(
        PreTradeRiskConfig(
            max_gross=0.8,
            max_abs_weight=0.8,
            max_turnover=None,
            drawdown_start=1.0,
            drawdown_stop=1.0,
        )
    )
    result = run_shared_portfolio_replay(
        dataset,
        policy,
        start_index=0,
        stop_index=5,
        initial_capital=10_000.0,
        execution_cost=ExecutionCostConfig(),
        risk=risk,
        capture_ledger_evidence=True,
    )
    assert result.decisions[0].proposal_weights == pytest.approx((0.25,))
    assert result.decisions[1].requested_hold_symbols == (0,)
    assert result.decisions[2].proposal_weights == pytest.approx((0.0,))
    assert result.book.total_cost > 0
    assert result.book.fill_count == 2
    assert result.ledger_evidence is not None


def test_policy_rejects_mismatched_clocks_and_invalid_configs() -> None:
    dataset = _test_market()
    pack = _test_signals(dataset)
    policy = DivergenceCloudMtfPolicy(symbol="OTHER", signals=pack)
    risk = PreTradeRisk(
        PreTradeRiskConfig(
            max_gross=0.8,
            max_abs_weight=0.8,
            max_turnover=None,
            drawdown_start=1.0,
            drawdown_stop=1.0,
        )
    )
    with pytest.raises(ValueError, match="bound single symbol"):
        run_shared_portfolio_replay(
            dataset,
            policy,
            start_index=0,
            stop_index=3,
            risk=risk,
        )
    with pytest.raises(ValueError, match="target_abs_weight"):
        DivergenceCloudMtfConfig(target_abs_weight=float("nan"))


def test_no_pending_order_double_submission_on_repeat_signals() -> None:
    dataset = _test_market()
    pack = _test_signals(dataset)
    policy = DivergenceCloudMtfPolicy(symbol="BTCUSDT", signals=pack)
    result = run_shared_portfolio_replay(
        dataset,
        policy,
        start_index=0,
        stop_index=5,
        execution_cost=ExecutionCostConfig(),
    )
    # Only one entry and one exit despite holding a fixed weight and risk changes.
    assert result.book.fill_count <= 2


def test_module_exports_are_available() -> None:
    from trade_rl.data.features import (
        build_divergence_bank,
        build_divergence_cloud_mtf,
        detect_divergences,
        visible_ichimoku_cloud,
    )
    from trade_rl.strategies.divergence_cloud_mtf import (
        DivergenceCloudMtfPolicy as PublicPolicy,
    )

    assert all(
        callable(f)
        for f in (
            build_divergence_bank,
            build_divergence_cloud_mtf,
            detect_divergences,
            visible_ichimoku_cloud,
        )
    )
    assert PublicPolicy is DivergenceCloudMtfPolicy


def test_full_indicator_mtf_policy_runs_with_native_accounting() -> None:
    """Actual generated indicators -> signals -> native replay; artificial prices."""
    from trade_rl.data.features.divergence import DivergenceConfig
    from trade_rl.data.features.divergence_mtf import build_divergence_cloud_mtf
    from trade_rl.data.source import RawMarketSeries

    n = 24 * 110
    t = np.datetime64("2025-01-01T01:00", "ns") + np.arange(n) * np.timedelta64(1, "h")
    x = np.arange(n, dtype=np.float64)
    close = 100.0 + 0.04 * x + 3 * np.sin(x / 8) + 2 * np.sin(x / 41)
    open_price = close - 0.15
    high = np.maximum(open_price, close) + 0.5
    low = np.minimum(open_price, close) - 0.5
    volume = np.full(n, 50_000.0)
    raw = RawMarketSeries(
        timestamps=t,
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=volume,
        funding_rate=np.zeros(n),
        tradable=np.ones(n, dtype=np.bool_),
    )
    signals = build_divergence_cloud_mtf(
        raw,
        indicators=("RSI", "MACD_HIST", "MFI"),
        config=DivergenceConfig(pivot_period=3, min_indicators=1, validate_lines=False),
    )
    assert int(signals.bull_1h.sum() + signals.bear_1h.sum()) > 0
    assert int(signals.bull_4h.sum() + signals.bear_4h.sum()) > 0
    dataset = MarketDataset(
        dataset_id="d" * 64,
        symbols=("BTCUSDT",),
        timestamps=t,
        features=np.zeros((n, 1, 1), dtype=np.float32),
        global_features=np.zeros((n, 1), dtype=np.float32),
        open=open_price[:, None],
        high=high[:, None],
        low=low[:, None],
        close=close[:, None],
        volume=volume[:, None],
        funding_rate=np.zeros((n, 1)),
        tradable=np.ones((n, 1), dtype=np.bool_),
        feature_available=np.ones((n, 1, 1), dtype=np.bool_),
        feature_names=("dummy",),
        global_feature_names=("global",),
        periods_per_year=8760,
    )
    policy = DivergenceCloudMtfPolicy(symbol="BTCUSDT", signals=signals)
    result = run_shared_portfolio_replay(
        dataset,
        policy,
        start_index=0,
        stop_index=n - 1,
        initial_capital=10_000,
        execution_cost=ExecutionCostConfig(),
    )
    assert np.isfinite(result.book.portfolio_value)
    assert result.book.fill_count > 0
    assert result.book.total_cost > 0.0
    assert any(d.requested_hold_symbols == (0,) for d in result.decisions)
