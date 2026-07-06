import numpy as np
import pandas as pd
import pytest

from mars_lite.eval.replay_sim import (
    ExecutionOrder,
    ReplaySimulator,
    compare_bar_vs_replay,
)


def test_replay_simulator_fills_against_agg_trades_and_reports_sharpe():
    trades = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=5, freq="min"),
            "symbol": ["BTCUSDT"] * 5,
            "price": [100.0, 101.0, 102.0, 103.0, 104.0],
            "quantity": [1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    orders = [
        ExecutionOrder(
            timestamp=pd.Timestamp("2026-01-01 00:01:00"),
            symbol="BTCUSDT",
            side="buy",
            quantity=2.0,
        ),
        ExecutionOrder(
            timestamp=pd.Timestamp("2026-01-01 00:03:00"),
            symbol="BTCUSDT",
            side="sell",
            quantity=1.0,
        ),
    ]

    result = ReplaySimulator(fee_rate=0.0, max_participation_rate=1.0).simulate(
        trades, orders, initial_cash=1_000.0
    )

    assert len(result.fills) == 2
    assert result.fills[0].average_price == 101.5
    assert result.fills[0].filled_quantity == 2.0
    assert result.fills[1].average_price == 103.0
    assert result.final_position["BTCUSDT"] == 1.0
    assert result.final_equity == 1_004.0
    assert isinstance(result.sharpe, float)


def test_compare_bar_vs_replay_quantifies_sharpe_drift():
    # Drifting, oscillating price path with thin per-trade liquidity so that
    # unconstrained ("bar") execution and participation-constrained
    # ("replay") execution actually land on different fill prices, producing
    # a real Sharpe divergence rather than a manufactured one.
    n = 120
    rng_prices = 100.0 + np.cumsum(np.sin(np.arange(n) / 5.0) * 0.3 + 0.02)
    trades = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="min"),
            "symbol": ["BTCUSDT"] * n,
            "price": rng_prices,
            "quantity": [5.0] * n,
        }
    )
    orders = [
        ExecutionOrder(
            pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=i),
            "BTCUSDT",
            "buy" if (i // 5) % 2 == 0 else "sell",
            3.0,
        )
        for i in range(0, 110, 5)
    ]

    # "Bar simulator" proxy: unconstrained fills, i.e. the whole order is
    # assumed to execute instantly at a single price (the common bar-level
    # assumption), independent of downstream liquidity.
    bar_result = ReplaySimulator(fee_rate=0.0, max_participation_rate=1.0).simulate(
        trades, orders, initial_cash=10_000.0
    )
    # Replay simulator: same orders, but each fill is capped by a fraction of
    # per-trade liquidity, so large orders walk through several trades at
    # different prices - a genuinely different execution path.
    replay_result = ReplaySimulator(fee_rate=0.0, max_participation_rate=0.5).simulate(
        trades, orders, initial_cash=10_000.0
    )

    assert bar_result.returns != replay_result.returns  # not the same series

    comparison = compare_bar_vs_replay(
        bar_returns=np.asarray(bar_result.returns),
        replay_returns=np.asarray(replay_result.returns),
        tolerance=0.3,
    )

    assert (
        comparison["sharpe_diff"]
        == comparison["replay_sharpe"] - comparison["bar_sharpe"]
    )
    assert comparison["abs_sharpe_diff"] == pytest.approx(
        abs(comparison["sharpe_diff"])
    )
    # The two execution assumptions genuinely diverge (not a tautological
    # zero-diff comparison), yet stay within the acceptance tolerance.
    assert comparison["abs_sharpe_diff"] > 1e-6
    assert comparison["within_tolerance"] is True

    # Sanity check that within_tolerance actually discriminates: an
    # unreasonably tight tolerance on the same real drift must fail.
    tight_comparison = compare_bar_vs_replay(
        bar_returns=np.asarray(bar_result.returns),
        replay_returns=np.asarray(replay_result.returns),
        tolerance=1e-9,
    )
    assert tight_comparison["within_tolerance"] is False


def test_replay_simulator_invalid_inputs():
    with pytest.raises(ValueError, match="max_participation_rate"):
        ReplaySimulator(max_participation_rate=0.0)
    with pytest.raises(ValueError, match="max_participation_rate"):
        ReplaySimulator(max_participation_rate=-0.1)

    with pytest.raises(ValueError, match="fee_rate"):
        ReplaySimulator(fee_rate=-0.001)

    with pytest.raises(ValueError, match="slippage_bps"):
        ReplaySimulator(slippage_bps=-1.0)

    sim = ReplaySimulator()
    trades = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=2, freq="min"),
            "symbol": ["BTCUSDT"] * 2,
            "price": [100.0, 101.0],
            "quantity": [1.0, 1.0],
        }
    )
    with pytest.raises(ValueError, match="initial_cash"):
        sim.simulate(trades, [], initial_cash=0)
    with pytest.raises(ValueError, match="initial_cash"):
        sim.simulate(trades, [], initial_cash=-100)

    with pytest.raises(ValueError, match="order quantity"):
        sim.simulate(
            trades, [ExecutionOrder(pd.Timestamp("2026-01-01"), "BTCUSDT", "buy", 0.0)]
        )

    # Missing columns validation
    bad_trades = pd.DataFrame({"timestamp": [], "symbol": [], "price": []})
    with pytest.raises(ValueError, match="agg_trades is missing columns"):
        sim.simulate(bad_trades, [])


def test_compare_bar_vs_replay_invalid_inputs():
    returns_a = np.array([0.01, -0.01])
    returns_b = np.array([0.02, -0.02])
    with pytest.raises(ValueError, match="tolerance"):
        compare_bar_vs_replay(returns_a, returns_b, tolerance=0)
    with pytest.raises(ValueError, match="tolerance"):
        compare_bar_vs_replay(returns_a, returns_b, tolerance=-0.1)
    with pytest.raises(ValueError, match="annualization_factor"):
        compare_bar_vs_replay(returns_a, returns_b, annualization_factor=0)
    with pytest.raises(ValueError, match="annualization_factor"):
        compare_bar_vs_replay(returns_a, returns_b, annualization_factor=-1)
