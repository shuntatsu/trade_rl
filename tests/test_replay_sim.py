import numpy as np
import pandas as pd
import pytest

from mars_lite.eval.replay_sim import ExecutionOrder, ReplaySimulator


def _trades():
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=6, freq="min"),
            "symbol": ["BTCUSDT"] * 6,
            "price": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "quantity": [10.0] * 6,
        }
    )


def test_liquidity_is_not_double_consumed():
    orders = [
        ExecutionOrder(pd.Timestamp("2026-01-01 00:00"), "BTCUSDT", "buy", 5.0),
        ExecutionOrder(pd.Timestamp("2026-01-01 00:00"), "BTCUSDT", "buy", 5.0),
    ]
    result = ReplaySimulator(
        fee_rate=0.0, max_participation_rate=0.5
    ).simulate(_trades(), orders, initial_cash=100_000.0)
    assert result.fills[0].average_price == 100.0
    assert result.fills[1].average_price == 101.0


def test_latency_expiry_and_limit_conditions_use_actual_trade_times():
    orders = [
        ExecutionOrder(
            pd.Timestamp("2026-01-01 00:00"),
            "BTCUSDT",
            "buy",
            2.0,
            latency_seconds=60,
            max_delay_seconds=120,
        ),
        ExecutionOrder(
            pd.Timestamp("2026-01-01 00:00"),
            "BTCUSDT",
            "buy",
            1.0,
            order_type="limit",
            limit_price=100.5,
            max_delay_seconds=180,
        ),
    ]
    result = ReplaySimulator(fee_rate=0.0, max_participation_rate=1.0).simulate(
        _trades(), orders, initial_cash=10_000.0
    )
    assert result.fills[0].first_fill_timestamp == pd.Timestamp("2026-01-01 00:01")
    assert result.fills[0].last_fill_timestamp == pd.Timestamp("2026-01-01 00:01")
    assert result.fills[1].first_fill_timestamp == pd.Timestamp("2026-01-01 00:00")


def test_equity_curve_is_uniform_one_minute_grid():
    result = ReplaySimulator(fee_rate=0.0, equity_frequency="1min").simulate(
        _trades(),
        [ExecutionOrder(pd.Timestamp("2026-01-01 00:01"), "BTCUSDT", "buy", 1.0)],
        initial_cash=1000.0,
    )
    diffs = np.diff(np.array(result.equity_timestamps, dtype="datetime64[ns]"))
    assert len(result.equity_curve) == 6
    assert len(result.returns) == 5
    assert all(diff == np.timedelta64(1, "m") for diff in diffs)
    assert result.annualization_factor == pytest.approx(365.25 * 24 * 60)


def test_final_equity_and_fill_timestamp_are_consistent():
    result = ReplaySimulator(fee_rate=0.0, max_participation_rate=1.0).simulate(
        _trades(),
        [ExecutionOrder(pd.Timestamp("2026-01-01 00:01"), "BTCUSDT", "buy", 2.0)],
        initial_cash=1000.0,
    )
    assert result.fills[0].average_price == 101.0
    assert result.final_position["BTCUSDT"] == 2.0
    assert result.final_equity == pytest.approx(1008.0)


def test_maker_fee_and_taker_fee_are_applied_separately():
    trades = _trades().iloc[:1]
    result = ReplaySimulator(
        fee_rate=0.001,
        maker_fee_rate=0.0002,
        max_participation_rate=1.0,
    ).simulate(
        trades,
        [
            ExecutionOrder(
                pd.Timestamp("2026-01-01"), "BTCUSDT", "buy", 1.0, maker=True
            ),
            ExecutionOrder(
                pd.Timestamp("2026-01-01"), "BTCUSDT", "buy", 1.0, maker=False
            ),
        ],
        initial_cash=1000.0,
    )
    assert result.fills[0].fee_paid == pytest.approx(0.02)
    assert result.fills[1].fee_paid == pytest.approx(0.1)


def test_invalid_limit_order_is_rejected():
    with pytest.raises(ValueError, match="limit_price"):
        ReplaySimulator().simulate(
            _trades(),
            [
                ExecutionOrder(
                    pd.Timestamp("2026-01-01"),
                    "BTCUSDT",
                    "buy",
                    1.0,
                    order_type="limit",
                )
            ],
        )
