from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from trade_rl.integrations.training_telemetry import environment_market_snapshot


def test_explicit_market_index_preserves_terminal_interval_after_auto_reset() -> None:
    timestamps = np.datetime64("2026-07-21T08:00:00", "ns") + np.arange(
        6
    ) * np.timedelta64(5, "m")
    open_price = np.column_stack((np.arange(100.0, 106.0), np.arange(200.0, 206.0)))
    close = open_price + 0.5
    dataset = SimpleNamespace(
        n_bars=6,
        symbols=("BTCUSDT", "ETHUSDT"),
        timestamps=timestamps,
        open=open_price,
        high=open_price + np.asarray([2.0, 3.0]),
        low=open_price - np.asarray([1.0, 2.0]),
        close=close,
    )
    environment = SimpleNamespace(
        unwrapped=SimpleNamespace(dataset=dataset, current_index=1)
    )

    snapshot = environment_market_snapshot(
        environment,
        bars_advanced=2,
        market_index=5,
    )

    assert snapshot["telemetry_market_index"] == 5
    assert snapshot["telemetry_market_time"] == "2026-07-21T08:25:00.000000000"
    assert snapshot["telemetry_ohlc"] == pytest.approx((104.0, 107.0, 103.0, 105.5))
