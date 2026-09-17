from dataclasses import replace

import numpy as np

from tests.evaluation.test_shared_cash_replay import _market
from trade_rl.data.features import price_channels


def test_channels_exclude_current_bar_and_preserve_economics() -> None:
    source = _market(np.array([[10.0], [11.0], [12.0], [20.0], [19.0], [9.0]]))
    assert hasattr(price_channels, "with_price_channels")
    data = price_channels.with_price_channels(source, entry_bars=3, exit_bars=2)
    assert data.identity_verified
    np.testing.assert_array_equal(data.close, source.close)
    assert not data.feature_available[2, 0, -4]
    assert data.feature_available[3, 0, -4]
    assert data.features[3, 0, -4] == np.float32(20 / 12 - 1)
    assert data.features[5, 0, -1] == np.float32(9 / 12 - 1)


def test_future_changes_cannot_change_earlier_channel_inputs() -> None:
    source = _market(np.arange(10.0, 18.0).reshape(-1, 1))
    values = source.close.copy()
    values[6:] *= 10
    future = replace(source, close=values, high=np.maximum(source.high, values))
    before = price_channels.with_price_channels(source, entry_bars=3, exit_bars=2)
    after = price_channels.with_price_channels(future, entry_bars=3, exit_bars=2)
    np.testing.assert_array_equal(before.features[:6], after.features[:6])


def test_missing_historical_candle_blocks_channel_until_it_leaves_window() -> None:
    source = _market(np.arange(10.0, 18.0).reshape(-1, 1))
    available = np.ones_like(source.close, dtype=bool)
    available[2] = False
    source = replace(source, information_available=available)
    data = price_channels.with_price_channels(source, entry_bars=3, exit_bars=2)
    assert not data.feature_available[5, 0, -4]
    assert data.feature_available[6, 0, -4]
