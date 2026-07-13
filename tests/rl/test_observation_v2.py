from __future__ import annotations

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.rl.observations import build_observation, observation_layout
from trade_rl.simulation.accounting import BookState
from trade_rl.strategies.trend import TrendTargets


def market() -> MarketDataset:
    close = np.array(
        [
            [100.0, 100.0],
            [101.0, 99.0],
            [102.0, 98.0],
            [103.0, 97.0],
        ]
    )
    open_price = np.vstack([close[0], close[:-1]])
    feature_available = np.ones((4, 2, 2), dtype=np.bool_)
    feature_available[1, 0, 1] = False
    tradable = np.ones((4, 2), dtype=np.bool_)
    tradable[2, 1] = False
    return MarketDataset(
        dataset_id="d" * 64,
        symbols=("A", "B"),
        timestamps=np.datetime64("2026-01-01T01:00:00", "ns")
        + np.arange(4) * np.timedelta64(1, "h"),
        features=np.arange(16, dtype=np.float32).reshape(4, 2, 2),
        global_features=np.ones((4, 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, close) + 1.0,
        low=np.minimum(open_price, close) - 1.0,
        close=close,
        volume=np.full((4, 2), 1_000.0),
        funding_rate=np.zeros((4, 2)),
        tradable=tradable,
        feature_available=feature_available,
        feature_names=("ret", "rsi"),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def positioned_book(weights: np.ndarray) -> BookState:
    prices = np.array([100.0, 100.0])
    book = BookState.zero(2, 1_000.0, prices)
    book.execute(
        fill_prices=prices,
        target_quantities=weights * 1_000.0 / prices,
        cost_amount=0.0,
        turnover=float(np.abs(weights).sum()),
    )
    return book


def test_observation_contains_masks_both_books_and_risk_state() -> None:
    dataset = market()
    hybrid = positioned_book(np.array([0.5, -0.2]))
    shadow = positioned_book(np.array([0.3, -0.3]))
    trends = TrendTargets(
        fast=np.array([0.4, -0.4]),
        base=np.array([0.3, -0.3]),
        slow=np.array([0.2, -0.2]),
    )

    observation = build_observation(
        dataset=dataset,
        index=1,
        trends=trends,
        alpha=np.array([0.1, -0.1]),
        hybrid=hybrid,
        shadow=shadow,
        start_index=0,
        end_index=3,
        hybrid_risk_scale=0.8,
        shadow_risk_scale=0.9,
    )

    layout = observation_layout(dataset)
    assert observation.shape == (layout.size,)
    per_symbol = observation[: dataset.n_symbols * layout.per_symbol_width].reshape(
        dataset.n_symbols,
        layout.per_symbol_width,
    )
    assert per_symbol[0, dataset.n_features] == pytest.approx(0.5)
    assert per_symbol[0, dataset.n_features + 1] == pytest.approx(1.0)
    assert per_symbol[1, dataset.n_features + 1] == pytest.approx(0.0)
    np.testing.assert_allclose(per_symbol[:, -3], hybrid.weights)
    np.testing.assert_allclose(per_symbol[:, -2], shadow.weights)
    np.testing.assert_allclose(per_symbol[:, -1], hybrid.weights - shadow.weights)

    globals_ = observation[-layout.global_width :]
    assert globals_[-3] == pytest.approx(0.8)
    assert globals_[-2] == pytest.approx(0.9)
    assert globals_[-1] == pytest.approx(0.5)
