from dataclasses import replace

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import ExecutionCostConfig, MarketExecutor
from trade_rl.simulation.orders.model import OrderBookState
from trade_rl.simulation.stateful.execution import execute_stateful_orders


def market():
    shape = (4, 1)
    prices = np.full(shape, 100.0)
    dividend = np.zeros(shape)
    dividend[1, 0] = 2.0
    return MarketDataset(
        dataset_id="c" * 64,
        symbols=("S0",),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(4) * np.timedelta64(1, "h"),
        features=np.ones((4, 1, 1)),
        global_features=np.zeros((4, 1)),
        open=prices,
        high=prices,
        low=prices,
        close=prices,
        volume=np.full(shape, 1000.0),
        funding_rate=np.zeros(shape),
        tradable=np.ones(shape, dtype=bool),
        feature_available=np.ones((4, 1, 1), dtype=bool),
        feature_names=("signal",),
        global_feature_names=("regime",),
        periods_per_year=8760,
        dividend=dividend,
    )


def marked_book():
    return BookState(
        np.array([1.0]),
        900.0,
        np.array([100.0]),
        1000.0,
        as_of_index=0,
        as_of_dataset_id="c" * 64,
    )


def test_known_clock_advances_across_bars_clones_and_rejects_duplicate_carry():
    executor = MarketExecutor(market(), ExecutionCostConfig.zero())
    original = marked_book()
    first = execute_stateful_orders(
        executor, original, OrderBookState.empty(), [], start_index=0, bars=1
    )
    assert first.book.as_of_index == first.next_index == 1
    assert first.book.cash == 902.0
    assert original.as_of_index == 0 and original.cash == 900.0
    assert first.book.clone().as_of_dataset_id == "c" * 64
    assert first.book.clone().as_of_index == 1
    with pytest.raises(ValueError, match="processing.*clock"):
        execute_stateful_orders(
            executor, first.book, first.order_book, [], start_index=0, bars=1
        )
    second = execute_stateful_orders(
        executor, first.book, first.order_book, [], start_index=first.next_index, bars=2
    )
    assert second.book.as_of_index == second.next_index == 3
    assert second.book.cash == 902.0
    assert second.book.fill_count == 0
    with pytest.raises(ValueError, match="processing.*clock"):
        execute_stateful_orders(
            MarketExecutor(
                replace(market(), dataset_id="d" * 64), ExecutionCostConfig.zero()
            ),
            original,
            OrderBookState.empty(),
            [],
            start_index=0,
            bars=1,
        )


def test_unmarked_legacy_book_retains_its_existing_optional_clock():
    executor = MarketExecutor(market(), ExecutionCostConfig.zero())
    original = BookState(np.array([1.0]), 900.0, np.array([100.0]), 1000.0)
    result = execute_stateful_orders(
        executor, original, OrderBookState.empty(), [], start_index=0, bars=2
    )
    assert result.book.as_of_index is None and result.book.as_of_dataset_id is None
    assert result.book.cash == 902.0


@pytest.mark.parametrize(
    "index,dataset_id",
    [
        (None, "c" * 64),
        (0, None),
        (True, "c" * 64),
        (-1, "c" * 64),
        (1.5, "c" * 64),
        (0, "bad"),
    ],
)
def test_processing_clock_rejects_incomplete_or_invalid_bootstrap(index, dataset_id):
    with pytest.raises(ValueError):
        BookState(
            np.array([0.0]),
            1000.0,
            np.array([100.0]),
            1000.0,
            as_of_index=index,
            as_of_dataset_id=dataset_id,
        )
