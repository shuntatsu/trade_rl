from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from trade_rl.data.market import MarketCalendarKind, MarketDataset
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import ExecutionCostConfig, MarketExecutor


def test_borrow_and_cash_interest_use_actual_elapsed_time() -> None:
    timestamps = np.array(
        [
            "2026-01-02T16:00:00",
            "2026-01-05T09:00:00",
            "2026-01-05T10:00:00",
        ],
        dtype="datetime64[ns]",
    )
    prices = np.full((3, 1), 100.0)
    dataset = MarketDataset(
        dataset_id="a" * 64,
        symbols=("A",),
        timestamps=timestamps,
        features=np.zeros((3, 1, 1), dtype=np.float32),
        global_features=np.zeros((3, 1), dtype=np.float32),
        open=prices,
        high=prices,
        low=prices,
        close=prices,
        volume=np.full((3, 1), 10_000.0),
        funding_rate=np.zeros((3, 1)),
        tradable=np.ones((3, 1), dtype=np.bool_),
        feature_available=np.ones((3, 1, 1), dtype=np.bool_),
        feature_names=("x",),
        global_feature_names=("g",),
        periods_per_year=1_638,
        calendar_kind=MarketCalendarKind.SESSION,
        nominal_bar_hours=1.0,
        borrow_rate=np.full((3, 1), 0.365),
        cash_rate=np.full(3, 0.365),
    )
    book = BookState.from_weights(
        weights=np.array([-0.5]),
        capital=1_000.0,
        prices=prices[0],
        contract_multipliers=dataset.contract_multipliers,
    )
    result = MarketExecutor(
        dataset,
        replace(ExecutionCostConfig.zero(), borrow_rate_multiplier=1.0),
    ).execute_interval(book, np.array([-0.5]), start_index=0, bars=1)

    year_fraction = 65.0 / (365.0 * 24.0)
    expected_borrow = 500.0 * 0.365 * year_fraction
    # Initial short creates 1,500 cash; interest is charged/credited over 65 hours.
    expected_interest = 1_500.0 * 0.365 * year_fraction
    assert result.interval_borrow_cost == pytest.approx(expected_borrow)
    assert result.interval_cash_interest == pytest.approx(expected_interest)
