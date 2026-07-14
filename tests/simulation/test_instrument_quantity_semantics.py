from __future__ import annotations

import numpy as np

from trade_rl.data.contracts import VolumeUnit
from trade_rl.data.market import MarketDataset
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import ExecutionCostConfig, MarketExecutor


def _dataset() -> MarketDataset:
    n = 4
    prices = np.tile(np.array([10.0, 20.0, 30.0]), (n, 1))
    return MarketDataset(
        dataset_id="a" * 64,
        symbols=("BASE", "QUOTE", "CONTRACT"),
        timestamps=np.datetime64("2026-01-01", "ns") + np.arange(n) * np.timedelta64(1, "h"),
        features=np.zeros((n, 3, 1), dtype=np.float32),
        global_features=np.zeros((n, 1), dtype=np.float32),
        open=prices,
        high=prices,
        low=prices,
        close=prices,
        volume=np.tile(np.array([2.0, 1_000.0, 5.0]), (n, 1)),
        funding_rate=np.zeros((n, 3)),
        tradable=np.ones((n, 3), dtype=np.bool_),
        feature_available=np.ones((n, 3, 1), dtype=np.bool_),
        feature_names=("dummy",),
        global_feature_names=("global",),
        periods_per_year=8_760,
        volume_units=(VolumeUnit.BASE_ASSET, VolumeUnit.QUOTE_NOTIONAL, VolumeUnit.CONTRACTS),
        contract_multipliers=np.array([1.0, 1.0, 0.1]),
    )


def test_quantity_notional_round_trip_respects_contract_multiplier() -> None:
    dataset = _dataset()
    quantities = np.array([2.0, 3.0, 4.0])
    notionals = dataset.quantity_notional(1, quantities)
    np.testing.assert_allclose(notionals, [20.0, 60.0, 12.0])
    np.testing.assert_allclose(dataset.notional_to_quantity(1, notionals), quantities)


def test_book_state_values_contracts_with_multiplier() -> None:
    book = BookState.from_weights(
        weights=np.array([0.0, 0.0, 0.5]),
        capital=1_000.0,
        prices=np.array([10.0, 20.0, 30.0]),
        contract_multipliers=np.array([1.0, 1.0, 0.1]),
    )
    assert book.quantities[2] == 1_000.0 * 0.5 / (30.0 * 0.1)
    assert book.position_values[2] == 500.0


def test_executor_uses_quote_notional_capacity_without_double_multiplication() -> None:
    dataset = _dataset()
    executor = MarketExecutor(
        dataset,
        ExecutionCostConfig.zero(),
    )
    book = BookState.zero(
        dataset.n_symbols,
        10_000.0,
        dataset.close[0],
        contract_multipliers=dataset.contract_multipliers,
    )
    result = executor.execute_interval(
        book,
        np.array([0.0, 1.0, 0.0]),
        start_index=0,
        bars=1,
    )
    # QUOTE volume is already 1,000 quote currency, so max filled notional is 1,000.
    assert result.filled_notional_by_symbol[1] == 1_000.0
