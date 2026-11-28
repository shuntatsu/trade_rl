"""Shared stateful fill and terminal valuation capacity authority."""

from __future__ import annotations

import numpy as np

from trade_rl.data.contracts import VolumeUnit
from trade_rl.data.market import MarketDataset


def _capacity_reference_for_symbol(
    dataset: MarketDataset,
    *,
    processing_index: int,
    processing_open_prices: np.ndarray,
    symbol: int,
    processing_bar_volume_capacity: bool,
) -> tuple[float, float, float, float | None]:
    """Return raw volume, quote pool, reference price and native quantity pool.

    Causal mode accepts only the immediately preceding observed bar, published
    by its own close. No previous or unavailable reference means zero capacity;
    the processing open is only an inert positive reference in that case.
    Same-bar mode remains an explicit non-PIT compatibility assumption.
    """
    unit = dataset.volume_units[symbol]
    reference_index = (
        processing_index if processing_bar_volume_capacity else processing_index - 1
    )
    if not processing_bar_volume_capacity and (
        reference_index < 0
        or not dataset.resolved_array("information_available")[reference_index, symbol]
    ):
        return (
            0.0,
            0.0,
            float(processing_open_prices[symbol]),
            None if unit is VolumeUnit.QUOTE_NOTIONAL else 0.0,
        )

    price = float(
        processing_open_prices[symbol]
        if processing_bar_volume_capacity
        else dataset.close[reference_index, symbol]
    )
    volume = float(dataset.volume[reference_index, symbol])
    # Convert only the requested symbol. An unrelated (possibly unpublished)
    # row must not affect the result or cause overflow during vector conversion.
    prices = np.ones(dataset.n_symbols, dtype=np.float64)
    volumes = np.zeros(dataset.n_symbols, dtype=np.float64)
    prices[symbol] = price
    volumes[symbol] = volume
    notional = float(
        dataset.market_notional(reference_index, prices, volume=volumes)[symbol]
    )
    quantity: float | None = None
    if unit is VolumeUnit.BASE_ASSET:
        multiplier = float(dataset.resolved_array("contract_multipliers")[symbol])
        quantity = volume / multiplier
    elif unit is VolumeUnit.CONTRACTS:
        quantity = volume
    return volume, notional, price, quantity
