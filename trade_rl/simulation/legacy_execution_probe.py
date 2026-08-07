"""Deterministic legacy execution fixture used for dual-shadow conformance."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.simulation import MarketExecutor
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.execution_canonicalization import (
    CanonicalEconomicClosure,
    CanonicalFillSignature,
)
from trade_rl.simulation.legacy_trace_adapter import canonicalize_legacy_fill_events
from trade_rl.simulation.orders import OrderBookState
from trade_rl.simulation.target_execution import execute_target_statefully


@dataclass(frozen=True, slots=True)
class LegacyExecutionProbeResult:
    fills: tuple[CanonicalFillSignature, ...]
    economics: CanonicalEconomicClosure


def run_legacy_flat_long_flat_probe() -> LegacyExecutionProbeResult:
    """Run a tiny legacy fixture economically aligned with the Nautilus probe."""

    dataset = _dataset()
    executor = MarketExecutor(dataset, _cost())
    initial_capital = 1_000.0
    initial = BookState.zero(1, initial_capital, dataset.close[0])

    opened = execute_target_statefully(
        executor,
        initial,
        OrderBookState.empty(),
        np.array([0.1], dtype=np.float64),
        start_index=0,
        bars=1,
        target_identity="dual-shadow-open",
    )
    closed = execute_target_statefully(
        executor,
        opened.book,
        opened.order_book,
        np.array([0.0], dtype=np.float64),
        start_index=1,
        bars=1,
        target_identity="dual-shadow-flat",
    )

    fills = canonicalize_legacy_fill_events(
        (*opened.order_events, *closed.order_events),
        price_tick=0.1,
        lot_size=0.001,
    )
    final_equity = closed.book.portfolio_value
    economics = CanonicalEconomicClosure(
        fee_minor=_minor(closed.book.total_cost),
        funding_minor=_minor(closed.book.funding_pnl),
        realized_pnl_minor=_minor(final_equity - initial_capital),
        final_equity_minor=_minor(final_equity),
        terminal_position_lots=int(round(float(closed.book.quantities[0]) / 0.001)),
        terminal_open_orders=len(closed.order_book.active_orders),
    )
    return LegacyExecutionProbeResult(fills=fills, economics=economics)


def _dataset() -> MarketDataset:
    open_prices = np.array([[100.0], [100.1], [104.9], [105.0]], dtype=np.float64)
    close = np.array([[100.0], [105.0], [105.0], [105.0]], dtype=np.float64)
    high = np.maximum(open_prices, close) + 1.0
    low = np.minimum(open_prices, close) - 1.0
    volume = np.full_like(close, 1_000.0)
    hour_ns = 60 * 60 * 1_000_000_000
    timestamps = np.array(
        [0, hour_ns, 2 * hour_ns, 3 * hour_ns],
        dtype="datetime64[ns]",
    )
    return MarketDataset(
        dataset_id="f" * 64,
        symbols=("BTCUSDT",),
        timestamps=timestamps,
        features=np.zeros((4, 1, 1), dtype=np.float32),
        global_features=np.zeros((4, 1), dtype=np.float32),
        open=open_prices,
        high=high,
        low=low,
        close=close,
        volume=volume,
        funding_rate=np.zeros_like(close),
        tradable=np.ones_like(close, dtype=np.bool_),
        feature_available=np.ones((4, 1, 1), dtype=np.bool_),
        feature_names=("probe",),
        global_feature_names=("probe",),
        periods_per_year=8_760,
    )


def _cost() -> ExecutionCostConfig:
    return ExecutionCostConfig(
        fee_rate=0.0,
        maker_fee_rate=0.0002,
        taker_fee_rate=0.0004,
        spread_rate=0.0,
        impact_rate=0.0,
        max_participation_rate=1.0,
        slippage_std=0.0,
        order_type="market",
        path_mode="neutral",
        processing_bar_volume_capacity=True,
        partial_fill_carry=True,
    )


def _minor(value: float, *, precision: int = 8) -> int:
    quantum = Decimal(1).scaleb(-precision)
    normalized = Decimal(str(float(value))).quantize(quantum, rounding=ROUND_HALF_EVEN)
    return int(normalized * (Decimal(10) ** precision))


__all__ = ["LegacyExecutionProbeResult", "run_legacy_flat_long_flat_probe"]
