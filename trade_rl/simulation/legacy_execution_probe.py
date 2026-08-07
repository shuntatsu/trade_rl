"""Deterministic legacy execution fixtures used for dual-shadow conformance."""

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
from trade_rl.simulation.orders import (
    OrderBookState,
    OrderEvent,
    OrderIntent,
    OrderType,
    TimeInForce,
)
from trade_rl.simulation.stateful_execution import execute_stateful_orders
from trade_rl.simulation.target_execution import execute_target_statefully
from trade_rl.simulation.target_exposure_controller import TargetExposureChildOrder

_QUANTITY_TOLERANCE = 1e-12


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


def run_legacy_flat_long_flat_short_flat_probe() -> LegacyExecutionProbeResult:
    """Execute an explicit reduce-to-flat sign reversal through the legacy engine."""

    child_orders = (
        TargetExposureChildOrder(quantity=1.0, reduce_only=False),
        TargetExposureChildOrder(quantity=-1.0, reduce_only=True),
        TargetExposureChildOrder(quantity=-1.0, reduce_only=False),
        TargetExposureChildOrder(quantity=1.0, reduce_only=True),
    )
    return run_legacy_child_order_sequence_probe(
        child_orders,
        dataset_id="e" * 64,
        target_identity_prefix="dual-shadow-sign-flip",
    )


def run_legacy_child_order_sequence_probe(
    child_orders: tuple[TargetExposureChildOrder, ...],
    *,
    dataset_id: str = "d" * 64,
    target_identity_prefix: str = "dual-shadow-child-order",
) -> LegacyExecutionProbeResult:
    """Execute controller-approved child orders through the legacy fill engine."""

    if not child_orders:
        raise ValueError("child_orders must be non-empty")
    dataset = _sequence_dataset(dataset_id=dataset_id, child_orders=child_orders)
    executor = MarketExecutor(dataset, _cost())
    initial_capital = 1_000.0
    book = BookState.zero(1, initial_capital, dataset.close[0])
    order_book = OrderBookState.empty()
    events: list[OrderEvent] = []

    for submit_index, child_order in enumerate(child_orders):
        _assert_safe_child_order(book, child_order)
        intent = OrderIntent.create(
            dataset_id=dataset.dataset_id,
            target_identity=f"{target_identity_prefix}-{submit_index}",
            execution_policy_digest=executor.cost.execution_policy_digest,
            symbol_index=0,
            requested_quantity=child_order.quantity,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.IOC,
            limit_price=None,
            stop_price=None,
            submit_index=submit_index,
            eligible_index=submit_index,
            expiry_index=None,
            submission_reference_price=float(dataset.close[submit_index, 0]),
            decision_equity=book.portfolio_value,
        )
        result = execute_stateful_orders(
            executor,
            book,
            order_book,
            (intent,),
            start_index=submit_index,
            bars=1,
        )
        events.extend(result.order_events)
        book = result.book
        order_book = result.order_book

    fills = canonicalize_legacy_fill_events(
        events,
        price_tick=0.1,
        lot_size=0.001,
    )
    final_equity = book.portfolio_value
    economics = CanonicalEconomicClosure(
        fee_minor=_minor(book.total_cost),
        funding_minor=_minor(book.funding_pnl),
        realized_pnl_minor=_minor(final_equity - initial_capital),
        final_equity_minor=_minor(final_equity),
        terminal_position_lots=int(round(float(book.quantities[0]) / 0.001)),
        terminal_open_orders=len(order_book.active_orders),
    )
    return LegacyExecutionProbeResult(fills=fills, economics=economics)


def _assert_safe_child_order(
    book: BookState,
    child_order: TargetExposureChildOrder,
) -> None:
    quantity = float(child_order.quantity)
    if not np.isfinite(quantity) or abs(quantity) <= _QUANTITY_TOLERANCE:
        raise ValueError("child order quantity must be finite and non-zero")
    if not child_order.reduce_only:
        return

    realized = float(book.quantities[0])
    if abs(realized) <= _QUANTITY_TOLERANCE:
        raise RuntimeError("reduce-only child order cannot execute while flat")
    if realized * quantity >= 0.0:
        raise RuntimeError("reduce-only child order must oppose the realized position")
    if abs(quantity) > abs(realized) + _QUANTITY_TOLERANCE:
        raise RuntimeError("reduce-only child order cannot cross through flat")


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


def _sequence_dataset(
    *,
    dataset_id: str,
    child_orders: tuple[TargetExposureChildOrder, ...],
) -> MarketDataset:
    fill_open_prices = [100.1 if order.quantity > 0.0 else 99.9 for order in child_orders]
    open_prices = np.asarray([[100.0], *[[price] for price in fill_open_prices]], dtype=np.float64)
    close = np.full_like(open_prices, 100.0)
    high = np.maximum(open_prices, close) + 1.0
    low = np.minimum(open_prices, close) - 1.0
    volume = np.full_like(close, 1_000.0)
    hour_ns = 60 * 60 * 1_000_000_000
    timestamps = np.asarray(
        [index * hour_ns for index in range(len(child_orders) + 1)],
        dtype="datetime64[ns]",
    )
    rows = len(child_orders) + 1
    return MarketDataset(
        dataset_id=dataset_id,
        symbols=("BTCUSDT",),
        timestamps=timestamps,
        features=np.zeros((rows, 1, 1), dtype=np.float32),
        global_features=np.zeros((rows, 1), dtype=np.float32),
        open=open_prices,
        high=high,
        low=low,
        close=close,
        volume=volume,
        funding_rate=np.zeros_like(close),
        tradable=np.ones_like(close, dtype=np.bool_),
        feature_available=np.ones((rows, 1, 1), dtype=np.bool_),
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


__all__ = [
    "LegacyExecutionProbeResult",
    "run_legacy_child_order_sequence_probe",
    "run_legacy_flat_long_flat_probe",
    "run_legacy_flat_long_flat_short_flat_probe",
]
