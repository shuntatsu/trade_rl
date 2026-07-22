"""Trigger, capacity, fill, and fill-evidence processing by symbol."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from trade_rl.simulation.bar_path import (
    BarPath,
    PathMode,
    TriggerDecision,
    TriggerSegment,
    evaluate_trigger,
    select_bar_path,
)
from trade_rl.simulation.liquidity import (
    LiquidityPriority,
    LiquidityRequest,
    allocate_symbol_capacity,
)
from trade_rl.simulation.orders import OrderStatus, OrderType, PendingOrder
from trade_rl.simulation.stateful_bar_lifecycle import StatefulBarContext
from trade_rl.simulation.stateful_runtime import StatefulExecutionRuntime

if TYPE_CHECKING:
    from trade_rl.simulation.execution import MarketExecutor


_TOLERANCE = 1e-12


def _priority(
    order: PendingOrder,
    *,
    processing_index: int,
    newly_triggered: bool,
) -> LiquidityPriority:
    if order.intent.order_type is OrderType.STOP_MARKET:
        if newly_triggered:
            return LiquidityPriority.NEWLY_TRIGGERED_STOP
        return LiquidityPriority.PREVIOUSLY_TRIGGERED_STOP
    if order.intent.order_type is OrderType.MARKET:
        return LiquidityPriority.MARKET
    if order.intent.eligible_index < processing_index:
        return LiquidityPriority.OLDER_LIMIT
    return LiquidityPriority.NEWER_LIMIT


def _configured_fraction(
    runtime: StatefulExecutionRuntime,
    segment: TriggerSegment | None,
) -> float:
    if segment is None:
        return 0.0
    index = {
        TriggerSegment.OPEN: 0,
        TriggerSegment.FIRST_EXTREME: 1,
        TriggerSegment.SECOND_EXTREME: 2,
        TriggerSegment.CLOSE: 3,
    }[segment]
    return float(runtime.executor.cost.trigger_volume_fractions[index])


def _execution_cost(
    runtime: StatefulExecutionRuntime,
    order: PendingOrder,
    *,
    processing_index: int,
    filled_notional: float,
    participation_rate: float,
) -> float:
    executor = runtime.executor
    symbol = order.intent.symbol_index
    dataset = executor.dataset
    maker = order.intent.order_type is OrderType.LIMIT
    venue_fee = (
        executor.cost.maker_fee_rate
        + dataset.resolved_array("maker_fee_rate")[processing_index, symbol]
        if maker
        else executor.cost.taker_fee_rate
        + dataset.resolved_array("taker_fee_rate")[processing_index, symbol]
    )
    spread_multiplier = 0.5 if maker else 1.0
    impact = executor.cost.impact_rate * math.sqrt(participation_rate)
    slippage = float(executor._slippage_rates(1)[0])
    unit_cost = executor.cost.multiplier * (
        executor.cost.fee_rate
        + dataset.resolved_array("fee_rate")[processing_index, symbol]
        + venue_fee
        + spread_multiplier
        * (
            executor.cost.spread_rate
            + dataset.resolved_array("spread_rate")[processing_index, symbol]
        )
        + impact
        + slippage
    )
    return filled_notional * unit_cost


class StatefulSymbolFillProcessor:
    """Process trigger paths and shared symbol liquidity for accepted orders."""

    def __init__(self, executor: MarketExecutor) -> None:
        self.executor = executor

    def process_symbols(
        self,
        runtime: StatefulExecutionRuntime,
        context: StatefulBarContext,
        accepted: list[PendingOrder],
    ) -> set[str]:
        executor = runtime.executor
        dataset = executor.dataset
        processing_index = context.processing_index
        attempted_order_ids: set[str] = set()
        for symbol in range(dataset.n_symbols):
            symbol_orders = tuple(
                order for order in accepted if order.intent.symbol_index == symbol
            )
            if not symbol_orders:
                continue
            directions = frozenset(
                1 if order.remaining_quantity > 0.0 else -1 for order in symbol_orders
            )
            path = select_bar_path(
                open_price=float(dataset.open[processing_index, symbol]),
                high=float(dataset.high[processing_index, symbol]),
                low=float(dataset.low[processing_index, symbol]),
                close=float(dataset.close[processing_index, symbol]),
                mode=PathMode(executor.cost.path_mode),
                active_directions=directions,
            )
            requests: list[LiquidityRequest] = []
            metadata: dict[
                str,
                tuple[PendingOrder, TriggerDecision, BarPath, float],
            ] = {}
            for accepted_order in symbol_orders:
                order = runtime.require_active_order(accepted_order.order_id)
                original_trigger_index = order.trigger_index
                trigger = evaluate_trigger(order, path)
                trigger = replace(
                    trigger,
                    available_volume_fraction=_configured_fraction(
                        runtime, trigger.segment
                    ),
                )
                newly_triggered = trigger.triggered and original_trigger_index is None
                if newly_triggered:
                    updated = order.mark_triggered(processing_index=processing_index)
                    runtime.order_book = runtime.order_book.replace(updated)
                    runtime.append_event(
                        previous=order,
                        updated=updated,
                        event_type="triggered",
                        processing_index=processing_index,
                        trigger_segment=(
                            None if trigger.segment is None else trigger.segment.value
                        ),
                        available_volume_fraction=(trigger.available_volume_fraction),
                        path=path,
                    )
                    order = updated
                attempted_order_ids.add(order.order_id)
                if not trigger.executable:
                    runtime.append_event(
                        previous=order,
                        updated=order,
                        event_type="no_fill",
                        processing_index=processing_index,
                        trigger_segment=None,
                        available_volume_fraction=0.0,
                        reason=trigger.reason,
                        path=path,
                    )
                    continue

                prices = context.open_prices.copy()
                assert trigger.execution_price is not None
                prices[symbol] = trigger.execution_price
                directions_vector = np.zeros(dataset.n_symbols, dtype=np.float64)
                directions_vector[symbol] = order.remaining_quantity
                rounded_price = float(
                    executor._round_prices(
                        prices,
                        index=processing_index,
                        directions=directions_vector,
                    )[symbol]
                )
                requests.append(
                    LiquidityRequest(
                        order_id=order.order_id,
                        remaining_quantity=order.remaining_quantity,
                        execution_price=rounded_price,
                        available_volume_fraction=(trigger.available_volume_fraction),
                        priority=_priority(
                            order,
                            processing_index=processing_index,
                            newly_triggered=newly_triggered,
                        ),
                        eligible_index=order.intent.eligible_index,
                    )
                )
                metadata[order.order_id] = (order, trigger, path, rounded_price)

            if not requests:
                continue
            market_notional = float(
                dataset.market_notional(
                    processing_index,
                    context.open_prices,
                )[symbol]
            )
            allocations, capacity = allocate_symbol_capacity(
                requests=requests,
                processing_volume=float(dataset.volume[processing_index, symbol]),
                processing_market_notional=market_notional,
                price=float(context.open_prices[symbol]),
                contract_multiplier=float(
                    dataset.resolved_array("contract_multipliers")[symbol]
                ),
                participation_limit=float(
                    min(
                        executor.cost.max_participation_rate,
                        dataset.resolved_array("max_participation_rate")[
                            processing_index, symbol
                        ],
                    )
                ),
                lot_size=float(context.lot_size[symbol]),
                minimum_notional=float(context.minimum_notional[symbol]),
            )
            runtime.capacities.append(capacity)
            for allocation in allocations:
                _, trigger, order_path, execution_price = metadata[allocation.order_id]
                order = runtime.require_active_order(allocation.order_id)
                if abs(allocation.filled_quantity) <= _TOLERANCE:
                    runtime.append_event(
                        previous=order,
                        updated=order,
                        event_type="no_fill",
                        processing_index=processing_index,
                        capacity_before=allocation.capacity_before,
                        capacity_after=allocation.capacity_after,
                        trigger_segment=(
                            None if trigger.segment is None else trigger.segment.value
                        ),
                        available_volume_fraction=(trigger.available_volume_fraction),
                        reason=allocation.no_fill_reason,
                        path=order_path,
                    )
                    continue

                cost_amount = _execution_cost(
                    runtime,
                    order,
                    processing_index=processing_index,
                    filled_notional=allocation.filled_notional,
                    participation_rate=allocation.participation_rate,
                )
                target_quantities = runtime.book.quantities.copy()
                target_quantities[symbol] += allocation.filled_quantity
                fill_prices = context.open_prices.copy()
                fill_prices[symbol] = execution_price
                runtime.book.execute(
                    fill_prices=fill_prices,
                    target_quantities=target_quantities,
                    cost_amount=cost_amount,
                    turnover=(allocation.filled_notional / context.period_start_value),
                )
                executor._update_margin(runtime.book)
                updated = order.apply_fill(
                    quantity=allocation.filled_quantity,
                    notional=allocation.filled_notional,
                    processing_index=processing_index,
                )
                runtime.order_book = runtime.order_book.replace(updated)
                runtime.append_event(
                    previous=order,
                    updated=updated,
                    event_type=(
                        "filled"
                        if updated.status is OrderStatus.FILLED
                        else "partial_fill"
                    ),
                    processing_index=processing_index,
                    filled_quantity=allocation.filled_quantity,
                    execution_price=execution_price,
                    filled_notional=allocation.filled_notional,
                    capacity_before=allocation.capacity_before,
                    capacity_after=allocation.capacity_after,
                    participation_rate=allocation.participation_rate,
                    trigger_segment=(
                        None if trigger.segment is None else trigger.segment.value
                    ),
                    available_volume_fraction=(trigger.available_volume_fraction),
                    path=order_path,
                )
                runtime.total_cost += cost_amount
                runtime.filled_notional += allocation.filled_notional
                runtime.filled_by_symbol[symbol] += allocation.filled_notional
                runtime.participation_by_symbol[symbol] = max(
                    runtime.participation_by_symbol[symbol],
                    allocation.participation_rate,
                )
                runtime.cost_by_symbol[symbol] += cost_amount
                runtime.fill_count += 1
                runtime.max_participation = max(
                    runtime.max_participation,
                    allocation.participation_rate,
                )
                if updated.status is OrderStatus.FILLED:
                    runtime.completed_fills += 1
        return attempted_order_ids
