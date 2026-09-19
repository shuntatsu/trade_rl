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
    LiquidityAllocation,
    LiquidityPriority,
    LiquidityRequest,
    allocate_symbol_capacity,
)
from trade_rl.simulation.orders.model import OrderStatus, OrderType, PendingOrder
from trade_rl.simulation.quantities import accepted_fill_quantity, exact_quantity
from trade_rl.simulation.stateful.bar_lifecycle import StatefulBarContext
from trade_rl.simulation.stateful.runtime import StatefulExecutionRuntime

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


def _capacity_reference(
    runtime: StatefulExecutionRuntime,
    context: StatefulBarContext,
    *,
    symbol: int,
) -> tuple[float, float, float]:
    executor = runtime.executor
    dataset = executor.dataset
    processing_index = context.processing_index
    if executor.cost.processing_bar_volume_capacity:
        reference_index = processing_index
        reference_prices = context.open_prices
    else:
        reference_index = processing_index - 1
        if reference_index < 0:
            return 0.0, 0.0, float(context.open_prices[symbol])
        reference_prices = dataset.close[reference_index]
    market_notional = float(
        dataset.market_notional(reference_index, reference_prices)[symbol]
    )
    return (
        float(dataset.volume[reference_index, symbol]),
        market_notional,
        float(reference_prices[symbol]),
    )


def _execution_cost(
    runtime: StatefulExecutionRuntime,
    order: PendingOrder,
    *,
    trigger: TriggerDecision,
    processing_index: int,
    filled_notional: float,
    participation_rate: float,
) -> float:
    executor = runtime.executor
    symbol = order.intent.symbol_index
    dataset = executor.dataset
    maker = order.intent.order_type is OrderType.LIMIT and not (
        trigger.segment is TriggerSegment.OPEN
        and order.intent.eligible_index == processing_index
    )
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
                rule = executor.market_order_rule(symbol)
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
                        reduce_only=order.intent.reduce_only,
                        minimum_notional=executor.order_minimum_notional(
                            order.intent, float(context.minimum_notional[symbol])
                        ),
                        minimum_quantity=0.0 if rule is None else rule.minimum_quantity,
                        maximum_quantity=None
                        if rule is None
                        else rule.maximum_quantity,
                    )
                )
                metadata[order.order_id] = (order, trigger, path, rounded_price)

            if not requests:
                continue
            capacity_volume, capacity_market_notional, capacity_price = (
                _capacity_reference(runtime, context, symbol=symbol)
            )
            allocations, capacity = allocate_symbol_capacity(
                requests=requests,
                processing_volume=capacity_volume,
                processing_market_notional=capacity_market_notional,
                price=capacity_price,
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
                initial_position=runtime.book.exact_quantities[symbol],
            )
            remaining_capacity = capacity.initial_capacity_notional
            for allocation in allocations:
                _, trigger, order_path, execution_price = metadata[allocation.order_id]
                order = runtime.require_active_order(allocation.order_id)
                allocation = self._recheck_closing_allocation(
                    runtime, order, allocation
                )
                # Keep the original reservations for later orders. Released
                # capacity stays unused, and evidence reflects actual fills.
                allocation = replace(
                    allocation,
                    capacity_before=remaining_capacity,
                    capacity_after=max(
                        0.0, remaining_capacity - allocation.filled_notional
                    ),
                )
                remaining_capacity = allocation.capacity_after
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
                    if allocation.no_fill_reason in {
                        "reduce_only_exhausted",
                        "reduce_only_inventory_changed",
                    }:
                        self._expire_closing_remainder(
                            runtime, order, processing_index, allocation.no_fill_reason
                        )
                    continue

                cost_amount = _execution_cost(
                    runtime,
                    order,
                    trigger=trigger,
                    processing_index=processing_index,
                    filled_notional=allocation.filled_notional,
                    participation_rate=allocation.participation_rate,
                )
                # Validate the immutable order update before changing the book.
                updated = order.apply_fill(
                    quantity=allocation.filled_quantity,
                    notional=allocation.filled_notional,
                    processing_index=processing_index,
                    lot_size=allocation.lot_size,
                    lot_count=allocation.filled_lot_count,
                )
                fill_prices = context.open_prices.copy()
                fill_prices[symbol] = execution_price
                runtime.book.execute_fill(
                    symbol_index=symbol,
                    quantity=allocation.filled_quantity,
                    lot_size=allocation.lot_size,
                    lot_count=allocation.filled_lot_count,
                    fill_prices=fill_prices,
                    cost_amount=cost_amount,
                    turnover=(allocation.filled_notional / context.period_start_value),
                )
                executor._update_margin(runtime.book)
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
                    reason=updated.terminal_reason,
                    path=order_path,
                )
                runtime.total_cost += cost_amount
                runtime.filled_notional += allocation.filled_notional
                multipliers = runtime.book.contract_multipliers
                if multipliers is None:
                    raise RuntimeError("filled book is missing contract multipliers")
                runtime.filled_reference_notional += (
                    abs(allocation.filled_quantity)
                    * order.intent.submission_reference_price
                    * float(multipliers[symbol])
                )
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
                elif allocation.reduce_only_exhausted:
                    self._expire_closing_remainder(runtime, updated, processing_index)
            runtime.capacities.append(
                replace(
                    capacity,
                    consumed_capacity_notional=capacity.initial_capacity_notional
                    - remaining_capacity,
                    remaining_capacity_notional=remaining_capacity,
                )
            )
        return attempted_order_ids

    @staticmethod
    def _recheck_closing_allocation(
        runtime: StatefulExecutionRuntime,
        order: PendingOrder,
        allocation: LiquidityAllocation,
    ) -> LiquidityAllocation:
        if not order.intent.reduce_only:
            return allocation
        position = runtime.book.exact_quantities[order.intent.symbol_index]
        filled = accepted_fill_quantity(
            allocation.filled_quantity,
            lot_size=allocation.lot_size,
            lot_count=allocation.filled_lot_count,
        )
        reason = None
        if position * exact_quantity(order.remaining_quantity) >= 0:
            reason = "reduce_only_exhausted"
        elif abs(filled) > abs(position):
            reason = "reduce_only_inventory_changed"
        if reason is None:
            return allocation
        # Margin handling after an earlier fill can flatten the real book,
        # invalidating the allocator's projected inventory even within one bar.
        return replace(
            allocation,
            filled_quantity=0.0,
            filled_notional=0.0,
            participation_rate=0.0,
            no_fill_reason=reason,
            filled_lot_count=None,
            lot_size=0.0,
            reduce_only_exhausted=True,
        )

    @staticmethod
    def _expire_closing_remainder(
        runtime: StatefulExecutionRuntime,
        order: PendingOrder,
        processing_index: int,
        reason: str = "reduce_only_exhausted",
    ) -> None:
        updated = order.expire(processing_index=processing_index, reason=reason)
        runtime.order_book = runtime.order_book.replace(updated)
        runtime.expired_count += 1
        runtime.append_event(
            previous=order,
            updated=updated,
            event_type="expired",
            processing_index=processing_index,
            reason=reason,
        )
