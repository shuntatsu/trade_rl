"""Stateful OHLCV order execution orchestration."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Sequence

import numpy as np

from trade_rl.simulation.accounting import BookState, EconomicTerminationReason
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
    SymbolCapacityEvidence,
    allocate_symbol_capacity,
)
from trade_rl.simulation.order_admission import OrderAdmissionPolicy
from trade_rl.simulation.orders import (
    OrderBookState,
    OrderEvent,
    OrderIntent,
    OrderStatus,
    OrderType,
    PendingOrder,
    TimeInForce,
)

if TYPE_CHECKING:
    from trade_rl.simulation.execution import MarketExecutor

_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class StatefulExecutionResult:
    book: BookState
    order_book: OrderBookState
    next_index: int
    bars_advanced: int
    order_events: tuple[OrderEvent, ...]
    capacity_evidence: tuple[SymbolCapacityEvidence, ...]
    interval_cost: float
    interval_funding: float
    interval_borrow_cost: float
    interval_dividend: float
    interval_cash_interest: float
    interval_net_return: float
    interval_log_return: float
    requested_notional: float
    filled_notional: float
    completed_fill_count: int
    rejected_count: int
    expired_count: int
    fill_count: int
    max_participation: float
    termination_reason: str | None


def _timestamp_ns(executor: MarketExecutor, index: int) -> int:
    timestamp = executor.dataset.timestamps[index]
    return int(timestamp.astype("datetime64[ns]").astype(np.int64))


def _event(
    executor: MarketExecutor,
    *,
    sequence: int,
    previous: PendingOrder,
    updated: PendingOrder,
    event_type: str,
    processing_index: int,
    filled_quantity: float = 0.0,
    execution_price: float | None = None,
    filled_notional: float = 0.0,
    capacity_before: float = 0.0,
    capacity_after: float = 0.0,
    participation_rate: float = 0.0,
    trigger_segment: str | None = None,
    available_volume_fraction: float = 0.0,
    reason: str | None = None,
    path: BarPath | None = None,
) -> OrderEvent:
    return OrderEvent(
        schema_version="order_event_v1",
        sequence=sequence,
        order_id=updated.order_id,
        replaced_order_id=updated.intent.replaced_order_id,
        dataset_id=updated.intent.dataset_id,
        execution_policy_digest=updated.intent.execution_policy_digest,
        symbol_index=updated.intent.symbol_index,
        event_type=event_type,
        processing_index=processing_index,
        timestamp_ns=_timestamp_ns(executor, processing_index),
        previous_status=previous.status,
        new_status=updated.status,
        requested_quantity=updated.intent.requested_quantity,
        remaining_quantity=updated.remaining_quantity,
        filled_quantity=filled_quantity,
        execution_price=execution_price,
        filled_notional=filled_notional,
        capacity_before=capacity_before,
        capacity_after=capacity_after,
        participation_rate=participation_rate,
        trigger_segment=trigger_segment,
        available_volume_fraction=available_volume_fraction,
        reason=reason,
        path_mode=(executor.cost.path_mode if path is None else path.mode.value),
        path_points=() if path is None else path.points,
    )


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
    executor: MarketExecutor,
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
    return float(executor.cost.trigger_volume_fractions[index])


def _cancel_active_orders(
    executor: MarketExecutor,
    state: OrderBookState,
    events: list[OrderEvent],
    *,
    processing_index: int,
    reason: str,
    symbol_mask: np.ndarray | None = None,
) -> OrderBookState:
    result = state
    for order in tuple(result.active_orders):
        if symbol_mask is not None and not symbol_mask[order.intent.symbol_index]:
            continue
        updated = order.cancel(processing_index=processing_index, reason=reason)
        result = result.replace(updated)
        events.append(
            _event(
                executor,
                sequence=len(events),
                previous=order,
                updated=updated,
                event_type="cancelled",
                processing_index=processing_index,
                reason=reason,
            )
        )
    return result


def _execution_cost(
    executor: MarketExecutor,
    order: PendingOrder,
    *,
    processing_index: int,
    filled_notional: float,
    participation_rate: float,
) -> float:
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


def execute_stateful_orders(
    executor: MarketExecutor,
    book: BookState,
    order_book: OrderBookState,
    intents: Sequence[OrderIntent],
    *,
    start_index: int,
    bars: int,
) -> StatefulExecutionResult:
    """Execute persistent orders over one or more processing bars."""

    dataset = executor.dataset
    if bars <= 0:
        raise ValueError("bars must be positive")
    if start_index < 0 or start_index + bars >= dataset.n_bars:
        raise ValueError("execution interval is outside the dataset")
    if book.quantities.shape != (dataset.n_symbols,):
        raise ValueError("book quantities do not match market symbols")

    result_book = book.clone()
    state = order_book
    events: list[OrderEvent] = []
    capacities: list[SymbolCapacityEvidence] = []
    for intent in intents:
        pending = PendingOrder.from_intent(intent)
        state = state.add(pending)
        events.append(
            _event(
                executor,
                sequence=len(events),
                previous=pending,
                updated=pending,
                event_type="submitted",
                processing_index=intent.submit_index,
            )
        )

    starting_value = result_book.portfolio_value
    if not math.isfinite(starting_value) or starting_value <= 0.0:
        raise ValueError("stateful execution requires positive starting equity")
    result_multipliers = result_book.contract_multipliers
    if result_multipliers is None:
        raise ValueError("stateful execution requires contract multipliers")
    requested_notional = 0.0
    for order in state.active_orders:
        requested_notional += (
            abs(order.remaining_quantity)
            * order.intent.submission_reference_price
            * float(result_multipliers[order.intent.symbol_index])
        )
    total_cost = 0.0
    total_funding = 0.0
    total_borrow = 0.0
    total_dividend = 0.0
    total_cash_interest = 0.0
    filled_notional = 0.0
    completed_fills = 0
    rejected_count = 0
    expired_count = 0
    fill_count = 0
    max_participation = 0.0

    admission = OrderAdmissionPolicy(
        expected_dataset_id=dataset.dataset_id,
        expected_execution_policy_digest=executor.execution_policy_digest,
        allow_short=executor.cost.allow_short,
        max_leverage=executor.cost.max_leverage,
    )

    for offset in range(bars):
        previous_index = start_index + offset
        processing_index = previous_index + 1
        period_start_value = max(result_book.portfolio_value, _TOLERANCE)

        split = dataset.resolved_array("split_factor")[processing_index]
        split_mask = np.abs(split - 1.0) > _TOLERANCE
        if np.any(split_mask):
            state = _cancel_active_orders(
                executor,
                state,
                events,
                processing_index=processing_index,
                reason="split_adjustment_required",
                symbol_mask=split_mask,
            )
        result_book.apply_split(split)

        inactive = ~dataset.resolved_array("asset_active")[processing_index]
        if np.any(inactive):
            state = _cancel_active_orders(
                executor,
                state,
                events,
                processing_index=processing_index,
                reason="inactive_asset",
                symbol_mask=inactive,
            )
            if np.any(inactive & (np.abs(result_book.quantities) > _TOLERANCE)):
                result_book.settle_positions(
                    mask=inactive,
                    prices=dataset.open[processing_index],
                    recovery=dataset.resolved_array("delisting_recovery")[
                        processing_index
                    ],
                )

        open_prices = dataset.open[processing_index]
        result_book.revalue(open_prices)
        value_at_open = max(result_book.portfolio_value, 0.0)
        gap_return = value_at_open / period_start_value - 1.0

        tick, lot, minimum = executor.effective_rule_arrays(index=processing_index)
        projected_book = result_book.clone()
        accepted: list[PendingOrder] = []
        for order in tuple(
            sorted(
                state.active_orders,
                key=lambda item: (item.intent.eligible_index, item.order_id),
            )
        ):
            if (
                order.intent.expiry_index is not None
                and processing_index > order.intent.expiry_index
            ):
                updated = order.expire(
                    processing_index=processing_index,
                    reason="time_in_force_expired",
                )
                state = state.replace(updated)
                expired_count += 1
                events.append(
                    _event(
                        executor,
                        sequence=len(events),
                        previous=order,
                        updated=updated,
                        event_type="expired",
                        processing_index=processing_index,
                        reason=updated.terminal_reason,
                    )
                )
                continue
            if processing_index < order.intent.eligible_index:
                updated = order.mark_latency_wait(processing_index=processing_index)
                state = state.replace(updated)
                events.append(
                    _event(
                        executor,
                        sequence=len(events),
                        previous=order,
                        updated=updated,
                        event_type="latency_wait",
                        processing_index=processing_index,
                        reason="latency_wait",
                    )
                )
                continue

            symbol = order.intent.symbol_index
            decision = admission.evaluate(
                order.intent,
                remaining_quantity=order.remaining_quantity,
                book=projected_book,
                processing_index=processing_index,
                asset_active=bool(
                    dataset.resolved_array("asset_active")[processing_index, symbol]
                ),
                tradable=bool(dataset.tradable[processing_index, symbol]),
                buy_allowed=bool(
                    dataset.resolved_array("buy_allowed")[processing_index, symbol]
                ),
                sell_allowed=bool(
                    dataset.resolved_array("sell_allowed")[processing_index, symbol]
                ),
                borrow_available=bool(
                    dataset.resolved_array("borrow_available")[processing_index, symbol]
                ),
                tick_size=float(tick[symbol]),
                lot_size=float(lot[symbol]),
                minimum_notional=float(minimum[symbol]),
                reference_prices=open_prices,
            )
            if not decision.accepted:
                updated = order.reject(
                    processing_index=processing_index,
                    reason=decision.reason or "admission_rejected",
                )
                state = state.replace(updated)
                rejected_count += 1
                events.append(
                    _event(
                        executor,
                        sequence=len(events),
                        previous=order,
                        updated=updated,
                        event_type="rejected",
                        processing_index=processing_index,
                        reason=updated.terminal_reason,
                    )
                )
                continue

            current = order
            if order.status in {OrderStatus.SUBMITTED, OrderStatus.LATENCY_WAIT}:
                current = order.mark_eligible(processing_index=processing_index)
                state = state.replace(current)
                events.append(
                    _event(
                        executor,
                        sequence=len(events),
                        previous=order,
                        updated=current,
                        event_type="eligible",
                        processing_index=processing_index,
                    )
                )
            accepted.append(current)
            admitted = decision.admitted_quantity
            projected_book.quantities[symbol] += admitted
            projected_multipliers = projected_book.contract_multipliers
            if projected_multipliers is None:
                raise ValueError("projected order book requires contract multipliers")
            cash_delta = (
                admitted * open_prices[symbol] * float(projected_multipliers[symbol])
            )
            projected_book.cash -= cash_delta
            projected_book.revalue(open_prices)

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
                order = next(
                    item
                    for item in state.active_orders
                    if item.order_id == accepted_order.order_id
                )
                original_trigger_index = order.trigger_index
                trigger = evaluate_trigger(order, path)
                trigger = replace(
                    trigger,
                    available_volume_fraction=_configured_fraction(
                        executor, trigger.segment
                    ),
                )
                newly_triggered = trigger.triggered and original_trigger_index is None
                if newly_triggered:
                    updated = order.mark_triggered(processing_index=processing_index)
                    state = state.replace(updated)
                    events.append(
                        _event(
                            executor,
                            sequence=len(events),
                            previous=order,
                            updated=updated,
                            event_type="triggered",
                            processing_index=processing_index,
                            trigger_segment=(
                                None
                                if trigger.segment is None
                                else trigger.segment.value
                            ),
                            available_volume_fraction=(
                                trigger.available_volume_fraction
                            ),
                            path=path,
                        )
                    )
                    order = updated
                attempted_order_ids.add(order.order_id)
                if not trigger.executable:
                    events.append(
                        _event(
                            executor,
                            sequence=len(events),
                            previous=order,
                            updated=order,
                            event_type="no_fill",
                            processing_index=processing_index,
                            trigger_segment=None,
                            available_volume_fraction=0.0,
                            reason=trigger.reason,
                            path=path,
                        )
                    )
                    continue

                prices = open_prices.copy()
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

            if requests:
                market_notional = float(
                    dataset.market_notional(
                        processing_index,
                        open_prices,
                    )[symbol]
                )
                allocations, capacity = allocate_symbol_capacity(
                    requests=requests,
                    processing_volume=float(dataset.volume[processing_index, symbol]),
                    processing_market_notional=market_notional,
                    price=float(open_prices[symbol]),
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
                    lot_size=float(lot[symbol]),
                    minimum_notional=float(minimum[symbol]),
                )
                capacities.append(capacity)
                for allocation in allocations:
                    order, trigger, order_path, execution_price = metadata[
                        allocation.order_id
                    ]
                    order = next(
                        item
                        for item in state.active_orders
                        if item.order_id == order.order_id
                    )
                    if abs(allocation.filled_quantity) <= _TOLERANCE:
                        events.append(
                            _event(
                                executor,
                                sequence=len(events),
                                previous=order,
                                updated=order,
                                event_type="no_fill",
                                processing_index=processing_index,
                                capacity_before=allocation.capacity_before,
                                capacity_after=allocation.capacity_after,
                                trigger_segment=(
                                    None
                                    if trigger.segment is None
                                    else trigger.segment.value
                                ),
                                available_volume_fraction=(
                                    trigger.available_volume_fraction
                                ),
                                reason=allocation.no_fill_reason,
                                path=order_path,
                            )
                        )
                        continue

                    cost_amount = _execution_cost(
                        executor,
                        order,
                        processing_index=processing_index,
                        filled_notional=allocation.filled_notional,
                        participation_rate=allocation.participation_rate,
                    )
                    target_quantities = result_book.quantities.copy()
                    target_quantities[symbol] += allocation.filled_quantity
                    fill_prices = open_prices.copy()
                    fill_prices[symbol] = execution_price
                    result_book.execute(
                        fill_prices=fill_prices,
                        target_quantities=target_quantities,
                        cost_amount=cost_amount,
                        turnover=allocation.filled_notional / period_start_value,
                    )
                    executor._update_margin(result_book)
                    updated = order.apply_fill(
                        quantity=allocation.filled_quantity,
                        notional=allocation.filled_notional,
                        processing_index=processing_index,
                    )
                    state = state.replace(updated)
                    event_type = (
                        "filled"
                        if updated.status is OrderStatus.FILLED
                        else "partial_fill"
                    )
                    events.append(
                        _event(
                            executor,
                            sequence=len(events),
                            previous=order,
                            updated=updated,
                            event_type=event_type,
                            processing_index=processing_index,
                            filled_quantity=allocation.filled_quantity,
                            execution_price=execution_price,
                            filled_notional=allocation.filled_notional,
                            capacity_before=allocation.capacity_before,
                            capacity_after=allocation.capacity_after,
                            participation_rate=allocation.participation_rate,
                            trigger_segment=(
                                None
                                if trigger.segment is None
                                else trigger.segment.value
                            ),
                            available_volume_fraction=(
                                trigger.available_volume_fraction
                            ),
                            path=order_path,
                        )
                    )
                    total_cost += cost_amount
                    filled_notional += allocation.filled_notional
                    fill_count += 1
                    max_participation = max(
                        max_participation, allocation.participation_rate
                    )
                    if updated.status is OrderStatus.FILLED:
                        completed_fills += 1

        for order_id in tuple(sorted(attempted_order_ids)):
            matching = tuple(
                order for order in state.active_orders if order.order_id == order_id
            )
            if not matching:
                continue
            order = matching[0]
            reason: str | None = None
            if order.intent.time_in_force is TimeInForce.IOC:
                reason = "ioc_remainder"
            elif (
                not executor.cost.partial_fill_carry
                and order.status is OrderStatus.PARTIALLY_FILLED
            ):
                reason = "partial_fill_carry_disabled"
            if reason is not None:
                updated = order.expire(
                    processing_index=processing_index,
                    reason=reason,
                )
                state = state.replace(updated)
                expired_count += 1
                events.append(
                    _event(
                        executor,
                        sequence=len(events),
                        previous=order,
                        updated=updated,
                        event_type="expired",
                        processing_index=processing_index,
                        reason=reason,
                    )
                )

        if result_book.insolvent:
            state = _cancel_active_orders(
                executor,
                state,
                events,
                processing_index=processing_index,
                reason="economic_termination",
            )
            executor._flatten_after_termination(result_book, open_prices)

        intrabar_asset_returns = (
            dataset.resolved_array("mark_price")[processing_index] / open_prices - 1.0
        )
        _ = gap_return + float(np.dot(result_book.weights, intrabar_asset_returns))
        dividend_amount = result_book.apply_dividend(
            dataset.resolved_array("dividend")[processing_index]
        )
        cash_interest = result_book.apply_cash_interest(
            float(dataset.resolved_array("cash_rate")[processing_index]),
            year_fraction=dataset.elapsed_year_fraction(
                previous_index,
                processing_index,
            ),
        )
        total_dividend += dividend_amount
        total_cash_interest += cash_interest
        funding_amount, borrow_amount = executor._charge_carry(
            result_book,
            index=processing_index,
        )
        total_funding += funding_amount
        total_borrow += borrow_amount
        result_book.mark_to_market(
            mark_prices=dataset.resolved_array("mark_price")[processing_index],
            funding_amount=funding_amount,
            period_start_value=period_start_value,
        )
        executor._update_margin(result_book)
        if result_book.insolvent:
            state = _cancel_active_orders(
                executor,
                state,
                events,
                processing_index=processing_index,
                reason="economic_termination",
            )
            executor._flatten_after_termination(
                result_book,
                dataset.resolved_array("mark_price")[processing_index],
            )

    ending_value = max(result_book.portfolio_value, 0.0)
    interval_net_return = max(
        ending_value / starting_value - 1.0,
        -1.0 + 1e-12,
    )
    reason = (
        None
        if result_book.termination_reason is None
        else EconomicTerminationReason(result_book.termination_reason).value
    )
    return StatefulExecutionResult(
        book=result_book,
        order_book=state,
        next_index=start_index + bars,
        bars_advanced=bars,
        order_events=tuple(events),
        capacity_evidence=tuple(capacities),
        interval_cost=total_cost,
        interval_funding=total_funding,
        interval_borrow_cost=total_borrow,
        interval_dividend=total_dividend,
        interval_cash_interest=total_cash_interest,
        interval_net_return=interval_net_return,
        interval_log_return=math.log1p(interval_net_return),
        requested_notional=requested_notional,
        filled_notional=filled_notional,
        completed_fill_count=completed_fills,
        rejected_count=rejected_count,
        expired_count=expired_count,
        fill_count=fill_count,
        max_participation=max_participation,
        termination_reason=reason,
    )
