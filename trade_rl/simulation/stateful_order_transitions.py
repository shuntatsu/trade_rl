"""Admission and non-fill transitions for stateful orders."""

from __future__ import annotations

from typing import TYPE_CHECKING

from trade_rl.simulation.order_admission import OrderAdmissionPolicy
from trade_rl.simulation.orders import OrderStatus, PendingOrder, TimeInForce
from trade_rl.simulation.stateful_bar_lifecycle import StatefulBarContext
from trade_rl.simulation.stateful_runtime import StatefulExecutionRuntime

if TYPE_CHECKING:
    from trade_rl.simulation.execution import MarketExecutor


class StatefulOrderTransitionProcessor:
    """Prepare active orders for symbol-level trigger and fill processing."""

    def __init__(self, executor: MarketExecutor) -> None:
        self.executor = executor
        dataset = executor.dataset
        self.admission = OrderAdmissionPolicy(
            expected_dataset_id=dataset.dataset_id,
            expected_execution_policy_digest=executor.execution_policy_digest,
            allow_short=executor.cost.allow_short,
            max_leverage=executor.cost.max_leverage,
        )

    def prepare_orders(
        self,
        runtime: StatefulExecutionRuntime,
        context: StatefulBarContext,
    ) -> list[PendingOrder]:
        executor = runtime.executor
        dataset = executor.dataset
        processing_index = context.processing_index
        projected_book = runtime.book.clone()
        accepted: list[PendingOrder] = []
        for order in tuple(
            sorted(
                runtime.order_book.active_orders,
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
                runtime.order_book = runtime.order_book.replace(updated)
                runtime.expired_count += 1
                runtime.append_event(
                    previous=order,
                    updated=updated,
                    event_type="expired",
                    processing_index=processing_index,
                    reason=updated.terminal_reason,
                )
                continue
            if processing_index < order.intent.eligible_index:
                updated = order.mark_latency_wait(processing_index=processing_index)
                runtime.order_book = runtime.order_book.replace(updated)
                runtime.append_event(
                    previous=order,
                    updated=updated,
                    event_type="latency_wait",
                    processing_index=processing_index,
                    reason="latency_wait",
                )
                continue

            symbol = order.intent.symbol_index
            decision = self.admission.evaluate(
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
                    dataset.resolved_array("borrow_available")[
                        processing_index, symbol
                    ]
                ),
                tick_size=float(context.tick_size[symbol]),
                lot_size=float(context.lot_size[symbol]),
                minimum_notional=float(context.minimum_notional[symbol]),
                reference_prices=context.open_prices,
            )
            if not decision.accepted:
                updated = order.reject(
                    processing_index=processing_index,
                    reason=decision.reason or "admission_rejected",
                )
                runtime.order_book = runtime.order_book.replace(updated)
                runtime.rejected_count += 1
                runtime.append_event(
                    previous=order,
                    updated=updated,
                    event_type="rejected",
                    processing_index=processing_index,
                    reason=updated.terminal_reason,
                )
                continue

            current = order
            if order.status in {OrderStatus.SUBMITTED, OrderStatus.LATENCY_WAIT}:
                current = order.mark_eligible(processing_index=processing_index)
                runtime.order_book = runtime.order_book.replace(current)
                runtime.append_event(
                    previous=order,
                    updated=current,
                    event_type="eligible",
                    processing_index=processing_index,
                )
            accepted.append(current)
            admitted = decision.admitted_quantity
            projected_book.quantities[symbol] += admitted
            multipliers = projected_book.contract_multipliers
            if multipliers is None:
                raise ValueError("projected order book requires contract multipliers")
            projected_book.cash -= (
                admitted
                * context.open_prices[symbol]
                * float(multipliers[symbol])
            )
            projected_book.revalue(context.open_prices)
        return accepted

    def expire_attempted_remainders(
        self,
        runtime: StatefulExecutionRuntime,
        *,
        processing_index: int,
        attempted_order_ids: set[str],
    ) -> None:
        for order_id in tuple(sorted(attempted_order_ids)):
            order = runtime.active_order(order_id)
            if order is None:
                continue
            reason: str | None = None
            if order.intent.time_in_force is TimeInForce.IOC:
                reason = "ioc_remainder"
            elif (
                not runtime.executor.cost.partial_fill_carry
                and order.status is OrderStatus.PARTIALLY_FILLED
            ):
                reason = "partial_fill_carry_disabled"
            if reason is None:
                continue
            updated = order.expire(
                processing_index=processing_index,
                reason=reason,
            )
            runtime.order_book = runtime.order_book.replace(updated)
            runtime.expired_count += 1
            runtime.append_event(
                previous=order,
                updated=updated,
                event_type="expired",
                processing_index=processing_index,
                reason=reason,
            )
