"""Admission and non-fill transitions for stateful orders."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

import numpy as np

from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.orders.admission import OrderAdmissionPolicy
from trade_rl.simulation.orders.model import OrderStatus, PendingOrder, TimeInForce
from trade_rl.simulation.stateful.bar_lifecycle import StatefulBarContext
from trade_rl.simulation.stateful.runtime import StatefulExecutionRuntime

if TYPE_CHECKING:
    from trade_rl.simulation.execution import MarketExecutor


@dataclass(slots=True)
class _AdmissionBookProjection:
    """Small mutable book view for sequential same-bar admission checks."""

    quantities: np.ndarray
    cash: float
    mark_prices: np.ndarray
    contract_multipliers: np.ndarray
    insolvent: bool
    exact_quantities: list[Fraction]
    actual_equity: float
    actual_gross_notional: float

    @classmethod
    def from_book(
        cls,
        book: BookState,
        *,
        mark_prices: np.ndarray,
    ) -> _AdmissionBookProjection:
        multipliers = book.contract_multipliers
        if multipliers is None:
            raise ValueError("projected order book requires contract multipliers")
        exact_quantities = list(book.exact_quantities)
        quantities = book.quantities.copy()
        resolved_prices = np.asarray(mark_prices, dtype=np.float64).reshape(-1).copy()
        resolved_multipliers = (
            np.asarray(
                multipliers,
                dtype=np.float64,
            )
            .reshape(-1)
            .copy()
        )
        equity = float(
            book.cash + (quantities * resolved_prices * resolved_multipliers).sum()
        )
        if not np.isfinite(equity):
            raise ValueError("portfolio value became non-finite")
        return cls(
            quantities=quantities,
            cash=book.cash,
            mark_prices=resolved_prices,
            contract_multipliers=resolved_multipliers,
            insolvent=book.insolvent or equity <= 0.0,
            exact_quantities=exact_quantities,
            actual_equity=equity,
            actual_gross_notional=float(
                np.abs(quantities * resolved_prices * resolved_multipliers).sum()
            ),
        )

    @property
    def portfolio_value(self) -> float:
        return float(
            self.cash
            + (self.quantities * self.mark_prices * self.contract_multipliers).sum()
        )

    def apply_admitted_order(
        self,
        *,
        symbol_index: int,
        quantity: float,
        admitted_exact_quantity: Fraction,
        price: float,
    ) -> None:
        self.exact_quantities[symbol_index] += admitted_exact_quantity
        self.quantities[symbol_index] += quantity
        self.cash -= quantity * price * float(self.contract_multipliers[symbol_index])
        equity = self.portfolio_value
        if not np.isfinite(equity):
            raise ValueError("portfolio value became non-finite")
        if not self.insolvent and equity <= 0.0:
            self.insolvent = True


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
        active_orders = runtime.order_book.active_orders
        if not active_orders:
            if runtime.book.insolvent:
                # begin_bar may flatten a margin-called book by replacing its
                # public quantities; reconcile the exact-lot view before returning.
                _ = runtime.book.exact_quantities
            return []

        actual_positions = runtime.book.exact_quantities
        projected_book = _AdmissionBookProjection.from_book(
            runtime.book,
            mark_prices=context.open_prices,
        )
        accepted: list[PendingOrder] = []
        for order in tuple(
            sorted(
                active_orders,
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
            rule = executor.market_order_rule(symbol)
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
                    dataset.resolved_array("borrow_available")[processing_index, symbol]
                ),
                tick_size=float(context.tick_size[symbol]),
                lot_size=float(context.lot_size[symbol]),
                minimum_notional=executor.order_minimum_notional(
                    order.intent, float(context.minimum_notional[symbol])
                ),
                minimum_quantity=0.0 if rule is None else rule.minimum_quantity,
                maximum_quantity=None if rule is None else rule.maximum_quantity,
                market_only=rule is not None,
                reference_prices=context.open_prices,
                actual_position=actual_positions[symbol],
            )
            if not decision.accepted:
                reason = decision.reason or "admission_rejected"
                if order.status in {
                    OrderStatus.TRIGGERED,
                    OrderStatus.PARTIALLY_FILLED,
                }:
                    updated = order.expire(
                        processing_index=processing_index,
                        reason=reason,
                    )
                    runtime.expired_count += 1
                    event_type = "expired"
                else:
                    updated = order.reject(
                        processing_index=processing_index,
                        reason=reason,
                    )
                    runtime.rejected_count += 1
                    event_type = "rejected"
                runtime.order_book = runtime.order_book.replace(updated)
                runtime.append_event(
                    previous=order,
                    updated=updated,
                    event_type=event_type,
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
            assert decision.admitted_exact_quantity is not None
            projected_book.apply_admitted_order(
                symbol_index=symbol,
                quantity=admitted,
                admitted_exact_quantity=decision.admitted_exact_quantity,
                price=float(context.open_prices[symbol]),
            )
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
