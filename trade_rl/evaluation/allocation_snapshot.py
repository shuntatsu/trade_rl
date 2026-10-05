"""Observer-only source validation for live independent allocation accounts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, fields
from typing import Any

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.allocation import _context
from trade_rl.risk.pretrade import PreTradeRisk
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import MarketExecutor
from trade_rl.simulation.orders.model import OrderBookState, PendingOrder
from trade_rl.simulation.quantities import exact_quantity, parse_quantity
from trade_rl.strategies.allocation_snapshot import AllocationAccountSnapshot


def _timestamp(value: np.datetime64) -> str:
    time = value.astype("datetime64[ns]")
    if np.isnat(time) or time.astype(value.dtype) != value:
        raise ValueError("snapshot timestamp is outside exact nanosecond range")
    return str(np.datetime_as_string(time, unit="ns"))


def _order_facts(order: PendingOrder) -> dict[str, Any]:
    facts = asdict(order)
    requested = exact_quantity(order.intent.requested_quantity)
    exact_cumulative = order.exact_cumulative_filled_quantity
    assert exact_cumulative is not None
    cumulative = parse_quantity(exact_cumulative)
    facts["exact_requested_quantity"] = str(requested)
    facts["exact_remaining_quantity"] = str(requested - cumulative)
    return facts


def snapshot_allocation_account(
    executor: MarketExecutor,
    book: BookState,
    order_book: OrderBookState,
    *,
    account_id: str,
    pretrade_risk: PreTradeRisk,
    symbol_index: int,
    start_index: int,
) -> AllocationAccountSnapshot:
    """Copy an admitted live MARKET/zero-latency independent-symbol account.

    Requires a known Dataset/index clock and caller-refreshed canonical margin.
    No original book cache, account, order or executor randomness is changed.
    The bootstrap clock/account ID remain caller declarations, not authenticity.
    Vectors expose only the selected symbol; the full book remains digest-bound.
    """
    dataset = executor.dataset
    book.validate_processing_clock(
        dataset_id=dataset.dataset_id, index=start_index, require_known=True
    )
    for order in (*order_book.active_orders, *order_book.terminal_orders):
        # Raw dataclass construction does not enforce canonical intent identity.
        PendingOrder.from_mapping(asdict(order))
    # clone()/exact_quantities may refresh BookState's replacement cache.
    detached = deepcopy(book)
    context = _context(
        executor,
        detached,
        order_book,
        account_id=account_id,
        pretrade_risk=pretrade_risk,
        symbol_index=symbol_index,
        start_index=start_index,
    )
    time = _timestamp(dataset.timestamps[start_index])
    available = _timestamp(
        dataset.resolved_array("available_at")[start_index, symbol_index]
    )
    information_available = dataset.information_available
    assert information_available is not None
    if not information_available[start_index, symbol_index] or available > time:
        raise ValueError("current snapshot market source is unavailable")
    margin_check = detached.clone()
    executor._update_margin(margin_check)
    for name in (
        "margin_used",
        "maintenance_margin",
        "maintenance_requirement",
        "margin_deficit",
    ):
        if getattr(detached, name) != getattr(margin_check, name):
            raise ValueError("snapshot requires refreshed canonical margin facts")
    for order in order_book.active_orders:
        if order.intent.execution_policy_digest != executor.execution_policy_digest:
            raise ValueError("snapshot order belongs to another execution policy")
    facts: dict[str, object] = {}
    for field in fields(detached):
        if field.name == "_exact_quantities":
            continue
        value = getattr(detached, field.name)
        facts[field.name] = (
            [float(value[symbol_index])] if isinstance(value, np.ndarray) else value
        )
    facts["exact_quantities"] = (str(detached.exact_quantities[symbol_index]),)
    facts["equity"] = detached.portfolio_value
    facts["current_drawdown"] = max(
        0.0, 1 - detached.portfolio_value / detached.peak_value
    )
    return AllocationAccountSnapshot(
        account_id=context.account_id,
        dataset_id=dataset.dataset_id,
        symbol=context.symbol,
        symbol_index=symbol_index,
        decision_index=start_index,
        decision_time=time,
        available_at=available,
        execution_policy_digest=executor.execution_policy_digest,
        risk_digest=content_digest(pretrade_risk.config),
        source_state_digest=context.state_digest,
        observable_tradable=bool(
            dataset.observable_tradable(start_index)[symbol_index]
        ),
        book_facts=facts,
        active_orders=tuple(_order_facts(order) for order in order_book.active_orders),
    )


__all__ = ["snapshot_allocation_account"]
