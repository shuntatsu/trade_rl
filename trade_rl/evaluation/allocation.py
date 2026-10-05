"""Opt-in independent-account allocator → hard risk → canonical execution.

The current-close return forecast is a declared surrogate: MARKET orders first
fill on the next processing bar. This adapter neither fits forecasts nor runs
a Study, and its supplied estimates are not economic evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from fractions import Fraction

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.risk.pretrade import PreTradeRisk, RiskConstrainedTarget
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import MarketExecutor
from trade_rl.simulation.orders.model import OrderBookState, OrderType
from trade_rl.simulation.stateful.execution import StatefulExecutionResult
from trade_rl.simulation.targets.execution import (
    execute_quantity_hold_statefully,
    execute_target_statefully,
)
from trade_rl.strategies.allocation import (
    AfterCostTargetAllocator,
    AllocationContext,
    AllocationInputs,
    AllocationProposal,
)


def _context(
    executor: MarketExecutor,
    book: BookState,
    order_book: OrderBookState,
    *,
    account_id: str,
    pretrade_risk: PreTradeRisk,
    symbol_index: int,
    start_index: int,
) -> AllocationContext:
    dataset = executor.dataset
    if (
        isinstance(start_index, bool)
        or not isinstance(start_index, int)
        or not 0 <= start_index < dataset.n_bars - 1
    ):
        raise ValueError("decision index is outside the executable dataset")
    if (
        isinstance(symbol_index, bool)
        or not isinstance(symbol_index, int)
        or not 0 <= symbol_index < dataset.n_symbols
    ):
        raise ValueError("selected symbol is outside the dataset")
    if executor.cost.order_type != "market" or executor.cost.order_latency_bars != 0:
        raise ValueError(
            "allocation v1 supports only MARKET orders with zero extra latency"
        )
    if (
        not pretrade_risk.config.drawdown_start
        < pretrade_risk.config.drawdown_stop
        <= 0.20
    ):
        raise ValueError("allocation requires the 20% drawdown risk guardrail")
    if (
        book.termination_reason is not None
        or book.insolvent
        or book.portfolio_value <= 0.0
    ):
        raise ValueError("allocation requires a live, solvent account")
    if book.quantities.shape != (dataset.n_symbols,) or not np.array_equal(
        np.asarray(book.contract_multipliers),
        dataset.resolved_array("contract_multipliers"),
    ):
        raise ValueError("book shape or contract multipliers do not match dataset")
    if not np.array_equal(
        book.mark_prices, dataset.resolved_array("mark_price")[start_index]
    ):
        raise ValueError("book marks do not match the current decision valuation")
    margin_check = book.clone()
    executor._update_margin(margin_check)
    if margin_check.termination_reason is not None or margin_check.insolvent:
        raise ValueError("canonical margin check rejects this initial account")
    margin_check.refresh_drawdown()
    if book.max_drawdown + 1e-12 < margin_check.max_drawdown:
        raise ValueError("book drawdown evidence has not been refreshed")
    quantities = book.exact_quantities
    if any(quantity != 0 for i, quantity in enumerate(quantities) if i != symbol_index):
        raise ValueError("allocation v1 supports only independent_symbol accounts")
    for order in order_book.active_orders:
        if order.intent.symbol_index != symbol_index:
            raise ValueError("independent account has another symbol's pending order")
        if order.intent.order_type is not OrderType.MARKET:
            raise ValueError("allocation v1 rejects non-MARKET/protective orders")
        if order.intent.dataset_id != dataset.dataset_id:
            raise ValueError("pending order belongs to another dataset")
        if (
            order.intent.submit_index > start_index
            or order.intent.eligible_index > start_index + 1
            or (
                order.intent.expiry_index is not None
                and order.intent.expiry_index < start_index + 1
            )
            or (
                order.last_processed_index is not None
                and order.last_processed_index > start_index
            )
            or (order.trigger_index is not None and order.trigger_index > start_index)
        ):
            raise ValueError("pending order clock is incompatible with this decision")
    if any(
        order.intent.submit_index > start_index
        or (
            order.last_processed_index is not None
            and order.last_processed_index > start_index
        )
        or (order.trigger_index is not None and order.trigger_index > start_index)
        for order in order_book.terminal_orders
    ):
        raise ValueError("terminal order clock is ahead of the decision")
    # Bind the full canonical book, including accepted exact quantity evidence.
    # No free-cash reservation, reconstructed P&L or second ledger is invented.
    book_facts: dict[str, object] = {}
    for field in fields(book):
        if field.name == "_exact_quantities":
            continue
        value = getattr(book, field.name)
        book_facts[field.name] = (
            value.tolist() if isinstance(value, np.ndarray) else value
        )
    book_facts["exact_quantities"] = tuple(str(q) for q in quantities)
    state_digest = content_digest(
        {
            "schema": "independent_allocation_context_v1",
            "account_id": account_id,
            "dataset_id": dataset.dataset_id,
            "execution_policy": executor.execution_policy_digest,
            "risk_config": pretrade_risk.config,
            "symbol_index": symbol_index,
            "decision_index": start_index,
            "decision_time": str(dataset.timestamps[start_index]),
            "book": book_facts,
            "active_orders": tuple(asdict(order) for order in order_book.active_orders),
            "terminal_orders": tuple(
                asdict(order) for order in order_book.terminal_orders
            ),
        }
    )
    remaining = sum(
        (Fraction(str(order.remaining_quantity)) for order in order_book.active_orders),
        Fraction(0),
    )
    return AllocationContext(
        account_id=account_id,
        symbol=dataset.symbols[symbol_index],
        decision_time=dataset.timestamps[start_index],
        state_digest=state_digest,
        current_weight=float(book.weights[symbol_index]),
        quantity=str(quantities[symbol_index]),
        cash=book.cash,
        equity=book.portfolio_value,
        pending_remaining=str(remaining),
    )


def propose_nonrl_target(
    executor: MarketExecutor,
    book: BookState,
    order_book: OrderBookState,
    *,
    account_id: str,
    inputs: AllocationInputs,
    allocator: AfterCostTargetAllocator,
    pretrade_risk: PreTradeRisk,
    symbol_index: int,
    start_index: int,
) -> AllocationProposal:
    """Optimize against actual holdings with static limits before hard risk."""
    context = _context(
        executor,
        book,
        order_book,
        account_id=account_id,
        pretrade_risk=pretrade_risk,
        symbol_index=symbol_index,
        start_index=start_index,
    )
    if inputs.horizon_end < executor.dataset.timestamps[start_index + 1]:
        raise ValueError("return horizon ends before the first processing bar")
    cap = min(pretrade_risk.config.max_abs_weight, executor.cost.max_leverage)
    # Static bounds are intersected here; drawdown projection occurs only once,
    # at execution, so a held position is not implicitly scaled twice.
    resolved = replace(
        allocator,
        lower_weight=max(allocator.lower_weight, -cap),
        upper_weight=min(allocator.upper_weight, cap),
    )
    return resolved.propose(inputs, context)


@dataclass(frozen=True, slots=True)
class NonRLExecutionResult:
    proposal: AllocationProposal
    risk_target: RiskConstrainedTarget
    execution: StatefulExecutionResult


def execute_nonrl_proposal(
    executor: MarketExecutor,
    book: BookState,
    order_book: OrderBookState,
    proposal: AllocationProposal,
    *,
    account_id: str,
    pretrade_risk: PreTradeRisk,
    start_index: int,
) -> NonRLExecutionResult:
    """Reject changed context, apply final risk once, then advance one bar."""
    if proposal.context.symbol not in executor.dataset.symbols:
        raise ValueError("proposal symbol does not belong to this dataset")
    symbol_index = executor.dataset.symbols.index(proposal.context.symbol)
    fresh = propose_nonrl_target(
        executor,
        book,
        order_book,
        account_id=account_id,
        inputs=proposal.inputs,
        allocator=proposal.allocator,
        pretrade_risk=pretrade_risk,
        symbol_index=symbol_index,
        start_index=start_index,
    )
    if fresh != proposal:
        raise ValueError("stale or altered allocation proposal/context")
    target, execution = _execute_allocation_target(
        executor,
        book,
        order_book,
        target_weight=proposal.target_weight,
        decision_digest=proposal.decision_digest,
        pretrade_risk=pretrade_risk,
        symbol_index=symbol_index,
        start_index=start_index,
    )
    return NonRLExecutionResult(proposal, target, execution)


def _execute_allocation_target(
    executor: MarketExecutor,
    book: BookState,
    order_book: OrderBookState,
    *,
    target_weight: float,
    decision_digest: str,
    pretrade_risk: PreTradeRisk,
    symbol_index: int,
    start_index: int,
) -> tuple[RiskConstrainedTarget, StatefulExecutionResult]:
    """Execute an admitted scalar decision through final risk exactly once."""
    proposed_weights = np.zeros(executor.dataset.n_symbols, dtype=np.float64)
    proposed_weights[symbol_index] = target_weight
    target = pretrade_risk.constrain(
        proposed_weights, current=book.weights, drawdown=book.max_drawdown
    )
    if np.array_equal(target.weights, book.weights):
        execution = execute_quantity_hold_statefully(
            executor,
            book,
            order_book,
            symbol_index=symbol_index,
            start_index=start_index,
        )
    else:
        identity = content_digest(
            {
                "decision": decision_digest,
                "target": target.weights.tolist(),
            }
        )
        execution = execute_target_statefully(
            executor,
            book,
            order_book,
            target.weights,
            start_index=start_index,
            bars=1,
            target_identity=identity,
        )
    return target, execution


__all__ = ["NonRLExecutionResult", "propose_nonrl_target", "execute_nonrl_proposal"]
