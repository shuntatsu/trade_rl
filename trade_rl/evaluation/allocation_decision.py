"""Verified common allocation decisions and canonical action transitions."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from trade_rl.evaluation.allocation import _execute_allocation_target
from trade_rl.evaluation.forecast_allocation import (
    HorizonCostEstimates,
    propose_forecast_target,
)
from trade_rl.risk.pretrade import PreTradeRisk, RiskConstrainedTarget
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import MarketExecutor
from trade_rl.simulation.orders.model import OrderBookState
from trade_rl.simulation.stateful.execution import StatefulExecutionResult
from trade_rl.strategies.allocation import AfterCostTargetAllocator
from trade_rl.strategies.allocation_action import (
    AllocationActionContract,
    AllocationActionProposal,
    AllocationDecision,
    _number,
)
from trade_rl.strategies.dataset_scope import validated_feature_indices
from trade_rl.strategies.forecasts.simple_stream import FrozenSimpleReturnStream


def prepare_allocation_decision(
    executor: MarketExecutor,
    book: BookState,
    order_book: OrderBookState,
    *,
    account_id: str,
    stream: FrozenSimpleReturnStream,
    estimates: HorizonCostEstimates,
    allocator: AfterCostTargetAllocator,
    pretrade_risk: PreTradeRisk,
    symbol_index: int,
    start_index: int,
    expected_horizon_seconds: int,
    action_contract: AllocationActionContract,
    initial_capital: float,
    remaining_steps: int,
    feature_indices: tuple[int, ...],
) -> AllocationDecision:
    """Admit the forecast baseline and this decision's selected RL features."""
    baseline = propose_forecast_target(
        executor,
        book,
        order_book,
        account_id=account_id,
        stream=stream,
        estimates=estimates,
        allocator=allocator,
        pretrade_risk=pretrade_risk,
        symbol_index=symbol_index,
        start_index=start_index,
        expected_horizon_seconds=expected_horizon_seconds,
    )
    dataset = executor.dataset
    indices = validated_feature_indices(dataset, feature_indices)
    if (
        not np.all(dataset.feature_available[start_index, symbol_index, list(indices)])
        or dataset.resolved_array("available_at")[start_index, symbol_index]
        > dataset.timestamps[start_index]
    ):
        raise ValueError(
            "selected decision features are unavailable at the current source clock"
        )
    gross = sum(
        (
            abs(Fraction(str(order.remaining_quantity)))
            for order in order_book.active_orders
        ),
        Fraction(0),
    )
    # A fixed-capital notional feature; this reserves no cash and updates no ledger.
    capital = _number(initial_capital, "initial_capital")
    if capital <= 0.0:
        raise ValueError("initial_capital must be finite and positive")
    pending_gross = (
        float(gross)
        * float(book.mark_prices[symbol_index])
        * float(np.asarray(book.contract_multipliers)[symbol_index])
    )
    return AllocationDecision(
        baseline=baseline,
        action_contract=action_contract,
        feature_names=tuple(dataset.feature_names[i] for i in indices),
        feature_values=tuple(
            float(v) for v in dataset.features[start_index, symbol_index, list(indices)]
        ),
        max_drawdown=book.max_drawdown,
        initial_capital=capital,
        remaining_steps=remaining_steps,
        pending_gross=pending_gross / capital,
        pending_count=len(order_book.active_orders),
    )


@dataclass(frozen=True, slots=True)
class AllocationActionExecutionResult:
    proposal: AllocationActionProposal
    risk_target: RiskConstrainedTarget
    execution: StatefulExecutionResult

    @property
    def decision(self) -> AllocationDecision:
        return self.proposal.decision

    @property
    def action(self) -> int:
        return self.proposal.raw_action


def execute_allocation_action(
    executor: MarketExecutor,
    book: BookState,
    order_book: OrderBookState,
    decision: AllocationDecision,
    action: object,
    *,
    account_id: str,
    stream: FrozenSimpleReturnStream,
    estimates: HorizonCostEstimates,
    allocator: AfterCostTargetAllocator,
    pretrade_risk: PreTradeRisk,
    symbol_index: int,
    start_index: int,
    expected_horizon_seconds: int,
    action_contract: AllocationActionContract,
    initial_capital: float,
    remaining_steps: int,
    feature_indices: tuple[int, ...],
) -> AllocationActionExecutionResult:
    """Rebind all inputs before applying the selected action and final hard risk."""
    fresh = prepare_allocation_decision(
        executor,
        book,
        order_book,
        account_id=account_id,
        stream=stream,
        estimates=estimates,
        allocator=allocator,
        pretrade_risk=pretrade_risk,
        symbol_index=symbol_index,
        start_index=start_index,
        expected_horizon_seconds=expected_horizon_seconds,
        action_contract=action_contract,
        initial_capital=initial_capital,
        remaining_steps=remaining_steps,
        feature_indices=feature_indices,
    )
    if not isinstance(decision, AllocationDecision) or fresh != decision:
        raise ValueError("stale or altered allocation decision/context")
    proposal = decision.propose(action)
    identity = (
        decision.baseline.decision_digest
        if proposal.target_weight == decision.baseline.target_weight
        else proposal.digest
    )
    risk_target, execution = _execute_allocation_target(
        executor,
        book,
        order_book,
        target_weight=proposal.target_weight,
        decision_digest=identity,
        pretrade_risk=pretrade_risk,
        symbol_index=symbol_index,
        start_index=start_index,
    )
    return AllocationActionExecutionResult(proposal, risk_target, execution)


__all__ = [
    "prepare_allocation_decision",
    "execute_allocation_action",
    "AllocationActionExecutionResult",
]
