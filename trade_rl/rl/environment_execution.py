"""Stateful target execution services for the residual market environment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.rl.observations import ObservationExecutionState
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation import MarketExecutor
from trade_rl.simulation.execution import ExecutionCostConfig, ExecutionResult
from trade_rl.simulation.orders import OrderBookState
from trade_rl.simulation.stateful_execution import StatefulExecutionResult
from trade_rl.simulation.target_execution import execute_target_statefully

_LIQUIDATION_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class TargetExecutionRequest:
    """Immutable identity and timing inputs for one target execution."""

    target: np.ndarray
    start_index: int
    decision_step_index: int
    bars: int
    book_kind: str


class EnvironmentExecutionCoordinator:
    """Reconcile targets and execute them through the maintained order engine."""

    def __init__(
        self,
        dataset: MarketDataset,
        execution_cost: ExecutionCostConfig,
        *,
        initial_capital: float,
    ) -> None:
        if initial_capital <= 0.0:
            raise ValueError("initial_capital must be positive")
        self.dataset = dataset
        self.execution_cost = execution_cost
        self.initial_capital = float(initial_capital)

    def execute_target(
        self,
        *,
        executor: MarketExecutor,
        book: BookState,
        order_book: OrderBookState,
        request: TargetExecutionRequest,
    ) -> StatefulExecutionResult:
        target_vector = np.asarray(request.target, dtype=np.float64).reshape(-1)
        if target_vector.shape != (self.dataset.n_symbols,):
            raise ValueError("target does not match dataset symbols")
        if request.bars <= 0:
            raise ValueError("bars must be positive")
        target_identity = content_digest(
            {
                "book_kind": request.book_kind,
                "dataset_id": self.dataset.dataset_id,
                "decision_step_index": request.decision_step_index,
                "submit_index": request.start_index,
                "target": tuple(float(value) for value in target_vector),
                "schema_version": "environment_order_target_v1",
            }
        )
        return execute_target_statefully(
            executor,
            book,
            order_book,
            target_vector,
            start_index=request.start_index,
            bars=request.bars,
            target_identity=target_identity,
        )

    @staticmethod
    def merge_liquidation_return(liquidation: ExecutionResult) -> BookState:
        result = liquidation.book
        if abs(liquidation.interval_net_return) <= 1e-15:
            return result
        if result.returns_history:
            previous = result.returns_history[-1]
            result.returns_history[-1] = (1.0 + previous) * (
                1.0 + liquidation.interval_net_return
            ) - 1.0
        else:
            result.returns_history.append(liquidation.interval_net_return)
        result.peak_value = max(result.peak_value, result.portfolio_value)
        result.max_drawdown = max(
            result.max_drawdown,
            1.0 - max(result.portfolio_value, 0.0) / max(result.peak_value, 1e-12),
        )
        return result

    @staticmethod
    def liquidation_complete(liquidation: ExecutionResult) -> bool:
        return bool(
            liquidation.unfilled_turnover <= _LIQUIDATION_TOLERANCE
            and np.all(np.abs(liquidation.book.quantities) <= _LIQUIDATION_TOLERANCE)
        )

    def execution_observation_state(
        self,
        *,
        position_age: np.ndarray,
        requested_weights: np.ndarray,
        result: ExecutionResult | StatefulExecutionResult,
        previous_weights: np.ndarray,
    ) -> tuple[ObservationExecutionState, np.ndarray]:
        requested = np.asarray(requested_weights, dtype=np.float64).reshape(-1)
        requested_notional = result.requested_notional_by_symbol
        filled_notional = result.filled_notional_by_symbol
        if requested_notional.shape != (self.dataset.n_symbols,):
            requested_notional = np.zeros(self.dataset.n_symbols, dtype=np.float64)
        if filled_notional.shape != (self.dataset.n_symbols,):
            filled_notional = np.zeros(self.dataset.n_symbols, dtype=np.float64)
        fill_ratio = np.ones(self.dataset.n_symbols, dtype=np.float64)
        positive = requested_notional > 1e-12
        fill_ratio[positive] = np.minimum(
            1.0,
            filled_notional[positive] / requested_notional[positive],
        )
        total_requested = max(float(requested_notional.sum()), 1e-12)
        unfilled = (
            np.maximum(requested_notional - filled_notional, 0.0) / total_requested
        )
        participation = result.participation_by_symbol
        if participation.shape != (self.dataset.n_symbols,):
            participation = np.zeros(self.dataset.n_symbols, dtype=np.float64)
        cost = result.cost_by_symbol
        if cost.shape != (self.dataset.n_symbols,):
            cost = np.zeros(self.dataset.n_symbols, dtype=np.float64)
        cost = cost / max(self.initial_capital, 1e-12)
        current_weights = result.book.weights
        changed = np.abs(current_weights - previous_weights) > 1e-10
        held = np.abs(current_weights) > 1e-10
        next_position_age = np.where(
            held,
            np.where(changed, 0.0, position_age + result.bars_advanced),
            0.0,
        )
        return (
            ObservationExecutionState(
                requested_weights=requested,
                fill_ratio=fill_ratio,
                unfilled_turnover=unfilled,
                participation=participation,
                execution_cost=cost,
                position_age=next_position_age.copy(),
            ),
            next_position_age,
        )
