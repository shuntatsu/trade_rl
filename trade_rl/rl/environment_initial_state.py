"""Fresh invocation-local mutable state for ``ResidualMarketEnv``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.rl.actions import ActionSpec
from trade_rl.rl.diagnostics import ActionDiagnosticsAccumulator
from trade_rl.rl.environment_config import ResidualMarketEnvConfig
from trade_rl.rl.observations import ObservationExecutionState
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.orders import OrderBookState


@dataclass(frozen=True, slots=True)
class EnvironmentInitialStateRequest:
    """Validated inputs required to build one environment's initial state."""

    dataset: MarketDataset
    config: ResidualMarketEnvConfig
    action_spec: ActionSpec
    minimum_start_index: int


@dataclass(frozen=True, slots=True)
class EnvironmentInitialState:
    """Fresh values copied into the maintained environment facade."""

    start_index: int
    end_index: int
    current_index: int
    hybrid: BookState
    shadow: BookState
    decision_step_index: int
    episode_seed: int
    episode_hours: float
    initial_state_mode: str
    previous_action: np.ndarray
    pending_hybrid_target: np.ndarray | None
    pending_shadow_target: np.ndarray | None
    hybrid_order_book: OrderBookState
    shadow_order_book: OrderBookState
    position_age: np.ndarray
    execution_state: ObservationExecutionState
    action_diagnostics: ActionDiagnosticsAccumulator
    has_reset: bool


class EnvironmentInitialStateFactory:
    """Create independent mutable values for one environment instance."""

    @staticmethod
    def create(request: EnvironmentInitialStateRequest) -> EnvironmentInitialState:
        dataset = request.dataset
        start_index = request.minimum_start_index
        initial_prices = dataset.close[start_index]
        hybrid = BookState.zero(
            dataset.n_symbols,
            request.config.initial_capital,
            initial_prices,
            contract_multipliers=dataset.resolved_array("contract_multipliers"),
        )
        return EnvironmentInitialState(
            start_index=start_index,
            end_index=start_index + 1,
            current_index=start_index,
            hybrid=hybrid,
            shadow=hybrid.clone(),
            decision_step_index=0,
            episode_seed=request.config.execution_cost.random_seed,
            episode_hours=request.config.episode_hours,
            initial_state_mode="cash",
            previous_action=np.zeros(request.action_spec.size, dtype=np.float32),
            pending_hybrid_target=None,
            pending_shadow_target=None,
            hybrid_order_book=OrderBookState.empty(),
            shadow_order_book=OrderBookState.empty(),
            position_age=np.zeros(dataset.n_symbols, dtype=np.float64),
            execution_state=ObservationExecutionState.zero(dataset.n_symbols),
            action_diagnostics=ActionDiagnosticsAccumulator(),
            has_reset=False,
        )


__all__ = [
    "EnvironmentInitialState",
    "EnvironmentInitialStateFactory",
    "EnvironmentInitialStateRequest",
]
