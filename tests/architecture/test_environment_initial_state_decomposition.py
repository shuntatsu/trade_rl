from __future__ import annotations

import inspect

from trade_rl.rl.environment import ResidualMarketEnv
from trade_rl.rl.environment_initial_state import (
    EnvironmentInitialState,
    EnvironmentInitialStateFactory,
    EnvironmentInitialStateRequest,
)


def test_initial_state_module_owns_maintained_boundary() -> None:
    assert EnvironmentInitialStateRequest.__module__ == (
        "trade_rl.rl.environment_initial_state"
    )
    assert EnvironmentInitialState.__module__ == "trade_rl.rl.environment_initial_state"
    assert EnvironmentInitialStateFactory.__module__ == (
        "trade_rl.rl.environment_initial_state"
    )


def test_environment_constructor_delegates_initial_state_once() -> None:
    source = inspect.getsource(ResidualMarketEnv.__init__)

    assert source.count("EnvironmentInitialStateFactory.create(") == 1
    assert source.count("EnvironmentInitialStateRequest(") == 1
    for forbidden in (
        "BookState.zero(",
        "OrderBookState.empty(",
        "ObservationExecutionState.zero(",
        "ActionDiagnosticsAccumulator(",
        "np.zeros(self.action_spec.size",
        "np.zeros(dataset.n_symbols",
    ):
        assert forbidden not in source
    assert source.index("self._environment_digest =") < source.index(
        "EnvironmentInitialStateFactory.create("
    )
    assert len(source.splitlines()) <= 170


def test_initial_state_factory_owns_creation_order() -> None:
    source = inspect.getsource(EnvironmentInitialStateFactory.create)
    markers = (
        "start_index = request.minimum_start_index",
        "initial_prices = dataset.close[start_index]",
        "BookState.zero(",
        "shadow=hybrid.clone()",
        "previous_action=np.zeros(",
        "hybrid_order_book=OrderBookState.empty()",
        "shadow_order_book=OrderBookState.empty()",
        "position_age=np.zeros(",
        "execution_state=ObservationExecutionState.zero(",
        "action_diagnostics=ActionDiagnosticsAccumulator()",
    )

    positions = tuple(source.index(marker) for marker in markers)
    assert positions == tuple(sorted(positions))
