from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tests.simulation.test_stateful_execution import _executor, _market, _zero_book
from trade_rl.rl.environment_execution import (
    EnvironmentExecutionCoordinator,
    ExecutionDualShadowRequest,
    ExecutionDualShadowSnapshot,
    TargetExecutionRequest,
)
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.orders import OrderBookState


@dataclass
class _FakeDualShadow:
    identity_digest: str = "d" * 64

    def __post_init__(self) -> None:
        self.requests: list[ExecutionDualShadowRequest] = []
        self.resets: list[tuple[int, float, tuple[float, ...]]] = []

    def reset(
        self,
        *,
        start_index: int,
        initial_capital: float,
        initial_quantities: tuple[float, ...],
    ) -> None:
        self.resets.append((start_index, initial_capital, initial_quantities))

    def observe(self, request: ExecutionDualShadowRequest) -> ExecutionDualShadowSnapshot:
        self.requests.append(request)
        return ExecutionDualShadowSnapshot(
            runtime_identity="fake",
            worker_pid=123,
            structural_parity=True,
            candidate_terminal_quantities=request.legacy_terminal_quantities,
            legacy_terminal_quantities=request.legacy_terminal_quantities,
        )


def test_execution_coordinator_observes_only_hybrid_target_without_changing_result(
) -> None:
    dataset = _market()
    observer = _FakeDualShadow()
    coordinator = EnvironmentExecutionCoordinator(
        dataset,
        ExecutionCostConfig.zero(),
        initial_capital=1_000.0,
        dual_shadow=observer,
    )
    coordinator.reset_dual_shadow(
        start_index=0,
        initial_quantities=(0.0,),
    )
    book = _zero_book(dataset)

    hybrid = coordinator.execute_target(
        executor=_executor(dataset),
        book=book,
        order_book=OrderBookState.empty(),
        request=TargetExecutionRequest(
            target=np.array([0.1]),
            start_index=0,
            decision_step_index=0,
            bars=1,
            book_kind="hybrid",
        ),
    )
    coordinator.execute_target(
        executor=_executor(dataset),
        book=book,
        order_book=OrderBookState.empty(),
        request=TargetExecutionRequest(
            target=np.array([0.0]),
            start_index=0,
            decision_step_index=0,
            bars=1,
            book_kind="shadow",
        ),
    )

    assert observer.resets == [(0, 1_000.0, (0.0,))]
    assert len(observer.requests) == 1
    request = observer.requests[0]
    assert request.target == (0.1,)
    assert request.start_index == 0
    assert request.end_index == hybrid.next_index
    assert request.allocated_equity == 1_000.0
    assert request.legacy_terminal_quantities == tuple(hybrid.book.quantities)
    assert coordinator.latest_dual_shadow_snapshot is not None
    assert coordinator.latest_dual_shadow_snapshot.structural_parity is True
