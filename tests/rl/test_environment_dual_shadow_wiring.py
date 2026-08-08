from __future__ import annotations

from dataclasses import dataclass

from tests.rl.test_environment_identity import market
from trade_rl.rl.dual_shadow_environment import ExecutionDualShadowResidualMarketEnv
from trade_rl.rl.environment import ResidualMarketEnvConfig
from trade_rl.rl.environment_execution import (
    ExecutionDualShadowRequest,
    ExecutionDualShadowSnapshot,
)
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.trend import TrendConfig, TrendStrategy


@dataclass
class _Observer:
    identity_digest: str

    def __post_init__(self) -> None:
        self.resets: list[tuple[int, float, tuple[float, ...]]] = []
        self.close_calls = 0

    def reset(
        self,
        *,
        start_index: int,
        initial_capital: float,
        initial_quantities: tuple[float, ...],
    ) -> None:
        self.resets.append((start_index, initial_capital, initial_quantities))

    def observe(
        self, request: ExecutionDualShadowRequest
    ) -> ExecutionDualShadowSnapshot:
        return ExecutionDualShadowSnapshot(
            runtime_identity="fake",
            worker_pid=1,
            structural_parity=True,
            candidate_terminal_quantities=request.legacy_terminal_quantities,
            legacy_terminal_quantities=request.legacy_terminal_quantities,
        )

    def close(self) -> None:
        self.close_calls += 1


def _environment(observer: _Observer) -> ExecutionDualShadowResidualMarketEnv:
    return ExecutionDualShadowResidualMarketEnv(
        market(),
        trend_strategy=TrendStrategy(
            TrendConfig(fast_lookback=2, base_lookback=4, slow_lookback=8)
        ),
        config=ResidualMarketEnvConfig(
            initial_capital=100_000.0,
            episode_bars=8,
            decision_every=2,
            execution_cost=ExecutionCostConfig.zero(),
            initial_state_modes=("cash",),
        ),
        execution_dual_shadow=observer,
    )


def test_environment_resets_dual_shadow_from_actual_initial_book() -> None:
    observer = _Observer("a" * 64)
    env = _environment(observer)

    env.reset(seed=7, options={"initial_state_mode": "cash"})

    assert observer.resets == [
        (env.start_index, env.initial_capital, (0.0, 0.0)),
    ]
    assert env.latest_execution_dual_shadow is None


def test_environment_identity_includes_dual_shadow_identity() -> None:
    first = _environment(_Observer("a" * 64))
    second = _environment(_Observer("b" * 64))

    assert first.environment_digest != second.environment_digest


def test_environment_close_releases_dual_shadow_runtime() -> None:
    observer = _Observer("c" * 64)
    env = _environment(observer)

    env.close()

    assert observer.close_calls == 1
