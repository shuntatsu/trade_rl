#!/usr/bin/env python3
"""Apply the reviewed ResidualMarketEnv service-delegation refactor once."""

from __future__ import annotations

from pathlib import Path

PATH = Path("trade_rl/rl/environment.py")


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, found {count}")
    return text.replace(old, new, 1)


def replace_region(
    text: str,
    start: str,
    end: str,
    replacement: str,
    *,
    label: str,
) -> str:
    start_index = text.find(start)
    end_index = text.find(end, start_index + len(start))
    if start_index < 0 or end_index < 0:
        raise RuntimeError(f"{label}: method boundary not found")
    return text[:start_index] + replacement + text[end_index:]


def main() -> None:
    text = PATH.read_text(encoding="utf-8")
    if "self._episode_sampler.sample" in text:
        print("ResidualMarketEnv decomposition already applied")
        return

    text = replace_once(
        text,
        '''from trade_rl.rl.episode import (
    complete_reward_history_steps,
    minimum_reward_start_index,
)
''',
        '''from trade_rl.rl.episode import (
    complete_reward_history_steps,
    minimum_reward_start_index,
)
from trade_rl.rl.environment_episode import EpisodeContractSampler
from trade_rl.rl.environment_execution import (
    EnvironmentExecutionCoordinator,
    TargetExecutionRequest,
)
from trade_rl.rl.environment_observation import (
    EnvironmentObservationAssembler,
    EnvironmentObservationRuntime,
)
from trade_rl.rl.environment_transition import EnvironmentTerminationCoordinator
''',
        label="service imports",
    )
    text = replace_once(
        text,
        '''from trade_rl.rl.observations import (
    OBSERVATION_SCHEMA,
    ObservationBuilder,
    ObservationExecutionState,
    ObservationInput,
    PendingOrderObservationState,
    PolicyObservationSnapshot,
    book_state_vector,
    observation_availability_mask,
    observation_passthrough_indices,
    observation_staleness_vector,
)
''',
        '''from trade_rl.rl.observations import (
    OBSERVATION_SCHEMA,
    ObservationBuilder,
    ObservationExecutionState,
    PendingOrderObservationState,
    PolicyObservationSnapshot,
    observation_passthrough_indices,
)
''',
        label="observation imports",
    )
    text = replace_once(
        text,
        '''from trade_rl.rl.sequence_observations import (
    SEQUENCE_OBSERVATION_SCHEMA,
    SequenceObservationBuilder,
    SequencePolicyPlane,
    SequenceWindowSpec,
    build_sequence_policy_plane,
    build_structured_current_observation,
    build_structured_policy_observation,
)
from trade_rl.rl.transition import classify_economic_transition
''',
        '''from trade_rl.rl.sequence_observations import (
    SEQUENCE_OBSERVATION_SCHEMA,
    SequenceObservationBuilder,
    SequencePolicyPlane,
    SequenceWindowSpec,
    build_sequence_policy_plane,
)
''',
        label="sequence imports",
    )
    text = replace_once(
        text,
        '''from trade_rl.simulation.order_reconciliation import reconcile_target
from trade_rl.simulation.orders import OrderBookState, OrderType, TimeInForce
''',
        '''from trade_rl.simulation.orders import OrderBookState
''',
        label="order imports",
    )
    text = replace_once(
        text,
        "\n_LIQUIDATION_TOLERANCE = 1e-12\n",
        "\n",
        label="local liquidation tolerance",
    )

    service_setup = '''        self._episode_sampler = EpisodeContractSampler(
            dataset,
            self.config,
            minimum_start_index=self._minimum_start_index,
        )
        self._execution_coordinator = EnvironmentExecutionCoordinator(
            dataset,
            self.config.execution_cost,
            initial_capital=self.config.initial_capital,
        )
        self._observation_assembler = EnvironmentObservationAssembler(
            dataset,
            observation_builder=self.observation_builder,
            layout=self.layout,
            normalizer=self.normalizer,
            sequence_observation_builder=self.sequence_observation_builder,
            sequence_policy_plane=self.sequence_policy_plane,
            sequence_normalizer=self.sequence_normalizer,
            action_size=self.action_spec.size,
            n_factors=self.action_spec.n_factors,
            finite_horizon=self.config.finite_horizon_observation,
        )
        self._termination_coordinator = EnvironmentTerminationCoordinator(
            config=self.config,
            reward_tracker=self.reward_tracker,
            pre_trade_risk=self.pre_trade_risk,
            hybrid_executor=self.hybrid_executor,
            shadow_executor=self.shadow_executor,
            execution_coordinator=self._execution_coordinator,
        )
'''
    text = replace_once(
        text,
        "        self._environment_digest = content_digest(self._digest_payload())\n",
        service_setup
        + "        self._environment_digest = content_digest(self._digest_payload())\n",
        label="service initialization",
    )
    text = replace_once(
        text,
        "        self._valid_start_cache: dict[tuple[float, int | None], np.ndarray] = {}\n",
        "",
        label="obsolete episode cache",
    )

    observation_methods = '''    def _observation_runtime(self) -> EnvironmentObservationRuntime:
        return EnvironmentObservationRuntime(
            current_index=self.current_index,
            start_index=self.start_index,
            end_index=self.end_index,
            hybrid=self.hybrid,
            shadow=self.shadow,
            hybrid_order_book=self._hybrid_order_book,
            execution_state=self._execution_state,
            previous_action=self._previous_action,
            pending_hybrid_target=self._pending_hybrid_target,
        )

    def _pending_order_observation_state(self) -> PendingOrderObservationState:
        return self._observation_assembler.pending_order_state(
            self._observation_runtime()
        )

    def _flat_observation_pair(self) -> tuple[np.ndarray, np.ndarray]:
        trends, alpha, factor_basis = self._market_inputs()
        return self._observation_assembler.flat_pair(
            self._observation_runtime(),
            trends=trends,
            alpha=alpha,
            factor_basis=factor_basis,
            pre_trade_risk=self.pre_trade_risk,
        )

    def observation_snapshot(self) -> PolicyObservationSnapshot:
        """Export the exact current training observation for serving parity."""

        if not self._has_reset:
            raise RuntimeError("environment must be reset before exporting observation")
        trends, alpha, factor_basis = self._market_inputs()
        return self._observation_assembler.snapshot(
            self._observation_runtime(),
            trends=trends,
            alpha=alpha,
            factor_basis=factor_basis,
            pre_trade_risk=self.pre_trade_risk,
            execution_policy_digest=self.execution_policy_digest,
        )

    def _observation(self) -> np.ndarray | dict[str, np.ndarray]:
        trends, alpha, factor_basis = self._market_inputs()
        return self._observation_assembler.observation(
            self._observation_runtime(),
            trends=trends,
            alpha=alpha,
            factor_basis=factor_basis,
            pre_trade_risk=self.pre_trade_risk,
        )

'''
    text = replace_region(
        text,
        "    def _pending_order_observation_state(",
        "    def _episode_end(",
        observation_methods,
        label="observation methods",
    )

    episode_methods = '''    def _episode_end(self, start: int, *, hours: float, bars: int | None) -> int:
        return self._episode_sampler.episode_end(start, hours=hours, bars=bars)

    def _valid_starts(self, *, hours: float, bars: int | None) -> np.ndarray:
        return self._episode_sampler.valid_starts(hours=hours, bars=bars)

    def _sample_episode_contract(
        self,
        options: dict[str, object],
    ) -> tuple[int, int, float]:
        contract = self._episode_sampler.sample(options, self.np_random)
        return contract.start_index, contract.end_index, contract.hours

'''
    text = replace_region(
        text,
        "    def _episode_end(",
        "    def _initial_weights(",
        episode_methods,
        label="episode methods",
    )

    execution_methods = '''    @staticmethod
    def _merge_liquidation_return(liquidation: ExecutionResult) -> BookState:
        return EnvironmentExecutionCoordinator.merge_liquidation_return(liquidation)

    @staticmethod
    def _liquidation_complete(liquidation: ExecutionResult) -> bool:
        return EnvironmentExecutionCoordinator.liquidation_complete(liquidation)

    def _execution_observation_state(
        self,
        *,
        requested_weights: np.ndarray,
        result: ExecutionResult | StatefulExecutionResult,
        previous_weights: np.ndarray,
    ) -> ObservationExecutionState:
        state, position_age = self._execution_coordinator.execution_observation_state(
            position_age=self._position_age,
            requested_weights=requested_weights,
            result=result,
            previous_weights=previous_weights,
        )
        self._position_age = position_age
        return state

    def _execute_stateful_target(
        self,
        *,
        executor: MarketExecutor,
        book: BookState,
        order_book: OrderBookState,
        target: np.ndarray,
        bars: int,
        book_kind: str,
    ) -> StatefulExecutionResult:
        return self._execution_coordinator.execute_target(
            executor=executor,
            book=book,
            order_book=order_book,
            request=TargetExecutionRequest(
                target=target,
                start_index=self.current_index,
                decision_step_index=self._decision_step_index,
                bars=bars,
                book_kind=book_kind,
            ),
        )

'''
    text = replace_region(
        text,
        "    @staticmethod\n    def _merge_liquidation_return(",
        "    def step(",
        execution_methods,
        label="execution methods",
    )

    old_transition_start = '''        hybrid_log_return = hybrid_execution.interval_log_return
        shadow_log_return = shadow_execution.interval_log_return
        hybrid_liquidation: ExecutionResult | None = None
'''
    old_transition_end = '''        terminated = economic_transition.terminated
        truncated = economic_transition.truncated
'''
    start_index = text.find(old_transition_start)
    end_index = text.find(old_transition_end, start_index)
    if start_index < 0 or end_index < 0:
        raise RuntimeError("step transition block not found")
    end_index += len(old_transition_end)
    transition_block = '''        transition = self._termination_coordinator.resolve(
            hybrid=self.hybrid,
            shadow=self.shadow,
            current_index=self.current_index,
            time_limit_reached=time_limit_reached,
            hybrid_log_return=hybrid_execution.interval_log_return,
            shadow_log_return=shadow_execution.interval_log_return,
        )
        self.hybrid = transition.hybrid
        self.shadow = transition.shadow
        hybrid_log_return = transition.hybrid_log_return
        shadow_log_return = transition.shadow_log_return
        hybrid_liquidation = transition.hybrid_liquidation
        shadow_liquidation = transition.shadow_liquidation
        liquidation_complete = transition.liquidation_complete
        liquidation_terminal = transition.liquidation_terminal
        emergency_deleverage = transition.emergency_deleverage
        hybrid_terminated = transition.hybrid_terminated
        shadow_terminated = transition.shadow_terminated
        economic_transition = transition.economic_transition
        terminated = economic_transition.terminated
        truncated = economic_transition.truncated
'''
    text = text[:start_index] + transition_block + text[end_index:]

    PATH.write_text(text, encoding="utf-8")
    print("Applied ResidualMarketEnv service decomposition")


if __name__ == "__main__":
    main()
