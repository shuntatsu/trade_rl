"""PPO-compatible allocation environment over canonical book/order transitions."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.contracts import MarketCalendarKind
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.allocation_decision import (
    execute_allocation_action,
    prepare_allocation_decision,
)
from trade_rl.evaluation.allocation_snapshot import snapshot_allocation_account
from trade_rl.evaluation.forecast_allocation import HorizonCostEstimates
from trade_rl.evaluation.objectives import BoundObjectiveClock, net_equity_increment
from trade_rl.evaluation.rl_allocation.binding import validate_runtime_binding
from trade_rl.risk import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation import BookState, MarketExecutor
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.orders.model import OrderBookState
from trade_rl.strategies.allocation import AfterCostTargetAllocator
from trade_rl.strategies.allocation_action import (
    AllocationActionContract,
    AllocationDecision,
)
from trade_rl.strategies.forecasts.simple_stream import FrozenSimpleReturnStream
from trade_rl.strategies.rl.allocation_observation_encoder_v2 import (
    encode_allocation_observation_v2,
)
from trade_rl.strategies.rl.allocation_observation_v2 import AllocationObservationSchema
from trade_rl.strategies.rl.allocation_policy import (
    ALLOCATION_OBSERVATION_FIELDS,
    AllocationRuntimeProfile,
    allocation_recipe_payload,
    encode_allocation_observation,
)
from trade_rl.strategies.rl.allocation_recipe_v2 import allocation_recipe_payload_v2


class AllocationTradingEnv(gym.Env):
    """One actual independent account; fixed-denominator profit reward, gamma=1.

    Decisions are [start_index, stop_index). Processing closes at stop_index.
    Finite-horizon termination retains marked holdings/pending orders explicitly.
    A rollout buffer boundary never resets or settles this environment.
    """

    def __init__(
        self,
        dataset: MarketDataset,
        *,
        stream: FrozenSimpleReturnStream,
        estimates: tuple[HorizonCostEstimates, ...],
        bound: BoundObjectiveClock,
        action_contract: AllocationActionContract,
        allocator: AfterCostTargetAllocator,
        execution_cost: ExecutionCostConfig,
        risk_config: PreTradeRiskConfig,
        feature_indices: tuple[int, ...],
        symbol_index: int,
        start_index: int,
        stop_index: int,
        account_id: str,
        expected_horizon_seconds: int = 3600,
        observation_schema: AllocationObservationSchema | None = None,
    ) -> None:
        self.dataset, self.stream, self.bound = dataset, stream, bound
        self._bound_digest = bound.digest
        self.action_contract, self.allocator = action_contract, allocator
        self.execution_cost, self.risk_config = execution_cost, risk_config
        self.feature_indices = tuple(feature_indices)
        self.observation_schema = observation_schema
        if (
            not self.feature_indices
            or len(set(self.feature_indices)) != len(self.feature_indices)
            or any(
                isinstance(i, bool)
                or not isinstance(i, int)
                or not 0 <= i < dataset.n_features
                for i in self.feature_indices
            )
        ):
            raise ValueError(
                "allocation observation features must be unique valid indices"
            )
        if (
            isinstance(symbol_index, bool)
            or not isinstance(symbol_index, int)
            or not 0 <= symbol_index < dataset.n_symbols
        ):
            raise ValueError("allocation symbol index is outside the Dataset")
        self.symbol_index, self.start_index, self.stop_index = (
            symbol_index,
            start_index,
            stop_index,
        )
        self.account_id, self.expected_horizon_seconds = (
            account_id,
            expected_horizon_seconds,
        )
        self.executor = MarketExecutor(
            dataset, execution_cost, insolvency_valuation="retain_debt"
        )
        self.risk = PreTradeRisk(risk_config)
        validate_runtime_binding(
            dataset,
            bound,
            self.executor,
            risk_config,
            self.recipe_digest,
            start_index=start_index,
            stop_index=stop_index,
        )
        self.initial_capital = bound.objective.capital.initial_equities[0]
        self._estimates = {int(c.decision_time.astype(np.int64)): c for c in estimates}
        if (
            len(self._estimates) != len(estimates)
            or len(estimates) != stop_index - start_index
        ):
            raise ValueError("cost declarations must cover each decision exactly once")
        expected_times = {
            int(t.astype("datetime64[ns]").astype(np.int64))
            for t in dataset.timestamps[start_index:stop_index]
        }
        if set(self._estimates) != expected_times:
            raise ValueError("cost declarations must cover the actual decision clocks")
        self._observation_shape = (
            len(self.feature_indices) + len(ALLOCATION_OBSERVATION_FIELDS)
            if self.observation_schema is None
            else len(self.observation_schema.fields),
        )
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=self._observation_shape,
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(4)
        self._terminated = True

    @property
    def recipe(self) -> dict[str, object]:
        capital, clock = self.bound.objective.capital, self.bound.clock
        arguments: dict[str, Any] = {}
        builder: Callable[..., dict[str, object]] = allocation_recipe_payload
        if self.observation_schema is not None:
            builder = allocation_recipe_payload_v2
            arguments["observation_schema"] = self.observation_schema
        return builder(
            self.action_contract,
            tuple(self.dataset.feature_names[i] for i in self.feature_indices),
            allocator=self.allocator,
            expected_horizon_seconds=self.expected_horizon_seconds,
            runtime_profile=AllocationRuntimeProfile(
                self.executor.execution_policy_digest,
                content_digest(self.risk.config),
                capital.initial_equities[0],
                capital.currency,
                clock.decision_interval_seconds,
                clock.economic_horizon_seconds,
                calendar_kind=MarketCalendarKind(self.dataset.calendar_kind).value,
                execution_bar_hours=self.dataset.bar_hours,
            ),
            **arguments,
        )

    @property
    def recipe_digest(self) -> str:
        return content_digest(self.recipe)

    def validate_binding(self) -> None:
        if self.executor.dataset is not self.dataset:
            raise ValueError("actual executor Dataset differs from declared runtime")
        if (
            self.executor.cost != self.execution_cost
            or self.risk.config != self.risk_config
        ):
            raise ValueError(
                "actual executor/risk profile differs from declared runtime"
            )
        if self.bound.digest != self._bound_digest:
            raise ValueError("bound objective or capital changed after construction")
        if self.initial_capital != self.bound.objective.capital.initial_equities[0]:
            raise ValueError("actual initial capital differs from bound objective")
        validate_runtime_binding(
            self.dataset,
            self.bound,
            self.executor,
            self.risk.config,
            self.recipe_digest,
            start_index=self.start_index,
            stop_index=self.stop_index,
        )

    def _arguments(self) -> dict[str, Any]:
        decision_time = self.dataset.timestamps[self.index].astype("datetime64[ns]")
        try:
            estimate = self._estimates[int(decision_time.astype(np.int64))]
        except KeyError as error:
            raise ValueError(
                "cost declaration missing at the current decision"
            ) from error
        return dict(
            account_id=self.account_id,
            stream=self.stream,
            estimates=estimate,
            allocator=self.allocator,
            pretrade_risk=self.risk,
            symbol_index=self.symbol_index,
            start_index=self.index,
            expected_horizon_seconds=self.expected_horizon_seconds,
            action_contract=self.action_contract,
            initial_capital=self.initial_capital,
            remaining_steps=self.stop_index - self.index,
            feature_indices=self.feature_indices,
        )

    def _prepare(self) -> AllocationDecision:
        return prepare_allocation_decision(
            self.executor, self.book, self.order_book, **self._arguments()
        )

    def _observation(self) -> np.ndarray:
        self.decision = self._prepare()
        if self.observation_schema is not None:
            snapshot = snapshot_allocation_account(
                self.executor,
                self.book,
                self.order_book,
                account_id=self.account_id,
                pretrade_risk=self.risk,
                symbol_index=self.symbol_index,
                start_index=self.index,
            )
            return encode_allocation_observation_v2(
                snapshot,
                self.decision,
                schema=self.observation_schema,
                episode_steps=self.stop_index - self.start_index,
            )
        return encode_allocation_observation(
            self.decision, episode_steps=self.stop_index - self.start_index
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        super().reset(seed=seed)
        if self.bound.digest != self._bound_digest:
            raise ValueError("bound objective or capital changed after construction")
        self.executor = MarketExecutor(
            self.dataset, self.execution_cost, insolvency_valuation="retain_debt"
        )
        # Policy/Gym seeds must not alter the declared execution economics.
        self.executor.reset_random_state(self.execution_cost.random_seed)
        self.risk = PreTradeRisk(self.risk_config)
        self.validate_binding()
        self.book = BookState(
            np.zeros(self.dataset.n_symbols),
            self.initial_capital,
            self.dataset.resolved_array("mark_price")[self.start_index],
            self.initial_capital,
            self.dataset.resolved_array("contract_multipliers"),
            as_of_index=self.start_index,
            as_of_dataset_id=self.dataset.dataset_id,
        )
        if self.observation_schema is not None:
            self.executor._update_margin(self.book)
        self.order_book, self.index = OrderBookState.empty(), self.start_index
        self._terminated = False
        return self._observation(), {"symbol": self.dataset.symbols[self.symbol_index]}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._terminated:
            raise RuntimeError("allocation episode is terminal; reset before step")
        self.validate_binding()
        before = self.book.portfolio_value
        result = execute_allocation_action(
            self.executor,
            self.book,
            self.order_book,
            self.decision,
            action,
            **self._arguments(),
        )
        self.book, self.order_book, self.index = (
            result.execution.book,
            result.execution.order_book,
            result.execution.next_index,
        )
        reward = net_equity_increment(
            before, self.book.portfolio_value, initial_capital=self.initial_capital
        )
        self._terminated = (
            self.index == self.stop_index or self.book.termination_reason is not None
        )
        observation = (
            np.zeros(self._observation_shape, dtype=np.float32)
            if self._terminated
            else self._observation()
        )
        return (
            observation,
            reward,
            self._terminated,
            False,
            {
                "execution": result.execution,
                "action_proposal": result.proposal,
                "risk_target": result.risk_target,
                "terminal_valuation": self.bound.objective.terminal_valuation,
            },
        )
